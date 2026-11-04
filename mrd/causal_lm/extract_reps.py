import logging
import os
import time
from pathlib import Path

import fire
import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from mrd.causal_lm.utils import (
    TASK_CONFIG,
    build_candidate_batch,
    collate_eval,
    get_label_id,
    load_task_dataset,
    normalize_task,
)
from mrd.common.logging import set_logger

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str):
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def resolve_layers(hidden_states_len: int, layers: list[int]) -> list[int]:
    out = []
    for layer in layers:
        idx = layer if layer >= 0 else hidden_states_len + layer
        if idx < 0 or idx >= hidden_states_len:
            raise ValueError(
                f"Layer index {layer} -> {idx} out of range "
                f"[0, {hidden_states_len - 1}]"
            )
        out.append(idx)

    seen = set()
    uniq = []
    for idx in out:
        if idx not in seen:
            uniq.append(idx)
            seen.add(idx)
    return uniq


def load_model_and_tokenizer(
    model_name_or_path: str,
    adapter_name_or_path: str | None,
    dtype: torch.dtype,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype,
    )

    if adapter_name_or_path is not None:
        logger.info(f"Loading adapter: {adapter_name_or_path}")
        model = PeftModel.from_pretrained(model, adapter_name_or_path)

    return model, tokenizer


def build_token_metadata(
    example_offset: int,
    gold: torch.Tensor,
    all_choices: list[list[str]],
    target_mask: torch.Tensor,
):
    row_example_idx = []
    row_choice_idx = []
    row_token_idx = []
    row_is_gold = []

    flat_idx = 0
    for i, choices in enumerate(all_choices):
        for j in range(len(choices)):
            mask = target_mask[flat_idx]
            positions = torch.nonzero(mask, as_tuple=False).flatten().tolist()

            for local_pos, _absolute_pos in enumerate(positions):
                row_example_idx.append(example_offset + i)
                row_choice_idx.append(j)
                row_token_idx.append(local_pos)
                row_is_gold.append(int(j == int(gold[i].item())))

            flat_idx += 1

    return {
        "row_example_idx": np.array(row_example_idx, dtype=np.int32),
        "row_choice_idx": np.array(row_choice_idx, dtype=np.int32),
        "row_token_idx": np.array(row_token_idx, dtype=np.int32),
        "row_is_gold": np.array(row_is_gold, dtype=np.int32),
    }


def main(
    model_name_or_path: str,
    task: str,
    adapter_name_or_path: str | None = None,
    split: str = "validation",
    n_eval: int | None = None,
    batch_size: int = 8,
    max_length: int = 512,
    seed: int = 42,
    device: str = "auto",
    dtype: str = "bfloat16",
    layers: list[int] | None = None,
    out: str | None = None,
    log_path: str | None = None,
):
    if layers is None:
        layers = [-1]

    task = normalize_task(task)
    if task not in TASK_CONFIG:
        raise ValueError(f"Unsupported task: {task}")

    if log_path is None:
        tag = model_name_or_path.replace("/", "--")
        if adapter_name_or_path is not None:
            adapter_tag = adapter_name_or_path.replace("/", "--")
            tag = f"{tag}__{adapter_tag}"
        log_path = Path("logs") / f"{Path(__file__).stem}_{tag}_{task}.log"
    else:
        log_path = Path(log_path)

    set_logger(str(log_path))

    if out is None:
        tag = model_name_or_path.replace("/", "--")
        out = f"reps/{tag}__{task}__{split}_seq.npz"

    os.makedirs(Path(out).parent, exist_ok=True)

    set_seed(seed)
    device_obj = resolve_device(device)
    torch_dtype = resolve_dtype(dtype)

    n_eval_msg = "full dataset" if n_eval is None else str(n_eval)

    logger.info("=" * 80)
    logger.info("Causal LM sequence-level representation extraction")
    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"Adapter: {adapter_name_or_path}")
    logger.info(f"Task: {task} split={split} n_eval={n_eval_msg}")
    logger.info(f"Layers request: {layers}")
    logger.info(f"Out: {out}")
    logger.info("=" * 80)

    start = time.time()

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        adapter_name_or_path=adapter_name_or_path,
        dtype=torch_dtype,
    )
    model.to(device_obj)
    model.eval()

    ds = load_task_dataset(task, split)
    if n_eval is not None:
        n_eval = min(n_eval, len(ds))
        ds = ds.shuffle(seed=seed).select(range(n_eval))

    logger.info(f"N examples loaded: {len(ds)}")

    labels = np.array([get_label_id(task, ex) for ex in ds], dtype=np.int32)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_eval(
            batch=batch,
            task=task,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
    )

    resolved_layers = None
    candidate_seq_buf = {}

    row_example_idx_buf = []
    row_choice_idx_buf = []
    row_token_idx_buf = []
    row_is_gold_buf = []

    example_offset = 0

    for prompts, all_choices, gold in loader:
        batch = build_candidate_batch(
            tokenizer=tokenizer,
            prompts=prompts,
            all_choices=all_choices,
            max_length=max_length,
        )

        input_ids = batch["input_ids"].to(device_obj)
        attention_mask = batch["attention_mask"].to(device_obj)

        # target_mask marks candidate tokens in input space.
        # For causal LM scoring, token at position p is predicted by hidden/logit at p-1. # noqa: E501
        # We therefore extract hidden states at positions p-1 for candidate tokens p.
        target_mask = batch["target_mask"].to(device_obj)
        score_hidden_mask = torch.zeros_like(target_mask)
        score_hidden_mask[:, :-1] = target_mask[:, 1:]

        with torch.no_grad():
            out_obj = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hs_all = out_obj.hidden_states

        if resolved_layers is None:
            resolved_layers = resolve_layers(len(hs_all), layers)
            logger.info(f"Resolved layers: {resolved_layers}")
            for layer in resolved_layers:
                candidate_seq_buf[layer] = []

        mask = score_hidden_mask.bool()

        for layer in resolved_layers:
            hidden = hs_all[layer]
            candidate_seq = hidden[mask]
            candidate_seq_buf[layer].append(
                candidate_seq.detach().cpu().float().numpy()
            )

        meta = build_token_metadata(
            example_offset=example_offset,
            gold=gold,
            all_choices=all_choices,
            target_mask=target_mask.cpu(),
        )

        row_example_idx_buf.append(meta["row_example_idx"])
        row_choice_idx_buf.append(meta["row_choice_idx"])
        row_token_idx_buf.append(meta["row_token_idx"])
        row_is_gold_buf.append(meta["row_is_gold"])

        example_offset += len(all_choices)

    packed = {
        "labels": labels,
        "layers": np.array(
            resolved_layers if resolved_layers is not None else [],
            dtype=np.int32,
        ),
        "row_example_idx": np.concatenate(row_example_idx_buf, axis=0),
        "row_choice_idx": np.concatenate(row_choice_idx_buf, axis=0),
        "row_token_idx": np.concatenate(row_token_idx_buf, axis=0),
        "row_is_gold": np.concatenate(row_is_gold_buf, axis=0),
    }

    for layer in resolved_layers or []:
        packed[f"candidate_seq_l{layer}"] = np.concatenate(
            candidate_seq_buf[layer],
            axis=0,
        )

    # Sanity check: metadata rows must match representation rows.
    rep_keys = [k for k in packed if k.startswith("candidate_seq_l")]
    for key in rep_keys:
        if packed[key].shape[0] != packed["row_example_idx"].shape[0]:
            raise ValueError(
                f"Row mismatch for {key}: "
                f"{packed[key].shape[0]} vs {packed['row_example_idx'].shape[0]}"
            )

    np.savez(out, **packed)

    logger.info(f"Saved: {out}")
    logger.info(f"N_examples={len(labels)}")
    logger.info(f"N_token_rows={packed['row_example_idx'].shape[0]}")
    logger.info(f"Done in {time.time() - start:.1f}s")

    return out


if __name__ == "__main__":
    fire.Fire(main)
