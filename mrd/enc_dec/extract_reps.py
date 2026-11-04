import logging
import os
import time
from pathlib import Path

import fire
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from mrd.common.logging import set_logger
from mrd.enc_dec.utils import (
    TASK_CONFIG,
    collate_eval,
    get_label_id,
    load_task_dataset,
    normalize_task,
)

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


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


def repeat_rows(x: torch.Tensor, counts: list[int]) -> torch.Tensor:
    idx = torch.arange(x.size(0), device=x.device)
    idx = torch.repeat_interleave(
        idx,
        torch.tensor(counts, device=x.device),
    )
    return x.index_select(0, idx)


def build_targets(tokenizer, all_choices, max_length: int):
    flat_choices = [c for choices in all_choices for c in choices]

    ids_list = []
    mask_list = []

    for text in flat_choices:
        tok = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = tok["input_ids"][0]
        ids_list.append(ids)
        mask_list.append(torch.ones_like(ids))

    target_ids = pad_sequence(
        ids_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    target_mask = pad_sequence(
        mask_list,
        batch_first=True,
        padding_value=0,
    )

    return target_ids, target_mask


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
            length = int(mask.sum().item())

            for pos in range(length):
                row_example_idx.append(example_offset + i)
                row_choice_idx.append(j)
                row_token_idx.append(pos)
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
    split: str = "validation",
    n_eval: int | None = None,
    batch_size: int = 16,
    max_length: int = 128,
    seed: int = 42,
    device: str = "auto",
    layers: list[int] | None = None,
    out: str | None = None,
    log_path: str | None = None,
):
    if layers is None:
        layers = [-1]

    task = normalize_task(task)
    if task not in TASK_CONFIG:
        raise ValueError(f"Unsupported task: {task}")

    tag = model_name_or_path.replace("/", "--")

    if log_path is None:
        log_path = Path("logs") / f"{Path(__file__).stem}_{tag}_{task}.log"
    else:
        log_path = Path(log_path)

    set_logger(str(log_path))

    if out is None:
        out = f"reps/{tag}__{task}__{split}.npz"

    os.makedirs(Path(out).parent, exist_ok=True)

    set_seed(seed)
    device_obj = resolve_device(device)

    logger.info(f"Using device: {device_obj}")
    logger.info("=" * 80)
    logger.info("T5 sequence-level representation extraction")
    logger.info(f"Model: {model_name_or_path}")
    n_eval_msg = "full dataset" if n_eval is None else str(n_eval)
    logger.info(f"Task: {task} split={split} n_eval={n_eval_msg}")
    logger.info(f"Layers request: {layers}")
    logger.info(f"Out: {out}")
    logger.info("=" * 80)

    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        legacy=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
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
    decoder_seq_buf = {}
    row_example_idx_buf = []
    row_choice_idx_buf = []
    row_token_idx_buf = []
    row_is_gold_buf = []

    example_offset = 0

    for enc, gold, all_choices in loader:
        num_choices = [len(x) for x in all_choices]

        target_ids, target_mask = build_targets(
            tokenizer=tokenizer,
            all_choices=all_choices,
            max_length=max_length,
        )

        input_ids = enc["input_ids"].to(device_obj)
        attention_mask = enc["attention_mask"].to(device_obj)

        input_ids = repeat_rows(input_ids, num_choices)
        attention_mask = repeat_rows(attention_mask, num_choices)

        target_ids = target_ids.to(device_obj)
        target_mask = target_mask.to(device_obj)

        with torch.no_grad():
            out_obj = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        dec_hs_all = out_obj.decoder_hidden_states

        if resolved_layers is None:
            resolved_layers = resolve_layers(len(dec_hs_all), layers)
            logger.info(f"Resolved layers: {resolved_layers}")
            for layer in resolved_layers:
                decoder_seq_buf[layer] = []

        mask = target_mask.bool()

        for layer in resolved_layers:
            dec_hidden = dec_hs_all[layer]
            decoder_seq = dec_hidden[mask]
            decoder_seq_buf[layer].append(decoder_seq.detach().cpu().numpy())

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
        packed[f"decoder_seq_l{layer}"] = np.concatenate(
            decoder_seq_buf[layer],
            axis=0,
        )

    np.savez(out, **packed)

    logger.info(f"Saved: {out}")
    logger.info(f"N_examples={len(labels)}")
    logger.info(f"N_token_rows={packed['row_example_idx'].shape[0]}")
    logger.info(f"Done in {time.time() - start:.1f}s")

    return out


if __name__ == "__main__":
    fire.Fire(main)
