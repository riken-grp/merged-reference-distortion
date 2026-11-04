import csv
import json
import logging
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
    load_task_dataset,
    normalize_task,
)

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_t5_model(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=False, legacy=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    return model, tokenizer


def get_param_spec(model):
    return [(name, p.shape, p.dtype, p.numel()) for name, p in model.named_parameters()]


def verify_param_compatibility(base_model, ft_models):
    base_spec = get_param_spec(base_model)

    for i, model in enumerate(ft_models):
        ft_spec = get_param_spec(model)
        if len(ft_spec) != len(base_spec):
            raise ValueError(f"Model {i} has different parameter count")

        for base_item, ft_item in zip(base_spec, ft_spec):
            b_name, b_shape, _, _ = base_item
            f_name, f_shape, _, _ = ft_item
            if b_name != f_name:
                raise ValueError(f"Parameter name mismatch: {b_name} vs {f_name}")
            if b_shape != f_shape:
                raise ValueError(f"Shape mismatch for {b_name}: {b_shape} vs {f_shape}")

    return base_spec


def model_to_vector(model) -> torch.Tensor:
    return torch.cat(
        [
            p.detach().cpu().to(torch.float32).reshape(-1)
            for _, p in model.named_parameters()
        ]
    )


def vector_to_state_dict(merged_vec, base_model, spec):
    state_dict = base_model.state_dict()
    out = {}

    offset = 0
    for name, shape, dtype, numel in spec:
        chunk = merged_vec[offset : offset + numel]
        offset += numel
        out[name] = chunk.view(shape).to(dtype=dtype)

    for key, value in state_dict.items():
        if key not in out:
            out[key] = value.detach().cpu()

    return out


def repeat_rows(x: torch.Tensor, counts: list[int]) -> torch.Tensor:
    idx = torch.arange(x.size(0), device=x.device)
    idx = torch.repeat_interleave(idx, torch.tensor(counts, device=x.device))
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
    target_mask = pad_sequence(mask_list, batch_first=True, padding_value=0)
    return target_ids, target_mask


def build_dataset(task: str, split: str, n_eval: int | None, seed: int):
    ds = load_task_dataset(task, split)
    if n_eval is not None:
        n = min(n_eval, len(ds))
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


@torch.no_grad()
def extract_task_hidden_rows(
    model,
    tokenizer,
    ds,
    task: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
):
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

    rows = []

    model.to(device)
    model.eval()

    for batch_idx, (enc, _gold, all_choices) in enumerate(loader):
        if batch_idx % 20 == 0:
            logger.info(f"[hidden] {task} batch={batch_idx}")

        num_choices = [len(x) for x in all_choices]

        target_ids, target_mask = build_targets(
            tokenizer=tokenizer,
            all_choices=all_choices,
            max_length=max_length,
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        input_ids = repeat_rows(input_ids, num_choices)
        attention_mask = repeat_rows(attention_mask, num_choices)

        target_ids = target_ids.to(device)
        target_mask = target_mask.to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        h = out.decoder_hidden_states[-1]
        mask = target_mask.bool()
        rows.append(h[mask].detach().cpu().float().numpy())

        del out, h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(rows, axis=0)


def compute_distortion_scores(h_merge: np.ndarray, h_ref: np.ndarray) -> dict:
    if h_merge.shape != h_ref.shape:
        raise ValueError(f"Shape mismatch: {h_merge.shape} vs {h_ref.shape}")

    diff = h_merge - h_ref

    raw_mse = float(np.mean(np.sum(diff * diff, axis=1)))
    ref_energy = float(np.mean(np.sum(h_ref * h_ref, axis=1)))

    norm_mse = raw_mse / max(ref_energy, 1e-12)

    mean_delta_norm = float(np.mean(np.linalg.norm(diff, axis=1)))
    mean_ref_norm = float(np.mean(np.linalg.norm(h_ref, axis=1)))

    return {
        "raw_mse": raw_mse,
        "norm_mse": norm_mse,
        "mean_delta_norm": mean_delta_norm,
        "mean_ref_norm": mean_ref_norm,
        "n_rows": int(h_ref.shape[0]),
        "hidden_dim": int(h_ref.shape[1]),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def main(
    base: str,
    ft_models: list[str],
    tasks: list[str],
    lambda_grid: list[float],
    split: str = "train",
    n_eval: int | None = 200,
    batch_size: int = 16,
    max_length: int = 128,
    out_dir: str = "merged_ta_repselect",
    work_dir: str | None = None,
    seed: int = 42,
    device: str = "auto",
    log_path: str | None = None,
    force: bool = False,
):
    out_dir = Path(out_dir)
    work_dir = (
        Path(work_dir)
        if work_dir is not None
        else out_dir.parent / f"{out_dir.name}_work"
    )

    if out_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite out_dir: {out_dir}")

    if work_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite work_dir: {work_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if log_path is None:
        log_path = work_dir / "select_ta_scale.log"
    else:
        log_path = Path(log_path)

    set_logger(str(log_path))
    set_seed(seed)

    start = time.time()
    device_obj = resolve_device(device)

    tasks = [normalize_task(t) for t in tasks]
    for task in tasks:
        if task not in TASK_CONFIG:
            raise ValueError(f"Unsupported task: {task}")

    logger.info("=" * 80)
    logger.info("T5 TA scale selection")
    logger.info(f"Base: {base}")
    logger.info(f"FT models: {ft_models}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Lambda grid: {lambda_grid}")
    logger.info(f"Split: {split} n_eval={n_eval}")
    logger.info(f"Out dir: {out_dir}")
    logger.info(f"Work dir: {work_dir}")
    logger.info("=" * 80)

    base_model, tokenizer = load_t5_model(base)
    ft_loaded = [load_t5_model(path)[0] for path in ft_models]

    spec = verify_param_compatibility(base_model, ft_loaded)

    flat_base = model_to_vector(base_model)
    flat_fts = torch.vstack([model_to_vector(m) for m in ft_loaded])
    delta = (flat_fts - flat_base).sum(dim=0)

    logger.info(f"[TA] ||delta||={delta.norm().item():.6e}")

    datasets = {
        task: build_dataset(task=task, split=split, n_eval=n_eval, seed=seed)
        for task in tasks
    }

    logger.info("Precomputing reference hidden states")
    ref_rows_by_task = {}

    for task, ft_path, ft_model in zip(tasks, ft_models, ft_loaded):
        # logger.info(f"[REF] task={task} model={ft_path}")
        # h_ref = extract_task_hidden_rows(
        #     model=ft_model,
        #     tokenizer=tokenizer,
        #     ds=datasets[task],
        #     task=task,
        #     batch_size=batch_size,
        #     max_length=max_length,
        #     device=device_obj,
        # )
        # ref_rows_by_task[task] = h_ref
        # np.save(work_dir / f"{task}_ref_hidden.npy", h_ref)
        # logger.info(f"[REF] {task} hidden shape={h_ref.shape}")
        h_ref = extract_task_hidden_rows(
            model=ft_model,
            tokenizer=tokenizer,
            ds=datasets[task],
            task=task,
            batch_size=batch_size,
            max_length=max_length,
            device=device_obj,
        )

        ref_rows_by_task[task] = h_ref
        np.save(work_dir / f"{task}_ref_hidden.npy", h_ref)
        logger.info(f"[REF] {task} hidden shape={h_ref.shape}")

        # Important: avoid keeping all fine-tuned models on GPU.
        ft_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lambda_rows = []
    task_rows = []

    best_lambda = None
    best_score = float("inf")
    best_vec = None

    for lam in lambda_grid:
        lam = float(lam)
        logger.info("=" * 80)
        logger.info(f"[LAMBDA] {lam:.4f}")

        merged_vec = flat_base + lam * delta
        merged_state_dict = vector_to_state_dict(merged_vec, base_model, spec)

        merged_model, _ = load_t5_model(base)
        merged_model.load_state_dict(merged_state_dict, strict=False)
        merged_model.to(device_obj)
        merged_model.eval()

        per_task_scores = []

        for task in tasks:
            logger.info(f"[MERGED] lambda={lam:.4f} task={task}")
            h_merge = extract_task_hidden_rows(
                model=merged_model,
                tokenizer=tokenizer,
                ds=datasets[task],
                task=task,
                batch_size=batch_size,
                max_length=max_length,
                device=device_obj,
            )

            h_ref = ref_rows_by_task[task]
            scores = compute_distortion_scores(h_merge=h_merge, h_ref=h_ref)
            scores["lambda"] = lam
            scores["task"] = task

            task_rows.append(scores)
            per_task_scores.append(scores)

            logger.info(
                f"[SCORE] lambda={lam:.4f} task={task} "
                f"raw={scores['raw_mse']:.6f} norm={scores['norm_mse']:.6f}"
            )

        mean_raw = float(np.mean([s["raw_mse"] for s in per_task_scores]))
        mean_norm = float(np.mean([s["norm_mse"] for s in per_task_scores]))

        lambda_row = {
            "lambda": lam,
            "mean_raw_mse": mean_raw,
            "mean_norm_mse": mean_norm,
        }
        lambda_rows.append(lambda_row)

        logger.info(
            f"[LAMBDA SUMMARY] lambda={lam:.4f} "
            f"mean_raw={mean_raw:.6f} mean_norm={mean_norm:.6f}"
        )

        if mean_norm < best_score:
            best_score = mean_norm
            best_lambda = lam
            best_vec = merged_vec.detach().clone()
            logger.info(f"[NEW BEST] lambda={best_lambda:.4f} score={best_score:.6f}")

        del merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_vec is None or best_lambda is None:
        raise RuntimeError("No valid lambda selected")

    for row in lambda_rows:
        row["selected_by_norm_mse"] = int(float(row["lambda"]) == float(best_lambda))

    best_state_dict = vector_to_state_dict(best_vec, base_model, spec)
    save_model, save_tokenizer = load_t5_model(base)
    save_model.load_state_dict(best_state_dict, strict=False)

    save_model.save_pretrained(out_dir)
    save_tokenizer.save_pretrained(out_dir)

    write_csv(
        work_dir / "lambda_scores.csv",
        lambda_rows,
        ["lambda", "mean_raw_mse", "mean_norm_mse", "selected_by_norm_mse"],
    )

    write_csv(
        work_dir / "task_scores.csv",
        task_rows,
        [
            "lambda",
            "task",
            "raw_mse",
            "norm_mse",
            "mean_delta_norm",
            "mean_ref_norm",
            "n_rows",
            "hidden_dim",
        ],
    )

    write_json(
        work_dir / "summary.json",
        {
            "base": base,
            "ft_models": ft_models,
            "tasks": tasks,
            "lambda_grid": lambda_grid,
            "split": split,
            "n_eval": n_eval,
            "batch_size": batch_size,
            "max_length": max_length,
            "seed": seed,
            "selected_lambda": best_lambda,
            "selected_mean_norm_mse": best_score,
            "out_dir": str(out_dir),
            "work_dir": str(work_dir),
            "elapsed_sec": time.time() - start,
        },
    )

    logger.info(f"[SELECTED] lambda={best_lambda:.4f} mean_norm_mse={best_score:.6f}")
    logger.info(f"Saved merged model to {out_dir}")
    logger.info(f"Wrote scores to {work_dir}")
    logger.info(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    fire.Fire(main)
