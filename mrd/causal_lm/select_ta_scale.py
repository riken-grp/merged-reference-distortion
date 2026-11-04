import csv
import gc
import json
import logging
import shutil
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


def load_adapter_state(path: Path) -> dict[str, torch.Tensor]:
    safetensors_path = path / "adapter_model.safetensors"
    bin_path = path / "adapter_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))

    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")

    raise FileNotFoundError(f"No adapter weights found in {path}")


def save_adapter_state(state: dict[str, torch.Tensor], out_dir: Path):
    try:
        from safetensors.torch import save_file

        save_file(
            {k: v.contiguous() for k, v in state.items()},
            str(out_dir / "adapter_model.safetensors"),
        )
    except Exception:
        torch.save(state, out_dir / "adapter_model.bin")


def verify_state_dicts(states: list[dict[str, torch.Tensor]]):
    keys = list(states[0].keys())

    for i, state in enumerate(states[1:], start=1):
        if list(state.keys()) != keys:
            raise ValueError(f"Adapter key mismatch at adapter {i}")

        for key in keys:
            if state[key].shape != states[0][key].shape:
                raise ValueError(f"Shape mismatch for {key}")


def merge_ta_states(
    states: list[dict[str, torch.Tensor]], lam: float
) -> dict[str, torch.Tensor]:
    verify_state_dicts(states)

    out = {}
    for key in states[0].keys():
        vals = [state[key].detach().cpu() for state in states]

        if torch.is_floating_point(vals[0]):
            summed = torch.stack([v.to(torch.float32) for v in vals], dim=0).sum(dim=0)
            out[key] = (float(lam) * summed).to(dtype=vals[0].dtype)
        else:
            out[key] = vals[0]

    return out


def copy_adapter_metadata(src_dir: Path, out_dir: Path):
    required = ["adapter_config.json"]

    for name in required:
        src = src_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required adapter file: {src}")
        shutil.copy2(src, out_dir / name)

    optional_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]

    for name in optional_files:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def prepare_out_dir(out_dir: Path, force: bool):
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} exists. Use --force=True to overwrite.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def save_temp_adapter(
    state: dict[str, torch.Tensor], metadata_src: Path, temp_dir: Path
):
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    copy_adapter_metadata(metadata_src, temp_dir)
    save_adapter_state(state, temp_dir)


def load_base_with_adapter(
    model_name_or_path: str, adapter_path: Path, dtype: torch.dtype
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype,
    )
    base_model.config.use_cache = False

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.config.use_cache = False

    return model, tokenizer


def build_dataset(task: str, split: str, n_eval: int | None, seed: int):
    ds = load_task_dataset(task, split)
    if n_eval is not None:
        n = min(n_eval, len(ds))
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def make_score_hidden_mask(target_mask: torch.Tensor) -> torch.Tensor:
    # Causal LM: hidden state at position p-1 predicts token at position p.
    score_hidden_mask = torch.zeros_like(target_mask)
    score_hidden_mask[:, :-1] = target_mask[:, 1:]
    return score_hidden_mask


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

    for batch_idx, (prompts, all_choices, _gold) in enumerate(loader):
        if batch_idx % 20 == 0:
            logger.info(f"[hidden] {task} batch={batch_idx}")

        batch = build_candidate_batch(
            tokenizer=tokenizer,
            prompts=prompts,
            all_choices=all_choices,
            max_length=max_length,
        )

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_mask = batch["target_mask"].to(device)

        score_hidden_mask = make_score_hidden_mask(target_mask)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        h = out.hidden_states[-1]
        mask = score_hidden_mask.bool()
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

    return {
        "raw_mse": raw_mse,
        "norm_mse": norm_mse,
        "mean_delta_norm": float(np.mean(np.linalg.norm(diff, axis=1))),
        "mean_ref_norm": float(np.mean(np.linalg.norm(h_ref, axis=1))),
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
    model_name_or_path: str,
    adapter_paths: list[str],
    tasks: list[str],
    lambda_grid: list[float],
    out_dir: str,
    work_dir: str | None = None,
    split: str = "train",
    n_eval: int | None = 500,
    batch_size: int = 4,
    max_length: int = 512,
    seed: int = 42,
    device: str = "auto",
    dtype: str = "bfloat16",
    log_path: str | None = None,
    force: bool = False,
):
    out_dir = Path(out_dir)
    work_dir = (
        Path(work_dir)
        if work_dir is not None
        else out_dir.parent / f"{out_dir.name}_work"
    )

    if work_dir.exists():
        if not force:
            raise FileExistsError(f"{work_dir} exists. Use --force=True to overwrite.")
        shutil.rmtree(work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)

    if log_path is None:
        log_path = work_dir / "select_ta_scale.log"
    else:
        log_path = Path(log_path)

    set_logger(str(log_path))
    set_seed(seed)

    start = time.time()
    device_obj = resolve_device(device)
    torch_dtype = resolve_dtype(dtype)

    tasks = [normalize_task(t) for t in tasks]
    for task in tasks:
        if task not in TASK_CONFIG:
            raise ValueError(f"Unsupported task: {task}")

    adapter_dirs = [Path(p) for p in adapter_paths]

    logger.info("=" * 80)
    logger.info("Causal LM TA scale selection")
    logger.info(f"Base model: {model_name_or_path}")
    logger.info(f"Adapters: {[str(p) for p in adapter_dirs]}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Lambda grid: {lambda_grid}")
    logger.info(f"Split: {split} n_eval={n_eval}")
    logger.info(f"Out dir: {out_dir}")
    logger.info(f"Work dir: {work_dir}")
    logger.info("=" * 80)

    states = []
    for path in adapter_dirs:
        logger.info(f"Loading adapter state: {path}")
        states.append(load_adapter_state(path))
    verify_state_dicts(states)

    datasets = {
        task: build_dataset(task=task, split=split, n_eval=n_eval, seed=seed)
        for task in tasks
    }

    logger.info("Precomputing reference hidden states")
    ref_rows_by_task = {}

    for task, adapter_path in zip(tasks, adapter_dirs):
        logger.info(f"[REF] task={task} adapter={adapter_path}")

        model, tokenizer = load_base_with_adapter(
            model_name_or_path=model_name_or_path,
            adapter_path=adapter_path,
            dtype=torch_dtype,
        )

        h_ref = extract_task_hidden_rows(
            model=model,
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

        # del model, tokenizer
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        model.to("cpu")
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    temp_dir = work_dir / ".tmp_adapter"

    lambda_rows = []
    task_rows = []

    best_lambda = None
    best_score = float("inf")
    best_state = None

    for lam in lambda_grid:
        lam = float(lam)

        logger.info("=" * 80)
        logger.info(f"[LAMBDA] {lam:.4f}")

        state = merge_ta_states(states, lam=lam)
        save_temp_adapter(state, metadata_src=adapter_dirs[0], temp_dir=temp_dir)

        model, tokenizer = load_base_with_adapter(
            model_name_or_path=model_name_or_path,
            adapter_path=temp_dir,
            dtype=torch_dtype,
        )

        per_task_scores = []

        for task in tasks:
            logger.info(f"[MERGED] lambda={lam:.4f} task={task}")

            h_merge = extract_task_hidden_rows(
                model=model,
                tokenizer=tokenizer,
                ds=datasets[task],
                task=task,
                batch_size=batch_size,
                max_length=max_length,
                device=device_obj,
            )

            scores = compute_distortion_scores(
                h_merge=h_merge,
                h_ref=ref_rows_by_task[task],
            )
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
            best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            logger.info(f"[NEW BEST] lambda={best_lambda:.4f} score={best_score:.6f}")

        # del model, tokenizer
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        model.to("cpu")
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if best_state is None or best_lambda is None:
        raise RuntimeError("No valid lambda selected")

    for row in lambda_rows:
        row["selected_by_norm_mse"] = int(float(row["lambda"]) == float(best_lambda))

    prepare_out_dir(out_dir, force=force)
    copy_adapter_metadata(adapter_dirs[0], out_dir)
    save_adapter_state(best_state, out_dir)

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

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
            "method": "causal_lm_ta_repselect",
            "model_name_or_path": model_name_or_path,
            "adapter_paths": [str(p) for p in adapter_dirs],
            "tasks": tasks,
            "lambda_grid": lambda_grid,
            "split": split,
            "n_eval": n_eval,
            "batch_size": batch_size,
            "max_length": max_length,
            "seed": seed,
            "dtype": dtype,
            "selected_lambda": best_lambda,
            "selected_mean_norm_mse": best_score,
            "out_dir": str(out_dir),
            "work_dir": str(work_dir),
            "elapsed_sec": time.time() - start,
        },
    )

    logger.info(f"[SELECTED] lambda={best_lambda:.4f} mean_norm_mse={best_score:.6f}")
    logger.info(f"Saved merged adapter to {out_dir}")
    logger.info(f"Wrote scores to {work_dir}")
    logger.info(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    fire.Fire(main)
