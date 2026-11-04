import csv
import gc
import logging
import shutil
from pathlib import Path

import fire
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import set_seed

from mrd.causal_lm.select_ta_scale import (
    build_dataset,
    load_adapter_state,
    load_base_with_adapter,
    merge_ta_states,
    resolve_device,
    resolve_dtype,
    save_temp_adapter,
    verify_state_dicts,
)
from mrd.causal_lm.utils import (
    TASK_CONFIG,
    collate_eval,
    normalize_task,
    score_candidate_batch,
)
from mrd.common.logging import set_logger

logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_choice_scores(
    model,
    tokenizer,
    dataset,
    task: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    length_normalization: bool,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_eval(
            batch=batch,
            task=task,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
    )

    model.to(device)
    model.eval()

    rows = []

    for batch_idx, (
        prompts,
        all_choices,
        _gold,
    ) in enumerate(loader):
        if batch_idx % 20 == 0:
            logger.info(
                "[scores] task=%s batch=%d",
                task,
                batch_idx,
            )

        scores = score_candidate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            all_choices=all_choices,
            max_length=max_length,
            device=device,
            length_normalization=length_normalization,
        )

        rows.append(scores.detach().cpu().float().numpy())

    return np.concatenate(
        rows,
        axis=0,
    )


def compute_baseline_scores(
    merged_scores: np.ndarray,
    reference_scores: np.ndarray,
) -> dict:
    if merged_scores.shape != reference_scores.shape:
        raise ValueError(
            f"Shape mismatch: " f"{merged_scores.shape} vs {reference_scores.shape}"
        )

    merged_predictions = merged_scores.argmax(axis=1)
    reference_predictions = reference_scores.argmax(axis=1)

    agreement = float(np.mean(merged_predictions == reference_predictions))

    shifted = merged_scores - merged_scores.max(
        axis=1,
        keepdims=True,
    )

    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(
        axis=1,
        keepdims=True,
    )

    confidence = float(probabilities.max(axis=1).mean())

    return {
        "agreement": agreement,
        "confidence": confidence,
        "n_examples": int(merged_scores.shape[0]),
    }


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main(
    model_name_or_path: str,
    adapter_paths: list[str],
    tasks: list[str],
    lambda_grid: list[float],
    work_dir: str,
    split: str = "train",
    n_eval: int | None = 200,
    batch_size: int = 4,
    max_length: int = 512,
    seed: int = 42,
    device: str = "auto",
    dtype: str = "bfloat16",
    length_normalization: bool = False,
    log_path: str | None = None,
    force: bool = False,
):
    work_path = Path(work_dir)

    if work_path.exists():
        if not force:
            raise FileExistsError(
                f"{work_path} exists. " "Use --force=True to overwrite."
            )

        shutil.rmtree(work_path)

    work_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if log_path is None:
        log_path = str(work_path / "select_ta_scale_baselines.log")

    set_logger(log_path)
    set_seed(seed)

    device_obj = resolve_device(device)
    torch_dtype = resolve_dtype(dtype)

    tasks = [normalize_task(task) for task in tasks]

    if len(tasks) != len(adapter_paths):
        raise ValueError("tasks and adapter_paths " "must have the same length")

    for task in tasks:
        if task not in TASK_CONFIG:
            raise ValueError(f"Unsupported task: {task}")

    adapter_dirs = [Path(path) for path in adapter_paths]

    logger.info("=" * 80)
    logger.info("Phi TA scale-selection baselines")
    logger.info(
        "Tasks: %s",
        tasks,
    )
    logger.info(
        "Lambda grid: %s",
        lambda_grid,
    )
    logger.info("=" * 80)

    states = [load_adapter_state(path) for path in adapter_dirs]

    verify_state_dicts(states)

    datasets = {
        task: build_dataset(
            task=task,
            split=split,
            n_eval=n_eval,
            seed=seed,
        )
        for task in tasks
    }

    logger.info("Precomputing reference predictions")

    reference_scores = {}

    for task, adapter_path in zip(
        tasks,
        adapter_dirs,
    ):
        logger.info(
            "[REF] task=%s adapter=%s",
            task,
            adapter_path,
        )

        model, tokenizer = load_base_with_adapter(
            model_name_or_path=(model_name_or_path),
            adapter_path=adapter_path,
            dtype=torch_dtype,
        )

        reference_scores[task] = collect_choice_scores(
            model=model,
            tokenizer=tokenizer,
            dataset=datasets[task],
            task=task,
            batch_size=batch_size,
            max_length=max_length,
            device=device_obj,
            length_normalization=(length_normalization),
        )

        model.to("cpu")
        del model, tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    temp_dir = work_path / ".tmp_adapter"

    lambda_rows = []
    task_rows = []

    for lam in lambda_grid:
        lam = float(lam)

        logger.info("=" * 80)
        logger.info(
            "[LAMBDA] %.4f",
            lam,
        )

        state = merge_ta_states(
            states,
            lam=lam,
        )

        save_temp_adapter(
            state=state,
            metadata_src=adapter_dirs[0],
            temp_dir=temp_dir,
        )

        model, tokenizer = load_base_with_adapter(
            model_name_or_path=(model_name_or_path),
            adapter_path=temp_dir,
            dtype=torch_dtype,
        )

        per_task = []

        for task in tasks:
            merged_scores = collect_choice_scores(
                model=model,
                tokenizer=tokenizer,
                dataset=datasets[task],
                task=task,
                batch_size=batch_size,
                max_length=max_length,
                device=device_obj,
                length_normalization=(length_normalization),
            )

            metrics = compute_baseline_scores(
                merged_scores=(merged_scores),
                reference_scores=(reference_scores[task]),
            )

            row = {
                "lambda": lam,
                "task": task,
                **metrics,
            }

            task_rows.append(row)
            per_task.append(row)

            logger.info(
                "[SCORE] lambda=%.4f " "task=%s " "agreement=%.4f " "confidence=%.4f",
                lam,
                task,
                metrics["agreement"],
                metrics["confidence"],
            )

        lambda_rows.append(
            {
                "lambda": lam,
                "mean_agreement": float(
                    np.mean([row["agreement"] for row in per_task])
                ),
                "mean_confidence": float(
                    np.mean([row["confidence"] for row in per_task])
                ),
            }
        )

        model.to("cpu")
        del model, tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    best_agreement = max(
        lambda_rows,
        key=lambda row: (row["mean_agreement"]),
    )["lambda"]

    best_confidence = max(
        lambda_rows,
        key=lambda row: (row["mean_confidence"]),
    )["lambda"]

    for row in lambda_rows:
        row["selected_by_agreement"] = int(row["lambda"] == best_agreement)

        row["selected_by_confidence"] = int(row["lambda"] == best_confidence)

    write_csv(
        work_path / "lambda_scores.csv",
        lambda_rows,
        [
            "lambda",
            "mean_agreement",
            "mean_confidence",
            "selected_by_agreement",
            "selected_by_confidence",
        ],
    )

    write_csv(
        work_path / "task_scores.csv",
        task_rows,
        [
            "lambda",
            "task",
            "agreement",
            "confidence",
            "n_examples",
        ],
    )

    logger.info(
        "[SELECTED agreement] lambda=%.4f",
        best_agreement,
    )

    logger.info(
        "[SELECTED confidence] lambda=%.4f",
        best_confidence,
    )


if __name__ == "__main__":
    fire.Fire(main)
