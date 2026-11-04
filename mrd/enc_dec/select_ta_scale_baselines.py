import csv
import logging
import shutil
from pathlib import Path

import fire
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import set_seed

from mrd.common.logging import set_logger
from mrd.enc_dec.select_ta_scale import (
    build_dataset,
    load_t5_model,
    model_to_vector,
    resolve_device,
    vector_to_state_dict,
    verify_param_compatibility,
)
from mrd.enc_dec.utils import (
    TASK_CONFIG,
    collate_eval,
    normalize_task,
    score_choices_batched,
)

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

    for batch_idx, (encoded, _gold, all_choices) in enumerate(loader):
        if batch_idx % 20 == 0:
            logger.info(
                "[scores] task=%s batch=%d",
                task,
                batch_idx,
            )

        scores = score_choices_batched(
            model=model,
            tokenizer=tokenizer,
            enc=encoded,
            all_choices=all_choices,
            device=device,
            max_length=max_length,
            length_normalization=length_normalization,
        )

        rows.append(scores.detach().cpu().float().numpy())

    return np.concatenate(rows, axis=0)


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
    base: str,
    ft_models: list[str],
    tasks: list[str],
    lambda_grid: list[float],
    work_dir: str,
    split: str = "train",
    n_eval: int | None = 200,
    batch_size: int = 16,
    max_length: int = 128,
    seed: int = 42,
    device: str = "auto",
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

    tasks = [normalize_task(task) for task in tasks]

    if len(tasks) != len(ft_models):
        raise ValueError("tasks and ft_models must have the same length")

    for task in tasks:
        if task not in TASK_CONFIG:
            raise ValueError(f"Unsupported task: {task}")

    logger.info("=" * 80)
    logger.info("T5 TA scale-selection baselines")
    logger.info("Tasks: %s", tasks)
    logger.info("Lambda grid: %s", lambda_grid)
    logger.info("=" * 80)

    base_model, tokenizer = load_t5_model(base)

    ft_loaded = [load_t5_model(path)[0] for path in ft_models]

    spec = verify_param_compatibility(
        base_model,
        ft_loaded,
    )

    flat_base = model_to_vector(base_model)

    flat_fts = torch.vstack([model_to_vector(model) for model in ft_loaded])

    delta = (flat_fts - flat_base).sum(dim=0)

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

    for task, model in zip(
        tasks,
        ft_loaded,
    ):
        scores = collect_choice_scores(
            model=model,
            tokenizer=tokenizer,
            dataset=datasets[task],
            task=task,
            batch_size=batch_size,
            max_length=max_length,
            device=device_obj,
            length_normalization=length_normalization,
        )

        reference_scores[task] = scores

        model.to("cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lambda_rows = []
    task_rows = []

    for lam in lambda_grid:
        lam = float(lam)

        logger.info("=" * 80)
        logger.info(
            "[LAMBDA] %.4f",
            lam,
        )

        merged_vec = flat_base + lam * delta

        state_dict = vector_to_state_dict(
            merged_vec,
            base_model,
            spec,
        )

        merged_model, _ = load_t5_model(base)

        merged_model.load_state_dict(
            state_dict,
            strict=False,
        )

        per_task = []

        for task in tasks:
            scores = collect_choice_scores(
                model=merged_model,
                tokenizer=tokenizer,
                dataset=datasets[task],
                task=task,
                batch_size=batch_size,
                max_length=max_length,
                device=device_obj,
                length_normalization=length_normalization,
            )

            metrics = compute_baseline_scores(
                merged_scores=scores,
                reference_scores=reference_scores[task],
            )

            row = {
                "lambda": lam,
                "task": task,
                **metrics,
            }

            task_rows.append(row)
            per_task.append(row)

            logger.info(
                "[SCORE] lambda=%.4f task=%s " "agreement=%.4f confidence=%.4f",
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

        merged_model.to("cpu")
        del merged_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_agreement = max(
        lambda_rows,
        key=lambda row: row["mean_agreement"],
    )["lambda"]

    best_confidence = max(
        lambda_rows,
        key=lambda row: row["mean_confidence"],
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
