import logging
from pathlib import Path

import numpy as np

from mrd.common.anisotropy import (
    check_metadata,
    infer_rep_keys,
    load_npz,
    write_csv,
)

logger = logging.getLogger(__name__)


def sample_rows(
    x: np.ndarray,
    n_rows: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        x.shape[0],
        size=n_rows,
        replace=False,
    )
    return x[indices]


def compute_global_svd(delta: np.ndarray) -> dict:
    mean = delta.mean(axis=0, keepdims=True)
    centered = delta - mean

    _, singular_values, basis = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    eigenvalues = singular_values**2 / max(
        centered.shape[0] - 1,
        1,
    )

    total = float(eigenvalues.sum())

    if total <= 0.0:
        explained = np.zeros_like(eigenvalues)
    else:
        explained = eigenvalues / total

    cumulative = np.cumsum(explained)

    denominator = float((eigenvalues**2).sum())

    if denominator == 0.0:
        participation_ratio = 0.0
    else:
        participation_ratio = float(eigenvalues.sum() ** 2) / denominator

    return {
        "mean": mean.astype(np.float32),
        "basis": basis.astype(np.float32),
        "explained_ratio": explained,
        "cumulative_ratio": cumulative,
        "participation_ratio": participation_ratio,
    }


def cumulative_at(
    cumulative: np.ndarray,
    k: int,
) -> float:
    if len(cumulative) == 0:
        return 0.0

    index = min(k, len(cumulative)) - 1
    return float(cumulative[index])


def build_global_basis(
    reference_npzs: list[str],
    merged_npzs: list[str],
    tasks: list[str],
    out_dir: str,
    rep_prefix: str,
    rep_key: str | None = None,
    n_rows_per_task: int | None = None,
    seed: int = 42,
) -> None:
    if len(reference_npzs) != len(merged_npzs):
        raise ValueError("reference_npzs and merged_npzs must have the same length")

    if len(tasks) != len(reference_npzs):
        raise ValueError(
            "tasks, reference_npzs, and merged_npzs " "must have the same length"
        )

    deltas = []
    task_rows = []
    chosen_rep_key = rep_key

    for i, task in enumerate(tasks):
        reference = load_npz(reference_npzs[i])
        merged = load_npz(merged_npzs[i])

        check_metadata(reference, merged)

        if chosen_rep_key is None:
            keys = infer_rep_keys(
                reference,
                rep_prefix,
            )

            if len(keys) != 1:
                raise ValueError(
                    f"Multiple representation keys found: {keys}. "
                    "Pass --rep_key explicitly."
                )

            chosen_rep_key = keys[0]

        if chosen_rep_key not in reference:
            raise ValueError(
                f"Missing representation key "
                f"'{chosen_rep_key}' in reference for {task}"
            )

        if chosen_rep_key not in merged:
            raise ValueError(
                f"Missing representation key "
                f"'{chosen_rep_key}' in merged model for {task}"
            )

        ref_reps = reference[chosen_rep_key]
        merged_reps = merged[chosen_rep_key]

        if ref_reps.shape != merged_reps.shape:
            raise ValueError(
                f"Shape mismatch for {task}: "
                f"{ref_reps.shape} vs {merged_reps.shape}"
            )

        if ref_reps.ndim != 2:
            raise ValueError(
                f"Expected a 2D representation matrix for {task}, "
                f"got {ref_reps.shape}"
            )

        delta = merged_reps.astype(np.float32) - ref_reps.astype(np.float32)

        deltas.append(delta)

        task_rows.append(
            {
                "task": task,
                "n_rows_available": int(delta.shape[0]),
                "hidden_dim": int(delta.shape[1]),
                "n_examples": int(reference["labels"].shape[0]),
            }
        )

        logger.info(
            "%s: rows=%d dim=%d",
            task,
            delta.shape[0],
            delta.shape[1],
        )

    hidden_dims = {delta.shape[1] for delta in deltas}

    if len(hidden_dims) != 1:
        raise ValueError(f"Hidden dimensions differ across tasks: {hidden_dims}")

    min_rows = min(delta.shape[0] for delta in deltas)

    if n_rows_per_task is None:
        n_rows = min_rows
    else:
        n_rows = min(
            n_rows_per_task,
            min_rows,
        )

    logger.info(
        "Balanced rows per task: %d",
        n_rows,
    )

    sampled = []

    for i, delta in enumerate(deltas):
        sampled_delta = sample_rows(
            delta,
            n_rows=n_rows,
            seed=seed + i,
        )

        sampled.append(sampled_delta)
        task_rows[i]["n_rows_sampled"] = int(sampled_delta.shape[0])

    global_delta = np.concatenate(
        sampled,
        axis=0,
    )

    logger.info(
        "Global distortion matrix: %s",
        global_delta.shape,
    )

    stats = compute_global_svd(global_delta)

    out_path = Path(out_dir)

    np.save(
        out_path / "basis.npy",
        stats["basis"],
    )
    np.save(
        out_path / "mean.npy",
        stats["mean"],
    )

    explained = stats["explained_ratio"]
    cumulative = stats["cumulative_ratio"]

    curve_rows = [
        {
            "rep_key": chosen_rep_key,
            "component": i + 1,
            "explained_ratio": f"{float(ev):.10f}",
            "cumulative_ratio": f"{float(cev):.10f}",
        }
        for i, (ev, cev) in enumerate(zip(explained, cumulative))
    ]

    write_csv(
        out_path / "explained_variance.csv",
        curve_rows,
        [
            "rep_key",
            "component",
            "explained_ratio",
            "cumulative_ratio",
        ],
    )

    summary_row = {
        "rep_key": chosen_rep_key,
        "n_tasks": len(tasks),
        "n_rows_total": int(global_delta.shape[0]),
        "n_rows_per_task": int(n_rows),
        "hidden_dim": int(global_delta.shape[1]),
        "top1_real": f"{cumulative_at(cumulative, 1):.10f}",
        "top5_real": f"{cumulative_at(cumulative, 5):.10f}",
        "top10_real": f"{cumulative_at(cumulative, 10):.10f}",
        "top20_real": f"{cumulative_at(cumulative, 20):.10f}",
        "participation_ratio_real": (f"{stats['participation_ratio']:.10f}"),
        "mean_delta_norm": (f"{np.linalg.norm(global_delta, axis=1).mean():.10f}"),
    }

    write_csv(
        out_path / "summary.csv",
        [summary_row],
        list(summary_row.keys()),
    )

    write_csv(
        out_path / "task_rows.csv",
        task_rows,
        [
            "task",
            "n_rows_available",
            "n_rows_sampled",
            "hidden_dim",
            "n_examples",
        ],
    )

    logger.info(
        "Saved basis: %s",
        out_path / "basis.npy",
    )
    logger.info(
        "Saved mean: %s",
        out_path / "mean.npy",
    )
