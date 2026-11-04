import csv
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CURVE_FIELDS = [
    "rep_key",
    "component",
    "real_explained_ratio",
    "real_cumulative_ratio",
    "random_explained_ratio",
    "random_cumulative_ratio",
    "shuffled_explained_ratio",
    "shuffled_cumulative_ratio",
]

SUMMARY_FIELDS = [
    "rep_key",
    "n_rows",
    "hidden_dim",
    "n_examples",
    "top1_real",
    "top5_real",
    "top10_real",
    "top1_random",
    "top5_random",
    "top10_random",
    "top1_shuffled",
    "top5_shuffled",
    "top10_shuffled",
    "participation_ratio_real",
    "participation_ratio_random",
    "participation_ratio_shuffled",
    "mean_delta_norm",
]


def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def check_metadata(
    reference: dict[str, np.ndarray],
    merged: dict[str, np.ndarray],
) -> None:
    keys = [
        "labels",
        "row_example_idx",
        "row_choice_idx",
        "row_token_idx",
        "row_is_gold",
    ]

    for key in keys:
        if key not in reference or key not in merged:
            raise ValueError(f"Both .npz files must contain '{key}'")

        if not np.array_equal(reference[key], merged[key]):
            raise ValueError(f"Metadata mismatch for '{key}'")


def infer_rep_keys(
    data: dict[str, np.ndarray],
    prefix: str,
) -> list[str]:
    keys = sorted(key for key in data if key.startswith(prefix))

    if not keys:
        raise ValueError(f"No representation key matching '{prefix}*' found")

    return keys


def compute_statistics(x: np.ndarray) -> dict[str, np.ndarray | float]:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")

    centered = x - x.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)

    eigenvalues = singular_values**2 / max(centered.shape[0] - 1, 1)
    total = float(eigenvalues.sum())

    if total <= 0.0:
        explained = np.zeros_like(eigenvalues)
    else:
        explained = eigenvalues / total

    denominator = float((eigenvalues**2).sum())
    if denominator == 0.0:
        participation_ratio = 0.0
    else:
        participation_ratio = float(eigenvalues.sum() ** 2) / denominator

    return {
        "explained_ratio": explained,
        "cumulative_ratio": np.cumsum(explained),
        "participation_ratio": participation_ratio,
    }


def matched_norm_random(delta: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=delta.shape)

    delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
    noise_norm = np.linalg.norm(noise, axis=1, keepdims=True)
    noise_norm = np.maximum(noise_norm, 1e-12)

    return noise / noise_norm * delta_norm


def shuffled(delta: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty_like(delta)

    for i, row in enumerate(delta):
        out[i] = rng.permutation(row)

    return out


def topk_cumulative(stats: dict[str, np.ndarray | float], k: int) -> float:
    cumulative = stats["cumulative_ratio"]

    if not isinstance(cumulative, np.ndarray) or len(cumulative) == 0:
        return 0.0

    index = min(k, len(cumulative)) - 1
    return float(cumulative[index])


def build_curve_rows(
    rep_key: str,
    real_stats: dict[str, np.ndarray | float],
    random_stats: dict[str, np.ndarray | float],
    shuffled_stats: dict[str, np.ndarray | float],
) -> list[dict[str, str | int]]:
    real = real_stats["explained_ratio"]
    random = random_stats["explained_ratio"]
    shuffled_values = shuffled_stats["explained_ratio"]
    real_cumulative = real_stats["cumulative_ratio"]
    random_cumulative = random_stats["cumulative_ratio"]
    shuffled_cumulative = shuffled_stats["cumulative_ratio"]

    if not all(
        isinstance(value, np.ndarray)
        for value in [
            real,
            random,
            shuffled_values,
            real_cumulative,
            random_cumulative,
            shuffled_cumulative,
        ]
    ):
        raise TypeError("Expected array-valued spectrum statistics")

    rows = []

    for i in range(len(real)):
        rows.append(
            {
                "rep_key": rep_key,
                "component": i + 1,
                "real_explained_ratio": f"{real[i]:.10f}",
                "real_cumulative_ratio": f"{real_cumulative[i]:.10f}",
                "random_explained_ratio": f"{random[i]:.10f}",
                "random_cumulative_ratio": f"{random_cumulative[i]:.10f}",
                "shuffled_explained_ratio": f"{shuffled_values[i]:.10f}",
                "shuffled_cumulative_ratio": (f"{shuffled_cumulative[i]:.10f}"),
            }
        )

    return rows


def build_summary_row(
    rep_key: str,
    delta: np.ndarray,
    n_examples: int,
    real_stats: dict[str, np.ndarray | float],
    random_stats: dict[str, np.ndarray | float],
    shuffled_stats: dict[str, np.ndarray | float],
) -> dict[str, str | int]:
    return {
        "rep_key": rep_key,
        "n_rows": int(delta.shape[0]),
        "hidden_dim": int(delta.shape[1]),
        "n_examples": n_examples,
        "top1_real": f"{topk_cumulative(real_stats, 1):.10f}",
        "top5_real": f"{topk_cumulative(real_stats, 5):.10f}",
        "top10_real": f"{topk_cumulative(real_stats, 10):.10f}",
        "top1_random": f"{topk_cumulative(random_stats, 1):.10f}",
        "top5_random": f"{topk_cumulative(random_stats, 5):.10f}",
        "top10_random": f"{topk_cumulative(random_stats, 10):.10f}",
        "top1_shuffled": f"{topk_cumulative(shuffled_stats, 1):.10f}",
        "top5_shuffled": f"{topk_cumulative(shuffled_stats, 5):.10f}",
        "top10_shuffled": f"{topk_cumulative(shuffled_stats, 10):.10f}",
        "participation_ratio_real": (f"{real_stats['participation_ratio']:.10f}"),
        "participation_ratio_random": (f"{random_stats['participation_ratio']:.10f}"),
        "participation_ratio_shuffled": (
            f"{shuffled_stats['participation_ratio']:.10f}"
        ),
        "mean_delta_norm": f"{np.linalg.norm(delta, axis=1).mean():.10f}",
    }


def write_csv(
    path: Path,
    rows: list[dict[str, str | int]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_distortion(
    reference_npz: str,
    merged_npz: str,
    out_dir: str,
    rep_prefix: str,
    rep_keys: list[str] | None = None,
    seed: int = 42,
) -> None:
    reference = load_npz(reference_npz)
    merged = load_npz(merged_npz)

    check_metadata(reference, merged)

    if rep_keys is None:
        rep_keys = infer_rep_keys(reference, rep_prefix)

    curve_rows = []
    summary_rows = []

    for rep_key in rep_keys:
        if rep_key not in reference or rep_key not in merged:
            raise ValueError(f"Representation key '{rep_key}' not found")

        ref_reps = reference[rep_key]
        merged_reps = merged[rep_key]

        if ref_reps.shape != merged_reps.shape:
            raise ValueError(
                f"Shape mismatch for {rep_key}: "
                f"{ref_reps.shape} vs {merged_reps.shape}"
            )

        delta = merged_reps - ref_reps
        random_delta = matched_norm_random(delta, seed)
        shuffled_delta = shuffled(delta, seed)

        real_stats = compute_statistics(delta)
        random_stats = compute_statistics(random_delta)
        shuffled_stats = compute_statistics(shuffled_delta)

        curve_rows.extend(
            build_curve_rows(
                rep_key,
                real_stats,
                random_stats,
                shuffled_stats,
            )
        )

        summary_rows.append(
            build_summary_row(
                rep_key,
                delta,
                int(reference["labels"].shape[0]),
                real_stats,
                random_stats,
                shuffled_stats,
            )
        )

    out_path = Path(out_dir)

    write_csv(
        out_path / "explained_variance.csv",
        curve_rows,
        CURVE_FIELDS,
    )
    write_csv(
        out_path / "summary.csv",
        summary_rows,
        SUMMARY_FIELDS,
    )

    logger.info("Wrote %s", out_path / "explained_variance.csv")
    logger.info("Wrote %s", out_path / "summary.csv")
