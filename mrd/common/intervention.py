import csv
import json
from pathlib import Path

import numpy as np
import torch


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    predictions = scores.argmax(axis=1)
    accuracy = float((predictions == labels).mean())

    gold_scores = scores[np.arange(len(labels)), labels]

    wrong_scores = scores.copy()
    wrong_scores[np.arange(len(labels)), labels] = -1e30
    best_wrong_scores = wrong_scores.max(axis=1)

    return {
        "acc": accuracy,
        "gold_score": float(gold_scores.mean()),
        "gold_margin": float((gold_scores - best_wrong_scores).mean()),
    }


def fit_distortion_basis(delta: np.ndarray) -> np.ndarray:
    centered = delta - delta.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return vh


def remove_projection(
    delta: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    return delta - delta @ basis.T @ basis


def make_random_basis(
    dim: int,
    k: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((dim, k))
    q, _ = np.linalg.qr(matrix)
    return q.T


def select_basis(
    basis_all: np.ndarray,
    condition: str,
    k: int,
    seed: int,
) -> np.ndarray:
    if condition == "topk":
        return basis_all[:k]

    if condition == "random":
        return make_random_basis(
            dim=basis_all.shape[1],
            k=k,
            seed=seed,
        )

    raise ValueError(f"Unsupported condition: {condition}")


def edit_hidden_sequence(
    h_ref: torch.Tensor,
    h_merged: torch.Tensor,
    mask: torch.Tensor,
    basis: np.ndarray,
) -> torch.Tensor:
    h_edit = h_merged.clone()
    mask = mask.bool()

    delta = (h_merged[mask] - h_ref[mask]).detach().cpu().float().numpy()

    delta_edit = remove_projection(
        delta,
        basis,
    )

    ref_rows = h_ref[mask].detach().cpu().float().numpy()
    edited_rows = ref_rows + delta_edit

    h_edit[mask] = torch.tensor(
        edited_rows,
        dtype=h_merged.dtype,
        device=h_merged.device,
    )

    return h_edit


def save_base_metrics(
    path: Path,
    ref_metrics: dict,
    merged_metrics: dict,
    metadata: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "ref_metrics": ref_metrics,
        "merged_metrics": merged_metrics,
        "metadata": metadata,
    }

    with path.open("w") as f:
        json.dump(
            payload,
            f,
            indent=2,
        )


def load_base_metrics(
    path: Path,
) -> tuple[dict, dict]:
    with path.open() as f:
        payload = json.load(f)

    return (
        payload["ref_metrics"],
        payload["merged_metrics"],
    )


def build_result_row(
    k: int,
    condition: str,
    ref_metrics: dict,
    merged_metrics: dict,
    edit_metrics: dict,
    elapsed_sec: float | None = None,
) -> dict:
    row = {
        "k": k,
        "condition": condition,
        "acc_ref": f"{ref_metrics['acc']:.10f}",
        "acc_merged": f"{merged_metrics['acc']:.10f}",
        "acc_edit": f"{edit_metrics['acc']:.10f}",
        "gold_score_ref": f"{ref_metrics['gold_score']:.10f}",
        "gold_score_merged": f"{merged_metrics['gold_score']:.10f}",
        "gold_score_edit": f"{edit_metrics['gold_score']:.10f}",
        "gold_margin_ref": f"{ref_metrics['gold_margin']:.10f}",
        "gold_margin_merged": f"{merged_metrics['gold_margin']:.10f}",
        "gold_margin_edit": f"{edit_metrics['gold_margin']:.10f}",
        "gold_score_gain": (
            f"{edit_metrics['gold_score'] - merged_metrics['gold_score']:.10f}"
        ),
        "gold_margin_gain": (
            f"{edit_metrics['gold_margin'] - merged_metrics['gold_margin']:.10f}"
        ),
    }

    if elapsed_sec is not None:
        row["elapsed_sec"] = f"{elapsed_sec:.3f}"

    return row


def write_result_csv(
    path: Path,
    row: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )
        writer.writeheader()
        writer.writerow(row)
