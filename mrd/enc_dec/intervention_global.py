import logging
import time
from pathlib import Path

import fire
import numpy as np
from transformers import set_seed

from mrd.common.intervention import (
    build_result_row,
    load_base_metrics,
    make_random_basis,
    save_base_metrics,
    write_result_csv,
)
from mrd.common.logging import set_logger
from mrd.enc_dec.intervention import (
    build_dataset,
    evaluate_edit_condition,
    evaluate_model,
    load_model_and_tokenizer,
    resolve_device,
)
from mrd.enc_dec.utils import TASK_CONFIG, normalize_task

logger = logging.getLogger(__name__)


def main(
    reference_model: str,
    merged_model: str,
    task: str,
    out_dir: str,
    stage: str = "eval",
    basis_path: str | None = None,
    metrics_path: str | None = None,
    condition: str = "topk",
    k: int = 10,
    split: str = "validation",
    n_eval: int | None = None,
    batch_size: int = 8,
    max_length: int = 128,
    seed: int = 42,
    device: str = "auto",
    length_normalization: bool = False,
    log_path: str | None = None,
):
    task = normalize_task(task)

    if task not in TASK_CONFIG:
        raise ValueError(f"Unsupported task: {task}")

    if stage not in {"base_metrics", "eval"}:
        raise ValueError(f"Unsupported stage: {stage}")

    if condition not in {"topk", "random"}:
        raise ValueError(f"Unsupported condition: {condition}")

    out_path = Path(out_dir)
    out_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if metrics_path is None:
        metrics_path_obj = out_path / "base_metrics.json"
    else:
        metrics_path_obj = Path(metrics_path)

    if log_path is None:
        log_path = str(out_path / f"intervention_global_{task}_{stage}.log")

    set_logger(log_path)
    set_seed(seed)

    device_obj = resolve_device(device)

    logger.info("=" * 80)
    logger.info("Encoder-decoder global-basis intervention")
    logger.info("Oracle-global version: uses h_ref during evaluation")
    logger.info("Stage: %s", stage)
    logger.info("Reference model: %s", reference_model)
    logger.info("Merged model: %s", merged_model)
    logger.info("Task: %s", task)
    logger.info("Split: %s", split)
    logger.info("Basis: %s", basis_path)
    logger.info("=" * 80)

    start = time.time()

    ref_model, merged_model_obj, tokenizer = load_model_and_tokenizer(
        reference_model=reference_model,
        merged_model=merged_model,
        dtype=None,
    )

    dataset = build_dataset(
        task=task,
        split=split,
        n_eval=n_eval,
        seed=seed,
    )

    logger.info("N examples: %d", len(dataset))

    metadata = {
        "reference_model": reference_model,
        "merged_model": merged_model,
        "task": task,
        "split": split,
        "n_eval": n_eval,
        "batch_size": batch_size,
        "max_length": max_length,
        "seed": seed,
        "length_normalization": length_normalization,
        "basis_path": basis_path,
        "oracle_global": True,
    }

    if stage == "base_metrics":
        if metrics_path_obj.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {metrics_path_obj}"
            )

        logger.info("Computing reference metrics")

        ref_metrics = evaluate_model(
            model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            task=task,
            batch_size=batch_size,
            max_length=max_length,
            device=device_obj,
            length_normalization=length_normalization,
        )

        logger.info("Computing merged metrics")

        merged_metrics = evaluate_model(
            model=merged_model_obj,
            tokenizer=tokenizer,
            dataset=dataset,
            task=task,
            batch_size=batch_size,
            max_length=max_length,
            device=device_obj,
            length_normalization=length_normalization,
        )

        save_base_metrics(
            path=metrics_path_obj,
            ref_metrics=ref_metrics,
            merged_metrics=merged_metrics,
            metadata=metadata,
        )

        logger.info("Saved base metrics: %s", metrics_path_obj)
        logger.info("Done in %.1fs", time.time() - start)

        return str(metrics_path_obj)

    if not metrics_path_obj.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path_obj}")

    if basis_path is None:
        raise ValueError("basis_path is required for stage='eval'")

    basis_path_obj = Path(basis_path)

    if not basis_path_obj.exists():
        raise FileNotFoundError(f"Missing basis file: {basis_path_obj}")

    basis_all = np.load(basis_path_obj).astype(np.float32)

    if basis_all.ndim != 2:
        raise ValueError(f"Expected 2D basis array, got shape {basis_all.shape}")

    if k > basis_all.shape[0]:
        raise ValueError(
            f"k={k} exceeds available basis rows " f"({basis_all.shape[0]})"
        )

    ref_metrics, merged_metrics = load_base_metrics(metrics_path_obj)

    if condition == "topk":
        basis = basis_all[:k]
    else:
        basis = make_random_basis(
            dim=basis_all.shape[1],
            k=k,
            seed=seed,
        ).astype(np.float32)

    logger.info(
        "Evaluating condition=%s k=%d",
        condition,
        k,
    )

    edit_metrics = evaluate_edit_condition(
        ref_model=ref_model,
        merged_model=merged_model_obj,
        tokenizer=tokenizer,
        dataset=dataset,
        task=task,
        batch_size=batch_size,
        max_length=max_length,
        device=device_obj,
        basis=basis,
        length_normalization=length_normalization,
    )

    row = build_result_row(
        k=k,
        condition=condition,
        ref_metrics=ref_metrics,
        merged_metrics=merged_metrics,
        edit_metrics=edit_metrics,
    )

    csv_path = out_path / f"eval_{condition}_k{k}.csv"

    if csv_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {csv_path}")

    write_result_csv(
        csv_path,
        row,
    )

    logger.info("Wrote %s", csv_path)
    logger.info("Done in %.1fs", time.time() - start)

    return str(csv_path)


if __name__ == "__main__":
    fire.Fire(main)
