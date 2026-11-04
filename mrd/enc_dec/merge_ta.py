import logging
import time
from pathlib import Path

import fire
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from mrd.common.logging import set_logger
from mrd.enc_dec.utils import TASK_CONFIG, evaluate_task, normalize_task

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_param_spec(model) -> list[tuple[str, torch.Size, torch.dtype, int]]:
    spec = []
    for name, param in model.named_parameters():
        spec.append((name, param.shape, param.dtype, param.numel()))
    return spec


def verify_param_compatibility(base_model, ft_models: list):
    base_spec = get_param_spec(base_model)

    for i, model in enumerate(ft_models):
        ft_spec = get_param_spec(model)
        if len(ft_spec) != len(base_spec):
            raise ValueError(
                f"Model {i} has different parameter count: "
                f"{len(ft_spec)} vs {len(base_spec)}"
            )

        for base_item, ft_item in zip(base_spec, ft_spec):
            b_name, b_shape, _, _ = base_item
            f_name, f_shape, _, _ = ft_item
            if b_name != f_name:
                raise ValueError(f"Parameter name mismatch: {b_name} vs {f_name}")
            if b_shape != f_shape:
                raise ValueError(f"Shape mismatch for {b_name}: {b_shape} vs {f_shape}")

    return base_spec


def model_to_vector(model) -> torch.Tensor:
    chunks = []
    for _, param in model.named_parameters():
        chunks.append(param.detach().cpu().to(torch.float32).reshape(-1))
    return torch.cat(chunks)


def vector_to_state_dict(
    merged_vec: torch.Tensor,
    base_model,
    spec: list[tuple[str, torch.Size, torch.dtype, int]],
) -> dict:
    state_dict = base_model.state_dict()
    out = {}

    offset = 0
    for name, shape, dtype, numel in spec:
        chunk = merged_vec[offset : offset + numel]
        offset += numel
        out[name] = chunk.view(shape).to(dtype=dtype)

    for key, value in state_dict.items():
        if key in out:
            continue
        out[key] = value.detach().cpu()

    return out


def load_t5_model(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    return model, tokenizer


def main(
    base: str,
    ft_models: list[str],
    tasks: list[str],
    lambda_grid: list[float],
    split: str = "train",
    n_eval: int | None = None,
    batch_size: int = 64,
    max_length: int = 128,
    out_dir: str = "merged_best",
    seed: int = 42,
    device: str = "auto",
    num_workers: int = 0,
    log_path: str | None = None,
    force: bool = False,
):
    out_dir = Path(out_dir)

    if out_dir.exists() and not force:
        logger.warning(f"[SKIP] {out_dir} exists. Use --force to overwrite.")
        return

    if log_path is None:
        log_path = Path("logs") / f"{Path(__file__).stem}.log"
    else:
        log_path = Path(log_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    set_logger(str(log_path))

    set_seed(seed)
    device_obj = resolve_device(device)

    tasks = [normalize_task(t) for t in tasks]
    for task in tasks:
        if task not in TASK_CONFIG:
            raise ValueError(f"Unsupported task: {task}")

    logger.info(f"Using device: {device_obj}")
    logger.info("=" * 80)
    logger.info("T5 TA Merge Lambda Search")
    logger.info(f"Base model: {base}")
    logger.info(f"Fine-tuned models: {ft_models}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Lambda grid: {lambda_grid}")
    logger.info(f"Tuning split: {split}")
    logger.info("=" * 80)

    start_time = time.time()

    base_model, tokenizer = load_t5_model(base)
    ft_loaded = [load_t5_model(path)[0] for path in ft_models]

    spec = verify_param_compatibility(base_model, ft_loaded)

    flat_base = model_to_vector(base_model)
    flat_fts = torch.vstack([model_to_vector(m) for m in ft_loaded])
    delta = (flat_fts - flat_base).sum(dim=0)

    logger.info(
        f"[TA] merge direction computed | "
        f"||delta||={delta.norm().item():.6e} | "
        f"n_tasks={len(ft_models)}"
    )

    best_acc = -1.0
    best_lambda = None
    best_vec = None

    for lam in lambda_grid:
        logger.info(f"lambda={lam:.4f} — evaluating")

        merged_vec = flat_base + float(lam) * delta
        merged_state_dict = vector_to_state_dict(merged_vec, base_model, spec)

        model, _ = load_t5_model(base)
        model.load_state_dict(merged_state_dict, strict=False)
        model.to(device_obj)
        model.eval()

        results = [
            evaluate_task(
                model=model,
                tokenizer=tokenizer,
                task=task,
                split=split,
                batch_size=batch_size,
                max_length=max_length,
                device=device_obj,
                max_eval_samples=n_eval,
                length_normalization=False,
                seed=seed,
                num_workers=num_workers,
            )
            for task in tasks
        ]

        mean_acc = float(np.mean([r["accuracy"] for r in results]))
        logger.info(f"lambda={lam:.4f}, mean_accuracy={mean_acc:.4f}")

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = float(lam)
            best_vec = merged_vec.detach().clone()
            logger.info(f"New best: lambda={lam:.4f}, mean_accuracy={mean_acc:.4f}")

        logger.info("-" * 80)

    if best_vec is None or best_lambda is None:
        raise RuntimeError("No valid model found during T5 TA lambda search.")

    best_state_dict = vector_to_state_dict(best_vec, base_model, spec)
    save_model, save_tokenizer = load_t5_model(base)
    save_model.load_state_dict(best_state_dict, strict=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_model.save_pretrained(out_dir)
    save_tokenizer.save_pretrained(out_dir)

    logger.info(f"[TA] Best lambda={best_lambda:.4f}, mean_accuracy={best_acc:.4f}")
    logger.info(f"Saved merged model to {out_dir}")
    logger.info(f"Done in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    fire.Fire(main)
