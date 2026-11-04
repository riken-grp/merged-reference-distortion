import logging
import shutil
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


def load_t5_model(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        legacy=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    return model, tokenizer


def get_param_spec(model) -> list[tuple[str, torch.Size, torch.dtype, int]]:
    return [
        (name, param.shape, param.dtype, param.numel())
        for name, param in model.named_parameters()
    ]


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
            base_name, base_shape, _, _ = base_item
            ft_name, ft_shape, _, _ = ft_item

            if base_name != ft_name:
                raise ValueError(f"Parameter name mismatch: {base_name} vs {ft_name}")

            if base_shape != ft_shape:
                raise ValueError(
                    f"Shape mismatch for {base_name}: {base_shape} vs {ft_shape}"
                )

    return base_spec


def model_to_vector(model) -> torch.Tensor:
    chunks = [
        param.detach().cpu().to(torch.float32).reshape(-1)
        for _, param in model.named_parameters()
    ]
    return torch.cat(chunks)


def vector_to_state_dict(
    vector: torch.Tensor,
    base_model,
    spec: list[tuple[str, torch.Size, torch.dtype, int]],
) -> dict[str, torch.Tensor]:
    state_dict = base_model.state_dict()
    merged_state = {}

    offset = 0

    for name, shape, dtype, numel in spec:
        chunk = vector[offset : offset + numel]
        offset += numel
        merged_state[name] = chunk.view(shape).to(dtype=dtype)

    for name, value in state_dict.items():
        if name not in merged_state:
            merged_state[name] = value.detach().cpu()

    return merged_state


def topk_values_mask(
    task_vectors: torch.Tensor,
    topk: float,
) -> torch.Tensor:
    """Keep the largest-magnitude entries in each task vector."""
    if topk > 1:
        topk /= 100.0

    _, vector_dim = task_vectors.shape

    n_keep = int(vector_dim * topk)
    kth = vector_dim - n_keep

    kth_values, _ = task_vectors.abs().kthvalue(
        kth,
        dim=1,
        keepdim=True,
    )

    mask = task_vectors.abs() >= kth_values
    return task_vectors * mask


def resolve_mass_sign(task_vectors: torch.Tensor) -> torch.Tensor:
    signs = torch.sign(task_vectors.sum(dim=0))

    majority_sign = torch.sign(signs.sum())
    signs[signs == 0] = majority_sign

    return signs


def disjoint_merge(
    task_vectors: torch.Tensor,
    signs: torch.Tensor,
    merge_func: str,
) -> torch.Tensor:
    keep = torch.where(
        signs.unsqueeze(0) > 0,
        task_vectors > 0,
        task_vectors < 0,
    )

    selected = task_vectors * keep

    if merge_func == "dis-sum":
        return selected.sum(dim=0)

    if merge_func == "dis-mean":
        counts = (selected != 0).sum(dim=0).float().clamp(min=1.0)
        return selected.sum(dim=0) / counts

    raise ValueError(f"Unsupported merge function: {merge_func}")


def parse_merge_spec(
    merge_spec: str,
) -> tuple[float, str, float]:
    parts = merge_spec.split("_")

    if len(parts) != 4:
        raise ValueError("Merge spec must look like 'topk20_mass_dis-sum_0.5'.")

    topk_spec, sign_method, merge_func, scale = parts

    if not topk_spec.startswith("topk"):
        raise ValueError(f"Unsupported pruning method: {topk_spec}")

    topk = float(topk_spec[len("topk") :])

    if sign_method != "mass":
        raise ValueError(f"Unsupported sign method: {sign_method}")

    if merge_func not in {"dis-sum", "dis-mean"}:
        raise ValueError(f"Unsupported merge function: {merge_func}")

    return topk, merge_func, float(scale)


def ties_merge(
    task_vectors: torch.Tensor,
    merge_spec: str,
) -> torch.Tensor:
    topk, merge_func, scale = parse_merge_spec(merge_spec)

    logger.info("Pruning: topk=%s", topk)
    pruned = topk_values_mask(task_vectors, topk)

    logger.info("Resolving sign: mass")
    signs = resolve_mass_sign(pruned)

    logger.info("Merging: %s", merge_func)
    merged = disjoint_merge(pruned, signs, merge_func)

    return scale * merged


def main(
    base: str,
    ft_models: list[str],
    tasks: list[str],
    merge_specs: list[str],
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

    log_path = Path(log_path or f"logs/{Path(__file__).stem}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    set_logger(str(log_path))

    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} exists. Use --force to overwrite.")
        shutil.rmtree(out_dir)

    set_seed(seed)
    device_obj = resolve_device(device)

    tasks = [normalize_task(task) for task in tasks]

    for task in tasks:
        if task not in TASK_CONFIG:
            raise ValueError(f"Unsupported task: {task}")

    logger.info("=" * 80)
    logger.info("T5 TIES merge search")
    logger.info("Base model: %s", base)
    logger.info("Fine-tuned models: %s", ft_models)
    logger.info("Tasks: %s", tasks)
    logger.info("Merge specs: %s", merge_specs)
    logger.info("Selection split: %s", split)
    logger.info("Device: %s", device_obj)
    logger.info("=" * 80)

    start_time = time.time()

    base_model, tokenizer = load_t5_model(base)

    ft_loaded = [load_t5_model(path)[0] for path in ft_models]

    param_spec = verify_param_compatibility(
        base_model,
        ft_loaded,
    )

    flat_base = model_to_vector(base_model)

    flat_ft = torch.stack([model_to_vector(model) for model in ft_loaded])

    task_vectors = flat_ft - flat_base

    logger.info(
        "Task vectors ready: n_tasks=%d, vector_dim=%d",
        len(ft_models),
        task_vectors.shape[1],
    )

    best_accuracy = -1.0
    best_spec = None
    best_vector = None

    for merge_spec in merge_specs:
        logger.info("-" * 80)
        logger.info("Evaluating spec=%s", merge_spec)

        merged_update = ties_merge(
            task_vectors,
            merge_spec,
        )

        merged_vector = flat_base + merged_update

        merged_state = vector_to_state_dict(
            merged_vector,
            base_model,
            param_spec,
        )

        model, _ = load_t5_model(base)

        model.load_state_dict(
            merged_state,
            strict=False,
        )

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

        mean_accuracy = float(np.mean([result["accuracy"] for result in results]))

        logger.info(
            "spec=%s, mean_accuracy=%.4f",
            merge_spec,
            mean_accuracy,
        )

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_spec = merge_spec
            best_vector = merged_vector.detach().clone()

            logger.info(
                "New best: spec=%s, mean_accuracy=%.4f",
                best_spec,
                best_accuracy,
            )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_vector is None or best_spec is None:
        raise RuntimeError("No valid TIES merge found.")

    best_state = vector_to_state_dict(
        best_vector,
        base_model,
        param_spec,
    )

    save_model, save_tokenizer = load_t5_model(base)

    save_model.load_state_dict(
        best_state,
        strict=False,
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_model.save_pretrained(out_dir)
    save_tokenizer.save_pretrained(out_dir)

    logger.info(
        "Best spec=%s, mean_accuracy=%.4f",
        best_spec,
        best_accuracy,
    )
    logger.info("Saved merged model to %s", out_dir)
    logger.info(
        "Done in %.1fs",
        time.time() - start_time,
    )


if __name__ == "__main__":
    fire.Fire(main)
