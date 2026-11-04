import json
import logging
import shutil
import time
from pathlib import Path

import fire
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from mrd.causal_lm.utils import TASK_CONFIG, evaluate_task, normalize_task
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


def load_adapter_state(
    path: Path,
) -> dict[str, torch.Tensor]:
    safetensors_path = path / "adapter_model.safetensors"
    bin_path = path / "adapter_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))

    if bin_path.exists():
        return torch.load(
            bin_path,
            map_location="cpu",
            weights_only=True,
        )

    raise FileNotFoundError(f"No adapter weights found in {path}")


def save_adapter_state(
    state: dict[str, torch.Tensor],
    out_dir: Path,
):
    from safetensors.torch import save_file

    save_file(
        {key: value.contiguous() for key, value in state.items()},
        str(out_dir / "adapter_model.safetensors"),
    )


def copy_adapter_metadata(
    src_dir: Path,
    out_dir: Path,
):
    shutil.copy2(
        src_dir / "adapter_config.json",
        out_dir / "adapter_config.json",
    )

    for name in [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]:
        src = src_dir / name

        if src.exists():
            shutil.copy2(
                src,
                out_dir / name,
            )


def verify_states(
    states: list[dict[str, torch.Tensor]],
) -> list[str]:
    keys = list(states[0])
    expected_keys = set(keys)

    for i, state in enumerate(
        states[1:],
        start=1,
    ):
        if set(state) != expected_keys:
            raise ValueError(f"Adapter key mismatch at adapter {i}")

        for key in keys:
            if state[key].shape != states[0][key].shape:
                raise ValueError(f"Shape mismatch for {key} at adapter {i}")

    return keys


def flatten_states(
    states: list[dict[str, torch.Tensor]],
) -> tuple[
    torch.Tensor,
    list[tuple[str, torch.Size, torch.dtype, int]],
]:
    keys = verify_states(states)

    spec = [
        (
            key,
            states[0][key].shape,
            states[0][key].dtype,
            states[0][key].numel(),
        )
        for key in keys
    ]

    vectors = []

    for state in states:
        chunks = [
            state[key].detach().cpu().to(torch.float32).reshape(-1) for key in keys
        ]

        vectors.append(torch.cat(chunks))

    return torch.stack(vectors), spec


def vector_to_state(
    vector: torch.Tensor,
    spec: list[tuple[str, torch.Size, torch.dtype, int]],
) -> dict[str, torch.Tensor]:
    state = {}
    offset = 0

    for key, shape, dtype, numel in spec:
        chunk = vector[offset : offset + numel]
        offset += numel

        state[key] = chunk.view(shape).to(dtype=dtype)

    return state


def topk_values_mask(
    task_vectors: torch.Tensor,
    topk: float,
) -> torch.Tensor:
    """Keep the largest-magnitude entries in each adapter vector."""
    if topk > 1:
        topk /= 100.0

    _, vector_dim = task_vectors.shape

    n_keep = int(vector_dim * topk)
    n_keep = max(
        1,
        min(n_keep, vector_dim),
    )

    _, indices = torch.topk(
        task_vectors.abs(),
        k=n_keep,
        dim=1,
    )

    mask = torch.zeros_like(
        task_vectors,
        dtype=torch.bool,
    )

    mask.scatter_(
        1,
        indices,
        True,
    )

    return task_vectors * mask


def resolve_mass_sign(
    task_vectors: torch.Tensor,
) -> torch.Tensor:
    signs = torch.sign(task_vectors.sum(dim=0))

    majority_sign = torch.sign(signs.sum())

    if majority_sign == 0:
        majority_sign = torch.tensor(
            1.0,
            device=signs.device,
        )

    signs = signs.clone()
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
        raise ValueError("Merge spec must look like " "'topk20_mass_dis-sum_0.5'.")

    topk_spec, sign_method, merge_func, scale = parts

    if not topk_spec.startswith("topk"):
        raise ValueError(f"Unsupported pruning method: {topk_spec}")

    topk = float(topk_spec[len("topk") :])

    if sign_method != "mass":
        raise ValueError(f"Unsupported sign method: {sign_method}")

    if merge_func not in {
        "dis-sum",
        "dis-mean",
    }:
        raise ValueError(f"Unsupported merge function: {merge_func}")

    return topk, merge_func, float(scale)


def ties_merge(
    task_vectors: torch.Tensor,
    merge_spec: str,
) -> torch.Tensor:
    topk, merge_func, scale = parse_merge_spec(merge_spec)

    logger.info(
        "Pruning: topk=%s",
        topk,
    )

    pruned = topk_values_mask(
        task_vectors,
        topk,
    )

    logger.info("Resolving sign: mass")

    signs = resolve_mass_sign(pruned)

    logger.info(
        "Merging: %s",
        merge_func,
    )

    merged = disjoint_merge(
        pruned,
        signs,
        merge_func,
    )

    return scale * merged


def load_model_with_adapter(
    model_name_or_path: str,
    adapter_path: Path,
    dtype: torch.dtype,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
    )

    return model, tokenizer


def main(
    model_name_or_path: str,
    adapter_paths: list[str],
    tasks: list[str],
    merge_specs: list[str],
    out_dir: str,
    split: str = "train",
    n_eval: int | None = 500,
    batch_size: int = 8,
    max_length: int = 512,
    seed: int = 42,
    dtype: str = "bfloat16",
    device: str = "auto",
    log_path: str | None = None,
    force: bool = False,
):
    out_dir = Path(out_dir)

    log_path = Path(log_path or f"logs/{Path(__file__).stem}.log")
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_logger(str(log_path))

    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} exists. " "Use --force to overwrite.")

        shutil.rmtree(out_dir)

    set_seed(seed)

    device_obj = resolve_device(device)

    torch_dtype = resolve_dtype(dtype)

    tasks = [normalize_task(task) for task in tasks]

    for task in tasks:
        if task not in TASK_CONFIG:
            raise ValueError(f"Unsupported task: {task}")

    adapter_dirs = [Path(path) for path in adapter_paths]

    logger.info("=" * 80)
    logger.info("Causal LM LoRA TIES merge search")
    logger.info(
        "Base model: %s",
        model_name_or_path,
    )
    logger.info(
        "Adapters: %s",
        [str(path) for path in adapter_dirs],
    )
    logger.info(
        "Tasks: %s",
        tasks,
    )
    logger.info(
        "Merge specs: %s",
        merge_specs,
    )
    logger.info(
        "Selection split: %s",
        split,
    )
    logger.info(
        "Device: %s",
        device_obj,
    )
    logger.info("=" * 80)

    start_time = time.time()

    states = [load_adapter_state(path) for path in adapter_dirs]

    flat_adapters, state_spec = flatten_states(states)

    temp_dir = out_dir.parent / f".tmp_{out_dir.name}_selection"

    best_accuracy = -1.0
    best_spec = None
    best_state = None

    for merge_spec in merge_specs:
        logger.info("-" * 80)
        logger.info(
            "Evaluating spec=%s",
            merge_spec,
        )

        merged_vector = ties_merge(
            flat_adapters,
            merge_spec,
        )

        merged_state = vector_to_state(
            merged_vector,
            state_spec,
        )

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        temp_dir.mkdir(parents=True)

        copy_adapter_metadata(
            adapter_dirs[0],
            temp_dir,
        )

        save_adapter_state(
            merged_state,
            temp_dir,
        )

        model, tokenizer = load_model_with_adapter(
            model_name_or_path=model_name_or_path,
            adapter_path=temp_dir,
            dtype=torch_dtype,
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
                seed=seed,
                num_workers=0,
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

            best_state = {
                key: value.detach().cpu().clone() for key, value in merged_state.items()
            }

            logger.info(
                "New best: spec=%s, mean_accuracy=%.4f",
                best_spec,
                best_accuracy,
            )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_state is None or best_spec is None:
        raise RuntimeError("No valid TIES merge found.")

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    copy_adapter_metadata(
        adapter_dirs[0],
        out_dir,
    )

    save_adapter_state(
        best_state,
        out_dir,
    )

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    summary = {
        "method": "adapter_ties",
        "model_name_or_path": model_name_or_path,
        "adapter_paths": [str(path) for path in adapter_dirs],
        "tasks": tasks,
        "merge_specs": merge_specs,
        "selection_split": split,
        "n_eval": n_eval,
        "best_spec": best_spec,
        "best_mean_accuracy": best_accuracy,
        "out_dir": str(out_dir),
        "elapsed_seconds": time.time() - start_time,
    }

    with (out_dir / "merge_summary.json").open("w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    logger.info(
        "Best spec=%s, mean_accuracy=%.4f",
        best_spec,
        best_accuracy,
    )

    logger.info(
        "Saved merged adapter to %s",
        out_dir,
    )

    logger.info(
        "Done in %.1fs",
        time.time() - start_time,
    )


if __name__ == "__main__":
    fire.Fire(main)
