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


def load_adapter_state(path: Path) -> dict[str, torch.Tensor]:
    safetensors_path = path / "adapter_model.safetensors"
    bin_path = path / "adapter_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))

    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")

    raise FileNotFoundError(
        f"No adapter weights found in {path}. "
        "Expected adapter_model.safetensors or adapter_model.bin."
    )


def save_adapter_state(state: dict[str, torch.Tensor], out_dir: Path):
    try:
        from safetensors.torch import save_file

        save_path = out_dir / "adapter_model.safetensors"
        save_file({k: v.contiguous() for k, v in state.items()}, str(save_path))
    except Exception:
        save_path = out_dir / "adapter_model.bin"
        torch.save(state, save_path)

    logger.info(f"Saved adapter weights to {save_path}")


def verify_state_dicts(states: list[dict[str, torch.Tensor]]):
    keys = list(states[0].keys())

    for i, state in enumerate(states[1:], start=1):
        if list(state.keys()) != keys:
            missing = set(keys) - set(state.keys())
            extra = set(state.keys()) - set(keys)
            raise ValueError(
                f"Adapter key mismatch at adapter {i}: "
                f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
            )

        for key in keys:
            if state[key].shape != states[0][key].shape:
                raise ValueError(
                    f"Shape mismatch for {key}: "
                    f"{state[key].shape} vs {states[0][key].shape}"
                )


def merge_ta_states(
    states: list[dict[str, torch.Tensor]],
    lam: float,
) -> dict[str, torch.Tensor]:
    """
    LoRA adapters have zero base adapter, so TA is:

        adapter = lambda * sum_i adapter_i
    """
    verify_state_dicts(states)

    out = {}
    keys = states[0].keys()

    for key in keys:
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
    state: dict[str, torch.Tensor],
    metadata_src: Path,
    temp_dir: Path,
):
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    copy_adapter_metadata(metadata_src, temp_dir)
    save_adapter_state(state, temp_dir)


def load_base_with_adapter(
    model_name_or_path: str,
    adapter_path: Path,
    dtype: torch.dtype,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    return model, tokenizer


def main(
    model_name_or_path: str,
    adapter_paths: list[str],
    tasks: list[str],
    lambda_grid: list[float],
    out_dir: str,
    split: str = "train",
    n_eval: int | None = 500,
    batch_size: int = 8,
    max_length: int = 512,
    length_normalization: bool = False,
    seed: int = 42,
    device: str = "auto",
    dtype: str = "bfloat16",
    num_workers: int = 0,
    log_path: str | None = None,
    force: bool = False,
):
    out_dir = Path(out_dir)

    if log_path is None:
        log_path = Path("logs") / f"{Path(__file__).stem}.log"
    else:
        log_path = Path(log_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    set_logger(str(log_path))

    set_seed(seed)
    device_obj = resolve_device(device)
    torch_dtype = resolve_dtype(dtype)

    tasks = [normalize_task(t) for t in tasks]
    for task in tasks:
        if task not in TASK_CONFIG:
            raise ValueError(f"Unsupported task: {task}")

    adapter_dirs = [Path(p) for p in adapter_paths]

    logger.info("=" * 80)
    logger.info("Causal LM LoRA task arithmetic")
    logger.info(f"Base model: {model_name_or_path}")
    logger.info(f"Adapter paths: {[str(p) for p in adapter_dirs]}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Lambda grid: {lambda_grid}")
    logger.info(f"Selection split: {split}")
    logger.info(f"Out dir: {out_dir}")
    logger.info("=" * 80)

    start = time.time()

    states = []
    for path in adapter_dirs:
        logger.info(f"Loading adapter: {path}")
        states.append(load_adapter_state(path))

    verify_state_dicts(states)

    best_acc = -1.0
    best_lambda = None
    best_state = None

    temp_dir = out_dir.parent / f".tmp_{out_dir.name}_selection"

    for lam in lambda_grid:
        logger.info("-" * 80)
        logger.info(f"[TA] lambda={lam:.4f}")

        state = merge_ta_states(states, lam=lam)
        save_temp_adapter(state, metadata_src=adapter_dirs[0], temp_dir=temp_dir)

        model, tokenizer = load_base_with_adapter(
            model_name_or_path=model_name_or_path,
            adapter_path=temp_dir,
            dtype=torch_dtype,
        )
        model.to(device_obj)
        model.eval()

        results = []
        for task in tasks:
            result = evaluate_task(
                model=model,
                tokenizer=tokenizer,
                task=task,
                split=split,
                batch_size=batch_size,
                max_length=max_length,
                device=device_obj,
                max_eval_samples=n_eval,
                length_normalization=length_normalization,
                seed=seed,
                num_workers=num_workers,
            )
            results.append(result)

        mean_acc = float(np.mean([r["accuracy"] for r in results]))
        logger.info(f"[TA] lambda={lam:.4f}, mean_accuracy={mean_acc:.4f}")

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = float(lam)
            best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            logger.info(
                f"[TA] New best lambda={best_lambda:.4f}, "
                f"mean_accuracy={best_acc:.4f}"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_state is None or best_lambda is None:
        raise RuntimeError("No valid adapter found during TA search.")

    prepare_out_dir(out_dir, force=force)
    copy_adapter_metadata(adapter_dirs[0], out_dir)
    save_adapter_state(best_state, out_dir)

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    summary = {
        "method": "adapter_task_arithmetic",
        "model_name_or_path": model_name_or_path,
        "adapter_paths": [str(p) for p in adapter_dirs],
        "tasks": tasks,
        "lambda_grid": lambda_grid,
        "selection_split": split,
        "n_eval": n_eval,
        "best_lambda": best_lambda,
        "best_mean_accuracy": best_acc,
        "out_dir": str(out_dir),
        "elapsed_seconds": time.time() - start,
    }

    with (out_dir / "merge_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"[TA] Best lambda={best_lambda:.4f}, mean_accuracy={best_acc:.4f}")
    logger.info(f"Saved merged adapter to {out_dir}")
    logger.info(f"Done in {summary['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    fire.Fire(main)
