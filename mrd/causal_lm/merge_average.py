import json
import logging
import shutil
import time
from pathlib import Path

import fire
import torch

from mrd.common.logging import set_logger

logger = logging.getLogger(__name__)


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

    logger.info(f"Saved averaged adapter weights to {save_path}")


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


def average_adapter_states(
    states: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    verify_state_dicts(states)

    out = {}
    keys = states[0].keys()

    for key in keys:
        vals = [state[key].detach().cpu() for state in states]

        if torch.is_floating_point(vals[0]):
            avg = torch.stack([v.to(torch.float32) for v in vals], dim=0).mean(dim=0)
            out[key] = avg.to(dtype=vals[0].dtype)
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


def main(
    adapter_paths: list[str],
    out_dir: str,
    log_path: str | None = None,
    force: bool = False,
):
    out_dir = Path(out_dir)

    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} exists. Use --force=True to overwrite.")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if log_path is None:
        log_path = Path("logs") / f"{Path(__file__).stem}.log"
    else:
        log_path = Path(log_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    set_logger(str(log_path))

    start = time.time()

    adapter_dirs = [Path(p) for p in adapter_paths]

    logger.info("=" * 80)
    logger.info("Causal LM LoRA adapter averaging")
    logger.info(f"Adapter paths: {[str(p) for p in adapter_dirs]}")
    logger.info(f"Out dir: {out_dir}")
    logger.info("=" * 80)

    states = []
    for path in adapter_dirs:
        logger.info(f"Loading adapter: {path}")
        states.append(load_adapter_state(path))

    logger.info("Averaging adapter weights")
    avg_state = average_adapter_states(states)

    logger.info("Copying adapter metadata")
    copy_adapter_metadata(adapter_dirs[0], out_dir)

    save_adapter_state(avg_state, out_dir)

    summary = {
        "method": "adapter_average",
        "adapter_paths": [str(p) for p in adapter_dirs],
        "out_dir": str(out_dir),
        "n_adapters": len(adapter_dirs),
        "n_tensors": len(avg_state),
        "elapsed_seconds": time.time() - start,
    }

    with (out_dir / "merge_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Wrote {out_dir / 'merge_summary.json'}")
    logger.info(f"Done in {summary['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    fire.Fire(main)
