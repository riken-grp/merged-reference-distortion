import logging
import time
from pathlib import Path

import fire
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from mrd.common.logging import set_logger

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model_and_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, legacy=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    return model, tokenizer


def verify_state_dicts(state_dicts: list[dict]):
    keys = list(state_dicts[0].keys())

    for i, sd in enumerate(state_dicts[1:], start=1):
        if list(sd.keys()) != keys:
            raise ValueError(f"State dict key mismatch at model {i}")

        for key in keys:
            if sd[key].shape != state_dicts[0][key].shape:
                raise ValueError(
                    f"Shape mismatch for {key}: "
                    f"{sd[key].shape} vs {state_dicts[0][key].shape}"
                )


def average_state_dicts(state_dicts: list[dict]) -> dict:
    verify_state_dicts(state_dicts)

    out = {}
    keys = state_dicts[0].keys()

    for key in keys:
        values = [sd[key].detach().cpu() for sd in state_dicts]

        if torch.is_floating_point(values[0]):
            stacked = torch.stack([v.to(torch.float32) for v in values], dim=0)
            out[key] = stacked.mean(dim=0).to(dtype=values[0].dtype)
        else:
            out[key] = values[0]

    return out


def main(
    base: str,
    ft_models: list[str],
    out_dir: str = "merged_average",
    seed: int = 42,
    log_path: str | None = None,
    force: bool = False,
):
    out_dir = Path(out_dir)

    if out_dir.exists() and not force:
        raise FileExistsError(f"{out_dir} exists. Use --force=True to overwrite.")

    if log_path is None:
        log_path = Path("logs") / f"{Path(__file__).stem}.log"
    else:
        log_path = Path(log_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    set_logger(str(log_path))

    set_seed(seed)

    logger.info("=" * 80)
    logger.info("T5 parameter averaging")
    logger.info(f"Base: {base}")
    logger.info(f"FT models: {ft_models}")
    logger.info(f"Out dir: {out_dir}")
    logger.info("=" * 80)

    start = time.time()

    base_model, base_tokenizer = load_model_and_tokenizer(base)
    ft_state_dicts = []

    for path in ft_models:
        logger.info(f"Loading FT model: {path}")
        model, _ = load_model_and_tokenizer(path)
        ft_state_dicts.append(model.state_dict())
        del model

    logger.info("Averaging state dicts")
    avg_state_dict = average_state_dicts(ft_state_dicts)

    logger.info("Loading averaged weights into base model")
    missing, unexpected = base_model.load_state_dict(avg_state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    base_model.tie_weights()

    out_dir.mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(out_dir)
    base_tokenizer.save_pretrained(out_dir)

    logger.info(f"Saved averaged model to {out_dir}")
    logger.info(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    fire.Fire(main)
