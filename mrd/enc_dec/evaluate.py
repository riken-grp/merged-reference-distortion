import logging
import time
from pathlib import Path

import fire
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from mrd.common.logging import set_logger
from mrd.enc_dec.utils import TASK_CONFIG, evaluate_task, normalize_task

logger = logging.getLogger(__name__)


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main(
    model_name_or_path: str,
    task: str,
    split: str = "validation",
    batch_size: int = 64,
    max_length: int = 128,
    max_eval_samples: int | None = None,
    length_normalization: bool = False,
    seed: int = 42,
    device: str = "auto",
    num_workers: int = 0,
    log_path: str | None = None,
):
    task = normalize_task(task)
    if task not in TASK_CONFIG:
        raise ValueError(f"Unsupported task: {task}")

    if log_path is None:
        tag = model_name_or_path.replace("/", "--")
        log_path = Path("logs") / f"{Path(__file__).stem}_{tag}_{task}.log"
    else:
        log_path = Path(log_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    set_logger(str(log_path))

    set_seed(seed)
    device_obj = resolve_device(device)

    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"Task: {task}")
    logger.info(f"Split: {split}")
    logger.info(f"Using device: {device_obj}")
    logger.info("=" * 80)

    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    result = evaluate_task(
        model=model,
        tokenizer=tokenizer,
        task=task,
        split=split,
        batch_size=batch_size,
        max_length=max_length,
        device=device_obj,
        max_eval_samples=max_eval_samples,
        length_normalization=length_normalization,
        seed=seed,
        num_workers=num_workers,
    )

    logger.info("=" * 80)
    logger.info(f"{task:>15s} accuracy: {result['accuracy']:.4f}")
    logger.info("=" * 80)
    logger.info(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    fire.Fire(main)
