import logging
import time
from pathlib import Path

import fire
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from mrd.causal_lm.utils import TASK_CONFIG, evaluate_task, normalize_task
from mrd.common.logging import set_logger

logger = logging.getLogger(__name__)


def resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main(
    model_name_or_path: str,
    task: str,
    split: str = "validation",
    batch_size: int = 4,
    max_length: int = 512,
    max_eval_samples: int | None = None,
    length_normalization: bool = False,
    adapter_name_or_path: str | None = None,
    seed: int = 42,
    device: str = "auto",
    num_workers: int = 0,
    dtype: str = "bfloat16",
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

    set_logger(str(log_path))
    set_seed(seed)

    device_obj = resolve_device(device)

    if dtype == "bfloat16":
        dtype = torch.bfloat16
    elif dtype == "float16":
        dtype = torch.float16
    elif dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    logger.info("=" * 80)
    logger.info("Candidate-ranking evaluation")
    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"Task: {task}")
    logger.info(f"Split: {split}")
    logger.info(f"Device: {device_obj}")
    logger.info(f"dtype: {dtype}")
    logger.info("=" * 80)

    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype,
    )

    if adapter_name_or_path is not None:
        logger.info(f"Loading adapter: {adapter_name_or_path}")
        model = PeftModel.from_pretrained(model, adapter_name_or_path)

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
