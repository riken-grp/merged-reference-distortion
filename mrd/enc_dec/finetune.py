import json
import logging
import time
import warnings
from pathlib import Path

import fire
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from mrd.common.logging import set_logger
from mrd.enc_dec.utils import (
    TASK_CONFIG,
    collate_train,
    evaluate_task,
    load_task_dataset,
    normalize_task,
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


logger = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_optimizer(model, lr: float, weight_decay: float):
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-8,
    )


def save_checkpoint(
    model,
    tokenizer,
    out_dir: Path,
    step: int,
    val_score: float,
    is_best: bool,
):
    if not is_best:
        return

    best_dir = out_dir / "best_checkpoint"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)

    meta = {
        "step": step,
        "val_score": val_score,
        "is_best": is_best,
    }
    with (best_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


def main(
    model_name_or_path: str,
    task: str,
    out_dir: str,
    train_batch_size: int = 32,
    gradient_accumulation_factor: int = 1,
    num_batches: int = 10000,
    checkpoint_frequency: int = 1000,
    early_stopping: bool = True,
    early_stopping_num_checkpoints_without_improvement: int = 5,
    lr: float = 1e-5,
    weight_decay: float = 0.0,
    max_length: int = 128,
    eval_split: str = "train",
    eval_batch_size: int = 64,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    length_normalization: bool = False,
    use_bfloat16_during_training: bool = True,
    seed: int = 42,
    device: str = "auto",
):
    task = normalize_task(task)
    if task not in TASK_CONFIG:
        raise ValueError(f"Unsupported task: {task}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / f"{Path(__file__).stem}.log"
    set_logger(str(log_path))
    set_seed(seed)
    device_obj = resolve_device(device)

    logger.info("=" * 80)
    logger.info("T5 single-task training")
    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"Task: {task}")
    logger.info(f"Out dir: {out_dir}")
    logger.info(f"Device: {device_obj}")
    logger.info("=" * 80)

    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        legacy=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model.to(device_obj)

    train_ds = load_task_dataset(task, "train")
    if max_train_samples is not None:
        n = min(max_train_samples, len(train_ds))
        train_ds = train_ds.shuffle(seed=seed).select(range(n))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_train(
            batch=batch,
            task=task,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
    )
    train_iter = iter(train_loader)

    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_bfloat16_during_training)

    best_score = -1.0
    no_improve = 0
    global_step = 0

    for micro_step in range(num_batches * gradient_accumulation_factor):
        step = micro_step // gradient_accumulation_factor + 1

        try:
            enc, target_ids = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            enc, target_ids = next(train_iter)

        enc = {k: v.to(device_obj) for k, v in enc.items()}
        target_ids = target_ids.to(device_obj)

        labels = target_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        model.train()
        if use_bfloat16_during_training:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    labels=labels,
                )
                loss = out.loss / gradient_accumulation_factor
            scaler.scale(loss).backward()
        else:
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=labels,
            )
            loss = out.loss / gradient_accumulation_factor
            loss.backward()

        if (micro_step + 1) % gradient_accumulation_factor != 0:
            continue

        if use_bfloat16_during_training:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad()
        global_step = step

        if global_step % checkpoint_frequency != 0:
            continue

        result = evaluate_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            split=eval_split,
            batch_size=eval_batch_size,
            max_length=max_length,
            device=device_obj,
            max_eval_samples=max_eval_samples,
            length_normalization=length_normalization,
            seed=seed,
            num_workers=0,
        )

        score = result["accuracy"]
        is_best = score > best_score
        if is_best:
            best_score = score
            no_improve = 0
        else:
            no_improve += 1

        save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            out_dir=out_dir,
            step=global_step,
            val_score=score,
            is_best=is_best,
        )

        logger.info(
            f"[CKPT] step={global_step} {eval_split}_accuracy={score:.4f} "
            f"best={best_score:.4f} no_improve={no_improve}"
        )

        if (
            early_stopping
            and no_improve >= early_stopping_num_checkpoints_without_improvement
        ):
            logger.info("Early stopping triggered")
            break

    elapsed = time.time() - start_time

    with (out_dir / "train_summary.json").open("w") as f:
        json.dump(
            {
                "task": task,
                "model_name_or_path": model_name_or_path,
                "selection_split": eval_split,
                "best_selection_accuracy": best_score,
                "final_step": global_step,
                "elapsed_seconds": elapsed,
                "elapsed_minutes": elapsed / 60,
            },
            f,
            indent=2,
        )

    logger.info(f"Fine-tuning finished ({elapsed / 60:.2f} min)")


if __name__ == "__main__":
    fire.Fire(main)
