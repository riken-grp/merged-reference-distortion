import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import fire
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from mrd.causal_lm.utils import (
    TASK_CONFIG,
    evaluate_task,
    get_label_id,
    get_prompt_and_choices,
    load_task_dataset,
    normalize_task,
)
from mrd.common.logging import set_logger

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)

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


def parse_target_modules(target_modules: Any) -> list[str]:
    if isinstance(target_modules, str):
        text = target_modules.strip()

        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("target_modules JSON must be a list.")
            return [str(x).strip() for x in parsed if str(x).strip()]

        return [x.strip() for x in text.split(",") if x.strip()]

    if isinstance(target_modules, (list, tuple)):
        return [str(x).strip() for x in target_modules if str(x).strip()]

    raise ValueError(f"Unsupported target_modules type: {type(target_modules)}")


def module_matches_target(module_name: str, target: str) -> bool:
    return module_name == target or module_name.endswith(f".{target}")


def count_target_matches(model, targets: list[str]) -> dict[str, int]:
    module_names = [name for name, _ in model.named_modules()]
    return {
        target: sum(module_matches_target(name, target) for name in module_names)
        for target in targets
    }


def collect_available_leaf_modules(model) -> list[str]:
    leaves = set()
    for name, module in model.named_modules():
        if not name:
            continue
        if hasattr(module, "weight"):
            leaves.add(name.split(".")[-1])
    return sorted(leaves)


def suggest_target_modules(model) -> list[str]:
    leaves = collect_available_leaf_modules(model)
    keywords = ("proj", "attn", "mlp", "linear", "fc")
    return [x for x in leaves if any(k in x.lower() for k in keywords)]


def validate_target_modules(
    model,
    targets: list[str],
    allow_missing_target_modules: bool,
) -> tuple[dict[str, int], list[str]]:
    match_counts = count_target_matches(model, targets)
    missing = [name for name, count in match_counts.items() if count == 0]

    logger.info("LoRA target module match counts:")
    for name, count in match_counts.items():
        logger.info(f"  {name}: {count}")

    if missing:
        suggestions = suggest_target_modules(model)
        message = (
            "Some requested LoRA target modules matched zero model modules:\n"
            + "\n".join(f"  {name}: 0 matches" for name in missing)
            + "\n\nAvailable likely target module suffixes include:\n"
            + "\n".join(f"  {name}" for name in suggestions[:80])
        )

        if not allow_missing_target_modules:
            raise ValueError(message)

        logger.warning(message)

    effective = [name for name, count in match_counts.items() if count > 0]
    if not effective:
        raise ValueError("No LoRA target modules matched the model.")

    return match_counts, effective


def collate_train(batch, task, tokenizer, max_length: int):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for ex in batch:
        prompt, choices = get_prompt_and_choices(task, ex)
        gold_id = get_label_id(task, ex)
        target = choices[gold_id]

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

        if tokenizer.eos_token_id is not None:
            target_ids = target_ids + [tokenizer.eos_token_id]

        full_ids = prompt_ids + target_ids

        if len(full_ids) > max_length:
            overflow = len(full_ids) - max_length
            prompt_ids = prompt_ids[overflow:]
            full_ids = prompt_ids + target_ids

        labels = [-100] * len(prompt_ids) + target_ids

        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        attention_mask_list.append(torch.ones(len(full_ids), dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask_list,
        batch_first=True,
        padding_value=0,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=-100,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def save_adapter(model, tokenizer, out_dir: Path, step: int, score: float):
    best_dir = out_dir / "best_adapter"
    best_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)

    with (best_dir / "meta.json").open("w") as f:
        json.dump(
            {
                "step": step,
                "selection_score": score,
            },
            f,
            indent=2,
        )


def main(
    model_name_or_path: str,
    task: str,
    out_dir: str,
    train_batch_size: int = 4,
    gradient_accumulation_factor: int = 1,
    num_batches: int = 1000,
    checkpoint_frequency: int = 250,
    early_stopping: bool = True,
    early_stopping_num_checkpoints_without_improvement: int = 5,
    lr: float = 2e-4,
    weight_decay: float = 0.0,
    max_length: int = 512,
    eval_split: str = "train",
    eval_batch_size: int = 4,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = 200,
    length_normalization: bool = False,
    seed: int = 42,
    device: str = "auto",
    dtype: str = "bfloat16",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Any = "o_proj",
    allow_missing_target_modules: bool = False,
):
    task = normalize_task(task)
    if task not in TASK_CONFIG:
        raise ValueError(f"Unsupported task: {task}")

    parsed_target_modules = parse_target_modules(target_modules)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_logger(str(out_dir / "finetune.log"))
    set_seed(seed)

    device_obj = resolve_device(device)
    torch_dtype = resolve_dtype(dtype)

    logger.info("=" * 80)
    logger.info("LoRA single-task fine-tuning")
    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"Task: {task}")
    logger.info(f"Out dir: {out_dir}")
    logger.info(f"Device: {device_obj}")
    logger.info(f"dtype: {dtype}")
    logger.info(f"Requested target_modules: {parsed_target_modules}")
    logger.info("=" * 80)

    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch_dtype,
    )

    target_match_counts, effective_target_modules = validate_target_modules(
        model=model,
        targets=parsed_target_modules,
        allow_missing_target_modules=allow_missing_target_modules,
    )

    logger.info(f"Effective target_modules: {effective_target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=effective_target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device_obj)

    train_ds = load_task_dataset(task, "train")
    if max_train_samples is not None:
        n = min(max_train_samples, len(train_ds))
        train_ds = train_ds.shuffle(seed=seed).select(range(n))

    train_loader = DataLoader(
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-8,
    )

    best_score = -1.0
    no_improve = 0
    global_step = 0

    for micro_step in range(num_batches * gradient_accumulation_factor):
        step = micro_step // gradient_accumulation_factor + 1

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device_obj) for k, v in batch.items()}

        model.train()
        out = model(**batch)
        loss = out.loss / gradient_accumulation_factor
        loss.backward()

        if (micro_step + 1) % gradient_accumulation_factor != 0:
            continue

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
            save_adapter(
                model=model,
                tokenizer=tokenizer,
                out_dir=out_dir,
                step=global_step,
                score=score,
            )
        else:
            no_improve += 1

        logger.info(
            f"[CKPT] step={global_step} "
            f"{eval_split}_accuracy={score:.4f} "
            f"best={best_score:.4f} no_improve={no_improve}"
        )

        if (
            early_stopping
            and no_improve >= early_stopping_num_checkpoints_without_improvement
        ):
            logger.info("Early stopping triggered")
            break

    elapsed = time.time() - start_time

    with (out_dir / "finetune_summary.json").open("w") as f:
        json.dump(
            {
                "task": task,
                "model_name_or_path": model_name_or_path,
                "selection_split": eval_split,
                "best_selection_accuracy": best_score,
                "final_step": global_step,
                "elapsed_seconds": elapsed,
                "elapsed_minutes": elapsed / 60,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "target_modules": effective_target_modules,
                "requested_target_modules": parsed_target_modules,
                "target_module_match_counts": target_match_counts,
            },
            f,
            indent=2,
        )

    logger.info(f"Fine-tuning finished ({elapsed / 60:.2f} min)")


if __name__ == "__main__":
    fire.Fire(main)
