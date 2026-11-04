import logging
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)

TASK_CONFIG = {
    "paws": {
        "dataset": ("paws", "labeled_final"),
        "split_map": {
            "train": "train",
            "validation": "validation",
            "test": "test",
        },
    },
    "wiki_qa": {
        "dataset": ("wiki_qa",),
        "split_map": {
            "train": "train",
            "validation": "validation",
            "test": "test",
        },
    },
    "quartz": {
        "dataset": ("quartz",),
        "split_map": {
            "train": "train",
            "validation": "validation",
            "test": "test",
        },
    },
    "qasc": {
        "dataset": ("qasc",),
        "split_map": {
            "train": "train",
            "validation": "validation",
            "test": "validation",
        },
    },
    "winogrande": {
        "dataset": ("winogrande", "winogrande_xl"),
        "split_map": {
            "train": "train",
            "validation": "validation",
            "test": "validation",
        },
    },
    "wsc": {
        "dataset": ("super_glue", "wsc.fixed"),
        "split_map": {
            "train": "train",
            "validation": "validation",
            "test": "validation",
        },
    },
}


def normalize_task(task: str) -> str:
    task = task.lower().replace("-", "_")
    if task.startswith("glue/"):
        task = task.split("/", 1)[1]
    return task


def load_task_dataset(task: str, split: str):
    info = TASK_CONFIG[task]
    hf_split = info["split_map"][split]
    ds_name = info["dataset"]

    if len(ds_name) == 2:
        return load_dataset(ds_name[0], ds_name[1], split=hf_split)
    return load_dataset(ds_name[0], split=hf_split)


def format_input(task: str, ex: dict) -> str:
    if task == "paws":
        return f"paws sentence1: {ex['sentence1']} " f"sentence2: {ex['sentence2']}"

    if task == "wiki_qa":
        return f"wiki_qa question: {ex['question']} " f"answer: {ex['answer']}"

    if task == "quartz":
        choices = ex["choices"]["text"]
        return (
            f"quartz question: {ex['question']} "
            f"context: {ex['para']} "
            f"choice1: {choices[0]} choice2: {choices[1]}"
        )

    if task == "qasc":
        choices = ex["choices"]["text"]
        pieces = [f"choice{i + 1}: {c}" for i, c in enumerate(choices)]
        return (
            f"qasc question: {ex['question']} "
            f"context: {ex['combinedfact']} " + " ".join(pieces)
        )

    if task == "winogrande":
        return (
            "Choose the correct option to fill in the blank. "
            f"Sentence: {ex['sentence']} "
            f"Option 1: {ex['option1']} "
            f"Option 2: {ex['option2']} "
            "Answer:"
        )

    if task == "wsc":
        return (
            f'In the following text, does "{ex["span2_text"]}" '
            f'refer to "{ex["span1_text"]}"? '
            f'Text: {ex["text"]} '
            "Answer:"
        )

    raise ValueError(f"Unsupported task: {task}")


def get_label_id(task: str, ex: dict) -> int:
    if task in {"paws", "wiki_qa", "wsc"}:
        return int(ex["label"])

    if task == "quartz":
        return 0 if ex["answerKey"] == "A" else 1

    if task == "qasc":
        alpha = "ABCDEFGH"
        return alpha.index(ex["answerKey"])

    if task == "winogrande":
        return int(ex["answer"]) - 1

    raise ValueError(f"Unsupported task: {task}")


def get_label_texts(task: str, ex: dict) -> list[str]:
    if task == "paws":
        return ["not equivalent", "equivalent"]

    if task == "wiki_qa":
        return ["no", "yes"]

    if task == "quartz":
        return list(ex["choices"]["text"])

    if task == "qasc":
        return list(ex["choices"]["text"])

    if task == "winogrande":
        return [ex["option1"], ex["option2"]]

    if task == "wsc":
        return ["no", "yes"]

    raise ValueError(f"Unsupported task: {task}")


def collate_train(batch, task: str, tokenizer, max_length: int):
    inputs = [format_input(task, ex) for ex in batch]
    labels = [get_label_id(task, ex) for ex in batch]
    targets = [get_label_texts(task, ex)[y] for ex, y in zip(batch, labels)]

    enc = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tgt = tokenizer(
        targets,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return enc, tgt["input_ids"]


def collate_eval(batch, task: str, tokenizer, max_length: int):
    inputs = [format_input(task, ex) for ex in batch]
    gold = [get_label_id(task, ex) for ex in batch]
    all_choices = [get_label_texts(task, ex) for ex in batch]

    enc = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return enc, torch.tensor(gold, dtype=torch.long), all_choices


def _repeat_rows(x: torch.Tensor, counts: list[int]) -> torch.Tensor:
    idx = torch.arange(x.size(0), device=x.device)
    idx = torch.repeat_interleave(
        idx,
        torch.tensor(counts, device=x.device),
    )
    return x.index_select(0, idx)


@torch.no_grad()
def score_choices_batched(
    model,
    tokenizer,
    enc: dict,
    all_choices: list[list[str]],
    device: torch.device,
    max_length: int,
    length_normalization: bool = False,
) -> torch.Tensor:
    batch_size = len(all_choices)
    num_choices_per_ex = [len(x) for x in all_choices]
    flat_choices = [c for choices in all_choices for c in choices]

    tgt_ids_list = []
    tgt_mask_list = []
    for text in flat_choices:
        tok = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tgt_ids_list.append(tok["input_ids"][0])
        tgt_mask_list.append(torch.ones_like(tok["input_ids"][0]))

    target_ids = pad_sequence(
        tgt_ids_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    target_mask = pad_sequence(
        tgt_mask_list,
        batch_first=True,
        padding_value=0,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    input_ids = _repeat_rows(input_ids, num_choices_per_ex)
    attention_mask = _repeat_rows(attention_mask, num_choices_per_ex)

    target_ids = target_ids.to(device)
    target_mask = target_mask.to(device)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=target_ids,
    )

    logits = out.logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)
    token_logp = token_logp * target_mask

    seq_scores = token_logp.sum(dim=-1)
    if length_normalization:
        lengths = target_mask.sum(dim=-1).clamp(min=1)
        seq_scores = seq_scores / lengths

    max_choices = max(num_choices_per_ex)
    scores = torch.full(
        (batch_size, max_choices),
        fill_value=-1e9,
        dtype=seq_scores.dtype,
        device=seq_scores.device,
    )

    offset = 0
    for i, n in enumerate(num_choices_per_ex):
        scores[i, :n] = seq_scores[offset : offset + n]
        offset += n

    return scores


@torch.no_grad()
def evaluate_task(
    model,
    tokenizer,
    task: str,
    split: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    max_eval_samples: int | None,
    length_normalization: bool,
    seed: int,
    num_workers: int = 0,
) -> dict:
    ds = load_task_dataset(task, split)

    if max_eval_samples is not None:
        n = min(max_eval_samples, len(ds))
        ds = ds.shuffle(seed=seed).select(range(n))

    logger.info(
        f"[EVAL] {task} split={split} n_examples={len(ds)} " f"batch_size={batch_size}"
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda batch: collate_eval(
            batch=batch,
            task=task,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
    )

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    start = time.time()

    for enc, gold, all_choices in loader:
        scores = score_choices_batched(
            model=model,
            tokenizer=tokenizer,
            enc=enc,
            all_choices=all_choices,
            device=device,
            max_length=max_length,
            length_normalization=length_normalization,
        )
        preds = scores.argmax(dim=1).cpu()
        correct += (preds == gold).sum().item()
        total += gold.size(0)

    acc = correct / max(total, 1)
    logger.info(
        f"[EVAL] {task} split={split} accuracy={acc:.4f} "
        f"time={time.time() - start:.1f}s"
    )
    return {
        "task": task,
        "split": split,
        "accuracy": acc,
        "correct": correct,
        "total": total,
    }
