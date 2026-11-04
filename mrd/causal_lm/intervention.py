import logging
import time
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from mrd.causal_lm.utils import (
    TASK_CONFIG,
    build_candidate_batch,
    collate_eval,
    get_label_id,
    load_task_dataset,
    normalize_task,
)
from mrd.common.intervention import (
    build_result_row,
    compute_metrics,
    edit_hidden_sequence,
    fit_distortion_basis,
    load_base_metrics,
    save_base_metrics,
    select_basis,
    write_result_csv,
)
from mrd.common.logging import set_logger

logger = logging.getLogger(__name__)

REF_ADAPTER_NAME = "ref"
MERGED_ADAPTER_NAME = "merged"


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


def load_model_and_tokenizer(
    model_name_or_path: str,
    reference_adapter: str,
    merged_adapter: str,
    dtype,
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
    base_model.config.use_cache = False

    model = PeftModel.from_pretrained(
        base_model,
        reference_adapter,
        adapter_name=REF_ADAPTER_NAME,
    )

    model.load_adapter(
        merged_adapter,
        adapter_name=MERGED_ADAPTER_NAME,
    )
    model.config.use_cache = False

    return model, tokenizer


def set_active_adapter(
    model,
    adapter_name: str,
) -> None:
    model.set_adapter(adapter_name)


def logits_from_hidden(
    model,
    hidden: torch.Tensor,
) -> torch.Tensor:
    return model.get_output_embeddings()(hidden)


def make_score_hidden_mask(
    target_mask: torch.Tensor,
) -> torch.Tensor:
    score_hidden_mask = torch.zeros_like(target_mask)

    score_hidden_mask[:, :-1] = target_mask[:, 1:]

    return score_hidden_mask


def compute_scores_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_mask: torch.Tensor,
    length_normalization: bool,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_target_mask = target_mask[:, 1:]

    log_probs = F.log_softmax(
        shift_logits,
        dim=-1,
    )

    token_logp = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    token_logp = token_logp * shift_target_mask
    scores = token_logp.sum(dim=-1)

    if length_normalization:
        lengths = shift_target_mask.sum(dim=-1).clamp(min=1)

        scores = scores / lengths

    return scores


def pack_choice_scores(
    seq_scores: torch.Tensor,
    num_choices: list[int],
) -> torch.Tensor:
    batch_size = len(num_choices)
    max_choices = max(num_choices)

    scores = torch.full(
        (batch_size, max_choices),
        fill_value=-1e9,
        dtype=seq_scores.dtype,
        device=seq_scores.device,
    )

    offset = 0

    for i, n_choices in enumerate(num_choices):
        scores[i, :n_choices] = seq_scores[offset : offset + n_choices]

        offset += n_choices

    return scores


def build_dataset(
    task: str,
    split: str,
    n_eval: int | None,
    seed: int,
):
    dataset = load_task_dataset(
        task,
        split,
    )

    if n_eval is not None:
        n_eval = min(
            n_eval,
            len(dataset),
        )

        dataset = dataset.shuffle(seed=seed).select(range(n_eval))

    return dataset


@torch.no_grad()
def collect_delta_rows(
    model,
    tokenizer,
    dataset,
    task: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_eval(
            batch=batch,
            task=task,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
    )

    rows = []

    model.to(device)
    model.eval()

    for batch_idx, (
        prompts,
        all_choices,
        _,
    ) in enumerate(loader):
        if batch_idx % 20 == 0:
            logger.info(
                "[basis] batch=%d",
                batch_idx,
            )

        batch = build_candidate_batch(
            tokenizer=tokenizer,
            prompts=prompts,
            all_choices=all_choices,
            max_length=max_length,
        )

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        target_mask = batch["target_mask"].to(device)

        score_hidden_mask = make_score_hidden_mask(target_mask)

        set_active_adapter(
            model,
            REF_ADAPTER_NAME,
        )

        out_ref = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        h_ref = out_ref.hidden_states[-1]

        set_active_adapter(
            model,
            MERGED_ADAPTER_NAME,
        )

        out_merged = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        h_merged = out_merged.hidden_states[-1]

        mask = score_hidden_mask.bool()

        delta = h_merged[mask] - h_ref[mask]

        rows.append(delta.detach().cpu().float().numpy())

        del (
            out_ref,
            out_merged,
            h_ref,
            h_merged,
            delta,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(
        rows,
        axis=0,
    )


@torch.no_grad()
def evaluate_adapter(
    model,
    tokenizer,
    dataset,
    task: str,
    adapter_name: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    length_normalization: bool,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_eval(
            batch=batch,
            task=task,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
    )

    labels = np.array(
        [get_label_id(task, example) for example in dataset],
        dtype=np.int32,
    )

    all_scores = []

    model.to(device)
    model.eval()

    set_active_adapter(
        model,
        adapter_name,
    )

    for batch_idx, (
        prompts,
        all_choices,
        _,
    ) in enumerate(loader):
        if batch_idx % 20 == 0:
            logger.info(
                "[eval-adapter:%s] batch=%d",
                adapter_name,
                batch_idx,
            )

        batch = build_candidate_batch(
            tokenizer=tokenizer,
            prompts=prompts,
            all_choices=all_choices,
            max_length=max_length,
        )

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        target_mask = batch["target_mask"].to(device)

        num_choices = batch["num_choices"]

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )

        seq_scores = compute_scores_from_logits(
            logits=output.logits,
            input_ids=input_ids,
            target_mask=target_mask,
            length_normalization=(length_normalization),
        )

        scores = pack_choice_scores(
            seq_scores,
            num_choices,
        )

        all_scores.append(scores.detach().float().cpu().numpy())

        del (
            output,
            seq_scores,
            scores,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores = np.concatenate(
        all_scores,
        axis=0,
    )

    return compute_metrics(
        scores,
        labels,
    )


@torch.no_grad()
def evaluate_edit_condition(
    model,
    tokenizer,
    dataset,
    task: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    basis: np.ndarray,
    length_normalization: bool,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_eval(
            batch=batch,
            task=task,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
    )

    labels = np.array(
        [get_label_id(task, example) for example in dataset],
        dtype=np.int32,
    )

    all_scores = []

    model.to(device)
    model.eval()

    for batch_idx, (
        prompts,
        all_choices,
        _,
    ) in enumerate(loader):
        if batch_idx % 20 == 0:
            logger.info(
                "[eval-edit] batch=%d",
                batch_idx,
            )

        batch = build_candidate_batch(
            tokenizer=tokenizer,
            prompts=prompts,
            all_choices=all_choices,
            max_length=max_length,
        )

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        target_mask = batch["target_mask"].to(device)

        num_choices = batch["num_choices"]

        score_hidden_mask = make_score_hidden_mask(target_mask)

        set_active_adapter(
            model,
            REF_ADAPTER_NAME,
        )

        out_ref = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        h_ref = out_ref.hidden_states[-1]

        set_active_adapter(
            model,
            MERGED_ADAPTER_NAME,
        )

        out_merged = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        h_merged = out_merged.hidden_states[-1]

        h_edit = edit_hidden_sequence(
            h_ref=h_ref,
            h_merged=h_merged,
            mask=score_hidden_mask,
            basis=basis,
        )

        logits = logits_from_hidden(
            model,
            h_edit,
        )

        seq_scores = compute_scores_from_logits(
            logits=logits,
            input_ids=input_ids,
            target_mask=target_mask,
            length_normalization=(length_normalization),
        )

        scores = pack_choice_scores(
            seq_scores,
            num_choices,
        )

        all_scores.append(scores.detach().float().cpu().numpy())

        del (
            out_ref,
            out_merged,
            h_ref,
            h_merged,
            h_edit,
            logits,
            seq_scores,
            scores,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores = np.concatenate(
        all_scores,
        axis=0,
    )

    return compute_metrics(
        scores,
        labels,
    )


def main(
    model_name_or_path: str,
    reference_adapter: str,
    merged_adapter: str,
    task: str,
    out_dir: str,
    stage: str,
    basis_path: str | None = None,
    metrics_path: str | None = None,
    condition: str = "topk",
    k: int = 10,
    split: str = "validation",
    n_eval: int | None = None,
    batch_size: int = 16,
    max_length: int = 512,
    seed: int = 42,
    device: str = "auto",
    dtype: str = "bfloat16",
    length_normalization: bool = False,
    log_path: str | None = None,
):
    task = normalize_task(task)

    if task not in TASK_CONFIG:
        raise ValueError(f"Unsupported task: {task}")

    if stage not in {
        "fit_basis",
        "eval",
    }:
        raise ValueError(f"Unsupported stage: {stage}")

    if condition not in {
        "topk",
        "random",
    }:
        raise ValueError(f"Unsupported condition: {condition}")

    out_path = Path(out_dir)

    out_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if basis_path is None:
        basis_path_obj = out_path / "basis.npy"
    else:
        basis_path_obj = Path(basis_path)

    if metrics_path is None:
        metrics_path_obj = out_path / "base_metrics.json"
    else:
        metrics_path_obj = Path(metrics_path)

    if log_path is None:
        log_path = str(Path("logs") / (f"{Path(__file__).stem}_" f"{task}_{stage}.log"))

    set_logger(log_path)
    set_seed(seed)

    device_obj = resolve_device(device)

    torch_dtype = resolve_dtype(dtype)

    logger.info("=" * 80)
    logger.info("Causal LM candidate-sequence intervention")
    logger.info(
        "Stage: %s",
        stage,
    )
    logger.info(
        "Base model: %s",
        model_name_or_path,
    )
    logger.info(
        "Reference adapter: %s",
        reference_adapter,
    )
    logger.info(
        "Merged adapter: %s",
        merged_adapter,
    )
    logger.info(
        "Task: %s",
        task,
    )
    logger.info(
        "Split: %s",
        split,
    )
    logger.info("=" * 80)

    start = time.time()

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=(model_name_or_path),
        reference_adapter=(reference_adapter),
        merged_adapter=(merged_adapter),
        dtype=torch_dtype,
    )

    dataset = build_dataset(
        task=task,
        split=split,
        n_eval=n_eval,
        seed=seed,
    )

    logger.info(
        "N examples: %d",
        len(dataset),
    )

    metadata = {
        "model_name_or_path": (model_name_or_path),
        "reference_adapter": (reference_adapter),
        "merged_adapter": merged_adapter,
        "task": task,
        "split": split,
        "n_eval": n_eval,
        "batch_size": batch_size,
        "max_length": max_length,
        "seed": seed,
        "device": device,
        "dtype": dtype,
        "length_normalization": (length_normalization),
    }

    if stage == "fit_basis":
        logger.info("Collecting candidate-token " "distortion rows")

        delta_rows = collect_delta_rows(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            task=task,
            batch_size=batch_size,
            max_length=max_length,
            device=device_obj,
        )

        logger.info(
            "Delta rows: %s",
            delta_rows.shape,
        )

        basis_all = fit_distortion_basis(delta_rows)

        logger.info("Computing reference metrics")

        ref_metrics = evaluate_adapter(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            task=task,
            adapter_name=REF_ADAPTER_NAME,
            batch_size=batch_size,
            max_length=max_length,
            device=device_obj,
            length_normalization=(length_normalization),
        )

        logger.info("Computing merged metrics")

        merged_metrics = evaluate_adapter(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            task=task,
            adapter_name=(MERGED_ADAPTER_NAME),
            batch_size=batch_size,
            max_length=max_length,
            device=device_obj,
            length_normalization=(length_normalization),
        )

        basis_path_obj.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            basis_path_obj,
            basis_all,
        )

        save_base_metrics(
            path=metrics_path_obj,
            ref_metrics=ref_metrics,
            merged_metrics=merged_metrics,
            metadata=metadata,
        )

        logger.info(
            "Saved basis: %s",
            basis_path_obj,
        )

        logger.info(
            "Saved base metrics: %s",
            metrics_path_obj,
        )

        logger.info(
            "Done in %.1fs",
            time.time() - start,
        )

        return str(basis_path_obj)

    if not basis_path_obj.exists():
        raise FileNotFoundError(f"Missing basis file: " f"{basis_path_obj}")

    if not metrics_path_obj.exists():
        raise FileNotFoundError(f"Missing metrics file: " f"{metrics_path_obj}")

    basis_all = np.load(basis_path_obj)

    ref_metrics, merged_metrics = load_base_metrics(metrics_path_obj)

    basis = select_basis(
        basis_all=basis_all,
        condition=condition,
        k=k,
        seed=seed,
    )

    logger.info(
        "Evaluating condition=%s k=%d",
        condition,
        k,
    )

    edit_metrics = evaluate_edit_condition(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        task=task,
        batch_size=batch_size,
        max_length=max_length,
        device=device_obj,
        basis=basis,
        length_normalization=(length_normalization),
    )

    row = build_result_row(
        k=k,
        condition=condition,
        ref_metrics=ref_metrics,
        merged_metrics=merged_metrics,
        edit_metrics=edit_metrics,
    )

    csv_path = out_path / f"eval_{condition}_k{k}.csv"

    write_result_csv(
        csv_path,
        row,
    )

    logger.info(
        "Wrote %s",
        csv_path,
    )

    logger.info(
        "Done in %.1fs",
        time.time() - start,
    )

    return str(csv_path)


if __name__ == "__main__":
    fire.Fire(main)
