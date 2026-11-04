import logging
import time
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

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
from mrd.enc_dec.utils import (
    TASK_CONFIG,
    collate_eval,
    get_label_id,
    load_task_dataset,
    normalize_task,
)

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def resolve_dtype(dtype: str | None):
    if dtype is None:
        return None

    if dtype == "bfloat16":
        return torch.bfloat16

    if dtype == "float16":
        return torch.float16

    if dtype == "float32":
        return torch.float32

    raise ValueError(f"Unsupported dtype: {dtype}")


def load_model_and_tokenizer(
    reference_model: str,
    merged_model: str,
    dtype,
):
    tokenizer = AutoTokenizer.from_pretrained(
        reference_model,
        use_fast=False,
        legacy=True,
    )

    model_kwargs = {}

    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    ref_model = AutoModelForSeq2SeqLM.from_pretrained(
        reference_model,
        **model_kwargs,
    )

    merged_model_obj = AutoModelForSeq2SeqLM.from_pretrained(
        merged_model,
        **model_kwargs,
    )

    ref_model.config.use_cache = False
    merged_model_obj.config.use_cache = False

    return ref_model, merged_model_obj, tokenizer


def repeat_rows(
    x: torch.Tensor,
    counts: list[int],
) -> torch.Tensor:
    indices = torch.arange(
        x.size(0),
        device=x.device,
    )

    indices = torch.repeat_interleave(
        indices,
        torch.tensor(
            counts,
            device=x.device,
        ),
    )

    return x.index_select(
        0,
        indices,
    )


def build_targets(
    tokenizer,
    all_choices,
    max_length: int,
):
    flat_choices = [choice for choices in all_choices for choice in choices]

    ids_list = []
    mask_list = []

    for text in flat_choices:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        ids = encoded["input_ids"][0]

        ids_list.append(ids)
        mask_list.append(torch.ones_like(ids))

    target_ids = pad_sequence(
        ids_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    target_mask = pad_sequence(
        mask_list,
        batch_first=True,
        padding_value=0,
    )

    return target_ids, target_mask


def compute_scores_from_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    length_normalization: bool,
) -> torch.Tensor:
    log_probs = F.log_softmax(
        logits,
        dim=-1,
    )

    token_logp = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    token_logp = token_logp * target_mask
    scores = token_logp.sum(dim=-1)

    if length_normalization:
        lengths = target_mask.sum(dim=-1).clamp(min=1)

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


def logits_from_hidden(
    model,
    hidden: torch.Tensor,
) -> torch.Tensor:
    if getattr(
        model.config,
        "tie_word_embeddings",
        False,
    ):
        hidden = hidden * (model.model_dim**-0.5)

    logits = model.lm_head(hidden)

    if hasattr(
        model,
        "final_logits_bias",
    ):
        logits = logits + model.final_logits_bias

    return logits


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
    ref_model,
    merged_model,
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

    ref_model.to(device)
    merged_model.to(device)

    ref_model.eval()
    merged_model.eval()

    rows = []

    for batch_idx, (
        encoded_inputs,
        _,
        all_choices,
    ) in enumerate(loader):
        if batch_idx % 20 == 0:
            logger.info(
                "[basis] batch=%d",
                batch_idx,
            )

        num_choices = [len(choices) for choices in all_choices]

        target_ids, target_mask = build_targets(
            tokenizer=tokenizer,
            all_choices=all_choices,
            max_length=max_length,
        )

        input_ids = encoded_inputs["input_ids"].to(device)

        attention_mask = encoded_inputs["attention_mask"].to(device)

        input_ids = repeat_rows(
            input_ids,
            num_choices,
        )

        attention_mask = repeat_rows(
            attention_mask,
            num_choices,
        )

        target_ids = target_ids.to(device)
        target_mask = target_mask.to(device)

        out_ref = ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        out_merged = merged_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        h_ref = out_ref.decoder_hidden_states[-1]
        h_merged = out_merged.decoder_hidden_states[-1]

        mask = target_mask.bool()

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
def evaluate_model(
    model,
    tokenizer,
    dataset,
    task: str,
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

    model.to(device)
    model.eval()

    all_scores = []

    for batch_idx, (
        encoded_inputs,
        _,
        all_choices,
    ) in enumerate(loader):
        if batch_idx % 20 == 0:
            logger.info(
                "[eval-model] batch=%d",
                batch_idx,
            )

        num_choices = [len(choices) for choices in all_choices]

        target_ids, target_mask = build_targets(
            tokenizer=tokenizer,
            all_choices=all_choices,
            max_length=max_length,
        )

        input_ids = encoded_inputs["input_ids"].to(device)

        attention_mask = encoded_inputs["attention_mask"].to(device)

        input_ids = repeat_rows(
            input_ids,
            num_choices,
        )

        attention_mask = repeat_rows(
            attention_mask,
            num_choices,
        )

        target_ids = target_ids.to(device)
        target_mask = target_mask.to(device)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
            return_dict=True,
            use_cache=False,
        )

        seq_scores = compute_scores_from_logits(
            logits=output.logits,
            target_ids=target_ids,
            target_mask=target_mask,
            length_normalization=length_normalization,
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
    ref_model,
    merged_model,
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

    ref_model.to(device)
    merged_model.to(device)

    ref_model.eval()
    merged_model.eval()

    all_scores = []

    for batch_idx, (
        encoded_inputs,
        _,
        all_choices,
    ) in enumerate(loader):
        if batch_idx % 20 == 0:
            logger.info(
                "[eval-edit] batch=%d",
                batch_idx,
            )

        num_choices = [len(choices) for choices in all_choices]

        target_ids, target_mask = build_targets(
            tokenizer=tokenizer,
            all_choices=all_choices,
            max_length=max_length,
        )

        input_ids = encoded_inputs["input_ids"].to(device)

        attention_mask = encoded_inputs["attention_mask"].to(device)

        input_ids = repeat_rows(
            input_ids,
            num_choices,
        )

        attention_mask = repeat_rows(
            attention_mask,
            num_choices,
        )

        target_ids = target_ids.to(device)
        target_mask = target_mask.to(device)

        out_ref = ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        out_merged = merged_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        h_ref = out_ref.decoder_hidden_states[-1]
        h_merged = out_merged.decoder_hidden_states[-1]

        h_edit = edit_hidden_sequence(
            h_ref=h_ref,
            h_merged=h_merged,
            mask=target_mask,
            basis=basis,
        )

        logits = logits_from_hidden(
            merged_model,
            h_edit,
        )

        seq_scores = compute_scores_from_logits(
            logits=logits,
            target_ids=target_ids,
            target_mask=target_mask,
            length_normalization=length_normalization,
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
    reference_model: str,
    merged_model: str,
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
    max_length: int = 128,
    seed: int = 42,
    device: str = "auto",
    dtype: str | None = None,
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
    logger.info("Encoder-decoder hidden-state intervention")
    logger.info(
        "Stage: %s",
        stage,
    )
    logger.info(
        "Reference model: %s",
        reference_model,
    )
    logger.info(
        "Merged model: %s",
        merged_model,
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

    ref_model, merged_model_obj, tokenizer = load_model_and_tokenizer(
        reference_model=reference_model,
        merged_model=merged_model,
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
        "reference_model": reference_model,
        "merged_model": merged_model,
        "task": task,
        "split": split,
        "n_eval": n_eval,
        "batch_size": batch_size,
        "max_length": max_length,
        "seed": seed,
        "device": device,
        "dtype": dtype,
        "length_normalization": length_normalization,
    }

    if stage == "fit_basis":
        logger.info("Collecting distortion rows")

        delta_rows = collect_delta_rows(
            ref_model=ref_model,
            merged_model=merged_model_obj,
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

        ref_metrics = evaluate_model(
            model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            task=task,
            batch_size=batch_size,
            max_length=max_length,
            device=device_obj,
            length_normalization=length_normalization,
        )

        logger.info("Computing merged metrics")

        merged_metrics = evaluate_model(
            model=merged_model_obj,
            tokenizer=tokenizer,
            dataset=dataset,
            task=task,
            batch_size=batch_size,
            max_length=max_length,
            device=device_obj,
            length_normalization=length_normalization,
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
        raise FileNotFoundError(f"Missing basis file: {basis_path_obj}")

    if not metrics_path_obj.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path_obj}")

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
        ref_model=ref_model,
        merged_model=merged_model_obj,
        tokenizer=tokenizer,
        dataset=dataset,
        task=task,
        batch_size=batch_size,
        max_length=max_length,
        device=device_obj,
        basis=basis,
        length_normalization=length_normalization,
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
