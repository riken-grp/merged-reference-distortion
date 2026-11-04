#!/bin/bash
set -euo pipefail

# Supported tasks:
# paws, wiki_qa, qasc, quartz, winogrande, wsc
TASK="${1:?Usage: $0 TASK}"

MODEL="microsoft/phi-3.5-mini-instruct"
EXP_DIR="experiments/phi35"
REFERENCE_ADAPTER="${EXP_DIR}/ft/${TASK}/best_adapter"
MERGED_ADAPTER="${EXP_DIR}/merged/ta_6tasks"

REPS_DIR="${EXP_DIR}/reps/train_n1000"
REF_OUT="${REPS_DIR}/${TASK}__ref__train_n1000_seq.npz"
MERGED_OUT="${REPS_DIR}/${TASK}__ta__train_n1000_seq.npz"

REF_LOG="${EXP_DIR}/logs/extract_reps_${TASK}_ref_train_n1000.log"
MERGED_LOG="${EXP_DIR}/logs/extract_reps_${TASK}_ta_train_n1000.log"

mkdir -p "${REPS_DIR}" "${EXP_DIR}/logs"

poetry run python -m mrd.causal_lm.extract_reps \
  "${MODEL}" \
  "${TASK}" \
  --adapter_name_or_path "${REFERENCE_ADAPTER}" \
  --split train \
  --n_eval 1000 \
  --batch_size 8 \
  --max_length 512 \
  --seed 42 \
  --dtype bfloat16 \
  --layers '[-1]' \
  --out "${REF_OUT}" \
  --log_path "${REF_LOG}"

poetry run python -m mrd.causal_lm.extract_reps \
  "${MODEL}" \
  "${TASK}" \
  --adapter_name_or_path "${MERGED_ADAPTER}" \
  --split train \
  --n_eval 1000 \
  --batch_size 8 \
  --max_length 512 \
  --seed 42 \
  --dtype bfloat16 \
  --layers '[-1]' \
  --out "${MERGED_OUT}" \
  --log_path "${MERGED_LOG}"
