#!/bin/bash
set -euo pipefail

# Supported tasks:
# paws, wiki_qa, qasc, quartz, winogrande, wsc
TASK="${1:?Usage: $0 TASK}"

EXP_DIR="experiments/t5-large"
REFERENCE_MODEL="${EXP_DIR}/ft/${TASK}/best_checkpoint"
MERGED_MODEL="${EXP_DIR}/merged/ta_6tasks"

REPS_DIR="${EXP_DIR}/reps/train_n1000"
REF_OUT="${REPS_DIR}/${TASK}__ref__train_n1000_seq.npz"
MERGED_OUT="${REPS_DIR}/${TASK}__ta__train_n1000_seq.npz"

REF_LOG="${EXP_DIR}/logs/extract_reps_${TASK}_ref_train_n1000.log"
MERGED_LOG="${EXP_DIR}/logs/extract_reps_${TASK}_ta_train_n1000.log"

mkdir -p "${REPS_DIR}" "${EXP_DIR}/logs"

poetry run python -m mrd.enc_dec.extract_reps \
  "${REFERENCE_MODEL}" \
  "${TASK}" \
  --split train \
  --n_eval 1000 \
  --batch_size 16 \
  --max_length 128 \
  --seed 42 \
  --layers '[-1]' \
  --out "${REF_OUT}" \
  --log_path "${REF_LOG}"

poetry run python -m mrd.enc_dec.extract_reps \
  "${MERGED_MODEL}" \
  "${TASK}" \
  --split train \
  --n_eval 1000 \
  --batch_size 16 \
  --max_length 128 \
  --seed 42 \
  --layers '[-1]' \
  --out "${MERGED_OUT}" \
  --log_path "${MERGED_LOG}"
