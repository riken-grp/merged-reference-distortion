#!/bin/bash
set -euo pipefail

# Supported tasks:
# paws, wiki_qa, qasc, quartz, winogrande, wsc
TASK="${1:?Usage: $0 TASK}"

EXP_DIR="experiments/t5-large"
MODEL="${EXP_DIR}/merged/ta_6tasks"
OUT="${EXP_DIR}/reps/${TASK}__ta__validation_seq.npz"
LOG_PATH="${EXP_DIR}/logs/extract_reps_${TASK}_ta_validation.log"

mkdir -p "${EXP_DIR}/reps" "${EXP_DIR}/logs"

poetry run python -m mrd.enc_dec.extract_reps \
  --model_name_or_path "${MODEL}" \
  --task "${TASK}" \
  --split validation \
  --batch_size 16 \
  --max_length 128 \
  --seed 42 \
  --layers '[-1]' \
  --out "${OUT}" \
  --log_path "${LOG_PATH}"
