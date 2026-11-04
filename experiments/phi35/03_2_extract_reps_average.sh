#!/bin/bash
set -euo pipefail

# Supported tasks:
# paws, wiki_qa, qasc, quartz, winogrande, wsc
TASK="${1:?Usage: $0 TASK}"

MODEL="microsoft/phi-3.5-mini-instruct"
EXP_DIR="experiments/phi35"
ADAPTER="${EXP_DIR}/merged/average_6tasks"
OUT="${EXP_DIR}/reps/${TASK}__average__validation_seq.npz"
LOG_PATH="${EXP_DIR}/logs/extract_reps_${TASK}_average_validation.log"

mkdir -p "${EXP_DIR}/reps" "${EXP_DIR}/logs"

poetry run python -m mrd.causal_lm.extract_reps \
  --model_name_or_path "${MODEL}" \
  --adapter_name_or_path "${ADAPTER}" \
  --task "${TASK}" \
  --split validation \
  --batch_size 8 \
  --max_length 512 \
  --seed 42 \
  --dtype bfloat16 \
  --layers '[-1]' \
  --out "${OUT}" \
  --log_path "${LOG_PATH}"
