#!/bin/bash
set -euo pipefail

# Supported tasks:
# paws, wiki_qa, qasc, quartz, winogrande, wsc
TASK="${1:?Usage: $0 TASK}"

MODEL="microsoft/phi-3.5-mini-instruct"
EXP_DIR="experiments/phi35"
REFERENCE_ADAPTER="${EXP_DIR}/ft/${TASK}/best_adapter"
MERGED_ADAPTER="${EXP_DIR}/merged/ta_6tasks"
OUT_DIR="${EXP_DIR}/results/intervention/ta/${TASK}"
LOG_PATH="${EXP_DIR}/logs/intervention_fit_basis_ta_${TASK}.log"

mkdir -p "${OUT_DIR}" "${EXP_DIR}/logs"

poetry run python -m mrd.causal_lm.intervention \
  "${MODEL}" \
  "${REFERENCE_ADAPTER}" \
  "${MERGED_ADAPTER}" \
  "${TASK}" \
  "${OUT_DIR}" \
  fit_basis \
  --split validation \
  --batch_size 16 \
  --max_length 512 \
  --seed 42 \
  --dtype bfloat16 \
  --log_path "${LOG_PATH}"
