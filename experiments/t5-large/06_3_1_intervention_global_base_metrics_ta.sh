#!/bin/bash
set -euo pipefail

# Supported tasks:
# paws, wiki_qa, qasc, quartz, winogrande, wsc
TASK="${1:?Usage: $0 TASK}"

EXP_DIR="experiments/t5-large"
REFERENCE_MODEL="${EXP_DIR}/ft/${TASK}/best_checkpoint"
MERGED_MODEL="${EXP_DIR}/merged/ta_6tasks"

OUT_DIR="${EXP_DIR}/results/intervention_global/ta/${TASK}"
METRICS_PATH="${OUT_DIR}/base_metrics.json"
LOG_PATH="${EXP_DIR}/logs/intervention_global_base_ta_${TASK}.log"

mkdir -p "${OUT_DIR}" "${EXP_DIR}/logs"

poetry run python -m mrd.enc_dec.intervention_global \
  "${REFERENCE_MODEL}" \
  "${MERGED_MODEL}" \
  "${TASK}" \
  "${OUT_DIR}" \
  --stage base_metrics \
  --metrics_path "${METRICS_PATH}" \
  --split validation \
  --batch_size 16 \
  --max_length 128 \
  --seed 42 \
  --log_path "${LOG_PATH}"
