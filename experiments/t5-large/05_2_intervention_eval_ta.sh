#!/bin/bash
set -euo pipefail

# Tasks: paws, wiki_qa, qasc, quartz, winogrande, wsc
# Conditions: topk, random
# K values: 1, 2, 5, 10, 20
# Example:
# ./experiments/t5-large/05_2_intervention_eval_ta.sh wsc topk 10

TASK="${1:?Usage: $0 TASK CONDITION K}"
CONDITION="${2:?Usage: $0 TASK CONDITION K}"
K="${3:?Usage: $0 TASK CONDITION K}"

EXP_DIR="experiments/t5-large"
REFERENCE_MODEL="${EXP_DIR}/ft/${TASK}/best_checkpoint"
MERGED_MODEL="${EXP_DIR}/merged/ta_6tasks"
OUT_DIR="${EXP_DIR}/results/intervention/ta/${TASK}"
BASIS_PATH="${OUT_DIR}/basis.npy"
METRICS_PATH="${OUT_DIR}/base_metrics.json"
LOG_PATH="${EXP_DIR}/logs/intervention_eval_ta_${TASK}_${CONDITION}_k${K}.log"

poetry run python -m mrd.enc_dec.intervention \
  "${REFERENCE_MODEL}" \
  "${MERGED_MODEL}" \
  "${TASK}" \
  "${OUT_DIR}" \
  eval \
  --basis_path "${BASIS_PATH}" \
  --metrics_path "${METRICS_PATH}" \
  --condition "${CONDITION}" \
  --k "${K}" \
  --split validation \
  --batch_size 16 \
  --max_length 128 \
  --seed 42 \
  --log_path "${LOG_PATH}"
