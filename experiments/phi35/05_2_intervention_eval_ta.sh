#!/bin/bash
set -euo pipefail

# Tasks: paws, wiki_qa, qasc, quartz, winogrande, wsc
# Conditions: topk, random
# K values: 1, 2, 5, 10, 20
# Example:
# ./experiments/phi35/05_2_intervention_eval_ta.sh wsc topk 10

TASK="${1:?Usage: $0 TASK CONDITION K}"
CONDITION="${2:?Usage: $0 TASK CONDITION K}"
K="${3:?Usage: $0 TASK CONDITION K}"

MODEL="microsoft/phi-3.5-mini-instruct"
EXP_DIR="experiments/phi35"
REFERENCE_ADAPTER="${EXP_DIR}/ft/${TASK}/best_adapter"
MERGED_ADAPTER="${EXP_DIR}/merged/ta_6tasks"
OUT_DIR="${EXP_DIR}/results/intervention/ta/${TASK}"
BASIS_PATH="${OUT_DIR}/basis.npy"
METRICS_PATH="${OUT_DIR}/base_metrics.json"
LOG_PATH="${EXP_DIR}/logs/intervention_eval_ta_${TASK}_${CONDITION}_k${K}.log"

poetry run python -m mrd.causal_lm.intervention \
  "${MODEL}" \
  "${REFERENCE_ADAPTER}" \
  "${MERGED_ADAPTER}" \
  "${TASK}" \
  "${OUT_DIR}" \
  eval \
  --basis_path "${BASIS_PATH}" \
  --metrics_path "${METRICS_PATH}" \
  --condition "${CONDITION}" \
  --k "${K}" \
  --split validation \
  --batch_size 16 \
  --max_length 512 \
  --seed 42 \
  --dtype bfloat16 \
  --log_path "${LOG_PATH}"
