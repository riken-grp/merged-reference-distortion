#!/bin/bash
set -euo pipefail

# Tasks: paws, wiki_qa, qasc, quartz, winogrande, wsc
# Conditions: topk, random
# Global-basis experiment uses k=10.
# Example:
# ./experiments/phi35/06_3_2_intervention_global_eval_ta.sh wsc topk

TASK="${1:?Usage: $0 TASK CONDITION}"
CONDITION="${2:?Usage: $0 TASK CONDITION}"

MODEL="microsoft/phi-3.5-mini-instruct"
EXP_DIR="experiments/phi35"
REFERENCE_ADAPTER="${EXP_DIR}/ft/${TASK}/best_adapter"
MERGED_ADAPTER="${EXP_DIR}/merged/ta_6tasks"

OUT_DIR="${EXP_DIR}/results/intervention_global/ta/${TASK}"
BASIS_PATH="${EXP_DIR}/results/global_basis/ta_6tasks_n1000/basis.npy"
METRICS_PATH="${OUT_DIR}/base_metrics.json"
LOG_PATH="${EXP_DIR}/logs/intervention_global_eval_ta_${TASK}_${CONDITION}_k10.log"

poetry run python -m mrd.causal_lm.intervention_global \
  "${MODEL}" \
  "${REFERENCE_ADAPTER}" \
  "${MERGED_ADAPTER}" \
  "${TASK}" \
  "${OUT_DIR}" \
  --stage eval \
  --basis_path "${BASIS_PATH}" \
  --metrics_path "${METRICS_PATH}" \
  --condition "${CONDITION}" \
  --k 10 \
  --split validation \
  --batch_size 16 \
  --max_length 512 \
  --seed 42 \
  --dtype bfloat16 \
  --log_path "${LOG_PATH}"
