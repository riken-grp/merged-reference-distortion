#!/bin/bash
set -euo pipefail

# Supported tasks:
# paws, wiki_qa, qasc, quartz, winogrande, wsc
TASK="${1:?Usage: $0 TASK}"

MODEL="microsoft/phi-3.5-mini-instruct"
ADAPTER_DIR="experiments/phi35/merged/ties_6tasks"
LOG_PATH="experiments/phi35/logs/evaluate_ties_${TASK}.log"

mkdir -p experiments/phi35/logs

poetry run python -m mrd.causal_lm.evaluate \
  --model_name_or_path "${MODEL}" \
  --adapter_name_or_path "${ADAPTER_DIR}" \
  --task "${TASK}" \
  --split validation \
  --batch_size 16 \
  --max_length 512 \
  --dtype bfloat16 \
  --log_path "${LOG_PATH}"
