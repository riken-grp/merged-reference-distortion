#!/bin/bash
set -euo pipefail

# Supported tasks:
# paws, wiki_qa, qasc, quartz, winogrande, wsc
TASK="${1:?Usage: $0 TASK}"

MODEL_DIR="experiments/t5-large/merged/ta_distortion_6tasks"
LOG_PATH="experiments/t5-large/logs/evaluate_ta_distortion_${TASK}.log"

mkdir -p experiments/t5-large/logs

poetry run python -m mrd.enc_dec.evaluate \
  --model_name_or_path "${MODEL_DIR}" \
  --task "${TASK}" \
  --split validation \
  --batch_size 32 \
  --max_length 128 \
  --log_path "${LOG_PATH}"
