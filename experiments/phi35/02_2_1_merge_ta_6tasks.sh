#!/bin/bash
set -euo pipefail

MODEL="microsoft/phi-3.5-mini-instruct"
EXP_DIR="experiments/phi35"
FT_DIR="${EXP_DIR}/ft"
OUT_DIR="${EXP_DIR}/merged/ta_6tasks"
LOG_PATH="${EXP_DIR}/logs/merge_ta_6tasks.log"

ADAPTER_PATHS='[
  "'${FT_DIR}'/paws/best_adapter",
  "'${FT_DIR}'/wiki_qa/best_adapter",
  "'${FT_DIR}'/qasc/best_adapter",
  "'${FT_DIR}'/quartz/best_adapter",
  "'${FT_DIR}'/winogrande/best_adapter",
  "'${FT_DIR}'/wsc/best_adapter"
]'

TASKS='["paws","wiki_qa","qasc","quartz","winogrande","wsc"]'
LAMBDA_GRID='[0.1,0.3,0.5,0.7,1.0]'

mkdir -p "${EXP_DIR}/merged" "${EXP_DIR}/logs"

poetry run python -m mrd.causal_lm.merge_ta \
  --model_name_or_path "${MODEL}" \
  --adapter_paths "${ADAPTER_PATHS}" \
  --tasks "${TASKS}" \
  --lambda_grid "${LAMBDA_GRID}" \
  --split train \
  --n_eval 200 \
  --batch_size 8 \
  --max_length 512 \
  --out_dir "${OUT_DIR}" \
  --seed 42 \
  --dtype bfloat16 \
  --log_path "${LOG_PATH}"
