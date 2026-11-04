#!/bin/bash
set -euo pipefail

MODEL="microsoft/phi-3.5-mini-instruct"
EXP_DIR="experiments/phi35"
FT_DIR="${EXP_DIR}/ft"
WORK_DIR="${EXP_DIR}/results/selection/ta_baselines_6tasks"
LOG_PATH="${EXP_DIR}/logs/select_ta_scale_baselines_6tasks.log"

ADAPTER_PATHS='[
  "'${FT_DIR}'/paws/best_adapter",
  "'${FT_DIR}'/wiki_qa/best_adapter",
  "'${FT_DIR}'/qasc/best_adapter",
  "'${FT_DIR}'/quartz/best_adapter",
  "'${FT_DIR}'/winogrande/best_adapter",
  "'${FT_DIR}'/wsc/best_adapter"
]'

TASKS='[
  "paws",
  "wiki_qa",
  "qasc",
  "quartz",
  "winogrande",
  "wsc"
]'

LAMBDA_GRID='[0.1,0.3,0.5,0.7,1.0]'

mkdir -p "${EXP_DIR}/results/selection" "${EXP_DIR}/logs"

poetry run python -m mrd.causal_lm.select_ta_scale_baselines \
  --model_name_or_path "${MODEL}" \
  --adapter_paths "${ADAPTER_PATHS}" \
  --tasks "${TASKS}" \
  --lambda_grid "${LAMBDA_GRID}" \
  --work_dir "${WORK_DIR}" \
  --split train \
  --n_eval 200 \
  --batch_size 8 \
  --max_length 512 \
  --seed 42 \
  --dtype bfloat16 \
  --log_path "${LOG_PATH}"
