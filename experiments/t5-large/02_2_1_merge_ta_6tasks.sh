#!/bin/bash
set -euo pipefail

MODEL="google-t5/t5-large"
EXP_DIR="experiments/t5-large"
FT_DIR="${EXP_DIR}/ft"
OUT_DIR="${EXP_DIR}/merged/ta_6tasks"
LOG_PATH="${EXP_DIR}/logs/merge_ta_6tasks.log"

FT_MODELS='[
  "'${FT_DIR}'/paws/best_checkpoint",
  "'${FT_DIR}'/wiki_qa/best_checkpoint",
  "'${FT_DIR}'/qasc/best_checkpoint",
  "'${FT_DIR}'/quartz/best_checkpoint",
  "'${FT_DIR}'/winogrande/best_checkpoint",
  "'${FT_DIR}'/wsc/best_checkpoint"
]'

TASKS='["paws","wiki_qa","qasc","quartz","winogrande","wsc"]'
LAMBDA_GRID='[0.1,0.3,0.5,0.7,1.0]'

mkdir -p "${EXP_DIR}/merged" "${EXP_DIR}/logs"

poetry run python -m mrd.enc_dec.merge_ta \
  --base "${MODEL}" \
  --ft_models "${FT_MODELS}" \
  --tasks "${TASKS}" \
  --lambda_grid "${LAMBDA_GRID}" \
  --split train \
  --n_eval 200 \
  --batch_size 32 \
  --max_length 128 \
  --out_dir "${OUT_DIR}" \
  --seed 42 \
  --log_path "${LOG_PATH}"
