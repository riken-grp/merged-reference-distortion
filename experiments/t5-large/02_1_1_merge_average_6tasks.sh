#!/bin/bash
set -euo pipefail

MODEL="google-t5/t5-large"
EXP_DIR="experiments/t5-large"
FT_DIR="${EXP_DIR}/ft"
OUT_DIR="${EXP_DIR}/merged/average_6tasks"
LOG_PATH="${EXP_DIR}/logs/merge_average_6tasks.log"

FT_MODELS='[
  "'${FT_DIR}'/paws/best_checkpoint",
  "'${FT_DIR}'/wiki_qa/best_checkpoint",
  "'${FT_DIR}'/qasc/best_checkpoint",
  "'${FT_DIR}'/quartz/best_checkpoint",
  "'${FT_DIR}'/winogrande/best_checkpoint",
  "'${FT_DIR}'/wsc/best_checkpoint"
]'

mkdir -p "${EXP_DIR}/merged" "${EXP_DIR}/logs"

poetry run python -m mrd.enc_dec.merge_average \
  --base "${MODEL}" \
  --ft_models "${FT_MODELS}" \
  --out_dir "${OUT_DIR}" \
  --seed 42 \
  --log_path "${LOG_PATH}"
