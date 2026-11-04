#!/bin/bash
set -euo pipefail

EXP_DIR="experiments/phi35"
FT_DIR="${EXP_DIR}/ft"
OUT_DIR="${EXP_DIR}/merged/average_6tasks"
LOG_PATH="${EXP_DIR}/logs/merge_average_6tasks.log"

ADAPTER_PATHS='[
  "'${FT_DIR}'/paws/best_adapter",
  "'${FT_DIR}'/wiki_qa/best_adapter",
  "'${FT_DIR}'/qasc/best_adapter",
  "'${FT_DIR}'/quartz/best_adapter",
  "'${FT_DIR}'/winogrande/best_adapter",
  "'${FT_DIR}'/wsc/best_adapter"
]'

mkdir -p "${EXP_DIR}/merged" "${EXP_DIR}/logs"

poetry run python -m mrd.causal_lm.merge_average \
  --adapter_paths "${ADAPTER_PATHS}" \
  --out_dir "${OUT_DIR}" \
  --log_path "${LOG_PATH}"
