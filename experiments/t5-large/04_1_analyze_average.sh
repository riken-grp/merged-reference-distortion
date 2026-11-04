#!/bin/bash
set -euo pipefail

# Supported tasks:
# paws, wiki_qa, qasc, quartz, winogrande, wsc
TASK="${1:?Usage: $0 TASK}"

EXP_DIR="experiments/t5-large"
REFERENCE_NPZ="${EXP_DIR}/reps/${TASK}__ref__validation_seq.npz"
MERGED_NPZ="${EXP_DIR}/reps/${TASK}__average__validation_seq.npz"
OUT_DIR="${EXP_DIR}/results/anisotropy/average/${TASK}"
LOG_PATH="${EXP_DIR}/logs/analyze_average_${TASK}.log"

mkdir -p "${EXP_DIR}/results/anisotropy/average" "${EXP_DIR}/logs"

poetry run python -m mrd.enc_dec.analyze \
  "${REFERENCE_NPZ}" \
  "${MERGED_NPZ}" \
  "${OUT_DIR}" \
  --rep_keys '["decoder_seq_l24"]' \
  --seed 42 \
  --log_path "${LOG_PATH}"
