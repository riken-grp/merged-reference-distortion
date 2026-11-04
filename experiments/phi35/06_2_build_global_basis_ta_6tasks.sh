#!/bin/bash
set -euo pipefail

EXP_DIR="experiments/phi35"
REPS_DIR="${EXP_DIR}/reps/train_n1000"
OUT_DIR="${EXP_DIR}/results/global_basis/ta_6tasks_n1000"
LOG_PATH="${EXP_DIR}/logs/analyze_global_ta_6tasks_train_n1000.log"

REFERENCE_NPZS='[
  "'${REPS_DIR}'/paws__ref__train_n1000_seq.npz",
  "'${REPS_DIR}'/wiki_qa__ref__train_n1000_seq.npz",
  "'${REPS_DIR}'/qasc__ref__train_n1000_seq.npz",
  "'${REPS_DIR}'/quartz__ref__train_n1000_seq.npz",
  "'${REPS_DIR}'/winogrande__ref__train_n1000_seq.npz",
  "'${REPS_DIR}'/wsc__ref__train_n1000_seq.npz"
]'

MERGED_NPZS='[
  "'${REPS_DIR}'/paws__ta__train_n1000_seq.npz",
  "'${REPS_DIR}'/wiki_qa__ta__train_n1000_seq.npz",
  "'${REPS_DIR}'/qasc__ta__train_n1000_seq.npz",
  "'${REPS_DIR}'/quartz__ta__train_n1000_seq.npz",
  "'${REPS_DIR}'/winogrande__ta__train_n1000_seq.npz",
  "'${REPS_DIR}'/wsc__ta__train_n1000_seq.npz"
]'

TASKS='["paws","wiki_qa","qasc","quartz","winogrande","wsc"]'

mkdir -p "${EXP_DIR}/results/global_basis" "${EXP_DIR}/logs"

poetry run python -m mrd.causal_lm.analyze_global \
  "${REFERENCE_NPZS}" \
  "${MERGED_NPZS}" \
  "${TASKS}" \
  "${OUT_DIR}" \
  --rep_key candidate_seq_l32 \
  --seed 42 \
  --log_path "${LOG_PATH}"
