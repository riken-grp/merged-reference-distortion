#!/bin/bash
set -euo pipefail

MODEL="google-t5/t5-large"
EXP_DIR="experiments/t5-large"
FT_DIR="${EXP_DIR}/ft"
OUT_DIR="${EXP_DIR}/merged/ties_6tasks"
LOG_PATH="${EXP_DIR}/logs/merge_ties_6tasks.log"

FT_MODELS='[
  "'${FT_DIR}'/paws/best_checkpoint",
  "'${FT_DIR}'/wiki_qa/best_checkpoint",
  "'${FT_DIR}'/qasc/best_checkpoint",
  "'${FT_DIR}'/quartz/best_checkpoint",
  "'${FT_DIR}'/winogrande/best_checkpoint",
  "'${FT_DIR}'/wsc/best_checkpoint"
]'

TASKS='[
  "paws",
  "wiki_qa",
  "qasc",
  "quartz",
  "winogrande",
  "wsc"
]'

MERGE_SPECS='[
  "topk10_mass_dis-sum_0.1",
  "topk20_mass_dis-sum_0.1",
  "topk30_mass_dis-sum_0.1",
  "topk10_mass_dis-sum_0.3",
  "topk20_mass_dis-sum_0.3",
  "topk30_mass_dis-sum_0.3",
  "topk10_mass_dis-sum_0.5",
  "topk20_mass_dis-sum_0.5",
  "topk30_mass_dis-sum_0.5",
  "topk10_mass_dis-sum_0.7",
  "topk20_mass_dis-sum_0.7",
  "topk30_mass_dis-sum_0.7",
  "topk10_mass_dis-sum_1.0",
  "topk20_mass_dis-sum_1.0",
  "topk30_mass_dis-sum_1.0"
]'

mkdir -p "${EXP_DIR}/merged" "${EXP_DIR}/logs"

poetry run python -m mrd.enc_dec.merge_ties \
  --base "${MODEL}" \
  --ft_models "${FT_MODELS}" \
  --tasks "${TASKS}" \
  --merge_specs "${MERGE_SPECS}" \
  --split train \
  --n_eval 200 \
  --batch_size 32 \
  --max_length 128 \
  --out_dir "${OUT_DIR}" \
  --seed 42 \
  --log_path "${LOG_PATH}"
