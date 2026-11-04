#!/bin/bash
set -euo pipefail

# Supported tasks:
# paws, wiki_qa, qasc, quartz, winogrande, wsc
TASK="${1:?Usage: $0 TASK}"

MODEL="microsoft/phi-3.5-mini-instruct"
OUT_DIR="experiments/phi35/ft/${TASK}"

poetry run python -m mrd.causal_lm.finetune \
  --model_name_or_path "${MODEL}" \
  --task "${TASK}" \
  --out_dir "${OUT_DIR}" \
  --train_batch_size 4 \
  --gradient_accumulation_factor 4 \
  --num_batches 3000 \
  --checkpoint_frequency 500 \
  --eval_split train \
  --eval_batch_size 8 \
  --max_eval_samples 500 \
  --max_length 512 \
  --lr 1e-4 \
  --seed 42 \
  --dtype bfloat16 \
  --target_modules "qkv_proj,o_proj"
