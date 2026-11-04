#!/bin/bash
set -euo pipefail

# Supported tasks:
# paws, wiki_qa, qasc, quartz, winogrande, wsc
TASK="${1:?Usage: $0 TASK}"

MODEL="google-t5/t5-large"
OUT_DIR="experiments/t5-large/ft/${TASK}"

poetry run python -m mrd.enc_dec.finetune \
  --model_name_or_path "${MODEL}" \
  --task "${TASK}" \
  --out_dir "${OUT_DIR}" \
  --num_batches 5000 \
  --checkpoint_frequency 2500 \
  --max_eval_samples 200 \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --max_length 128 \
  --lr 1e-5 \
  --weight_decay 0.0 \
  --seed 42
