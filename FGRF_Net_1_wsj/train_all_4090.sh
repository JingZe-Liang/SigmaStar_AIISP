#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
SAVE_DIR="${SAVE_DIR:-$SCRIPT_DIR/checkpoints_pseudogt_static_4090}"

cd "$SCRIPT_DIR"

python verify_data.py \
  --config config_train_128.json \
  --config config_train_645.json

# nvidia-smi GPU 1 is the RTX 4090 on this server; after filtering it is cuda:0.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python train.py \
  --config config_train_128.json \
  --config config_train_645.json \
  --device cuda:0 \
  --expected-gpu "RTX 4090" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --learning-rate "$LEARNING_RATE" \
  --amp \
  --save-dir "$SAVE_DIR" \
  "$@"
