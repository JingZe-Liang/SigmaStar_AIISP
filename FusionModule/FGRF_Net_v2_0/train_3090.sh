#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-4}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-2048}"
VAL_SAMPLES="${VAL_SAMPLES:-256}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
SAVE_DIR="${SAVE_DIR:-$SCRIPT_DIR/checkpoints_v2_3090}"
MAX_STEPS="${MAX_STEPS:-0}"
EXPECTED_GPU="${EXPECTED_GPU:-RTX 3090}"

cd "$SCRIPT_DIR"
PYTHONDONTWRITEBYTECODE=1 python verify_data.py \
  --config config_train_128.json \
  --config config_train_645.json

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONDONTWRITEBYTECODE=1 python train.py \
  --config config_train_128.json \
  --config config_train_645.json \
  --device cuda:0 \
  --expected-gpu "$EXPECTED_GPU" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --train-samples "$TRAIN_SAMPLES" \
  --val-samples "$VAL_SAMPLES" \
  --learning-rate "$LEARNING_RATE" \
  --save-dir "$SAVE_DIR" \
  --max-steps "$MAX_STEPS" \
  --amp
