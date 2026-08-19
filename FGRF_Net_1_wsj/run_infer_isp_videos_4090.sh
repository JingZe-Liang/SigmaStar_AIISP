#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISP_ROOT="/HardDisk/jingzeliang/projects/Focal_Breathing_Solving/playground/Competitors/RViDeformer-main/opencv_fixed_raw_compare_isp"
DEVICE="${DEVICE:-cuda:0}"
FPS="${FPS:-30}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SCRIPT_DIR/checkpoints_pseudogt_static_4090}"
CHECKPOINT="${1:-}"

if [[ -z "$CHECKPOINT" ]]; then
  shopt -s nullglob
  BEST_CANDIDATES=("$CHECKPOINT_DIR"/*best*.pt)
  shopt -u nullglob
  if (( ${#BEST_CANDIDATES[@]} > 0 )); then
    CHECKPOINT="${BEST_CANDIDATES[0]}"
  else
    CHECKPOINT="$CHECKPOINT_DIR/fgrf_epoch_050.pt"
    echo "No $CHECKPOINT_DIR/*best*.pt found; using final checkpoint: $CHECKPOINT" >&2
  fi
fi

if [[ "$CHECKPOINT" != /* ]]; then
  CHECKPOINT="$SCRIPT_DIR/$CHECKPOINT"
fi
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }
[[ -d "$ISP_ROOT" ]] || { echo "ISP root not found: $ISP_ROOT" >&2; exit 2; }

run_scene() {
  local scene="$1"
  local config_stem="${scene%x}"
  local config="$SCRIPT_DIR/config_train_${config_stem}.json"
  local output="$SCRIPT_DIR/inference_${scene}_best"
  echo "===== ${scene}: FGRF inference on ${DEVICE} ====="
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
    python "$SCRIPT_DIR/infer.py" \
      --config "$config" \
      --checkpoint "$CHECKPOINT" \
      --output-dir "$output" \
      --device "$DEVICE" \
      --save-raw

  echo "===== ${scene}: ISP highlight recovery + adaptive Reinhard -> MP4 ====="
  python "$SCRIPT_DIR/render_isp_video.py" \
    --raw-dir "$output/fused_raw_frames" \
    --config "$config" \
    --isp-root "$ISP_ROOT" \
    --output-mp4 "$output/${scene}_fused_isp_highlight_adaptive.mp4" \
    --fps "$FPS" \
    --enable-highlight-recovery \
    --white-balance folder \
    --tone-map adaptive-reinhard
}

run_scene 128x
run_scene 645x
echo "Done. RAW frames and MP4 files are under: $SCRIPT_DIR/inference_128x_best and $SCRIPT_DIR/inference_645x_best"
