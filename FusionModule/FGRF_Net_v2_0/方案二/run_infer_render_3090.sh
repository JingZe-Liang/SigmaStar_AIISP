#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCENE="${1:?Usage: bash run_infer_render_3090.sh <128x|645x|all>}"
GPU="${GPU:-0}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/checkpoints_sepconv_3090/fgrf_v2_best.pt}"
ISP_ROOT="${ISP_ROOT:-/HardDisk/jingzeliang/projects/Focal_Breathing_Solving/playground/Competitors/RViDeformer-main/opencv_fixed_raw_compare_isp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR}"
FPS="${FPS:-30}"
ENCODER="${ENCODER:-h264_nvenc}"
CQ="${CQ:-12}"
PIXEL_FORMAT="${PIXEL_FORMAT:-yuv420p}"
MAX_FRAMES="${MAX_FRAMES:-0}"
NO_LABELS="${NO_LABELS:-1}"

[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }
[[ -d "$ISP_ROOT" ]] || { echo "ISP root not found: $ISP_ROOT" >&2; exit 2; }

case "$SCENE" in
  128x|645x) SCENES=("$SCENE") ;;
  all) SCENES=(128x 645x) ;;
  *) echo "Scene must be 128x, 645x, or all: $SCENE" >&2; exit 2 ;;
esac

for scene in "${SCENES[@]}"; do
  config="$SCRIPT_DIR/config_train_${scene%x}.json"
  output="$OUTPUT_ROOT/inference_v2_$scene"
  infer_args=(--config "$config" --checkpoint "$CHECKPOINT" --output-dir "$output" --device cuda:0 --save-raw)
  render_args=(--config "$config" --fused-raw-dir "$output/fused_raw_frames" --isp-root "$ISP_ROOT" --output-mp4 "$output/${scene}_quad_fixed_scale.mp4" --fps "$FPS" --encoder "$ENCODER" --cq "$CQ" --pixel-format "$PIXEL_FORMAT" --overwrite)
  if [[ "$NO_LABELS" == "1" ]]; then render_args+=(--no-labels); fi
  if [[ "$MAX_FRAMES" != "0" ]]; then
    infer_args+=(--max-frames "$MAX_FRAMES")
    render_args+=(--max-frames "$MAX_FRAMES")
  fi
  if [[ "${FIXED_SCALE:-}" != "" ]]; then render_args+=(--fixed-scale "$FIXED_SCALE"); fi
  if [[ "${LOSSLESS:-0}" == "1" ]]; then render_args+=(--lossless); fi

  echo "===== $scene: v2 inference -> 16-bit fused RAW ====="
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" PYTHONDONTWRITEBYTECODE=1 \
    python "$SCRIPT_DIR/infer.py" "${infer_args[@]}"
  echo "===== $scene: fixed-scale 4K quad MP4 ====="
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" PYTHONDONTWRITEBYTECODE=1 \
    python "$SCRIPT_DIR/render_quad_fixed_isp.py" "${render_args[@]}"
done
