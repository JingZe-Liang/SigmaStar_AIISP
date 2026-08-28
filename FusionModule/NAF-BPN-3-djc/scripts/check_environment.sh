#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

"${python_bin}" -m py_compile \
  "${root}/model.py" \
  "${root}/data.py" \
  "${root}/losses.py" \
  "${root}/train.py" \
  "${root}/cache_motion.py" \
  "${root}/pipeline.py" \
  "${root}/preflight.py" \
  "${root}/motion_detection/robust_raw_md.py"
"${python_bin}" "${root}/preflight.py" --config "${root}/configs/cloud.json"
