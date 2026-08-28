#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
smoke_root="${root}/artifacts/smoke/${stamp}"
config="${root}/configs/cloud.json"

bash "${root}/scripts/check_environment.sh"
"${python_bin}" -u "${root}/train.py" --stage 1 --config "${config}" --smoke-test --output "${smoke_root}/stage1"
"${python_bin}" -u "${root}/cache_motion.py" --config "${config}" --sequence all
"${python_bin}" -u "${root}/train.py" --stage 2 --fold 128_to_645 --config "${config}" --smoke-test --output "${smoke_root}/stage2" --init-checkpoint "${smoke_root}/stage1/best.pth"
"${python_bin}" -u "${root}/train.py" --stage 2 --fold 128_to_645 --config "${config}" --smoke-test --output "${smoke_root}/stage2" --resume "${smoke_root}/stage2/last.pth"
echo "Smoke test completed: ${smoke_root}"
