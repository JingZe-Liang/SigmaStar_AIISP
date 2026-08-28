#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

"${python_bin}" -m pip install --upgrade pip
"${python_bin}" -m pip install -r "${root}/requirements.txt"
