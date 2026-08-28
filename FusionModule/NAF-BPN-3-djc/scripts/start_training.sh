#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
state_dir="${root}/runs/pipeline"
log_dir="${root}/logs"
pid_file="${state_dir}/pipeline.pid"

mkdir -p "${state_dir}" "${log_dir}"
if [[ -s "${pid_file}" ]]; then
  old_pid="$(cat "${pid_file}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "Training pipeline is already running: PID ${old_pid}" >&2
    exit 1
  fi
  rm -f "${pid_file}"
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${log_dir}/pipeline_${timestamp}.log"
nohup setsid "${python_bin}" -u "${root}/pipeline.py" --config "${root}/configs/cloud.json" "$@" >"${log_file}" 2>&1 < /dev/null &
pid="$!"
echo "${pid}" > "${pid_file}"
echo "Started NAFBPNNet pipeline."
echo "PID: ${pid}"
echo "Log: ${log_file}"
echo "Status: bash scripts/status_training.sh"
