#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pid_file="${root}/runs/pipeline/pipeline.pid"
state_file="${root}/runs/pipeline/pipeline_state.json"

if [[ -s "${pid_file}" ]]; then
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Pipeline: running (PID ${pid})"
  else
    echo "Pipeline: not running (stale PID ${pid})"
  fi
else
  echo "Pipeline: no PID file"
fi

if [[ -f "${state_file}" ]]; then
  echo "State:"
  cat "${state_file}"
fi

latest_log="$(find "${root}/logs" -maxdepth 1 -type f -name 'pipeline_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -n "${latest_log}" ]]; then
  echo "Latest log: ${latest_log}"
  tail -n 20 "${latest_log}"
fi
