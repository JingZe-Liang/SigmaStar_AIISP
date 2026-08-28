#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pid_file="${root}/runs/pipeline/pipeline.pid"

if [[ ! -s "${pid_file}" ]]; then
  echo "No pipeline PID file found." >&2
  exit 1
fi
pid="$(cat "${pid_file}")"
command="$(ps -o args= -p "${pid}" 2>/dev/null || true)"
if [[ "${command}" != *"${root}/pipeline.py"* ]]; then
  echo "Refusing to stop PID ${pid}: it is not this NAFBPNNet pipeline." >&2
  exit 1
fi
kill -- "-${pid}"
rm -f "${pid_file}"
echo "Sent SIGTERM to NAFBPNNet pipeline process group ${pid}."
