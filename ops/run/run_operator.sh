#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTROL_HOST="${TELEOP_CONTROL_HOST:-127.0.0.1}"
CONTROL_PORT="${TELEOP_CONTROL_PORT:-5590}"
TELEOP_CONDA_ENV="${TELEOP_CONDA_ENV:-pico-armhand-teleop}"
# shellcheck disable=SC1091
source "$REPO_ROOT/ops/lib/conda.sh"
CONDA_EXE_PATH="$(resolve_teleop_conda)"
GUI_PYTHON="$("$CONDA_EXE_PATH" run -n "$TELEOP_CONDA_ENV" python -c 'import sys; print(sys.executable)')"
backend_pid=""
gui_pid=""

group_alive() {
  kill -0 -- "-$1" 2>/dev/null
}

wait_for_group() {
  local pid=$1
  local attempts=$2
  local attempt
  for ((attempt = 0; attempt < attempts; attempt++)); do
    group_alive "$pid" || return 0
    sleep 0.1
  done
  return 1
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$backend_pid" ]]; then
    kill -INT -- "-$backend_pid" 2>/dev/null || true
  fi
  if [[ -n "$gui_pid" ]]; then
    kill -TERM -- "-$gui_pid" 2>/dev/null || true
  fi
  # Native MANUS/Wuji shutdown can take several seconds while subscriptions,
  # the device connection, and Core Integrated stop.  Let that cleanup finish
  # so WujiHand2Backend.close() can disable the hand before escalating.
  if [[ -n "$backend_pid" ]] && ! wait_for_group "$backend_pid" 100; then
    echo "Backend did not stop after SIGINT; sending SIGTERM..." >&2
    kill -TERM -- "-$backend_pid" 2>/dev/null || true
    wait_for_group "$backend_pid" 50 || {
      echo "Backend did not stop after SIGTERM; sending SIGKILL..." >&2
      kill -KILL -- "-$backend_pid" 2>/dev/null || true
    }
  fi
  [[ -z "$backend_pid" ]] || wait "$backend_pid" 2>/dev/null || true
  [[ -z "$gui_pid" ]] || wait "$gui_pid" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

occupied="$(lsof -t -iTCP:"$CONTROL_PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "$occupied" ]]; then
  echo "Control port $CONTROL_PORT is already used by PID(s): $occupied" >&2
  echo "Stop the previous operator before starting another one." >&2
  exit 1
fi

backend_command=(
  "$REPO_ROOT/ops/run/run_teleop.sh"
  --control-host "$CONTROL_HOST"
  --control-port "$CONTROL_PORT"
  "$@"
)
setsid "${backend_command[@]}" &
backend_pid=$!

setsid "$GUI_PYTHON" "$REPO_ROOT/apps/operator_gui/operator_gui.py" \
  --host "$CONTROL_HOST" --port "$CONTROL_PORT" &
gui_pid=$!

set +e
wait -n "$backend_pid" "$gui_pid"
status=$?
set -e
exit "$status"
