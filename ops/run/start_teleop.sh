#!/usr/bin/env bash
# One-command PICO + dual FR3 + MANUS/Wuji teleoperation.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
FRANKA_SERVICE=franka-control
OPERATOR_ARGS=()
for argument in "$@"; do
  if [[ "$argument" == "--fake-franka" ]]; then
    FRANKA_SERVICE=fake-franka-control
  else
    OPERATOR_ARGS+=("$argument")
  fi
done
SERVICES=("$FRANKA_SERVICE" teleop-control pico-bridge)
stack_started=false
operator_pid=""

pid_alive() {
  local pid=$1
  [[ -n "$pid" ]] && kill -0 -- "$pid" 2>/dev/null
}

wait_until_dead() {
  local pid=$1
  local attempts=$2
  local attempt
  for ((attempt = 0; attempt < attempts; attempt++)); do
    pid_alive "$pid" || return 0
    sleep 0.1
  done
  return 1
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if pid_alive "$operator_pid"; then
    kill -TERM -- "$operator_pid" 2>/dev/null || true
    if ! wait_until_dead "$operator_pid" 200; then
      echo "Operator supervisor did not stop; sending SIGKILL..." >&2
      kill -KILL -- "$operator_pid" 2>/dev/null || true
    fi
  fi
  [[ -z "$operator_pid" ]] || wait "$operator_pid" 2>/dev/null || true
  if [[ "$stack_started" == true ]]; then
    (cd "$REPO_ROOT/docker" && docker compose stop --timeout 10 "${SERVICES[@]}") || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

"$REPO_ROOT/ops/run/preflight.sh"
cd "$REPO_ROOT/docker"
docker compose up -d "${SERVICES[@]}"
stack_started=true
docker compose ps "${SERVICES[@]}"

cd "$REPO_ROOT"
"$REPO_ROOT/ops/run/run_operator.sh" "${OPERATOR_ARGS[@]}" &
operator_pid=$!
set +e
wait "$operator_pid"
status=$?
set -e
operator_pid=""
exit "$status"
