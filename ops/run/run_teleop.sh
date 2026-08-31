#!/usr/bin/env bash
# Host-side PICO motion-tracker arms plus MANUS-to-Wuji hands.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TELEOP_CONDA_ENV="${TELEOP_CONDA_ENV:-pico-armhand-teleop}"
# shellcheck disable=SC1091
source "$REPO_ROOT/config/devices.env"
DATA_ROOT="${TELEOP_DATA_ROOT:-}"
if [[ -z "$DATA_ROOT" && -f "$REPO_ROOT/docker/.env" ]]; then
  DATA_ROOT="$(awk -F= '$1 == "TELEOP_DATA_ROOT" {sub(/^[^=]*=/, ""); print; exit}' \
    "$REPO_ROOT/docker/.env")"
fi
if [[ -z "$DATA_ROOT" ]]; then
  echo "TELEOP_DATA_ROOT is not set and is missing from docker/.env" >&2
  exit 1
fi

RUN_PARENT="${TELEOP_DIAGNOSTICS_ROOT:-$DATA_ROOT/diagnostics}"
if [[ -n "${TELEOP_RUN_DIR:-}" ]]; then
  RUN_DIR="$TELEOP_RUN_DIR"
  if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to reuse TELEOP_RUN_DIR: $RUN_DIR" >&2
    exit 1
  fi
  mkdir -- "$RUN_DIR"
else
  mkdir -p -- "$RUN_PARENT"
  RUN_DIR="$(mktemp -d "$RUN_PARENT/$(date +%Y%m%d_%H%M%S).XXXXXX")"
fi
echo "RUN_DIR=$RUN_DIR"

ARM_ARGS=(--arm-source motion-trackers)
HAND_ARGS=(
  --hand-source wuji
  --hand-debug-log "$RUN_DIR/wuji_commands.jsonl"
  --wuji-left-address "$WUJI_LEFT_ADDRESS"
  --wuji-right-address "$WUJI_RIGHT_ADDRESS"
)
for argument in "$@"; do
  if [[ "$argument" == "--arm-source" || "$argument" == --arm-source=* ]]; then
    ARM_ARGS=()
  fi
  if [[ "$argument" == "--hand-source" || "$argument" == --hand-source=* ]]; then
    HAND_ARGS=()
  fi
done

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/ops/lib/conda.sh"
CONDA_EXE_PATH="$(resolve_teleop_conda)"
CONDA_BASE="$("$CONDA_EXE_PATH" info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$TELEOP_CONDA_ENV"
exec python -m teleop_runtime.cli \
  --config config/modes/pico.yaml \
  "${ARM_ARGS[@]}" \
  "${HAND_ARGS[@]}" \
  --debug-log "$RUN_DIR/arm_follow.jsonl" \
  "$@"
