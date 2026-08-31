#!/usr/bin/env bash
# MANUS-to-Wuji hands only: no PICO, FR3, Docker, or arm controller.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${TELEOP_CONDA_ENV:-pico-armhand-teleop}"
BRIDGE="$REPO_ROOT/adapters/manus/build/libmanus_skeleton_bridge.so"
# shellcheck disable=SC1091
source "$REPO_ROOT/config/devices.env"

if [[ ! -f "$BRIDGE" ]]; then
  echo "Missing MANUS bridge: $BRIDGE" >&2
  echo "Build it first with: $REPO_ROOT/adapters/manus/scripts/build.sh" >&2
  exit 1
fi
if ldd "$BRIDGE" 2>/dev/null | grep -q 'not found'; then
  echo "The MANUS bridge has unresolved native libraries:" >&2
  ldd "$BRIDGE" | grep 'not found' >&2
  exit 1
fi

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/ops/lib/conda.sh"
CONDA_EXE_PATH="$(resolve_teleop_conda)"
exec "$CONDA_EXE_PATH" run --no-capture-output -n "$CONDA_ENV" \
  python -m adapters.wuji.hand_only \
  --wuji-left-address "$WUJI_LEFT_ADDRESS" \
  --wuji-right-address "$WUJI_RIGHT_ADDRESS" \
  "$@"
