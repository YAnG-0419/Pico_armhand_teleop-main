#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${TELEOP_CONDA_ENV:-pico-armhand-teleop}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Host ROS Python and library paths are incompatible with this Conda process.
unset PYTHONPATH LD_LIBRARY_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
# shellcheck disable=SC1091
source "$REPO_ROOT/ops/lib/conda.sh"
CONDA_EXE_PATH="$(resolve_teleop_conda)"

"$CONDA_EXE_PATH" run --no-capture-output -n "$ENV_NAME" python -m pip install \
  'nlopt>=2.7' \
  'pin>=3.8.0' \
  'wuji-sdk>=0.10.0' \
  'wujihandpy>=1.8.0'

"$CONDA_EXE_PATH" run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import nlopt
import pinocchio
import wuji_sdk
import wujihandpy
print("Wuji Python dependencies import successfully")
PY
