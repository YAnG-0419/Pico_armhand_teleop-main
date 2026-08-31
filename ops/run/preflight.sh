#!/usr/bin/env bash
# Offline environment validation. This script never initializes a hardware SDK.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TELEOP_CONDA_ENV="${TELEOP_CONDA_ENV:-pico-armhand-teleop}"
# shellcheck disable=SC1091
source "$REPO_ROOT/ops/lib/conda.sh"
CONDA_EXE_PATH="$(resolve_teleop_conda)"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

[[ -f "$REPO_ROOT/docker/.env" ]] || fail \
  "docker/.env is missing; run: cp docker/.env.example docker/.env"
[[ -f "$REPO_ROOT/adapters/manus/config/Calibration_left.mcal" ]] || \
  fail "missing MANUS left calibration"
[[ -f "$REPO_ROOT/adapters/manus/config/Calibration_right.mcal" ]] || \
  fail "missing MANUS right calibration"
[[ -f "$REPO_ROOT/adapters/manus/build/libmanus_skeleton_bridge.so" ]] || \
  fail "MANUS bridge is not built; run: ./adapters/manus/scripts/build.sh"
[[ -f "$REPO_ROOT/ros_ws/install/setup.bash" ]] || \
  fail "ROS workspace is not built; run: ./ops/setup/build.sh"

command -v docker >/dev/null || fail "docker is not available"
docker info >/dev/null 2>&1 || fail "Docker daemon is unavailable"

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
  -u ROS_DISTRO -u ROS_VERSION -u ROS_PYTHON_VERSION \
  "$CONDA_EXE_PATH" run --no-capture-output -n "$TELEOP_CONDA_ENV" python -c \
  "import mujoco, nlopt, pinocchio, wuji_sdk; import xrobotoolkit_sdk; print('[PASS] host Python dependencies')"
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  "$CONDA_EXE_PATH" run --no-capture-output -n "$TELEOP_CONDA_ENV" python -c \
  "import PySide6; print('[PASS] operator GUI')"

(
  cd "$REPO_ROOT/docker"
  docker compose config --quiet
)
echo "[PASS] Docker Compose configuration"
echo "Preflight passed. No PICO, MANUS, Wuji, or Franka connection was opened."
