#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
environment=${TELEOP_CONDA_ENV:-pico-armhand-teleop}
# Host ROS Python and library paths are incompatible with this Conda process.
unset PYTHONPATH LD_LIBRARY_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
# shellcheck disable=SC1091
source "${repo}/ops/lib/conda.sh"
conda_exe="$(resolve_teleop_conda)"

if "$conda_exe" env list | awk '{print $1}' | grep -Fxq "${environment}"; then
  echo "Updating Conda environment: ${environment}"
  "$conda_exe" env update --name "${environment}" \
    --file "${repo}/environment.pico-armhand-teleop.yml" --prune
else
  echo "Creating Conda environment: ${environment}"
  "$conda_exe" env create --name "${environment}" \
    --file "${repo}/environment.pico-armhand-teleop.yml"
fi

"$conda_exe" run --name "${environment}" python -m pip install \
  "${repo}/vendor/xrobotoolkit_sdk"

"$conda_exe" run --name "${environment}" python -m pip install \
  -e "${repo}/ros_ws/src/teleop_core" \
  -e "${repo}/adapters/pico" \
  -e "${repo}/adapters/manus/python"
