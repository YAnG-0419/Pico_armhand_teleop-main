#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
source /opt/vendor_ws/install/setup.bash
if [[ -f /workspace/pico_armhand_teleop/ros_ws/install/setup.bash ]]; then
  source /workspace/pico_armhand_teleop/ros_ws/install/setup.bash
fi

exec "$@"
