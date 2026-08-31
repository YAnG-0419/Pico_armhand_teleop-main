#!/usr/bin/env bash
set -eo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/ros/humble/setup.bash
source /opt/vendor_ws/install/setup.bash
set -u
cd "${repo}/ros_ws"
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release "$@"
