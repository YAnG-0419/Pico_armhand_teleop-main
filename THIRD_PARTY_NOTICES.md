# Third-party notices

The realtime FR3 controller is adapted from Franka Robotics ROS 2 controller
code and retains its Apache-2.0 license declarations. The container downloads
pinned libfranka, franka_ros2, and franka_description releases during build.

The PICO adapter includes the XRoboToolkit PC Service Python binding and its
prebuilt Linux SDK library under `vendor/xrobotoolkit_sdk`; its upstream license
is retained there. The binary targets x86-64 Linux.

The MuJoCo/Pinocchio dual-FR3 model contains Franka description assets. Their
license and notice are retained in `adapters/pico/assets/dual_fr3`.

The MANUS adapter includes MANUS CoreSDK 3.1.1 headers and the integrated
x86-64 Linux library under `vendor/manus_sdk`. These proprietary files remain
governed by `vendor/manus_sdk/LICENSE.vendor`.

The Wuji retargeting implementation and robot assets were imported from
`wuji-retargeting` revision `66693c2f82f4b8ad40b6802620bab6488b41c18d`.
Their licenses are retained as `adapters/wuji/LICENSE` and
`adapters/wuji/WUJI_DESCRIPTION_LICENSE`.
