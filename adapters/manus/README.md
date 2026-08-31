# MANUS skeleton input

The native bridge exposes calibrated left/right MANUS skeleton frames to the
Wuji retargeting pipeline. It does not publish ROS topics or send hand commands.

Only one MANUS client may run at a time. Both calibration files must match the
connected gloves and operator:

- `config/Calibration_left.mcal`
- `config/Calibration_right.mcal`

Build the native bridge after cloning or moving the repository:

```bash
./adapters/manus/scripts/build.sh
```

The resulting `adapters/manus/build/libmanus_skeleton_bridge.so` embeds an
RPATH to this repository's vendored MANUS SDK, so it must be rebuilt rather
than copied from another checkout.
