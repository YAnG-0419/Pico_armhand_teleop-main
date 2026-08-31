# PICO Arm-Hand Teleoperation

This repository drives two Franka FR3 arms from PICO motion trackers and two
Wuji hands from MANUS gloves. The host Operator process stays ROS-free; ROS 2
and the real-time Franka controllers run in host-networked Docker containers.

```text
PICO trackers -> relative Cartesian mapping -> bimanual IK -> UDP
  -> pico_teleop_bridge -> ArmCommand -> safety gateway -> dual FR3

MANUS gloves -> canonical landmarks -> Wuji retargeting -> Wuji SDK
  -> dual Wuji Hand 2
```

Every side starts disengaged. PICO loss, stale robot state, a gateway rejection,
or GUI loss disengages control. Only `teleop_core` publishes the FR3 command
bus. MANUS/Wuji runs in a worker thread and cannot block the 100 Hz arm loop.

## Layout

- `teleop_runtime`: unified Operator backend.
- `adapters/pico`: PICO SDK input, pose mapping, IK, UDP client, and tests.
- `adapters/manus`: native MANUS skeleton bridge and Python frame contract.
- `adapters/wuji`: MANUS-to-Wuji retargeting, models, and hardware backends.
- `apps/operator_gui`: per-side arm and hand engagement GUI.
- `apps/left_wuji_dataset_recorder`: synchronized left arm/hand episode recorder.
- `ros_ws/src`: ROS messages, safety gateway, UDP bridge, and FR3 controllers.
- `config`: PICO tracking, safety limits, and workcell topology.
- `ops`: setup, preflight, and launch entry points.

## Setup

Supported host: x86-64 Ubuntu 22.04. PICO XRoboToolkit PC Service and MANUS
Core are external system services; their Linux SDK libraries are vendored here.

```bash
cp docker/.env.example docker/.env
./ops/setup/setup_teleop_env.sh
./adapters/manus/scripts/build.sh
./ops/setup/build.sh
```

If Conda is installed in a non-standard location, set
`TELEOP_CONDA_EXE=/absolute/path/to/conda`. `setup_wuji_env.sh` remains as an
idempotent Wuji dependency repair/check, but the main setup already installs
those dependencies.

Before building or starting hardware, review:

- `docker/.env`: data path, ROS domain, CPU set, container workcell path.
- `config/workcell/current.yaml`: both FR3 IPs and dedicated CPU sets.
- `config/devices.env`: PICO tracker identities and Wuji network addresses.
- `config/modes/pico.yaml`: tracking limits and tracker-to-control transforms.
- MANUS calibration files under `adapters/manus/config`.
- Explicit `--wuji-left-address` and `--wuji-right-address` values.

## Offline preflight

```bash
./ops/run/preflight.sh
```

The preflight checks files, imports, the ROS build, Docker, and Compose without
initializing PICO, MANUS, Wuji, or Franka hardware.

## Run

Start XRoboToolkit PC Service, the PICO headset application, and MANUS Core.
Keep the XRoboToolkit desktop demo closed because it competes for the same
stream. Then run:

```bash
./ops/run/start_teleop.sh
```

For offline integration with fake FR3 hardware while keeping real PICO,
MANUS, and Wuji devices:

```bash
./ops/run/start_teleop.sh --fake-franka
```

To enable runtime recording controls for the left FR3 + Wuji Hand 2, configure
an output directory. The recorder uses the existing Wuji connection and never
publishes commands:

```bash
mkdir -p /home/user/franka_teleop_data/datasets
./ops/run/start_teleop.sh \
  --wuji-sides left \
  --left-dataset-dir /home/user/franka_teleop_data/datasets
```

Use **Start recording** and **Stop and save** in Operator GUI. Each start creates
a new timestamped episode while teleoperation continues uninterrupted.

See [apps/left_wuji_dataset_recorder/README.md](apps/left_wuji_dataset_recorder/README.md)
for the schema and validation checklist.

To test PICO arms without connecting Wuji hardware, run the stack manually and
override the hand source:

```bash
cd docker
docker compose up -d fake-franka-control teleop-control pico-bridge
cd ..
./ops/run/run_operator.sh --hand-source none
```

## Network and bridge parameters

Operator and Docker are assumed to run on the same computer. Compose uses
`network_mode: host`, so `127.0.0.1:5560` for commands and
`127.0.0.1:5561` for feedback are intentional. Do not start another bridge on
these ports. A split-computer or non-host-network deployment requires an
explicit endpoint redesign; changing only one loopback address is insufficient.

See [docs/HARDWARE_DEPLOY.md](docs/HARDWARE_DEPLOY.md) for staged validation.
