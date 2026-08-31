#!/usr/bin/env python3
"""Inspect isolated G20 thumb-flexion configurations.

By default this opens a PyBullet viewer with three separated L20 hands:

  open          all five fingers open;
  tip-only      four fingers open, thumb MCP/IP fully flexed;
  base-and-tip  four fingers open, thumb CMC pitch and MCP/IP fully flexed.

It also prints the exact 21-name URDF qpos and 20-slot G20 command for each
pose. Nothing is sent to hardware unless --send-hardware is explicitly given.
"""

from __future__ import annotations

import argparse
import importlib.util
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "adapters" / "pico" / "src"))

from pico_bimanual_franka_teleop.env_guard import ensure_ros_free_process  # noqa: E402

ensure_ros_free_process()

from pico_bimanual_franka_teleop.hand_profiles import g20_urdf_path  # noqa: E402
from pico_bimanual_franka_teleop.hand_retarget import L20Retargeter  # noqa: E402
from pico_bimanual_franka_teleop.hand_stream import build_hand_packet  # noqa: E402

PRESETS = ("open", "tip-only", "base-and-tip")
SIDES = ("left", "right")


def _bridge_core():
    path = (
        REPO_ROOT
        / "ros_ws"
        / "src"
        / "linker_hand_bridge"
        / "linker_hand_bridge"
        / "core.py"
    )
    spec = importlib.util.spec_from_file_location("thumb_probe_bridge_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load bridge mapper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configuration(
    retargeter: L20Retargeter, preset: str
) -> tuple[list[str], np.ndarray]:
    """Return an explicit 21-name pose for one diagnostic preset."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}")
    names = list(retargeter.joint_names)
    qpos = np.zeros(retargeter.dof, dtype=np.float64)
    if preset in {"tip-only", "base-and-tip"}:
        mcp = names.index("thumb_mcp")
        distal_name = "thumb_ip" if retargeter.side == "left" else "thumb_dip"
        distal = names.index(distal_name)
        qpos[mcp] = retargeter.upper[mcp]
        qpos[distal] = retargeter.upper[distal]
    if preset == "base-and-tip":
        pitch = names.index("thumb_cmc_pitch")
        qpos[pitch] = retargeter.upper[pitch]
    return names, qpos


def print_configuration(
    side: str,
    preset: str,
    names: list[str],
    qpos: np.ndarray,
    mapper,
    command_names: tuple[str, ...],
) -> None:
    command = mapper.map_qpos(side, names, qpos)
    print()
    print(f"{side} / {preset}")
    print("  thumb qpos (rad):")
    for name, value in zip(names, qpos):
        if name.startswith("thumb"):
            print(f"    {name:18s} {value:8.4f}")
    print("  G20 slots (0=fully flexed, 255=open):")
    for index, (name, value) in enumerate(zip(command_names, command)):
        print(f"    {index:2d}  {name:28s} {value:6.0f}")


def _teleop_processes() -> list[str]:
    matches = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"teleop_runtime.cli" in command or b"teleop_hands.py" in command:
            matches.append(command.decode(errors="replace").strip())
    return matches


def send_hardware(
    *,
    side: str,
    names: list[str],
    target: np.ndarray,
    host: str,
    port: int,
    ramp_seconds: float,
    hold_seconds: float,
    rate: float,
) -> None:
    """Stream one explicitly selected pose through the normal hand bridge."""
    running = _teleop_processes()
    if running:
        print("Refusing hardware mode while a PICO teleop sender is running:", file=sys.stderr)
        for command in running:
            print(f"  {command}", file=sys.stderr)
        raise SystemExit(2)

    print()
    print("HARDWARE MODE")
    print("  This opens all four fingers, then bends the selected thumb.")
    print("  The normal bridge watchdog and slew limiter remain in the path.")
    print("  Keep the hand clear and the emergency stop reachable.")

    open_pose = np.zeros_like(target)
    interval = 1.0 / rate
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    address = (host, port)
    sequence = 0
    stream_id = f"thumb-configuration-{side}"

    def send(qpos: np.ndarray) -> None:
        nonlocal sequence
        sock.sendto(
            build_hand_packet(
                stream_id,
                sequence,
                time.time(),
                side,
                names,
                qpos,
            ),
            address,
        )
        sequence += 1

    try:
        print("  Opening the hand...")
        open_end = time.monotonic() + 2.0
        while time.monotonic() < open_end:
            started = time.monotonic()
            send(open_pose)
            time.sleep(max(0.0, interval - (time.monotonic() - started)))

        print(f"  Ramping to the target over {ramp_seconds:.1f} s...")
        steps = max(1, int(round(ramp_seconds * rate)))
        for step in range(1, steps + 1):
            started = time.monotonic()
            send(target * (step / steps))
            time.sleep(max(0.0, interval - (time.monotonic() - started)))

        print(f"  Holding for {hold_seconds:.1f} s...")
        hold_end = time.monotonic() + hold_seconds
        while time.monotonic() < hold_end:
            started = time.monotonic()
            send(target)
            time.sleep(max(0.0, interval - (time.monotonic() - started)))
        print("  Stream stopped. The bridge watchdog now stops publishing.")
    finally:
        sock.close()


def show_pybullet(configurations: list[tuple[str, str, list[str], np.ndarray]]) -> None:
    import pybullet as pb

    client = pb.connect(pb.GUI)
    if client < 0:
        raise RuntimeError("Could not open the PyBullet viewer")
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0, physicsClientId=client)

    preset_x = {"open": -0.32, "tip-only": 0.0, "base-and-tip": 0.32}
    side_y = {"left": 0.16, "right": -0.16}
    try:
        for side, preset, names, qpos in configurations:
            urdf = g20_urdf_path(REPO_ROOT / "assets", side)
            base = [preset_x[preset], side_y[side], 0.0]
            body = pb.loadURDF(
                str(urdf),
                basePosition=base,
                useFixedBase=True,
                flags=pb.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=client,
            )
            by_name = dict(zip(names, qpos))
            if side == "left" and "thumb_ip" in by_name:
                by_name["thumb_dip"] = by_name["thumb_ip"]
            for joint_index in range(
                pb.getNumJoints(body, physicsClientId=client)
            ):
                info = pb.getJointInfo(body, joint_index, physicsClientId=client)
                joint_name = info[1].decode("utf-8")
                if info[2] == pb.JOINT_FIXED or joint_name not in by_name:
                    continue
                pb.resetJointState(
                    body,
                    joint_index,
                    by_name[joint_name],
                    physicsClientId=client,
                )
            pb.addUserDebugText(
                f"{side}: {preset}",
                [base[0], base[1], 0.27],
                textColorRGB=[0.1, 0.1, 0.1],
                textSize=1.2,
                physicsClientId=client,
            )

        pb.resetDebugVisualizerCamera(
            cameraDistance=1.15,
            cameraYaw=45.0,
            cameraPitch=-25.0,
            cameraTargetPosition=[0.0, 0.0, 0.13],
            physicsClientId=client,
        )
        print()
        print("PyBullet viewer is open. Close it or press Ctrl-C here to exit.")
        while pb.isConnected(client):
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if pb.isConnected(client):
            pb.disconnect(client)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=(*SIDES, "both"), default="both")
    parser.add_argument("--preset", choices=(*PRESETS, "all"), default="all")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="print configurations without opening the PyBullet viewer",
    )
    parser.add_argument(
        "--send-hardware",
        action="store_true",
        help="stream one preset to the normal hand bridge",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5570)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--ramp-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    args = parser.parse_args()

    if args.rate <= 0.0 or args.ramp_seconds <= 0.0 or args.hold_seconds <= 0.0:
        parser.error("rate, ramp-seconds and hold-seconds must be positive")
    if args.send_hardware and (args.side == "both" or args.preset == "all"):
        parser.error("--send-hardware requires one side and one preset")

    selected_sides = SIDES if args.side == "both" else (args.side,)
    selected_presets = PRESETS if args.preset == "all" else (args.preset,)
    bridge_core = _bridge_core()
    mapper = bridge_core.G20Mapper()
    command_names = tuple(bridge_core.G20_JOINT_NAMES)
    configurations = []
    for side in selected_sides:
        urdf = g20_urdf_path(REPO_ROOT / "assets", side)
        with L20Retargeter(urdf, side) as retargeter:
            for preset in selected_presets:
                names, qpos = configuration(retargeter, preset)
                configurations.append((side, preset, names, qpos))
                print_configuration(
                    side,
                    preset,
                    names,
                    qpos,
                    mapper,
                    command_names,
                )

    if args.send_hardware:
        side, _, names, qpos = configurations[0]
        send_hardware(
            side=side,
            names=names,
            target=qpos,
            host=args.host,
            port=args.port,
            ramp_seconds=args.ramp_seconds,
            hold_seconds=args.hold_seconds,
            rate=args.rate,
        )
    elif not args.print_only:
        show_pybullet(configurations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
