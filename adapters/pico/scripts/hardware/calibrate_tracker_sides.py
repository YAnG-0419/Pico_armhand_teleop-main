#!/usr/bin/env python3
"""Assign motion trackers to sides by moving one hand at a time.

Wear or hold the trackers as usual, then:

    conda run --no-capture-output --name pico-armhand-teleop \
      python adapters/pico/scripts/hardware/calibrate_tracker_sides.py

Follow the prompts: move only the LEFT hand, then only the RIGHT hand.
The script prints the detected mapping; add --write to store it in
    config/devices.env (comments preserved).

Owns the single PICO SDK client - stop teleop and close RobotLinuxDemo
first, exactly like inspect_motion_trackers.py.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "adapters" / "pico" / "src"))

from pico_bimanual_franka_teleop.env_guard import ensure_ros_free_process  # noqa: E402

ensure_ros_free_process()

import numpy as np  # noqa: E402

from pico_bimanual_franka_teleop.tracker_calibration import (  # noqa: E402
    CalibrationError,
    assign_sides,
    replace_device_env_serials,
)

DEFAULT_CONFIG = REPO_ROOT / "config" / "devices.env"


def _desktop_gui_pids() -> list[int]:
    pids = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"RobotLinuxDemo.x86_64" in command:
            pids.append(int(process.name))
    return sorted(pids)


def _snapshot(xrt):
    """Consistent {serial: position} frame, or None (same retry discipline
    as the teleop input: the SDK serves cached data without a stable
    timestamp)."""
    for _ in range(3):
        timestamp_before = int(xrt.get_motion_timestamp_ns())
        count = int(xrt.num_motion_data_available())
        serials = list(xrt.get_motion_tracker_serial_numbers())
        poses = list(xrt.get_motion_tracker_pose())
        timestamp_after = int(xrt.get_motion_timestamp_ns())
        if timestamp_before <= 0 or timestamp_before != timestamp_after:
            continue
        if (
            count != len(serials)
            or count != len(poses)
            or len(set(serials)) != len(serials)
        ):
            return None
        return {
            str(serial): np.asarray(pose, dtype=float)[:3]
            for serial, pose in zip(serials, poses)
        }
    return None


def _record_phase(xrt, label: str, seconds: float) -> dict[str, float]:
    """Peak displacement per serial from the phase's first seen position."""
    for countdown in (3, 2, 1):
        print(f"  {label} in {countdown} ...", flush=True)
        time.sleep(1.0)
    print(f"  GO - {label} now.", flush=True)
    anchors: dict[str, np.ndarray] = {}
    peaks: dict[str, float] = {}
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        frame = _snapshot(xrt)
        if frame is not None:
            for serial, position in frame.items():
                anchor = anchors.setdefault(serial, position)
                displacement = float(np.linalg.norm(position - anchor))
                if displacement > peaks.get(serial, 0.0):
                    peaks[serial] = displacement
        time.sleep(0.02)
    print("  stop.", flush=True)
    for serial, peak in sorted(peaks.items()):
        print(f"    {serial}: moved {peak * 100:5.1f} cm", flush=True)
    return peaks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="devices.env to compare against and optionally rewrite",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="store the detected mapping into the config file",
    )
    parser.add_argument("--phase-seconds", type=float, default=5.0)
    args = parser.parse_args()

    gui_pids = _desktop_gui_pids()
    if gui_pids:
        print(
            "Refusing to start while the desktop RobotLinuxDemo GUI is "
            f"running (PID(s): {gui_pids}).",
            file=sys.stderr,
        )
        return 2

    import xrobotoolkit_sdk as xrt

    print("Initializing the XRoboToolkit SDK ...", flush=True)
    xrt.init()
    try:
        print("Waiting for two trackers ...", flush=True)
        deadline = time.monotonic() + 30.0
        frame = None
        while time.monotonic() < deadline:
            frame = _snapshot(xrt)
            if frame is not None and len(frame) >= 2:
                break
            time.sleep(0.2)
        if frame is None or len(frame) < 2:
            detected = sorted(frame) if frame else []
            print(
                f"Need two trackers, detected {detected}. Check headset "
                "tracking mode and tracker power.",
                file=sys.stderr,
            )
            return 1
        print(f"Detected trackers: {sorted(frame)}", flush=True)

        print("\nPhase 1: move ONLY the LEFT hand; keep the right still.")
        left_phase = _record_phase(
            xrt, "move the LEFT hand", args.phase_seconds
        )
        print("\nPhase 2: move ONLY the RIGHT hand; keep the left still.")
        right_phase = _record_phase(
            xrt, "move the RIGHT hand", args.phase_seconds
        )
    finally:
        xrt.close()

    try:
        mapping = assign_sides(left_phase, right_phase)
    except CalibrationError as error:
        print(f"\nCalibration failed: {error}", file=sys.stderr)
        return 1

    print(f"\nleft:  {mapping['left']}")
    print(f"right: {mapping['right']}")

    config_text = args.config.read_text(encoding="utf-8")
    try:
        updated = replace_device_env_serials(config_text, mapping)
    except CalibrationError as error:
        print(f"Cannot update {args.config}: {error}", file=sys.stderr)
        return 1
    if updated == config_text:
        print(f"{args.config} already has this mapping; nothing to do.")
        return 0
    if not args.write:
        print(f"Differs from {args.config}; re-run with --write to store it.")
        return 0

    args.config.write_text(updated, encoding="utf-8")
    print(f"Wrote {args.config}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
