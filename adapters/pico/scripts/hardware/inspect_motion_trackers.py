#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path


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
    for _ in range(3):
        timestamp_before = int(xrt.get_motion_timestamp_ns())
        count = int(xrt.num_motion_data_available())
        serials = tuple(
            str(value)
            for value in xrt.get_motion_tracker_serial_numbers()
        )
        timestamp_after = int(xrt.get_motion_timestamp_ns())
        if timestamp_before > 0 and timestamp_before == timestamp_after:
            if count == len(serials) and len(set(serials)) == len(serials):
                return timestamp_after, serials
            return None
    return None


def _show(serials: tuple[str, ...]) -> None:
    print(f"Detected {len(serials)} motion tracker(s):", flush=True)
    for index, serial in enumerate(serials):
        print(f"  [{index}] {serial}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Listen for Motion Tracker serial numbers using one SDK client."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="seconds to listen before exiting (default: 10)",
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")

    gui_pids = _desktop_gui_pids()
    if gui_pids:
        print(
            "Refusing to start while the desktop RobotLinuxDemo GUI is running "
            f"(PID(s): {gui_pids}). Close only the desktop GUI; keep "
            "RoboticsService and the headset app running.",
            file=sys.stderr,
        )
        return 2

    import xrobotoolkit_sdk as xrt

    print(
        "Initializing the XRoboToolkit SDK. Waiting for Motion Tracking data...",
        flush=True,
    )
    seen: tuple[str, ...] = ()
    try:
        xrt.init()
        print("SDK initialized; listening for tracker IDs.", flush=True)
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            snapshot = _snapshot(xrt)
            if snapshot is not None:
                _, serials = snapshot
                if serials != seen:
                    seen = serials
                    _show(seen)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopped by operator.", flush=True)
    finally:
        print("\nClosing the XRoboToolkit SDK.", flush=True)
        xrt.close()
        print("SDK closed.", flush=True)

    if not seen:
        print(
            "No Motion Tracker IDs received. This is expected when the PICO "
            "stream has not been started. Otherwise, check Object/Motion "
            "Tracking mode and data sending.",
            flush=True,
        )
        return 0
    print("Final tracker IDs:", flush=True)
    _show(seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
