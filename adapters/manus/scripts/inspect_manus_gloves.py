#!/usr/bin/env python3
"""Report which MANUS gloves are connected and delivering frames.

    conda run --no-capture-output --name pico-armhand-teleop \
      python adapters/manus/scripts/inspect_manus_gloves.py

Owns an in-process MANUS SDK client; stop any MANUS teleop first. A glove
that is connected but uncalibrated shows frames=0 - the bridge drops its
frames until Calibration_<side>.mcal exists in adapters/manus/config.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "adapters" / "pico" / "src"))
sys.path.insert(0, str(REPO_ROOT / "adapters" / "manus" / "python"))

from pico_bimanual_franka_teleop.env_guard import ensure_ros_free_process  # noqa: E402

ensure_ros_free_process()

from manus_teleop.skeleton import ManusBridge, SIDE_CODES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    bridge = ManusBridge(
        REPO_ROOT
        / "adapters"
        / "manus"
        / "build"
        / "libmanus_skeleton_bridge.so"
    )
    try:
        print("Connecting to MANUS Core ...", flush=True)
        bridge.connect(REPO_ROOT / "adapters" / "manus" / "config")
        counts = {side: 0 for side in SIDE_CODES}
        sequences: dict[str, set] = {side: set() for side in SIDE_CODES}
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            for side in SIDE_CODES:
                try:
                    frame = bridge.read(side, 0.0)
                except RuntimeError as error:
                    print(f"{side}: read failed: {error}", flush=True)
                    return 1
                if frame is not None and frame.sequence not in sequences[side]:
                    sequences[side].add(int(frame.sequence))
                    counts[side] += 1
            time.sleep(0.01)
        available = bridge.available_sides()
        print(f"gloves reported by the SDK: {list(available) or 'none'}")
        for side in SIDE_CODES:
            state = f"{counts[side]} frames in {args.duration:.0f} s"
            if counts[side] == 0:
                if side in available:
                    state += (
                        "   <- connected but no frames: is "
                        f"Calibration_{side}.mcal present in "
                        "adapters/manus/config?"
                    )
                else:
                    state += "   <- glove not connected to MANUS Core"
            print(f"{side}: {state}")
        return 0 if all(counts.values()) else 1
    finally:
        bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
