#!/usr/bin/env python3
"""Replay a recorded PICO hand log through retargeting and out over UDP.

Offline validation of the whole host-side hand path with no hardware and no
headset: recorded 26-joint skeletons are converted to canonical landmarks,
retargeted to L20 URDF poses, and sent as hand qpos datagrams. Point it at the
`linker_hand_bridge` in dry-run mode to exercise the complete chain up to, but
not including, the vendor driver.

Recordings come from `scripts/hardware/inspect_hand_tracking.py --log`.
"""

import argparse

import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "adapters" / "pico" / "src"))

from pico_bimanual_franka_teleop.env_guard import ensure_ros_free_process  # noqa: E402

ensure_ros_free_process()

from pico_bimanual_franka_teleop import hand_landmarks as hl  # noqa: E402
from pico_bimanual_franka_teleop.hand_profiles import g20_urdf_path  # noqa: E402
from pico_bimanual_franka_teleop.hand_retarget import L20Retargeter  # noqa: E402
from pico_bimanual_franka_teleop.hand_stream import build_hand_packet  # noqa: E402

SIDES = ("left", "right")


def load_frames(log: Path) -> dict[str, list[np.ndarray]]:
    frames: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
    skipped = {side: 0 for side in SIDES}
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for side in SIDES:
            if record.get(f"{side}_is_active") != 1:
                skipped[side] += 1
                continue
            skeleton = np.asarray(record[f"{side}_joints"], dtype=float)
            try:
                hl.validate_skeleton(skeleton)
            except ValueError:
                skipped[side] += 1
                continue
            frames[side].append(skeleton)
    for side in SIDES:
        print(
            f"  {side}: {len(frames[side])} usable frames, "
            f"{skipped[side]} skipped as inactive or invalid"
        )
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5570)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument(
        "--sides", default="both", choices=["left", "right", "both"]
    )
    parser.add_argument("--loop", action="store_true", help="replay indefinitely")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="retarget and report statistics without sending any datagram",
    )
    args = parser.parse_args()
    if args.rate <= 0:
        parser.error("--rate must be positive")
    if not args.log.is_file():
        parser.error(f"log not found: {args.log}")

    sides = SIDES if args.sides == "both" else (args.sides,)
    print(f"Loading {args.log}")
    frames = load_frames(args.log)
    if not any(frames[side] for side in sides):
        print("No usable frames for the selected sides.", file=sys.stderr)
        return 1

    assets = REPO_ROOT / "assets"
    retargeters = {
        side: L20Retargeter(g20_urdf_path(assets, side), side)
        for side in sides
        if frames[side]
    }
    sock = None if args.dry_run else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sequence = {side: 0 for side in sides}
    stats = {side: {"loss": [], "qpos": []} for side in sides}
    interval = 1.0 / args.rate

    print(
        f"Replaying at {args.rate:g} Hz to udp://{args.host}:{args.port}"
        if sock
        else f"Replaying at {args.rate:g} Hz without sending"
    )
    try:
        first = True
        while first or args.loop:
            first = False
            longest = max(len(frames[s]) for s in retargeters)
            for index in range(longest):
                started = time.monotonic()
                for side, retargeter in retargeters.items():
                    side_frames = frames[side]
                    if index >= len(side_frames):
                        continue
                    landmarks = hl.to_canonical_landmarks(side_frames[index])
                    qpos, info = retargeter.retarget(landmarks)
                    stats[side]["loss"].append(float(info["loss"]))
                    stats[side]["qpos"].append(qpos)
                    if sock is not None:
                        payload = build_hand_packet(
                            f"replay-{side}",
                            sequence[side],
                            time.time(),
                            side,
                            retargeter.joint_names,
                            qpos,
                        )
                        sock.sendto(payload, (args.host, args.port))
                    sequence[side] += 1
                remaining = interval - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopped by operator.")
    finally:
        if sock is not None:
            sock.close()
        for retargeter in retargeters.values():
            retargeter.close()

    print()
    for side in retargeters:
        losses = np.asarray(stats[side]["loss"])
        qpos = np.asarray(stats[side]["qpos"])
        print(f"{side}: sent {sequence[side]} packets")
        print(
            f"  loss   median={np.median(losses):.6f}  p95="
            f"{np.percentile(losses, 95):.6f}  max={losses.max():.6f}"
        )
        print(
            f"  qpos   min={qpos.min():+.3f}  max={qpos.max():+.3f}  "
            f"all finite={bool(np.isfinite(qpos).all())}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
