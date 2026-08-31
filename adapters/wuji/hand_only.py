#!/usr/bin/env python3
"""Run MANUS-to-Wuji hand teleoperation without PICO or FR3 arms."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "adapters" / "pico" / "src"))
sys.path.insert(0, str(REPO_ROOT / "adapters" / "manus" / "python"))

from pico_bimanual_franka_teleop.env_guard import (  # noqa: E402
    ensure_ros_free_process,
)

ensure_ros_free_process()

from pico_bimanual_franka_teleop.xr_input import KeyboardActivation  # noqa: E402

MODELS = ("wuji_hand", "wuji_hand_2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wuji-sides",
        "--sides",
        dest="wuji_sides",
        default="right",
        choices=("left", "right", "both"),
        help="hands to connect (default: right)",
    )
    for side in ("left", "right"):
        parser.add_argument(
            f"--wuji-{side}-model",
            default="wuji_hand_2",
            choices=MODELS,
        )
        parser.add_argument(
            f"--wuji-{side}-address",
            default="",
            metavar="IP:PORT",
            help=f"explicit network address for the {side} Wuji Hand 2",
        )
        parser.add_argument(
            f"--wuji-{side}-serial",
            default="",
            help=f"optional serial for the original USB {side} Wuji Hand",
        )
    parser.add_argument("--wuji-rate", type=float, default=30.0)
    parser.add_argument("--wuji-stale-timeout", type=float, default=0.25)
    parser.add_argument("--wuji-kp", type=float, default=3.0)
    parser.add_argument("--wuji-kd", type=float, default=0.1)
    parser.add_argument("--wuji-current-limit", type=float, default=1.5)
    parser.add_argument("--keyboard-device", default="/dev/tty")
    parser.add_argument("--debug-log")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="stop after this many seconds; zero runs until Q or Ctrl-C",
    )
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    sides = (
        ("left", "right")
        if args.wuji_sides == "both"
        else (args.wuji_sides,)
    )
    for side in sides:
        model = getattr(args, f"wuji_{side}_model")
        address = getattr(args, f"wuji_{side}_address")
        if model == "wuji_hand_2" and not address:
            parser.error(
                f"--wuji-{side}-address IP:PORT is required for wuji_hand_2"
            )
    if not 0.0 < args.wuji_rate <= 60.0:
        parser.error("--wuji-rate must be in (0, 60]")
    if args.wuji_stale_timeout <= 0.0:
        parser.error("--wuji-stale-timeout must be positive")
    if args.wuji_kp < 0.0 or args.wuji_kd < 0.0:
        parser.error("--wuji-kp and --wuji-kd must not be negative")
    if args.wuji_current_limit <= 0.0:
        parser.error("--wuji-current-limit must be positive")
    if args.duration < 0.0:
        parser.error("--duration must not be negative")
    args.selected_sides = sides
    return args


def run(args: argparse.Namespace) -> int:
    from .pipeline import WujiHandPipeline

    sides = args.selected_sides
    models = {side: getattr(args, f"wuji_{side}_model") for side in sides}
    addresses = {
        side: getattr(args, f"wuji_{side}_address")
        for side in sides
        if getattr(args, f"wuji_{side}_address")
    }
    serials = {
        side: getattr(args, f"wuji_{side}_serial")
        for side in sides
        if getattr(args, f"wuji_{side}_serial")
    }

    print("Wuji HAND-ONLY mode: PICO, FR3, Docker, and arm control are not started.")
    for side in sides:
        endpoint = addresses.get(side) or serials.get(side) or "USB auto-discovery"
        print(f"  {side}: model={models[side]}, device={endpoint}")
    print("The hand starts DISENGAGED; verify its free space before engaging.")

    pipeline = None
    keyboard = None
    try:
        pipeline = WujiHandPipeline(
            sides=sides,
            models=models,
            addresses=addresses,
            serials=serials,
            rate=args.wuji_rate,
            stale_timeout=args.wuji_stale_timeout,
            kp=args.wuji_kp,
            kd=args.wuji_kd,
            current_limit=args.wuji_current_limit,
            debug_log=args.debug_log,
        )
        keyboard = KeyboardActivation(args.keyboard_device, sides=sides)
        keyboard.show(
            f"Wuji hands ({', '.join(sides)}) DISENGAGED. "
            "L/R toggles one side, Space toggles selected sides; "
            "X stops; O requests open while disengaged; Q exits."
        )

        started = time.monotonic()
        deadline = started + args.duration if args.duration > 0.0 else None
        next_report = started
        previous_sent = {side: 0 for side in sides}
        while deadline is None or time.monotonic() < deadline:
            loop_started = time.monotonic()
            active = keyboard.poll()
            requests = keyboard.take_requests()
            if requests["open_hands"]:
                pipeline.request_open(loop_started)
            pipeline.tick(loop_started, active=active)

            if loop_started >= next_report:
                parts = []
                elapsed = max(loop_started - (next_report - 1.0), 1e-6)
                for side in sides:
                    status = pipeline.status.sides[side]
                    send_rate = (status.sent - previous_sent[side]) / elapsed
                    previous_sent[side] = status.sent
                    parts.append(
                        f"{side}: sending {send_rate:.1f} Hz, solve "
                        f"{status.solve_seconds * 1e3:.1f} ms"
                        if status.sending
                        else f"{side}: not sending ({status.fault})"
                    )
                keyboard.show(" | ".join(parts))
                next_report = loop_started + 1.0

            remaining = 0.01 - (time.monotonic() - loop_started)
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        # Disable and disconnect the hardware before restoring the terminal.
        if pipeline is not None:
            pipeline.close()
        if keyboard is not None:
            keyboard.close()
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
