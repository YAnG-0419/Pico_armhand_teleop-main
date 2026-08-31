#!/usr/bin/env python3
import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path

from pico_bimanual_franka_teleop.env_guard import ensure_ros_free_process

ensure_ros_free_process()

import subprocess

from pico_bimanual_franka_teleop.config import load_config
from pico_bimanual_franka_teleop.hand_worker import HandWorker
from pico_bimanual_franka_teleop.hardware import DualFr3HardwareTeleop
from pico_bimanual_franka_teleop.xr_input import PicoSession, create_pico_input

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def invoke_trigger(service: str, timeout: float = 120.0) -> tuple[bool, str]:
    """Call a ROS Trigger service through the tools container."""
    completed = subprocess.run(
        [
            "docker", "compose", "run", "--rm", "tools",
            "ros2", "service", "call",
            service, "std_srvs/srv/Trigger", "{}",
        ],
        cwd=REPO_ROOT / "docker",
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = (completed.stdout + completed.stderr).strip()
    succeeded = completed.returncode == 0 and "success=True" in completed.stdout
    return succeeded, output[-400:]


def invoke_reset(side: str | None = None) -> tuple[bool, str]:
    """Call /reset_to_initial_pose (optionally one side) via the container.

    The operator process is deliberately ROS-free (env_guard), so the reset goes
    through the same `docker compose run` path the runbook documents. The
    trajectory itself can take ~20 s for large displacements, plus container
    startup; the timeout is generous because killing the call does not stop the
    controller-side trajectory anyway.
    """
    service = "/reset_to_initial_pose" + (f"/{side}" if side else "")
    return invoke_trigger(service)


def invoke_capture_home() -> tuple[bool, str]:
    """Save both arms' current measured positions as the new home pose."""
    return invoke_trigger("/capture_initial_pose", timeout=30.0)


def main() -> None:
    # The supervisor backgrounds this process, so Bash may leave SIGINT ignored.
    # Restore both shutdown signals to the KeyboardInterrupt/finally path so a
    # recording is finalized before hardware and SDK resources are closed.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    parser = argparse.ArgumentParser(
        description=(
            "PICO teleoperation for dual Franka FR3 arms with optional "
            "MANUS-to-Wuji hand control."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--arm-source",
        required=True,
        choices=(
            "controllers",
            "motion-trackers",
            "hand-roots",
        ),
    )
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument(
        "--control-port",
        type=int,
        default=5590,
        help="JSON-TCP operator control port; the PySide6 GUI "
        "(apps/operator_gui) connects here (default: 5590)",
    )
    # Hand options are CLI arguments rather than YAML, matching how --arm-source is
    # handled: what is being driven is an explicit choice per run, and this keeps
    # existing configuration files valid.
    parser.add_argument(
        "--debug-log",
        default=None,
        help="write one JSONL row per control tick, capturing tracker pose, "
        "mapped target, commanded and measured state, for offline analysis of "
        "following quality",
    )
    parser.add_argument(
        "--hand-source",
        default="none",
        choices=("none", "wuji"),
        help="hand source integrated into this operator process (default: none)",
    )
    parser.add_argument(
        "--hand-debug-log",
        default=None,
        help="write emitted Wuji joint commands to JSONL (requires Wuji)",
    )
    parser.add_argument(
        "--record-left-dataset",
        type=Path,
        default=None,
        metavar="EPISODE.npz",
        help=(
            "record measured left FR3 joints, left Wuji Hand 2 joints, and "
            "left link8 pose into one synchronized NPZ episode"
        ),
    )
    parser.add_argument(
        "--left-dataset-dir",
        type=Path,
        default=None,
        metavar="DIRECTORY",
        help=(
            "enable GUI-controlled left FR3 + Wuji Hand 2 recording; each "
            "Start creates a timestamped episode in this directory"
        ),
    )
    parser.add_argument(
        "--hand-rate",
        type=float,
        default=30.0,
        help="hand commands per second per side, at most 60 Hz (default: 30)",
    )
    parser.add_argument(
        "--wuji-sides",
        choices=("left", "right", "both"),
        default="both",
        help="Wuji hardware sides to drive (only with --hand-source wuji)",
    )
    for side in ("left", "right"):
        parser.add_argument(
            f"--wuji-{side}-model",
            choices=("wuji_hand", "wuji_hand_2"),
            default="wuji_hand_2",
            help=f"physical {side} Wuji hand model",
        )
        parser.add_argument(
            f"--wuji-{side}-address",
            default="",
            help=f"{side} Wuji Hand 2 SDK address (IP:PORT)",
        )
        parser.add_argument(
            f"--wuji-{side}-serial",
            default="",
            help=f"{side} original Wuji Hand USB serial",
        )
    parser.add_argument("--wuji-kp", type=float, default=3.0)
    parser.add_argument("--wuji-kd", type=float, default=0.1)
    parser.add_argument(
        "--wuji-current-limit",
        type=float,
        default=1.5,
        help="Wuji Hand 2 per-joint current limit in amps",
    )
    args = parser.parse_args()
    if args.hand_debug_log and args.hand_source == "none":
        parser.error("--hand-debug-log requires a hand source")
    if not 0.0 < args.hand_rate <= 60.0:
        parser.error("--hand-rate must be in (0, 60]")
    if args.hand_source != "wuji" and any(
        (
            args.wuji_left_address,
            args.wuji_right_address,
            args.wuji_left_serial,
            args.wuji_right_serial,
        )
    ):
        parser.error("--wuji-*-address/serial requires --hand-source wuji")
    if args.hand_source == "wuji":
        selected_wuji_sides = (
            ("left", "right")
            if args.wuji_sides == "both"
            else (args.wuji_sides,)
        )
        for side in selected_wuji_sides:
            model = getattr(args, f"wuji_{side}_model")
            address = getattr(args, f"wuji_{side}_address")
            if model == "wuji_hand_2" and not address:
                parser.error(
                    f"--wuji-{side}-address IP:PORT is required for "
                    "wuji_hand_2"
                )
        if args.wuji_kp < 0.0 or args.wuji_kd < 0.0:
            parser.error("--wuji-kp and --wuji-kd must not be negative")
        if args.wuji_current_limit <= 0.0:
            parser.error("--wuji-current-limit must be positive")
    if args.record_left_dataset is not None and args.left_dataset_dir is not None:
        parser.error(
            "--record-left-dataset and --left-dataset-dir are mutually exclusive"
        )
    if args.record_left_dataset is not None or args.left_dataset_dir is not None:
        if args.hand_source != "wuji":
            parser.error("dataset recording requires --hand-source wuji")
        if args.wuji_sides == "right":
            parser.error("dataset recording requires the left Wuji side")
        if args.wuji_left_model != "wuji_hand_2":
            parser.error("dataset recording requires left Wuji Hand 2")

    debug_logger = None
    if args.debug_log:
        from pico_bimanual_franka_teleop.debug_log import FollowDebugLogger

        debug_logger = FollowDebugLogger(args.debug_log)
        print(f"debug log -> {args.debug_log}")

    config = load_config(args.config)

    from pico_bimanual_franka_teleop.control_server import (
        OperatorConsole,
        OperatorControlServer,
    )

    ui = OperatorConsole()
    server = OperatorControlServer((args.control_host, args.control_port), ui)
    server.start()
    print(
        f"operator control server on {args.control_host}:{args.control_port} "
        "- connect the GUI (apps/operator_gui) to engage"
    )

    pico_session = None
    arm_source = None
    hands = None
    dataset_recorder = None
    dataset_recorder_factory = None
    try:
        pico_session = PicoSession()
        arm_source = create_pico_input(
            config.input,
            args.arm_source,
            keyboard=ui,
            xrt_client=pico_session.client,
        )
        hand_pipeline = None
        if args.hand_source == "wuji":
            from adapters.wuji import WujiHandPipeline

            sides = selected_wuji_sides
            models = {
                side: getattr(args, f"wuji_{side}_model") for side in sides
            }
            addresses = {
                side: getattr(args, f"wuji_{side}_address") for side in sides
            }
            serials = {
                side: getattr(args, f"wuji_{side}_serial") for side in sides
            }
            hand_pipeline = WujiHandPipeline(
                sides=sides,
                models=models,
                addresses=addresses,
                serials=serials,
                rate=args.hand_rate,
                kp=args.wuji_kp,
                kd=args.wuji_kd,
                current_limit=args.wuji_current_limit,
                debug_log=args.hand_debug_log,
            )
        if hand_pipeline is not None:
            hands = HandWorker(
                hand_pipeline, tick_rate=config.host.control_rate
            )
        if args.record_left_dataset is not None:
            from apps.left_wuji_dataset_recorder import LeftWujiDatasetRecorder

            dataset_recorder = LeftWujiDatasetRecorder(
                args.record_left_dataset,
                hand_joint_names=hand_pipeline.joint_names["left"],
                control_rate_hz=config.host.control_rate,
            )
            print(f"left teleop dataset -> {dataset_recorder.output}")
        elif args.left_dataset_dir is not None:
            from apps.left_wuji_dataset_recorder import LeftWujiDatasetRecorder

            dataset_directory = args.left_dataset_dir.expanduser().resolve()
            hand_joint_names = tuple(hand_pipeline.joint_names["left"])

            def dataset_recorder_factory():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                return LeftWujiDatasetRecorder(
                    dataset_directory / f"episode_{timestamp}.npz",
                    hand_joint_names=hand_joint_names,
                    control_rate_hz=config.host.control_rate,
                )

            print(f"GUI-controlled left datasets -> {dataset_directory}")

        teleop = DualFr3HardwareTeleop(
            command_host=config.udp.command_host,
            command_port=config.udp.command_port,
            state_host=config.udp.state_host,
            state_port=config.udp.state_port,
            state_timeout=config.udp.state_timeout,
            translation_scale=config.host.translation_scale,
            rotation_scale=config.host.rotation_scale,
            control_rate=config.host.control_rate,
            max_joint_speed=config.host.max_joint_speed,
            robot_state_wait_timeout=config.host.robot_state_wait_timeout,
            arm_source=arm_source,
            operator=ui,
            hands=hands,
            debug_logger=debug_logger,
            dataset_recorder=dataset_recorder,
            dataset_recorder_factory=dataset_recorder_factory,
            reset_invoker=invoke_reset,
            capture_home_invoker=invoke_capture_home,
        )
        # The keyboard's `q` (and Ctrl-C) surface as KeyboardInterrupt; run()'s
        # finally block has already closed hands, robot, and input by the time
        # it reaches here.
        try:
            teleop.run()
        except KeyboardInterrupt:
            pass
    finally:
        try:
            if dataset_recorder is not None:
                try:
                    dataset_recorder.close()
                except Exception as error:  # noqa: BLE001 - safe shutdown continues
                    print(f"dataset finalization FAILED: {error}", file=sys.stderr)
            if hands is not None:
                hands.close()
            if arm_source is not None:
                arm_source.close()
            if pico_session is not None:
                pico_session.close()
        finally:
            server.close()
    print("\nteleop stopped")


if __name__ == "__main__":
    main()
