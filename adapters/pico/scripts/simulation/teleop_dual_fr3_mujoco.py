#!/usr/bin/env python3

import argparse

from pico_bimanual_franka_teleop.env_guard import ensure_ros_free_process

ensure_ros_free_process()

from pico_bimanual_franka_teleop.config import load_config
from pico_bimanual_franka_teleop.simulation import DualFr3Simulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm-source",
        required=True,
        choices=("controllers", "motion-trackers", "hand-roots", "mock"),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    simulation = DualFr3Simulation(
        translation_scale=config.host.translation_scale,
        rotation_scale=config.host.rotation_scale,
        control_rate=config.host.control_rate,
        max_joint_speed=config.host.max_joint_speed,
        input_config=config.input,
        input_type=args.arm_source,
    )
    simulation.run(duration=args.duration, headless=args.headless)


if __name__ == "__main__":
    main()
