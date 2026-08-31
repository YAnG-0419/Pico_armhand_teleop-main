#!/usr/bin/env python3

import argparse

from pico_bimanual_franka_teleop.env_guard import ensure_ros_free_process

ensure_ros_free_process()
from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = Path(__file__).resolve().parents[2] / "assets" / "dual_fr3" / "scene.xml"
MAX_HOME_ERROR = 0.005
MAX_CONTACT_FORCE = 1000.0


def contact_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    max_force = 0.0
    max_penetration = 0.0
    for index in range(data.ncon):
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        max_force = max(max_force, float(np.linalg.norm(wrench[:3])))
        max_penetration = max(max_penetration, max(0.0, -data.contact[index].dist))
    return max_force, max_penetration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print the first 100 integration steps in addition to the summary",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    home = data.qpos.copy()
    data.ctrl[:] = home
    mujoco.mj_forward(model, data)

    peak_force = 0.0
    peak_penetration = 0.0
    checkpoints: dict[int, float] = {}
    for step in range(1, 501):
        mujoco.mj_step(model, data)
        force, penetration = contact_metrics(model, data)
        peak_force = max(peak_force, force)
        peak_penetration = max(peak_penetration, penetration)
        error = data.qpos - home
        if args.verbose and step <= 100:
            error_text = " ".join(f"{value:+.6f}" for value in error)
            print(
                f"step={step:03d} t={data.time:.3f} ncon={data.ncon} "
                f"contact_force_max={force:.6f} penetration_max={penetration:.6f} "
                f"dq=[{error_text}]"
            )
        if step in (25, 500):
            checkpoints[step] = float(np.max(np.abs(error)))

    print(
        f"summary peak_contact_force={peak_force:.6f} "
        f"peak_penetration={peak_penetration:.6f} "
        f"max_dq_0.05s={checkpoints[25]:.6f} "
        f"max_dq_1.00s={checkpoints[500]:.6f}"
    )
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        raise RuntimeError("simulation produced non-finite state")
    if peak_force > MAX_CONTACT_FORCE or peak_penetration > 0.0:
        raise RuntimeError("home pose produced an unexpected collision")
    if max(checkpoints.values()) >= MAX_HOME_ERROR:
        raise RuntimeError("home-pose error exceeded 0.005 rad")


if __name__ == "__main__":
    main()
