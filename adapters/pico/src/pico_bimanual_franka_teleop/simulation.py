import time
from contextlib import nullcontext

import mujoco
import numpy as np
import pinocchio as pin

from .config import InputConfig
from .ik import BimanualPinkIK
from .paths import MJCF_PATH
from .pose_mapping import RelativePoseMapper
from .types import Pose, SIDES
from .xr_input import MockTeleopInput, create_pico_input


class DualFr3Simulation:
    def __init__(
        self,
        translation_scale: float,
        rotation_scale: float,
        control_rate: float,
        max_joint_speed: float,
        input_config: InputConfig,
        input_type: str,
    ) -> None:
        if control_rate <= 0.0:
            raise ValueError("Control rate must be positive")
        self.dt = 1.0 / control_rate
        self.model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.data.ctrl[:] = self.data.qpos
        mujoco.mj_forward(self.model, self.data)
        self.hold_q = self.data.qpos.copy()
        self.ik = BimanualPinkIK(dt=self.dt, max_joint_speed=max_joint_speed)
        self.mappers = {
            side: RelativePoseMapper(
                translation_scale=translation_scale,
                rotation_scale=rotation_scale,
            )
            for side in SIDES
        }
        if input_type == "mock":
            self.teleop_input = MockTeleopInput()
        else:
            self.teleop_input = create_pico_input(input_config, input_type)
        self.target_mocap = {
            side: self.model.body(f"{side}_target").mocapid[0]
            for side in SIDES
        }

    def _set_target_marker(self, side: str, target: Pose) -> None:
        mocap_id = self.target_mocap[side]
        quaternion = pin.Quaternion(target.rotation)
        self.data.mocap_pos[mocap_id] = target.position
        self.data.mocap_quat[mocap_id] = [
            quaternion.w,
            quaternion.x,
            quaternion.y,
            quaternion.z,
        ]

    def tick(self) -> None:
        sample = self.teleop_input.sample()
        # Hold the last commanded joints so released arms stay suspended.
        q = self.hold_q
        targets = {}
        for side in SIDES:
            current = self.ik.frame_pose(q, side)
            if sample is None:
                self.mappers[side].update(current, False, current)
                target = None
            else:
                # Arms only move while the corresponding input is active.
                target = self.mappers[side].update(
                    sample.poses[side],
                    sample.activations[side],
                    current,
                )
            self._set_target_marker(side, target or current)
            if target is not None:
                targets[side] = target
        if targets:
            self.hold_q = self.ik.step(q, targets)
        self.data.ctrl[:] = self.hold_q
        steps = max(1, round(self.dt / self.model.opt.timestep))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def run(self, duration: float | None = None, headless: bool = False) -> None:
        if duration is not None and duration <= 0.0:
            raise ValueError("Duration must be positive")
        viewer_context = nullcontext(None)
        if not headless:
            from mujoco import viewer

            viewer_context = viewer.launch_passive(self.model, self.data)
        started_at = time.monotonic()
        try:
            with viewer_context as active_viewer:
                while duration is None or time.monotonic() - started_at < duration:
                    loop_started = time.monotonic()
                    self.tick()
                    if active_viewer is not None:
                        if not active_viewer.is_running():
                            break
                        active_viewer.sync()
                    remaining = self.dt - (time.monotonic() - loop_started)
                    if remaining > 0.0:
                        time.sleep(remaining)
        finally:
            self.teleop_input.close()
