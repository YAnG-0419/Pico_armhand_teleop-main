"""Relative joint mapping shared by joint-space leader devices."""

from __future__ import annotations

import time

import numpy as np


class RelativeJointMapper:
    """Anchor a leader pose to measured robot joints on each engage edge."""

    def __init__(
        self,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        max_relative_delta: float,
        joint_sensitivity: np.ndarray | None = None,
        max_target_velocity: float | None = None,
        nominal_dt: float = 0.01,
    ) -> None:
        self.lower_limits = np.asarray(lower_limits, dtype=float)
        self.upper_limits = np.asarray(upper_limits, dtype=float)
        if self.lower_limits.shape != (7,) or self.upper_limits.shape != (7,):
            raise ValueError("Joint limits must each contain 7 values")
        if max_relative_delta <= 0:
            raise ValueError("max_relative_delta must be positive")
        if max_target_velocity is not None and max_target_velocity <= 0:
            raise ValueError("max_target_velocity must be positive")
        if nominal_dt <= 0:
            raise ValueError("nominal_dt must be positive")
        sensitivity = (
            np.ones(7, dtype=float)
            if joint_sensitivity is None
            else np.asarray(joint_sensitivity, dtype=float)
        )
        if (
            sensitivity.shape != (7,)
            or not np.all(np.isfinite(sensitivity))
            or np.any(sensitivity <= 0.0)
        ):
            raise ValueError("Joint sensitivity must contain seven positive values")
        self.max_relative_delta = float(max_relative_delta)
        self.joint_sensitivity = sensitivity.copy()
        self.max_target_velocity = (
            None if max_target_velocity is None else float(max_target_velocity)
        )
        self.nominal_dt = float(nominal_dt)
        self.active = False
        self._leader_anchor: np.ndarray | None = None
        self._robot_anchor: np.ndarray | None = None
        self._last_target: np.ndarray | None = None
        self._last_update: float | None = None

    def reset(self) -> None:
        self.active = False
        self._leader_anchor = None
        self._robot_anchor = None
        self._last_target = None
        self._last_update = None

    def update(
        self,
        leader_joints: np.ndarray,
        engaged: bool,
        measured_robot_joints: np.ndarray,
        now: float | None = None,
    ) -> np.ndarray | None:
        moment = time.monotonic() if now is None else float(now)
        leader = np.asarray(leader_joints, dtype=float)
        measured = np.asarray(measured_robot_joints, dtype=float)
        if leader.shape != (7,) or measured.shape != (7,):
            raise ValueError("Leader and robot inputs must each contain 7 joints")
        if not np.all(np.isfinite(leader)) or not np.all(np.isfinite(measured)):
            raise ValueError("Leader and robot inputs must be finite")
        if not engaged:
            self.reset()
            return None
        if not self.active:
            self._leader_anchor = leader.copy()
            self._robot_anchor = measured.copy()
            self._last_target = measured.copy()
            self._last_update = moment
            self.active = True
            return measured.copy()

        raw_delta = leader - self._leader_anchor
        delta = np.clip(
            raw_delta * self.joint_sensitivity,
            -self.max_relative_delta,
            self.max_relative_delta,
        )
        desired = np.clip(
            self._robot_anchor + delta,
            self.lower_limits,
            self.upper_limits,
        )
        if self.max_target_velocity is None:
            target = desired
        else:
            elapsed = max(moment - self._last_update, self.nominal_dt)
            max_step = self.max_target_velocity * elapsed
            target = self._last_target + np.clip(
                desired - self._last_target, -max_step, max_step
            )
        self._last_target = target.copy()
        self._last_update = moment
        return target

    def diagnostics(self, leader_joints: np.ndarray) -> dict:
        """Return JSON-safe mapping state for later logging integration."""
        leader = np.asarray(leader_joints, dtype=float)
        delta = None
        scaled_delta = None
        if self._leader_anchor is not None:
            raw_delta = leader - self._leader_anchor
            delta = raw_delta.tolist()
            scaled_delta = (raw_delta * self.joint_sensitivity).tolist()
        return {
            "active": self.active,
            "leader_anchor": (
                None if self._leader_anchor is None else self._leader_anchor.tolist()
            ),
            "robot_anchor": (
                None if self._robot_anchor is None else self._robot_anchor.tolist()
            ),
            "leader_delta": delta,
            "joint_sensitivity": self.joint_sensitivity.tolist(),
            "scaled_delta": scaled_delta,
        }
