from dataclasses import dataclass

import numpy as np

from .contract import COMMAND_JOINT_NAMES, SIDES, command_names


LOWER_LIMITS = np.array(
    [-2.9007400167, -1.8360900167, -2.9007400167, -3.0770200167,
     -2.87630335, 0.43982265, -3.05083335] * 2
)
UPPER_LIMITS = np.array(
    [2.9007400167, 1.8360900167, 2.9007400167, -0.1169370833,
     2.87630335, 4.62163335, 3.05083335] * 2
)


@dataclass(frozen=True)
class ValidatedCommand:
    names: tuple[str, ...]
    positions: tuple[float, ...]


class CommandSafetyGate:
    def __init__(self, max_joint_speed, max_initial_delta, nominal_dt,
                 contact_torque_thresholds):
        if max_joint_speed <= 0 or max_initial_delta <= 0 or nominal_dt <= 0:
            raise ValueError("Safety limits must be positive.")
        self.max_joint_speed = float(max_joint_speed)
        self.max_initial_delta = float(max_initial_delta)
        self.nominal_dt = float(nominal_dt)
        # Contact gating: a joint whose measured external torque exceeds its
        # threshold may only move the command in the unloading direction.
        # tau_ext is already Jacobian-mapped by libfranka, so the sign test
        # needs no kinematics: stepping a joint against its external torque
        # does positive work on the contact and presses harder. Without the
        # gate the slew limit walks the command into an obstacle at
        # max_joint_speed while the stiff impedance turns the deviation into
        # k_gains * error torque until the Franka reflex fires.
        thresholds = np.asarray(contact_torque_thresholds, dtype=float).reshape(-1)
        if (
            thresholds.size != 7
            or not np.all(np.isfinite(thresholds))
            or np.any(thresholds <= 0)
        ):
            raise ValueError(
                "contact_torque_thresholds needs 7 finite positive values (Nm)."
            )
        self.contact_torque_thresholds = thresholds
        self.active = {side: False for side in SIDES}
        self.last_output = {side: None for side in SIDES}
        self.last_time = {side: None for side in SIDES}
        # Per-message reasons for sides dropped by validate().
        self.side_faults: list[str] = []
        # 1-based joints currently held by the contact gate, per side; the
        # gateway logs the transitions so trials have direct evidence of
        # when gating engaged (2026-07-29: had to be inferred from EE
        # geometry after the fact).
        self.pressing_joints = {side: () for side in SIDES}

    def reset(self):
        for side in SIDES:
            self.active[side] = False
            self.last_output[side] = None
            self.last_time[side] = None
            self.pressing_joints[side] = ()

    def validate(self, active_sides, names, positions, measured_q, now,
                 external_torques=None):
        """Validate one command tick.

        external_torques maps side -> 7 external joint torques (Nm) from the
        robot state broadcaster, or None when that side's estimate is missing
        or stale. A missing estimate disables gating for that side (fail
        open): the reflex thresholds below still protect, while failing
        closed would freeze teleoperation on a dropped diagnostic topic.
        """
        active_sides = tuple(active_sides)
        names = tuple(names)
        positions = tuple(float(value) for value in positions)
        measured = np.asarray(measured_q, dtype=float)
        if measured.shape != (14,) or not np.all(np.isfinite(measured)):
            raise ValueError("Measured state must contain 14 finite positions.")
        if names != command_names(active_sides) or len(positions) != len(names):
            raise ValueError("Command names do not match active sides.")
        values = dict(zip(names, positions))
        self.side_faults = []
        output_names = []
        output_positions = []
        for side in SIDES:
            offset = 0 if side == "left" else 7
            if side not in active_sides:
                self.active[side] = False
                self.last_output[side] = None
                self.last_time[side] = None
                self.pressing_joints[side] = ()
                continue
            side_names = COMMAND_JOINT_NAMES[offset:offset + 7]
            try:
                target = np.asarray(
                    [values[name] for name in side_names], dtype=float
                )
                if not np.all(np.isfinite(target)):
                    raise ValueError(
                        f"{side} command contains non-finite positions."
                    )
                if np.any(target < LOWER_LIMITS[offset:offset + 7]) or np.any(
                    target > UPPER_LIMITS[offset:offset + 7]
                ):
                    raise ValueError(f"{side} command exceeds FR3 joint limits.")
                if not self.active[side]:
                    current = measured[offset:offset + 7]
                    if np.max(np.abs(target - current)) > self.max_initial_delta:
                        raise ValueError(
                            f"{side} first target is too far from measured state."
                        )
                    self.last_output[side] = current.copy()
                    self.last_time[side] = now - self.nominal_dt
                    self.active[side] = True
            except ValueError as error:
                # One side's fault must not silence the other: drop only
                # this side (2026-07-29 a rejected right re-engage froze the
                # left arm too, because the whole message was discarded).
                self.active[side] = False
                self.last_output[side] = None
                self.last_time[side] = None
                self.pressing_joints[side] = ()
                self.side_faults.append(str(error))
                continue
            dt = min(0.1, max(self.nominal_dt, now - self.last_time[side]))
            max_step = self.max_joint_speed * dt
            command = self.last_output[side] + np.clip(
                target - self.last_output[side], -max_step, max_step
            )
            torques = (external_torques or {}).get(side)
            if torques is not None:
                torques = np.asarray(torques, dtype=float)
                step = command - self.last_output[side]
                # Sign convention measured on hardware (2026-07-29 probe,
                # right arm: corr <= -0.93 on six joints, 1700+ samples each):
                # tau_ext_hat_filtered carries the sign OPPOSITE to the
                # external push, so a step with the same sign as the reported
                # torque presses harder. Hold that joint; the unloading
                # direction always passes, so retreat releases immediately.
                pressing = (
                    np.abs(torques) > self.contact_torque_thresholds
                ) & (np.sign(step) == np.sign(torques))
                command = np.where(pressing, self.last_output[side], command)
                self.pressing_joints[side] = tuple(
                    int(index) + 1 for index in np.nonzero(pressing)[0]
                )
            else:
                self.pressing_joints[side] = ()
            self.last_output[side] = command
            self.last_time[side] = now
            output_names.extend(side_names)
            output_positions.extend(command)
        if not output_names:
            return None
        return ValidatedCommand(tuple(output_names), tuple(float(v) for v in output_positions))
