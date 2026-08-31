from collections.abc import Mapping

import numpy as np
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import DampingTask, FrameTask, PostureTask
from qpsolvers.exceptions import QPError

from .paths import URDF_PATH
from .types import Pose, SIDES


END_EFFECTOR_FRAMES = {
    "left": "left_fr3v2_link7",
    "right": "right_fr3v2_link7",
}


class IKError(RuntimeError):
    pass


# A step is transparent when the commanded end effector lands within these
# tolerances of the target; anything worse deserves an attributed cause.
POSITION_ERROR_TOLERANCE = 0.010  # m
ORIENTATION_ERROR_TOLERANCE = 0.05  # rad
# A joint this close to a position limit stops contributing to some Cartesian
# directions; report it as the binding constraint.
LIMIT_MARGIN = 0.05  # rad
# Closing speed that separates catching-up from stuck. At a workspace edge
# the QP chatters at the velocity clamp without net progress (measured: a
# 1.5 m target settles at 1.11 m error with the clamp saturated every tick),
# so saturation alone cannot make that call; genuine catch-up closes at
# millimetres per tick.
PROGRESS_EPSILON = 0.0002  # m per tick, smoothed


def classify_step(diagnostics: Mapping) -> str:
    """Name the binding constraint behind one side's IK step.

    ``ok`` tracking within tolerance; ``joint-limit`` a joint is pinned at a
    position limit; ``speed-clamp`` the joint-speed clamp is saturated and
    the error is still shrinking (the commanded pose is catching up);
    ``workspace`` the error persists without progress - out of reach.
    """
    if (
        diagnostics["position_error"] <= POSITION_ERROR_TOLERANCE
        and diagnostics["orientation_error"] <= ORIENTATION_ERROR_TOLERANCE
    ):
        return "ok"
    if diagnostics["limit_joints"]:
        return "joint-limit"
    if (
        diagnostics["saturated_joints"]
        and diagnostics.get("progress", float("inf")) > PROGRESS_EPSILON
    ):
        return "speed-clamp"
    return "workspace"


class BimanualPinkIK:
    def __init__(self, dt: float, max_joint_speed: float) -> None:
        if dt <= 0.0 or max_joint_speed <= 0.0:
            raise ValueError("IK timestep and joint speed must be positive")
        self.dt = float(dt)
        self.max_joint_speed = float(max_joint_speed)
        self.model = pin.buildModelFromUrdf(str(URDF_PATH))
        self.data = self.model.createData()
        self.configuration = pink.Configuration(
            self.model,
            self.data,
            pin.neutral(self.model),
        )
        self.frame_tasks = {
            side: FrameTask(
                frame,
                position_cost=100.0,
                orientation_cost=20.0,
            )
            for side, frame in END_EFFECTOR_FRAMES.items()
        }
        # A fixed posture reference gives the null space somewhere to go. With a
        # 7-DoF arm every end-effector pose has a one-parameter family of elbow
        # configurations, and with no attractor the elbow random-walks: measured
        # on closed end-effector loops, q drifted 0.88 rad in one loop while the
        # end effector returned to within 0.00 mm. Drifted configurations end up
        # near joint limits where some directions stop responding, which the
        # operator experiences as ambiguity. The cost is far below the frame
        # tasks' so tracking stays practically exact; the reference defaults to
        # the first configuration seen and is normally the captured hardware
        # home. The cost was swept: 0.2 is too weak to pull the elbow back and
        # drift reached 2.3 rad, while 1.0 returned the configuration to within
        # 0.000 rad after ten adversarial loops at a worst-case tracking cost of
        # 1.5 mm at a 30 cm displacement; 3.0 already costs 12 mm. The same
        # equilibrium holds a 90-degree orientation step to within 0.875 deg,
        # measured, against 0.232 deg at cost 0.5 whose drift protection fails.
        # The attractor is a constant pull, so unlike the speed limit and the
        # damping task it does trade a sub-perceptual amount of steady-state
        # accuracy for a bounded elbow.
        self.posture_task = PostureTask(cost=1.0)
        self.posture_reference: np.ndarray | None = None
        # Per-side facts about the most recent step(), for the debug log and
        # the operator display: why the commanded pose is not on the target.
        self.last_diagnostics: dict[str, dict] = {}
        # Smoothed per-tick decrease of each side's position error; feeds the
        # catching-up-versus-stuck call in classify_step.
        self._error_progress: dict[str, tuple[float, float]] = {}
        self.damping_task = DampingTask(cost=10.0)
        self.joint_names = tuple(str(name) for name in self.model.names[1:])
        expected = tuple(
            f"{side}_fr3v2_joint{index}"
            for side in SIDES
            for index in range(1, 8)
        )
        if self.joint_names != expected:
            raise ValueError(f"Unexpected URDF joint order: {self.joint_names}")

    def set_posture_reference(self, q: np.ndarray) -> None:
        """Anchor the null-space attractor, normally at the hardware home."""
        values = np.asarray(q, dtype=float)
        if values.shape != (self.model.nq,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected {self.model.nq} finite joint positions")
        self.posture_reference = np.clip(
            values,
            self.model.lowerPositionLimit,
            self.model.upperPositionLimit,
        )

    def update(self, q: np.ndarray) -> None:
        values = np.asarray(q, dtype=float)
        if values.shape != (self.model.nq,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected {self.model.nq} finite joint positions")
        # MuJoCo/physics can drift a few ulps past joint limits; Pink rejects that.
        values = np.clip(
            values,
            self.model.lowerPositionLimit,
            self.model.upperPositionLimit,
        )
        self.configuration.update(values)

    def frame_pose(self, q: np.ndarray, side: str) -> Pose:
        if side not in END_EFFECTOR_FRAMES:
            raise ValueError(f"Unknown side: {side}")
        self.update(q)
        transform = self.configuration.get_transform_frame_to_world(
            END_EFFECTOR_FRAMES[side]
        )
        # Copy out of Pinocchio's mutable data cache: a later FK update must
        # not silently rewrite a pose retained by a mapper or debug logger.
        return Pose(transform.translation.copy(), transform.rotation.copy())

    def named_frame_pose(self, q: np.ndarray, frame: str) -> Pose:
        """Return any model frame pose without changing the control frame."""
        if not self.model.existFrame(frame):
            raise ValueError(f"Unknown model frame: {frame}")
        self.update(q)
        transform = self.configuration.get_transform_frame_to_world(frame)
        return Pose(transform.translation.copy(), transform.rotation.copy())

    def frame_poses(self, q: np.ndarray) -> dict[str, Pose]:
        """Return both end-effector poses from one FK update."""
        self.update(q)
        poses = {}
        for side, frame in END_EFFECTOR_FRAMES.items():
            transform = self.configuration.get_transform_frame_to_world(frame)
            poses[side] = Pose(
                transform.translation.copy(),
                transform.rotation.copy(),
            )
        return poses

    def step(self, q: np.ndarray, targets: Mapping[str, Pose]) -> np.ndarray:
        unknown = set(targets).difference(SIDES)
        if unknown:
            raise ValueError(f"Unknown target sides: {sorted(unknown)}")
        self.update(q)
        self.last_diagnostics = {}
        if not targets:
            self._error_progress.clear()
            return self.configuration.q.copy()

        if self.posture_reference is None:
            self.posture_reference = self.configuration.q.copy()
        self.posture_task.set_target(self.posture_reference)
        tasks = [self.posture_task, self.damping_task]
        for side, target in targets.items():
            self.frame_tasks[side].set_target(
                pin.SE3(target.rotation, target.position)
            )
            tasks.append(self.frame_tasks[side])
        try:
            velocity = solve_ik(
                self.configuration,
                tasks,
                self.dt,
                solver="quadprog",
                safety_break=True,
            )
        except (QPError, AssertionError) as exc:
            raise IKError(f"Pink failed to solve the bimanual target: {exc}") from exc
        raw_velocity = velocity
        velocity = np.clip(velocity, -self.max_joint_speed, self.max_joint_speed)
        result = pin.integrate(self.model, self.configuration.q, velocity * self.dt)
        result = np.clip(
            result,
            self.model.lowerPositionLimit,
            self.model.upperPositionLimit,
        )
        # A side without a target has no frame task pinning it, so the
        # posture attractor would walk its joints toward the reference at
        # the speed clamp - while the real arm, which is not being
        # commanded, stays put (measured 2026-07-29: 0.35 rad of phantom
        # drift within 0.75 s of a single-side engage, deadlocking the
        # other side's re-engage against the gateway's initial-delta
        # check). An uncommanded side's joints never move.
        for side, joints in (("left", slice(0, 7)), ("right", slice(7, 14))):
            if side not in targets:
                result[joints] = self.configuration.q[joints]
        self._record_diagnostics(result, raw_velocity, targets)
        return result

    def _record_diagnostics(
        self,
        q_commanded: np.ndarray,
        raw_velocity: np.ndarray,
        targets: Mapping[str, Pose],
    ) -> None:
        """Attribute each side's residual to its binding constraint.

        Runs one extra FK on the already-updated data (tens of microseconds
        against the 10 ms tick); everything else is plain array comparisons.
        """
        self.update(q_commanded)
        margins = np.minimum(
            q_commanded - self.model.lowerPositionLimit,
            self.model.upperPositionLimit - q_commanded,
        )
        diagnostics: dict[str, dict] = {}
        for side, target in targets.items():
            transform = self.configuration.get_transform_frame_to_world(
                END_EFFECTOR_FRAMES[side]
            )
            rotation_residual = pin.log3(
                target.rotation @ transform.rotation.T
            )
            joints = slice(0, 7) if side == "left" else slice(7, 14)
            saturated = tuple(
                int(index) + 1
                for index, value in enumerate(raw_velocity[joints])
                if abs(value) > self.max_joint_speed
            )
            limits = tuple(
                (int(index) + 1, float(margin))
                for index, margin in enumerate(margins[joints])
                if margin < LIMIT_MARGIN
            )
            position_error = float(
                np.linalg.norm(target.position - transform.translation)
            )
            # Smooth the per-tick error decrease: the clamp chatter at a
            # workspace edge alternates sign, so the average goes to zero
            # while genuine catch-up stays positive. A freshly engaged side
            # counts as catching up until the average exists.
            state = self._error_progress.get(side)
            if state is None:
                progress = float("inf")
                average = None
            else:
                previous_error, average = state
                sample = previous_error - position_error
                average = (
                    sample if average is None else 0.8 * average + 0.2 * sample
                )
                progress = average
            self._error_progress[side] = (position_error, average)
            diagnostics[side] = {
                "position_error": position_error,
                "orientation_error": float(np.linalg.norm(rotation_residual)),
                "saturated_joints": saturated,
                "limit_joints": limits,
                "progress": progress,
            }
        for side in tuple(self._error_progress):
            if side not in targets:
                del self._error_progress[side]
        self.last_diagnostics = diagnostics
