import threading
import time
from dataclasses import replace

import numpy as np
from teleop_core.safety import LOWER_LIMITS, UPPER_LIMITS

from .ik import BimanualPinkIK, IKError, classify_step
from .interfaces import ArmPoseSource, HandController, OperatorState
from .joint_mapping import RelativeJointMapper
from .pose_mapping import RelativePoseMapper
from .robot_udp import UdpRobotBackend
from .types import ArmSample, SIDES, TeleopSample


def reseed_inactive_joints(held, measured, activations, mapper_active):
    """Keep every inactive or newly engaging side aligned to hardware."""
    result = np.asarray(held, dtype=float).copy()
    measured = np.asarray(measured, dtype=float)
    for side, joints in (("left", slice(0, 7)), ("right", slice(7, 14))):
        if not activations.get(side, False) or not mapper_active.get(side, False):
            result[joints] = measured[joints]
    return result


def disengage_sample_sides(sample: ArmSample | None, sides) -> ArmSample | None:
    if sample is None:
        return None
    activations = dict(sample.activations)
    for side in sides:
        activations[side] = False
    return replace(sample, activations=activations)


class DualFr3HardwareTeleop:
    def __init__(
        self,
        command_host: str,
        command_port: int,
        state_host: str,
        state_port: int,
        state_timeout: float,
        translation_scale: float,
        rotation_scale: float,
        control_rate: float,
        max_joint_speed: float,
        robot_state_wait_timeout: float,
        arm_source: ArmPoseSource,
        operator: OperatorState,
        hands: HandController | None = None,
        debug_logger=None,
        dataset_recorder=None,
        dataset_recorder_factory=None,
        reset_invoker=None,
        capture_home_invoker=None,
    ) -> None:
        self.arm_source = arm_source
        self.operator = operator
        self._notify = operator.show
        self._set_dataset_status = getattr(
            operator, "set_dataset_recording", lambda **_status: None
        )
        self.dt = 1.0 / control_rate
        self.robot_state_wait_timeout = robot_state_wait_timeout
        try:
            self.robot = UdpRobotBackend(
                command_host=command_host,
                command_port=command_port,
                state_host=state_host,
                state_port=state_port,
                state_timeout=state_timeout,
            )
        except BaseException:
            if hands is not None:
                hands.close()
            arm_source.close()
            raise
        self.ik = BimanualPinkIK(dt=self.dt, max_joint_speed=max_joint_speed)
        self.joint_input = getattr(arm_source, "output_kind", "pose") == "joint"
        if self.joint_input:
            max_delta = float(getattr(arm_source, "max_relative_delta", 0.25))
            max_target_velocity = getattr(
                arm_source, "max_target_velocity", None
            )
            sensitivity_by_side = getattr(arm_source, "joint_sensitivity", {})
            self.mappers = {
                side: RelativeJointMapper(
                    LOWER_LIMITS[index : index + 7],
                    UPPER_LIMITS[index : index + 7],
                    max_delta,
                    joint_sensitivity=sensitivity_by_side.get(
                        side, np.ones(7, dtype=float)
                    ),
                    max_target_velocity=max_target_velocity,
                    nominal_dt=self.dt,
                )
                for side, index in (("left", 0), ("right", 7))
            }
        else:
            self.mappers = {
                side: RelativePoseMapper(
                    translation_scale=translation_scale,
                    rotation_scale=rotation_scale,
                )
                for side in SIDES
            }
        self.hold_q: np.ndarray | None = None

        self.hands = hands
        self.debug_logger = debug_logger
        self.dataset_recorder = dataset_recorder
        self.dataset_recorder_factory = dataset_recorder_factory
        self.dataset_finalize_thread: threading.Thread | None = None
        self.dataset_finalize_outcome: list[tuple[bool, str, str, int]] = []
        self._set_dataset_status(
            configured=(
                dataset_recorder is not None or dataset_recorder_factory is not None
            ),
            active=dataset_recorder is not None,
            finalizing=False,
            path=("" if dataset_recorder is None else str(dataset_recorder.output)),
            sample_count=0,
        )
        # Reset to the captured initial pose, requested from the keyboard. The
        # operator process is deliberately ROS-free, so the reset is delegated
        # to a blocking callable (a `ros2 service call` in the container) run on
        # a worker thread; the worker only appends to `reset_outcome`, and every
        # other state change stays on the control thread.
        self.reset_invoker = reset_invoker
        self.capture_home_invoker = capture_home_invoker
        self.reset_thread: threading.Thread | None = None
        self.reset_outcome: list[tuple[bool, str]] = []
        self.reset_operation = "reset"

    def _start_dataset_recording(self) -> None:
        if self.dataset_recorder is not None:
            self._notify("dataset recording is already active")
            return
        if self.dataset_finalize_thread is not None:
            self._notify("dataset recording is still finalizing")
            return
        if self.dataset_recorder_factory is None:
            self._notify(
                "dataset recording is not configured; restart with "
                "--left-dataset-dir"
            )
            return
        try:
            self.dataset_recorder = self.dataset_recorder_factory()
        except Exception as error:  # noqa: BLE001 - report to operator
            self._notify(f"dataset recording start FAILED: {error}")
            return
        self._set_dataset_status(
            active=True,
            finalizing=False,
            path=str(self.dataset_recorder.output),
            sample_count=0,
        )
        self._notify(f"dataset recording started: {self.dataset_recorder.output}")

    def _stop_dataset_recording(self) -> None:
        recorder = self.dataset_recorder
        if recorder is None:
            self._notify("dataset recording is not active")
            return
        self.dataset_recorder = None
        path = str(recorder.output)
        sample_count = recorder.sample_count
        self._set_dataset_status(
            active=False,
            finalizing=True,
            path=path,
            sample_count=sample_count,
        )
        self._notify(f"dataset recording stopped; finalizing {path}")

        def worker() -> None:
            try:
                recorder.close()
                outcome = (True, path, "", recorder.sample_count)
            except Exception as error:  # noqa: BLE001 - report on control thread
                outcome = (False, path, str(error), recorder.sample_count)
            self.dataset_finalize_outcome.append(outcome)

        self.dataset_finalize_thread = threading.Thread(
            target=worker, name="dataset-finalizer", daemon=True
        )
        self.dataset_finalize_thread.start()

    def _service_dataset_finalization(self) -> None:
        thread = self.dataset_finalize_thread
        if thread is None or thread.is_alive():
            return
        thread.join()
        self.dataset_finalize_thread = None
        succeeded, path, error, sample_count = (
            self.dataset_finalize_outcome.pop()
            if self.dataset_finalize_outcome
            else (False, "", "no finalization result", 0)
        )
        self._set_dataset_status(
            finalizing=False, path=path, sample_count=sample_count
        )
        if succeeded:
            self._notify(f"dataset saved: {path}")
        else:
            self._notify(f"dataset finalization FAILED: {error}")

    def _close_dataset_recording(self) -> None:
        """Finalize active/background recording during process shutdown."""
        recorder = self.dataset_recorder
        self.dataset_recorder = None
        if recorder is not None:
            recorder.close()
        thread = self.dataset_finalize_thread
        if thread is not None:
            thread.join(timeout=30.0)
            if thread.is_alive():
                raise RuntimeError("dataset finalizer did not stop within 30 seconds")
            self.dataset_finalize_thread = None

    def _start_reset(self, side: str | None = None) -> None:
        """Home both arms, or only `side`. Either way the whole session
        disengages for the duration: the reset trajectory owns the command
        bus (the gateway blocks while it is active), so the other arm simply
        holds where it is."""
        if self.reset_invoker is None:
            self._notify("reset requested, but no reset command is configured")
            return
        if self.reset_thread is not None:
            self._notify("reset already in progress")
            return
        self.operator.disable_all("resetting to initial pose")
        for mapper in self.mappers.values():
            mapper.reset()
        scope = f"{side} arm" if side else "arms"
        self.reset_operation = "reset"
        self._notify(f"reset: moving {scope} to the initial pose")

        def worker() -> None:
            try:
                outcome = self.reset_invoker(side)
            except Exception as error:  # noqa: BLE001 - report, never crash the loop
                outcome = (False, str(error))
            self.reset_outcome.append(outcome)

        self.reset_thread = threading.Thread(target=worker, daemon=True)
        self.reset_thread.start()

    def _start_capture_home(self) -> None:
        """Save the current measured pose of both arms as the new home."""
        if self.capture_home_invoker is None:
            self._notify(
                "home capture requested, but no capture command is configured"
            )
            return
        if self.reset_thread is not None:
            self._notify("reset or home capture already in progress")
            return
        self.operator.disable_all("capturing current arm pose as home")
        for mapper in self.mappers.values():
            mapper.reset()
        self.reset_operation = "home capture"
        self._notify("home capture: saving both arms' current measured pose")

        def worker() -> None:
            try:
                outcome = self.capture_home_invoker()
            except Exception as error:  # noqa: BLE001 - report, never crash the loop
                outcome = (False, str(error))
            self.reset_outcome.append(outcome)

        self.reset_thread = threading.Thread(target=worker, daemon=True)
        self.reset_thread.start()

    def _open_hands(self, sides: tuple[str, ...]) -> None:
        """Stop selected sides following, then stream their open pose."""
        selected = tuple(side for side in SIDES if side in sides)
        if self.hands is None:
            self._notify("hands: not running, start with --hand-source")
            return
        for side in selected:
            self.operator.deny(side, "opening hand")
        self.hands.request_open(sides=selected)
        self._notify("hands: opening " + "/".join(selected))

    def _service_reset(self) -> None:
        """Fold a finished reset back into the loop, on the control thread."""
        if self.reset_thread is None or self.reset_thread.is_alive():
            return
        self.reset_thread.join()
        self.reset_thread = None
        succeeded, message = (
            self.reset_outcome.pop() if self.reset_outcome else (False, "no result")
        )
        operation = self.reset_operation
        self.reset_operation = "reset"
        self._notify(
            f"{operation} {'done' if succeeded else 'FAILED'}: {message}"
        )
        # The arms are wherever the reset left them, so the pre-reset hold_q is
        # a lie. Dropping it makes the loop re-seed from measured state and
        # re-anchor the IK posture reference before anything can re-engage.
        self.hold_q = None

    @staticmethod
    def _ik_status(worst: dict) -> str:
        """One operator-readable clause per side, worst tick since the last
        report: the IK failure cause, or 'ok' while tracking is transparent."""
        parts = []
        for side in SIDES:
            diagnostics = worst.get(side)
            if diagnostics is None:
                continue
            cause = classify_step(diagnostics)
            if cause == "ok":
                parts.append(f"{side} ok")
                continue
            detail = (
                f"{diagnostics['position_error'] * 1e3:.0f}mm/"
                f"{np.degrees(diagnostics['orientation_error']):.0f}deg"
            )
            if cause == "joint-limit":
                joints = ",".join(
                    f"j{joint}" for joint, _ in diagnostics["limit_joints"]
                )
                parts.append(f"{side} LIMIT {joints} off {detail}")
            elif cause == "speed-clamp":
                parts.append(f"{side} clamped off {detail}")
            else:
                parts.append(f"{side} UNREACHABLE off {detail}")
        return " | ".join(parts)

    def run(self) -> None:
        next_status_report = 0.0
        state_missing_since: float | None = None
        # Worst IK step per side since the last status report; a transient
        # at the report instant must not hide a limit hit seconds earlier.
        ik_worst: dict[str, dict] = {}
        previous_hand_sent = (
            {} if self.hands is None else {side: 0 for side in self.hands.sides}
        )
        status_summary = getattr(self.arm_source, "status_summary", None)
        try:
            self.robot.wait_for_state(timeout=self.robot_state_wait_timeout)
            # Leave the constructor default "starting..." as soon as the
            # control loop has robot state. Idle ticks used to skip set_status
            # when the input source had no summary, which left the GUI stuck.
            self.operator.set_status("STATE | ready")
            if self.hands is not None:
                start_hands = getattr(self.hands, "start", None)
                if start_hands is not None:
                    start_hands()
            while True:
                started_at = time.monotonic()
                self._service_reset()
                self._service_dataset_finalization()
                q = self.robot.receive_state()
                if q is None:
                    reason = "robot state missing or stale"
                    if state_missing_since is None:
                        state_missing_since = started_at
                        self.operator.disable_all(reason)
                    missing_for = started_at - state_missing_since
                    self.operator.set_status(
                        f"FAULT | {reason} for {missing_for:.1f}s"
                    )
                    for mapper in self.mappers.values():
                        mapper.reset()
                    self.robot.send_command(
                        self.hold_q if self.hold_q is not None else self.ik.configuration.q,
                        (),
                    )
                    # The arms are disengaged, so the hands stop following too;
                    # a pending open request still streams.
                    if self.hands is not None:
                        self.hands.set_active(
                            {side: False for side in SIDES}
                        )
                    if missing_for >= self.robot_state_wait_timeout:
                        raise TimeoutError(
                            "Lost fresh dual-FR3 state for "
                            f"{missing_for:.1f}s"
                        )
                    time.sleep(self.dt)
                    continue
                state_missing_since = None
                if self.hold_q is None:
                    self.hold_q = np.asarray(q, dtype=float).copy()
                    # Anchor the IK null-space attractor at the pose the session
                    # started from, normally the captured hardware home.
                    self.ik.set_posture_reference(self.hold_q)
                for fault in self.robot.take_gateway_faults():
                    rejected = tuple(
                        side for side in SIDES if fault.startswith(f"{side} ")
                    )
                    if len(rejected) == 1:
                        self.operator.deny(
                            rejected[0], f"safety gateway: {fault}"
                        )
                    else:
                        self.operator.disable_all(
                            f"safety gateway: {fault}"
                        )
                sample = self.arm_source.sample()
                requests = self.operator.take_requests()
                if requests.get("start_dataset_recording"):
                    self._start_dataset_recording()
                if requests.get("stop_dataset_recording"):
                    self._stop_dataset_recording()
                open_sides = {
                    side
                    for side in SIDES
                    if requests.get("open_hands")
                    or requests.get(f"open_{side}_hand")
                }
                if open_sides:
                    self._open_hands(tuple(open_sides))
                    sample = disengage_sample_sides(sample, open_sides)
                if requests.get("reset"):
                    self._start_reset()
                elif requests.get("reset_left"):
                    self._start_reset("left")
                elif requests.get("reset_right"):
                    self._start_reset("right")
                if requests.get("capture_home"):
                    self._start_capture_home()
                if self.reset_thread is not None:
                    # Reset owns the arms, while capture must observe a stable
                    # disengaged pose. Do not act on stale activations.
                    self.operator.disable_all("arm operation in progress")
                    sample = None
                if self.reset_thread is None:
                    activations = (
                        {side: False for side in SIDES}
                        if sample is None
                        else sample.activations
                    )
                    self.hold_q = reseed_inactive_joints(
                        self.hold_q,
                        q,
                        activations,
                        {
                            side: self.mappers[side].active
                            for side in SIDES
                        },
                    )
                targets = {}
                if self.joint_input:
                    for side, joints in (
                        ("left", slice(0, 7)),
                        ("right", slice(7, 14)),
                    ):
                        leader = (
                            np.zeros(7)
                            if sample is None
                            else sample.positions[side]
                        )
                        target = self.mappers[side].update(
                            leader,
                            sample is not None and sample.activations[side],
                            q[joints],
                        )
                        if target is not None:
                            self.hold_q[joints] = target
                            targets[side] = target
                    self.ik.last_diagnostics = {}
                else:
                    current_poses = self.ik.frame_poses(self.hold_q)
                    for side in SIDES:
                        current = current_poses[side]
                        if sample is None:
                            self.mappers[side].update(current, False, current)
                            continue
                        target = self.mappers[side].update(
                            sample.poses[side], sample.activations[side], current
                        )
                        if target is not None:
                            targets[side] = target
                    try:
                        if targets:
                            self.hold_q = self.ik.step(self.hold_q, targets)
                    except IKError:
                        self.robot.send_command(q, ())
                        raise
                    for side, diagnostics in self.ik.last_diagnostics.items():
                        worst = ik_worst.get(side)
                        if worst is None or diagnostics["position_error"] > worst[
                            "position_error"
                        ]:
                            ik_worst[side] = diagnostics
                active_sides = tuple(side for side in SIDES if side in targets)
                self.robot.send_command(self.hold_q, active_sides)
                if self.debug_logger is not None:
                    raw_pose_reader = getattr(
                        self.arm_source, "debug_raw_poses", None
                    )
                    raw_poses = (
                        raw_pose_reader() if raw_pose_reader is not None else {}
                    )
                    feed_reader = getattr(
                        self.arm_source, "debug_feed_state", None
                    )
                    pose_sample = sample if isinstance(sample, TeleopSample) else None
                    pose_targets = {} if self.joint_input else targets
                    self.debug_logger.record(
                        time.monotonic(),
                        q,
                        self.hold_q,
                        {} if pose_sample is None else pose_sample.poses,
                        {} if sample is None else sample.activations,
                        pose_targets,
                        self.ik.frame_poses(self.hold_q),
                        raw_tracker_poses=raw_poses,
                        measured_ee_poses=self.ik.frame_poses(q),
                        ik_diagnostics=self.ik.last_diagnostics,
                        feed_state=(
                            feed_reader() if feed_reader is not None else None
                        ),
                    )
                hand_activations = {side: False for side in SIDES}
                if self.hands is not None:
                    poll_hands = getattr(self.operator, "poll_hands", None)
                    hand_activations = (
                        poll_hands()
                        if poll_hands is not None
                        else (
                            {side: False for side in SIDES}
                            if sample is None
                            else sample.activations
                        )
                    )
                    self.hands.set_active(hand_activations)
                if self.dataset_recorder is not None:
                    try:
                        hand_feedback = (
                            None
                            if self.hands is None
                            else self.hands.feedback_snapshot("left")
                        )
                        recorded_at = time.monotonic()
                        self.dataset_recorder.record(
                            wall_time_ns=time.time_ns(),
                            monotonic_time_s=recorded_at,
                            arm_q=np.asarray(q, dtype=float)[:7],
                            hand_feedback=hand_feedback,
                            ee_pose=self.ik.named_frame_pose(
                                q, "left_fr3v2_link8"
                            ),
                            arm_active=(
                                sample is not None
                                and bool(sample.activations.get("left", False))
                            ),
                            hand_active=bool(hand_activations.get("left", False)),
                        )
                    except Exception as error:  # noqa: BLE001 - never stop control
                        self._notify(f"dataset recording disabled: {error}")
                        self._stop_dataset_recording()
                now = time.monotonic()
                if now >= next_status_report:
                    if self.dataset_recorder is not None:
                        self._set_dataset_status(
                            sample_count=self.dataset_recorder.sample_count
                        )
                    input_summary = (
                        status_summary() if status_summary is not None else ""
                    )
                    source_name = getattr(self.arm_source, "source_name", "input")
                    parts = [
                        f"{source_name}: {input_summary}"
                        if input_summary
                        else f"{source_name}: ready"
                    ]
                    ik_summary = self._ik_status(ik_worst)
                    if ik_summary:
                        parts.append(f"ik: {ik_summary}")
                    ik_worst = {}
                    if self.hands is not None:
                        hand_parts = []
                        for side in self.hands.sides:
                            status = self.hands.status.sides[side]
                            sent = status.sent - previous_hand_sent[side]
                            previous_hand_sent[side] = status.sent
                            state = (
                                f"sending {sent:.0f}Hz"
                                if status.sending
                                else f"stopped ({status.fault})"
                            )
                            hand_parts.append(f"{side}={state}")
                        parts.append("hands: " + " | ".join(hand_parts))
                    self.operator.set_status("STATE | " + " | ".join(parts))
                    next_status_report = now + 1.0
                remaining = self.dt - (time.monotonic() - started_at)
                if remaining > 0.0:
                    time.sleep(remaining)
        finally:
            if self.debug_logger is not None:
                self.debug_logger.close()
            try:
                self._close_dataset_recording()
            except Exception as error:  # noqa: BLE001 - hardware still must close
                print(f"dataset finalization FAILED: {error}")
            # Close the hand pipeline before the SDK client it reads from.
            try:
                if self.hands is not None:
                    self.hands.close()
            finally:
                try:
                    self.robot.close()
                finally:
                    self.arm_source.close()
