import sys
import threading
import time

import numpy as np
import pytest

from pico_bimanual_franka_teleop import hardware
from pico_bimanual_franka_teleop.hand_worker import HandWorker
from pico_bimanual_franka_teleop.types import Pose, TeleopSample
from pico_bimanual_franka_teleop.xr_input import (
    ControllerInput,
    MotionTrackerInput,
    PicoSession,
)


class _Status:
    sides = {}


class _SlowHandPipeline:
    sides = ("left", "right")
    status = _Status()

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.active = []
        self.opened = []
        self.closed = False

    def tick(self, *, active) -> None:
        self.active.append(dict(active))
        self.entered.set()
        self.release.wait(timeout=1.0)

    def request_open(self, *, sides, duration) -> None:
        self.opened.append((sides, duration))

    def close(self) -> None:
        self.closed = True


def test_slow_hand_pipeline_never_blocks_arm_side_updates():
    pipeline = _SlowHandPipeline()
    worker = HandWorker(pipeline, tick_rate=100.0)
    worker.start()
    assert pipeline.entered.wait(timeout=1.0)

    started = time.monotonic()
    worker.set_active({"left": True, "right": False})
    worker.request_open(sides=("right",))
    elapsed = time.monotonic() - started

    assert elapsed < 0.02
    pipeline.release.set()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if pipeline.opened and any(state["left"] for state in pipeline.active):
            break
        time.sleep(0.005)
    worker.close()
    assert pipeline.opened == [(("right",), 2.0)]
    assert any(state == {"left": True, "right": False} for state in pipeline.active)
    assert pipeline.closed


def test_open_request_atomically_disengages_the_selected_hand():
    pipeline = _SlowHandPipeline()
    pipeline.release.set()
    worker = HandWorker(pipeline, tick_rate=100.0)
    worker.set_active({"left": True, "right": True})
    worker.request_open(sides=("right",))
    worker.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not pipeline.opened:
        time.sleep(0.005)
    worker.close()

    assert pipeline.opened == [(("right",), 2.0)]
    assert pipeline.active[0] == {"left": True, "right": False}


def test_hand_worker_caches_measured_feedback_with_receipt_time():
    class FeedbackPipeline:
        sides = ("left",)
        feedback_sides = ("left",)
        status = type("Status", (), {"errors": 0, "last_error": None})()

        def tick(self, *, active):
            pass

        def feedback_position(self, side):
            assert side == "left"
            return np.full(20, 0.2)

        def close(self):
            pass

    worker = HandWorker(FeedbackPipeline(), tick_rate=100.0)
    worker.start()
    deadline = time.monotonic() + 1.0
    snapshot = None
    while time.monotonic() < deadline and snapshot is None:
        snapshot = worker.feedback_snapshot("left")
        time.sleep(0.005)
    worker.close()

    assert snapshot is not None
    positions, received_at = snapshot
    assert positions == tuple([0.2] * 20)
    assert received_at <= time.monotonic()


def test_dataset_recording_can_start_stop_and_finalize_without_stopping_teleop():
    statuses = []
    messages = []

    class Recorder:
        output = "/data/episode.npz"
        sample_count = 23

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    recorder = Recorder()
    teleop = object.__new__(hardware.DualFr3HardwareTeleop)
    teleop.dataset_recorder = None
    teleop.dataset_recorder_factory = lambda: recorder
    teleop.dataset_finalize_thread = None
    teleop.dataset_finalize_outcome = []
    teleop._notify = messages.append
    teleop._set_dataset_status = lambda **status: statuses.append(status)

    teleop._start_dataset_recording()
    assert teleop.dataset_recorder is recorder
    teleop._stop_dataset_recording()
    assert teleop.dataset_recorder is None
    teleop.dataset_finalize_thread.join(timeout=1.0)
    teleop._service_dataset_finalization()

    assert recorder.closed
    assert teleop.dataset_finalize_thread is None
    assert statuses[-1] == {
        "finalizing": False,
        "path": "/data/episode.npz",
        "sample_count": 23,
    }
    assert messages[-1] == "dataset saved: /data/episode.npz"


class _FakePicoClient:
    def __init__(self) -> None:
        self.init_count = 0
        self.close_count = 0
        self.timestamp = 1

    def init(self) -> None:
        self.init_count += 1

    def close(self) -> None:
        self.close_count += 1

    def get_time_stamp_ns(self) -> int:
        return self.timestamp

    def get_left_controller_pose(self):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def get_right_controller_pose(self):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def get_left_grip(self) -> float:
        return 0.0

    def get_right_grip(self) -> float:
        return 0.0

    def get_motion_snapshot(self):
        pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        return 1, self.timestamp, ["LEFT", "RIGHT"], [pose, pose]


class _Operator:
    def __init__(self) -> None:
        self.closed = False

    def poll(self):
        return {"left": False, "right": False}

    def disable_all(self, _reason):
        return None

    def show(self, _message):
        return None

    def close(self):
        self.closed = True


def test_shared_pico_session_is_the_only_sdk_lifecycle_owner(monkeypatch):
    client = _FakePicoClient()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", client)
    monkeypatch.setattr(
        "pico_bimanual_franka_teleop.xr_input.desktop_gui_pids", lambda: []
    )

    session = PicoSession()
    source = ControllerInput(
        grip_threshold=0.5,
        ready_timeout=0.1,
        stale_timeout=0.25,
        use_grip=False,
        keyboard=_Operator(),
        xrt_client=session.client,
    )
    source.close()
    assert client.init_count == 1
    assert client.close_count == 0
    session.close()
    session.close()
    assert client.close_count == 1


def test_injected_motion_source_owns_neither_session_nor_operator():
    client = _FakePicoClient()
    operator = _Operator()
    identity = {
        "translation_xyz": [0.0, 0.0, 0.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    source = MotionTrackerInput(
        serials={"left": "LEFT", "right": "RIGHT"},
        tracker_to_control={"left": identity, "right": identity},
        ready_timeout=0.1,
        stale_timeout=0.25,
        frozen_timeout=1.0,
        max_position_jump=0.2,
        max_rotation_jump=1.0,
        max_linear_speed=3.0,
        max_angular_speed=12.0,
        keyboard=operator,
        xrt_client=client,
    )

    source.close()

    assert client.init_count == 0
    assert client.close_count == 0
    assert not operator.closed


def test_hardware_coordinator_runs_with_a_device_agnostic_arm_source(monkeypatch):
    closed = {"source": False, "robot": False}
    statuses = []

    class Source:
        def sample(self):
            pose = Pose(np.zeros(3), np.eye(3))
            return TeleopSample(
                {"left": pose, "right": pose},
                {"left": False, "right": False},
                time.monotonic(),
            )

        def close(self):
            closed["source"] = True

    class Operator:
        def take_requests(self):
            return {}

        def disable_all(self, _reason):
            return None

        def deny(self, _side, _reason):
            return None

        def show(self, _message):
            return None

        def set_status(self, status):
            statuses.append(status)

    class Robot:
        def __init__(self, **_kwargs):
            return None

        def wait_for_state(self, timeout):
            return np.zeros(14)

        def receive_state(self):
            return np.zeros(14)

        def take_gateway_faults(self):
            return ()

        def send_command(self, _q, _sides):
            raise KeyboardInterrupt

        def close(self):
            closed["robot"] = True

    class IK:
        def __init__(self, **_kwargs):
            self.configuration = type("Configuration", (), {"q": np.zeros(14)})()
            self.last_diagnostics = {}

        def set_posture_reference(self, _q):
            return None

        def frame_poses(self, _q):
            pose = Pose(np.zeros(3), np.eye(3))
            return {"left": pose, "right": pose}

    monkeypatch.setattr(hardware, "UdpRobotBackend", Robot)
    monkeypatch.setattr(hardware, "BimanualPinkIK", IK)
    teleop = hardware.DualFr3HardwareTeleop(
        command_host="unused",
        command_port=1,
        state_host="unused",
        state_port=2,
        state_timeout=0.1,
        translation_scale=1.0,
        rotation_scale=1.0,
        control_rate=100.0,
        max_joint_speed=0.5,
        robot_state_wait_timeout=0.1,
        arm_source=Source(),
        operator=Operator(),
    )

    with pytest.raises(KeyboardInterrupt):
        teleop.run()

    assert closed == {"source": True, "robot": True}
    assert statuses[0] == "STATE | ready"


def test_hardware_coordinator_reports_and_exits_after_state_loss(monkeypatch):
    statuses = []
    disabled = []
    closed = {"source": False, "robot": False}

    class Source:
        def close(self):
            closed["source"] = True

    class Operator:
        def disable_all(self, reason):
            disabled.append(reason)

        def set_status(self, status):
            statuses.append(status)

        def show(self, _message):
            return None

    class Robot:
        def __init__(self, **_kwargs):
            return None

        def wait_for_state(self, timeout):
            return np.zeros(14)

        def receive_state(self):
            return None

        def send_command(self, _q, active_sides):
            assert active_sides == ()

        def close(self):
            closed["robot"] = True

    class IK:
        def __init__(self, **_kwargs):
            self.configuration = type("Configuration", (), {"q": np.zeros(14)})()

    monkeypatch.setattr(hardware, "UdpRobotBackend", Robot)
    monkeypatch.setattr(hardware, "BimanualPinkIK", IK)
    teleop = hardware.DualFr3HardwareTeleop(
        command_host="unused",
        command_port=1,
        state_host="unused",
        state_port=2,
        state_timeout=0.01,
        translation_scale=1.0,
        rotation_scale=1.0,
        control_rate=1000.0,
        max_joint_speed=0.5,
        robot_state_wait_timeout=0.01,
        arm_source=Source(),
        operator=Operator(),
    )

    with pytest.raises(TimeoutError, match="Lost fresh dual-FR3 state"):
        teleop.run()

    assert disabled == ["robot state missing or stale"]
    assert statuses
    assert statuses[-1].startswith("FAULT | robot state missing or stale")
    assert closed == {"source": True, "robot": True}
