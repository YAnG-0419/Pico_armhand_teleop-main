import sys

import numpy as np
import pytest

from pico_bimanual_franka_teleop import xr_input


class _FakeXrt:
    def __init__(self) -> None:
        self.closed = False
        self.timestamp = 123
        self.motion_sequence = 1
        self.serials = ["RIGHT-SN", "LEFT-SN"]
        self.poses = [
            [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0],
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        ]

    def init(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def get_motion_timestamp_ns(self) -> int:
        return self.timestamp

    def get_time_stamp_ns(self) -> int:
        return self.timestamp

    def get_motion_frame_sequence(self) -> int:
        return self.motion_sequence

    def get_motion_snapshot(self):
        return (
            self.motion_sequence,
            self.timestamp,
            list(self.serials),
            [list(pose) for pose in self.poses],
        )

    def get_callback_error_count(self) -> int:
        return 0

    def num_motion_data_available(self) -> int:
        return len(self.serials)

    def get_motion_tracker_serial_numbers(self) -> list[str]:
        return list(self.serials)

    def get_motion_tracker_pose(self) -> list[list[float]]:
        return list(self.poses)


def _identity_transforms() -> dict[str, dict[str, list[float]]]:
    transform = {
        "translation_xyz": [0.0, 0.0, 0.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    return {"left": dict(transform), "right": dict(transform)}


def _create_tracker_input() -> xr_input.MotionTrackerInput:
    return xr_input.MotionTrackerInput(
        serials={"left": "LEFT-SN", "right": "RIGHT-SN"},
        tracker_to_control=_identity_transforms(),
        ready_timeout=0.1,
        stale_timeout=0.25,
        frozen_timeout=1.0,
        max_position_jump=0.2,
        max_rotation_jump=1.0,
        max_linear_speed=3.0,
        max_angular_speed=12.0,
    )


def _remove_tracker(fake_xrt: "_FakeXrt", serial: str) -> None:
    index = fake_xrt.serials.index(serial)
    fake_xrt.serials.pop(index)
    fake_xrt.poses.pop(index)


def test_motion_trackers_are_mapped_by_serial(monkeypatch) -> None:
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": True, "right": True}

    sample = tracker_input.sample()

    assert sample is not None
    np.testing.assert_allclose(sample.poses["left"].position, [-0.3, -0.1, 0.2])
    np.testing.assert_allclose(sample.poses["right"].position, [-0.6, -0.4, 0.5])
    assert sample.activations == {"left": True, "right": True}
    tracker_input.close()
    assert fake_xrt.closed


def test_single_tracker_startup_does_not_require_the_other(monkeypatch) -> None:
    fake_xrt = _FakeXrt()
    fake_xrt.serials = ["RIGHT-SN"]
    fake_xrt.poses = [fake_xrt.poses[0]]
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)

    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active["right"] = True
    sample = tracker_input.sample()

    assert sample is not None
    assert sample.activations == {"left": False, "right": True}
    assert tracker_input.readiness["left"].startswith("missing (LEFT-SN)")
    assert tracker_input.readiness["right"] == "ready"
    tracker_input.close()


def test_no_tracker_startup_proceeds_disarmed(monkeypatch) -> None:
    # A missing tracker must not hold the workcell hostage: the session
    # starts, arms cannot engage, and O/H keep working against the robot.
    fake_xrt = _FakeXrt()
    fake_xrt.serials = []
    fake_xrt.poses = []
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)

    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": True, "right": True}
    fake_xrt.timestamp += 20_000_000
    sample = tracker_input.sample()

    assert sample is not None
    assert sample.activations == {"left": False, "right": False}
    assert tracker_input.readiness["left"].startswith("missing (LEFT-SN)")
    assert tracker_input.readiness["right"].startswith("missing (RIGHT-SN)")
    tracker_input.close()


def test_engaging_a_missing_side_denies_only_that_side(monkeypatch) -> None:
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": True, "right": False}
    fake_xrt.timestamp += 20_000_000
    assert tracker_input.sample() is not None

    _remove_tracker(fake_xrt, "RIGHT-SN")
    tracker_input.keyboard.active["right"] = True
    fake_xrt.timestamp += 20_000_000
    sample = tracker_input.sample()

    assert sample is not None
    assert sample.activations == {"left": True, "right": False}
    assert tracker_input.keyboard.active["right"] is False


def test_tracker_loss_while_engaged_disengages_everything(monkeypatch) -> None:
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": True, "right": True}
    fake_xrt.timestamp += 20_000_000
    assert tracker_input.sample() is not None

    _remove_tracker(fake_xrt, "LEFT-SN")
    fake_xrt.timestamp += 20_000_000

    assert tracker_input.sample() is None
    assert tracker_input.keyboard.active == {"left": False, "right": False}


def test_disengaged_tracker_loss_does_not_stop_active_side(monkeypatch) -> None:
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": False, "right": True}
    fake_xrt.timestamp += 20_000_000
    assert tracker_input.sample() is not None

    _remove_tracker(fake_xrt, "LEFT-SN")
    fake_xrt.timestamp += 20_000_000
    sample = tracker_input.sample()

    assert sample is not None
    assert sample.activations == {"left": False, "right": True}


def test_reappearing_tracker_reseeds_without_tripping_the_jump_guard(
    monkeypatch,
) -> None:
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": False, "right": True}
    fake_xrt.timestamp += 20_000_000
    assert tracker_input.sample() is not None

    removed_pose = list(fake_xrt.poses[1])
    _remove_tracker(fake_xrt, "LEFT-SN")
    fake_xrt.timestamp += 20_000_000
    assert tracker_input.sample() is not None

    # The tracker reappears far from where it vanished, and is engaged.
    removed_pose[0] += 0.9
    fake_xrt.serials.append("LEFT-SN")
    fake_xrt.poses.append(removed_pose)
    tracker_input.keyboard.active["left"] = True
    fake_xrt.timestamp += 20_000_000
    sample = tracker_input.sample()

    assert sample is not None
    assert sample.activations == {"left": True, "right": True}


def test_motion_snapshot_rejects_inconsistent_count() -> None:
    fake_xrt = _FakeXrt()
    fake_xrt.serials = ["LEFT-SN"]
    tracker_input = object.__new__(xr_input.MotionTrackerInput)
    tracker_input.xrt = fake_xrt
    tracker_input.serials = {"left": "LEFT-SN", "right": "RIGHT-SN"}
    tracker_input.detected_serials = []
    tracker_input.readiness = {"left": "waiting", "right": "waiting"}

    assert tracker_input._snapshot() is None


def test_motion_snapshot_accepts_five_trackers() -> None:
    fake_xrt = _FakeXrt()
    fake_xrt.serials = [f"SN-{index}" for index in range(5)]
    fake_xrt.poses = [
        [float(index), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        for index in range(5)
    ]
    tracker_input = object.__new__(xr_input.MotionTrackerInput)
    tracker_input.xrt = fake_xrt
    tracker_input.serials = {"left": "SN-3", "right": "SN-1"}
    tracker_input.detected_serials = []
    tracker_input.readiness = {"left": "waiting", "right": "waiting"}

    snapshot = tracker_input._snapshot()

    assert snapshot is not None
    _, selected, serials = snapshot
    assert serials == fake_xrt.serials
    np.testing.assert_allclose(selected["left"][:3], [3.0, 0.0, 0.0])
    np.testing.assert_allclose(selected["right"][:3], [1.0, 0.0, 0.0])


def test_motion_input_accepts_bounded_motion(monkeypatch) -> None:
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": True, "right": True}
    fake_xrt.timestamp += 20_000_000
    fake_xrt.poses[1][0] += 0.01

    sample = tracker_input.sample()

    assert sample is not None
    assert sample.activations == {"left": True, "right": True}


def test_motion_input_disengages_on_pose_jump(monkeypatch) -> None:
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": True, "right": True}
    fake_xrt.timestamp += 20_000_000
    assert tracker_input.sample() is not None
    fake_xrt.timestamp += 20_000_000
    fake_xrt.poses[1][0] += 0.3

    assert tracker_input.sample() is None
    assert tracker_input.keyboard.active == {"left": False, "right": False}


def test_motion_input_recovers_after_timestamp_restart(monkeypatch) -> None:
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": True, "right": True}
    fake_xrt.timestamp = 1

    assert tracker_input.sample() is None
    assert tracker_input.keyboard.active == {"left": False, "right": False}

    tracker_input.keyboard.active = {"left": True, "right": True}
    fake_xrt.timestamp += 20_000_000
    sample = tracker_input.sample()

    assert sample is not None
    assert sample.activations == {"left": True, "right": True}


def test_inactive_tracker_can_move_freely(monkeypatch) -> None:
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": False, "right": True}
    fake_xrt.timestamp += 20_000_000
    fake_xrt.poses[1][0] += 0.5

    sample = tracker_input.sample()

    assert sample is not None
    assert sample.activations == {"left": False, "right": True}


def test_reenabled_tracker_reanchors_safety_baseline(monkeypatch) -> None:
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": False, "right": True}
    fake_xrt.timestamp += 20_000_000
    fake_xrt.poses[1][0] += 0.5
    assert tracker_input.sample() is not None

    tracker_input.keyboard.active = {"left": True, "right": True}
    fake_xrt.timestamp += 20_000_000
    fake_xrt.poses[1][0] += 0.5
    sample = tracker_input.sample()

    assert sample is not None
    assert sample.activations == {"left": True, "right": True}


def test_motion_input_disengages_on_frozen_pose(monkeypatch) -> None:
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": True, "right": True}
    fake_xrt.timestamp += 20_000_000
    assert tracker_input.sample() is not None
    tracker_input.last_position_changed_at["left"] -= 2.0
    tracker_input.last_rotation_changed_at["left"] -= 2.0
    fake_xrt.timestamp += 20_000_000

    assert tracker_input.sample() is None
    assert tracker_input.keyboard.active == {"left": False, "right": False}


def test_inactive_frozen_tracker_does_not_stop_active_side(monkeypatch) -> None:
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": False, "right": True}
    tracker_input.last_position_changed_at["left"] -= 2.0
    tracker_input.last_rotation_changed_at["left"] -= 2.0
    fake_xrt.timestamp += 20_000_000
    fake_xrt.poses[0][0] += 0.01

    sample = tracker_input.sample()

    assert sample is not None
    assert sample.activations == {"left": False, "right": True}


def test_frozen_position_with_live_rotation_disengages(monkeypatch) -> None:
    # The failure recorded on 2026-07-25: the tracker lost its optical fix, so
    # position updated 37 times in 45 s while the IMU kept streaming rotation
    # on 84% of ticks. A combined alive-if-anything-moves clock never faulted
    # and the arms tracked a sub-hertz position stream for half a minute.
    # Position and rotation liveness must be judged independently.
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": True, "right": True}
    fake_xrt.timestamp += 20_000_000
    assert tracker_input.sample() is not None

    # Rotation keeps jittering, position never moves, and the position clock
    # has aged past the frozen timeout.
    tracker_input.last_position_changed_at["left"] -= 2.0
    fake_xrt.poses[1][3] += 0.01  # IMU wiggle on the left tracker's quaternion
    fake_xrt.timestamp += 20_000_000

    assert tracker_input.sample() is None
    assert tracker_input.keyboard.active == {"left": False, "right": False}


def test_live_position_with_frozen_rotation_disengages(monkeypatch) -> None:
    # The mirror failure: a dead IMU with a live optical fix must fault too.
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": True, "right": True}
    fake_xrt.timestamp += 20_000_000
    assert tracker_input.sample() is not None

    tracker_input.last_rotation_changed_at["left"] -= 2.0
    fake_xrt.poses[1][0] += 0.005  # position moves, under the jump limit
    fake_xrt.timestamp += 20_000_000

    assert tracker_input.sample() is None
    assert tracker_input.keyboard.active == {"left": False, "right": False}


def test_callback_sequence_is_fresh_when_payload_timestamp_freezes(
    monkeypatch,
) -> None:
    # Observed on 20260730_144406: MotionTimeStampNs froze even though the
    # operator confirmed that both trackers remained live. The local callback
    # sequence, not that unreliable payload field, is the feed-liveness clock.
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": True, "right": True}
    fake_xrt.timestamp += 20_000_000
    assert tracker_input.sample() is not None

    tracker_input.last_motion_update_at -= 1.0
    fake_xrt.motion_sequence += 1
    sample = tracker_input.sample()

    assert sample is not None
    state = tracker_input.debug_feed_state()
    assert state["seq"] == fake_xrt.motion_sequence
    assert state["age"] < 0.1
    tracker_input.close()


def test_frozen_pose_fault_runs_when_sdk_motion_timestamp_is_frozen(
    monkeypatch,
) -> None:
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": True, "right": True}
    fake_xrt.timestamp += 20_000_000
    assert tracker_input.sample() is not None

    # Only rotation changes. The unchanged SDK timestamp must not bypass the
    # independent position-freeze check.
    tracker_input.last_position_changed_at["left"] -= 2.0
    fake_xrt.poses[1][3] += 0.01
    fake_xrt.motion_sequence += 1

    assert tracker_input.sample() is None
    assert tracker_input.readiness["left"] == "left tracker position is frozen"
    assert tracker_input.keyboard.active == {"left": False, "right": False}
    tracker_input.close()


def test_invalid_atomic_snapshot_disengages(monkeypatch) -> None:
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    tracker_input.keyboard.active = {"left": True, "right": True}
    fake_xrt.timestamp += 20_000_000
    assert tracker_input.sample() is not None

    monkeypatch.setattr(
        fake_xrt, "get_motion_snapshot", lambda: (999, 0, [], [])
    )
    assert tracker_input.sample() is None
    assert tracker_input.keyboard.active == {"left": False, "right": False}
    tracker_input.close()


def test_debug_feed_state_reports_snapshot_health(monkeypatch) -> None:
    fake_xrt = _FakeXrt()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    tracker_input = _create_tracker_input()
    fake_xrt.timestamp += 20_000_000
    assert tracker_input.sample() is not None
    state = tracker_input.debug_feed_state()
    assert state["ok"] is True
    assert state["n"] == 2
    assert state["frame_ts"] == fake_xrt.timestamp
    assert state["ts"] == fake_xrt.timestamp
    assert state["age"] is not None and state["age"] < 1.0

    monkeypatch.setattr(
        fake_xrt, "get_motion_snapshot", lambda: (6, 0, [], [])
    )
    tracker_input.sample()
    failed_state = tracker_input.debug_feed_state()
    assert failed_state["ok"] is False
    assert failed_state["ts"] == 0
    assert failed_state["seq"] == 6
    tracker_input.close()


class _FakeHandXrt:
    """Serves static-but-valid 26x7 skeletons for both hands."""

    def __init__(self) -> None:
        self.closed = False
        base = np.zeros((26, 7))
        base[:, 6] = 1.0
        base[:, 0] = np.linspace(0.0, 0.25, 26)
        base[1, :3] = [0.1, 0.2, 0.3]
        self.joints = {"left": base.copy(), "right": base.copy()}
        self.joints["right"][:, 1] += 0.1
        self.active = {"left": 1, "right": 1}

    def init(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def get_left_hand_tracking_state(self):
        return self.joints["left"].copy()

    def get_left_hand_is_active(self) -> int:
        return self.active["left"]

    def get_right_hand_tracking_state(self):
        return self.joints["right"].copy()

    def get_right_hand_is_active(self) -> int:
        return self.active["right"]


def _create_hand_root_input(
    monkeypatch, fake_xrt, smoothing_time_constant: float = 0.1
) -> xr_input.HandRootInput:
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    return xr_input.HandRootInput(
        ready_timeout=2.0,
        stale_timeout=0.25,
        frozen_timeout=1.0,
        max_position_jump=0.2,
        max_rotation_jump=1.5,
        smoothing_time_constant=smoothing_time_constant,
    )


def test_pose_ema_seeds_then_blends() -> None:
    import pinocchio as pin

    ema = xr_input.PoseEma(time_constant=1.0)
    start = xr_input.Pose(np.zeros(3), np.eye(3))
    assert ema.update(start, None) is start

    target = xr_input.Pose(
        np.array([1.0, 0.0, 0.0]), pin.exp3(np.array([0.0, 0.0, 1.0]))
    )
    blended = ema.update(target, elapsed=1.0)
    alpha = 1.0 - np.exp(-1.0)
    np.testing.assert_allclose(blended.position, [alpha, 0.0, 0.0])
    np.testing.assert_allclose(
        pin.log3(blended.rotation), [0.0, 0.0, alpha], atol=1e-12
    )

    # A gap much longer than the time constant re-seeds onto the new pose.
    far = xr_input.Pose(np.array([5.0, 5.0, 5.0]), np.eye(3))
    caught_up = ema.update(far, elapsed=100.0)
    np.testing.assert_allclose(caught_up.position, far.position, atol=1e-6)


def test_pose_ema_adapts_rotation_without_changing_position_filter() -> None:
    import pinocchio as pin

    parameters = {
        "rotation_slow_time_constant": 0.30,
        "rotation_fast_time_constant": 0.075,
        "rotation_error_low": 0.015,
        "rotation_error_high": 0.080,
    }
    start = xr_input.Pose(np.zeros(3), np.eye(3))

    quiet = xr_input.PoseEma(time_constant=0.10, **parameters)
    quiet.update(start, None)
    small_target = xr_input.Pose(
        np.ones(3), pin.exp3(np.array([0.01, 0.0, 0.0]))
    )
    small = quiet.update(small_target, elapsed=0.02)
    slow_alpha = 1.0 - np.exp(-0.02 / 0.30)
    position_alpha = 1.0 - np.exp(-0.02 / 0.10)
    np.testing.assert_allclose(pin.log3(small.rotation), [0.01 * slow_alpha, 0, 0])
    np.testing.assert_allclose(small.position, np.full(3, position_alpha))

    moving = xr_input.PoseEma(time_constant=0.10, **parameters)
    moving.update(start, None)
    large_target = xr_input.Pose(
        np.ones(3), pin.exp3(np.array([0.10, 0.0, 0.0]))
    )
    large = moving.update(large_target, elapsed=0.02)
    fast_alpha = 1.0 - np.exp(-0.02 / 0.075)
    np.testing.assert_allclose(pin.log3(large.rotation), [0.10 * fast_alpha, 0, 0])
    np.testing.assert_allclose(large.position, np.full(3, position_alpha))


def test_hand_root_input_maps_wrist_to_world(monkeypatch) -> None:
    fake_xrt = _FakeHandXrt()
    hand_input_source = _create_hand_root_input(monkeypatch, fake_xrt)
    hand_input_source.keyboard.active = {"left": True, "right": True}

    sample = hand_input_source.sample()

    assert sample is not None
    # XR (0.1, 0.2, 0.3) maps to robot world (-z, -x, y) = (-0.3, -0.1, 0.2).
    np.testing.assert_allclose(sample.poses["left"].position, [-0.3, -0.1, 0.2])
    assert sample.activations == {"left": True, "right": True}
    hand_input_source.close()
    assert fake_xrt.closed


def test_hand_root_smoothing_lags_a_step(monkeypatch) -> None:
    fake_xrt = _FakeHandXrt()
    # Large time constant so the single-step blend is clearly partial no matter
    # how much wall time the test takes between samples.
    hand_input_source = _create_hand_root_input(
        monkeypatch, fake_xrt, smoothing_time_constant=10.0
    )
    hand_input_source.keyboard.active = {"left": True, "right": True}
    before = hand_input_source.sample()
    assert before is not None
    old_position = before.poses["left"].position

    fake_xrt.joints["left"][1, 0] += 0.05
    after = hand_input_source.sample()

    assert after is not None
    new_raw = xr_input.xr_pose_to_world(fake_xrt.joints["left"][1]).position
    smoothed = after.poses["left"].position
    assert np.linalg.norm(smoothed - old_position) < np.linalg.norm(
        smoothed - new_raw
    )


def test_hand_root_jump_disengages(monkeypatch) -> None:
    fake_xrt = _FakeHandXrt()
    hand_input_source = _create_hand_root_input(monkeypatch, fake_xrt)
    hand_input_source.keyboard.active = {"left": True, "right": True}
    assert hand_input_source.sample() is not None

    # The 0.906 m skeleton teleport measured in hand_coexistence.jsonl.
    fake_xrt.joints["left"][1, 0] += 0.9

    assert hand_input_source.sample() is None
    assert hand_input_source.keyboard.active == {"left": False, "right": False}


def test_hand_root_loss_on_engaged_side_disengages(monkeypatch) -> None:
    fake_xrt = _FakeHandXrt()
    hand_input_source = _create_hand_root_input(monkeypatch, fake_xrt)
    hand_input_source.keyboard.active = {"left": True, "right": True}
    assert hand_input_source.sample() is not None

    fake_xrt.active["right"] = 0

    assert hand_input_source.sample() is None
    assert hand_input_source.keyboard.active == {"left": False, "right": False}


def test_hand_root_loss_on_disengaged_side_is_ignored(monkeypatch) -> None:
    fake_xrt = _FakeHandXrt()
    hand_input_source = _create_hand_root_input(monkeypatch, fake_xrt)
    hand_input_source.keyboard.active = {"left": False, "right": True}

    fake_xrt.active["left"] = 0

    sample = hand_input_source.sample()
    assert sample is not None
    assert sample.activations == {"left": False, "right": True}
    # The lost side still carries its last smoothed pose for the mapper to
    # ignore.
    assert sample.poses["left"] is not None


def test_hand_root_frozen_skeleton_disengages(monkeypatch) -> None:
    fake_xrt = _FakeHandXrt()
    hand_input_source = _create_hand_root_input(monkeypatch, fake_xrt)
    hand_input_source.keyboard.active = {"left": True, "right": True}
    assert hand_input_source.sample() is not None

    hand_input_source.liveness["right"].changed_at -= 2.0

    assert hand_input_source.sample() is None
    assert hand_input_source.keyboard.active == {"left": False, "right": False}


def test_create_pico_input_refuses_next_to_the_desktop_gui(monkeypatch) -> None:
    # The GUI and the Python SDK compete for the PC Service feedback stream; a
    # recorded session with the GUI in use degraded tracker positions to
    # sub-hertz while the GUI showed them moving accurately. Teleoperation is
    # where degraded input moves hardware, so it must refuse to start.
    monkeypatch.setattr(xr_input, "desktop_gui_pids", lambda: [4242])

    class _Config:
        controllers = None
        motion_trackers = None

    try:
        xr_input.create_pico_input(_Config(), "motion-trackers")
    except RuntimeError as error:
        assert "RobotLinuxDemo" in str(error)
        assert "4242" in str(error)
    else:
        raise AssertionError("expected create_pico_input to refuse")


class _FakeControllerXrt:
    def __init__(self) -> None:
        self.timestamp = 100
        self.left_grip = 0.0
        self.right_grip = 0.0
        self.left_pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        self.right_pose = [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0]

    def init(self) -> None:
        pass

    def close(self) -> None:
        pass

    def get_time_stamp_ns(self) -> int:
        return self.timestamp

    def get_left_controller_pose(self):
        return list(self.left_pose)

    def get_right_controller_pose(self):
        return list(self.right_pose)

    def get_left_grip(self) -> float:
        return self.left_grip

    def get_right_grip(self) -> float:
        return self.right_grip


class _GuiOperator:
    def __init__(self) -> None:
        self.active = {"left": False, "right": False}
        self.disabled_reason = None

    def poll(self):
        return dict(self.active)

    def disable_all(self, reason):
        self.disabled_reason = reason
        self.active = {"left": False, "right": False}

    def deny(self, side, _reason):
        self.active[side] = False

    def show(self, _message):
        pass

    def close(self):
        pass


def _create_controller_input(fake_xrt, gui, *, use_grip: bool):
    return xr_input.ControllerInput(
        grip_threshold=0.9,
        ready_timeout=0.1,
        stale_timeout=0.25,
        use_grip=use_grip,
        keyboard=gui,
        xrt_client=fake_xrt,
    )


def test_controller_status_summary_reports_liveness_and_engage() -> None:
    fake_xrt = _FakeControllerXrt()
    gui = _GuiOperator()
    source = _create_controller_input(fake_xrt, gui, use_grip=False)
    gui.active = {"left": True, "right": False}

    sample = source.sample()

    assert sample is not None
    assert source.source_name == "controllers"
    assert source.status_summary() == (
        "left=live, control=ON | right=live, control=off"
    )
    source.close()


def test_controller_gui_engage_moves_without_grip(monkeypatch) -> None:
    fake_xrt = _FakeControllerXrt()
    gui = _GuiOperator()
    source = _create_controller_input(fake_xrt, gui, use_grip=False)
    gui.active = {"left": True, "right": False}

    sample = source.sample()

    assert sample is not None
    assert sample.activations == {"left": True, "right": False}
    source.close()


def test_controller_grip_deadman_requires_gui_and_grip(monkeypatch) -> None:
    fake_xrt = _FakeControllerXrt()
    gui = _GuiOperator()
    source = _create_controller_input(fake_xrt, gui, use_grip=True)
    gui.active = {"left": True, "right": True}

    sample = source.sample()
    assert sample is not None
    assert sample.activations == {"left": False, "right": False}

    fake_xrt.timestamp += 1
    fake_xrt.left_grip = 0.95
    fake_xrt.right_grip = 0.2
    sample = source.sample()
    assert sample.activations == {"left": True, "right": False}

    fake_xrt.timestamp += 1
    gui.active = {"left": False, "right": False}
    fake_xrt.left_grip = 1.0
    fake_xrt.right_grip = 1.0
    sample = source.sample()
    assert sample.activations == {"left": False, "right": False}
    source.close()
