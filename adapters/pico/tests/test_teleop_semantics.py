"""Lock the teleoperation semantics the operator relies on.

On engage, the mapper anchors the current hand pose and the current end-effector
pose; thereafter deltas map one to one: rotate the hand 90 degrees about a world
axis and the commanded end effector rotates 90 degrees about that same world
axis. And the IK that executes those targets must neither jump configurations
nor let the elbow drift: with a 7-DoF arm every end-effector pose has a
one-parameter family of elbow configurations, and without an attractor the
configuration random-walked 0.88 rad in a single closed end-effector loop.
"""

import numpy as np
import pinocchio as pin
import pytest

from pico_bimanual_franka_teleop.hardware import (
    DualFr3HardwareTeleop,
    disengage_sample_sides,
    reseed_inactive_joints,
)
from pico_bimanual_franka_teleop.ik import BimanualPinkIK, classify_step
from pico_bimanual_franka_teleop.pose_mapping import RelativePoseMapper
from pico_bimanual_franka_teleop.types import Pose, TeleopSample

# A captured hardware home (2026-07-25). The live initial_pose.yaml may be
# newer; these tests only need a physically plausible dual-arm posture.
HOME_Q = np.array([
    -0.110970355570, -0.493441432714, 0.114729285240, -2.441572666170,
    0.119077377021, 2.436770200730, 0.340127378702,
    0.917802035809, -0.843023777008, -1.136612057686, -2.587259292603,
    0.054329961538, 3.444272756577, -1.538317084312,
])


def make_ik() -> BimanualPinkIK:
    ik = BimanualPinkIK(dt=0.01, max_joint_speed=0.5)
    ik.set_posture_reference(HOME_Q)
    return ik


def engage(mapper, hand, robot):
    assert mapper.update(hand, True, robot) is not None


def test_inactive_and_newly_engaging_sides_reseed_independently():
    held = np.arange(14, dtype=float)
    measured = held + 100.0

    # Left is already tracking and must retain its command. Right is newly
    # engaging, so its anchor must use measured hardware even though left
    # remains active.
    result = reseed_inactive_joints(
        held,
        measured,
        {"left": True, "right": True},
        {"left": True, "right": False},
    )
    np.testing.assert_array_equal(result[:7], held[:7])
    np.testing.assert_array_equal(result[7:], measured[7:])

    # An explicitly inactive side is continuously healed from measured state.
    result = reseed_inactive_joints(
        held,
        measured,
        {"left": False, "right": True},
        {"left": True, "right": True},
    )
    np.testing.assert_array_equal(result[:7], measured[:7])
    np.testing.assert_array_equal(result[7:], held[7:])


def test_open_hand_disengages_only_selected_side_before_opening():
    denied = []
    opened = []
    teleop = object.__new__(DualFr3HardwareTeleop)
    teleop.operator = type(
        "Operator",
        (),
        {"deny": lambda _self, side, reason: denied.append((side, reason))},
    )()
    teleop.hands = type(
        "Hands",
        (),
        {"request_open": lambda _self, *, sides: opened.append(sides)},
    )()
    teleop._notify = lambda _message: None

    teleop._open_hands(("right",))

    assert denied == [("right", "opening hand")]
    assert opened == [("right",)]


def test_home_arm_does_not_also_open_the_hand():
    opened = []
    teleop = object.__new__(DualFr3HardwareTeleop)
    teleop.operator = type(
        "Operator",
        (),
        {"disable_all": lambda _self, _reason: None},
    )()
    teleop.mappers = {
        side: type("Mapper", (), {"reset": lambda _self: None})()
        for side in ("left", "right")
    }
    teleop.hands = type(
        "Hands",
        (),
        {"request_open": lambda _self, **_kwargs: opened.append(True)},
    )()
    teleop.reset_invoker = lambda side: (True, side or "both")
    teleop.reset_thread = None
    teleop.reset_outcome = []
    teleop._notify = lambda _message: None

    teleop._start_reset("left")
    teleop.reset_thread.join(timeout=1.0)

    assert opened == []


def test_capture_home_disengages_and_invokes_capture_command():
    disabled = []
    mapper_resets = []
    messages = []
    teleop = object.__new__(DualFr3HardwareTeleop)
    teleop.operator = type(
        "Operator",
        (),
        {"disable_all": lambda _self, reason: disabled.append(reason)},
    )()
    teleop.mappers = {
        side: type(
            "Mapper",
            (),
            {"reset": lambda _self, selected=side: mapper_resets.append(selected)},
        )()
        for side in ("left", "right")
    }
    teleop.capture_home_invoker = lambda: (True, "saved")
    teleop.reset_thread = None
    teleop.reset_outcome = []
    teleop.reset_operation = "reset"
    teleop.hold_q = np.ones(14)
    teleop._notify = messages.append

    teleop._start_capture_home()
    teleop.reset_thread.join(timeout=1.0)
    teleop._service_reset()

    assert disabled == ["capturing current arm pose as home"]
    assert mapper_resets == ["left", "right"]
    assert teleop.hold_q is None
    assert messages[-1] == "home capture done: saved"


def test_open_hand_removes_stale_same_tick_activation():
    sample = TeleopSample(
        poses={
            "left": Pose(np.zeros(3), np.eye(3)),
            "right": Pose(np.ones(3), np.eye(3)),
        },
        activations={"left": True, "right": True},
        timestamp=1.0,
    )

    updated = disengage_sample_sides(sample, ("right",))

    assert updated.activations == {"left": True, "right": False}
    assert sample.activations == {"left": True, "right": True}


def test_translation_maps_one_to_one():
    mapper = RelativePoseMapper(translation_scale=1.0, rotation_scale=1.0)
    hand = Pose(np.array([0.1, 0.2, 1.0]), np.eye(3))
    robot = Pose(np.array([0.4, 0.5, 0.4]), np.eye(3))
    engage(mapper, hand, robot)
    moved = Pose(hand.position + np.array([0.07, -0.03, 0.11]), hand.rotation)
    target = mapper.update(moved, True, robot)
    np.testing.assert_allclose(
        target.position - robot.position, [0.07, -0.03, 0.11], atol=1e-12
    )


def test_rotation_maps_to_the_same_world_axis():
    # Rotate the hand 90 degrees about a world axis: the commanded end effector
    # rotates 90 degrees about that same world axis, regardless of either
    # frame's initial orientation.
    mapper = RelativePoseMapper(translation_scale=1.0, rotation_scale=1.0)
    hand = Pose(np.zeros(3), pin.exp3(np.array([0.2, 0.7, -0.4])))
    robot = Pose(np.array([0.4, 0.5, 0.4]), pin.exp3(np.array([-1.0, 0.3, 0.5])))
    engage(mapper, hand, robot)
    for world_axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])):
        rotation_vector = (np.pi / 2) * world_axis
        rotated = Pose(hand.position, pin.exp3(rotation_vector) @ hand.rotation)
        target = mapper.update(rotated, True, robot)
        delta = target.rotation @ robot.rotation.T
        np.testing.assert_allclose(pin.log3(delta), rotation_vector, atol=1e-9)


def test_reengage_reanchors_at_the_current_robot_pose():
    # Disengage, move the hand freely, re-engage: no jump. The new anchor is
    # wherever the robot currently is.
    mapper = RelativePoseMapper(translation_scale=1.0, rotation_scale=1.0)
    hand = Pose(np.zeros(3), np.eye(3))
    robot = Pose(np.array([0.4, 0.5, 0.4]), np.eye(3))
    engage(mapper, hand, robot)
    assert mapper.update(hand, False, robot) is None
    far_hand = Pose(np.array([5.0, 5.0, 5.0]), pin.exp3(np.array([0, 0, 2.0])))
    target = mapper.update(far_hand, True, robot)
    np.testing.assert_allclose(target.position, robot.position, atol=1e-12)
    np.testing.assert_allclose(target.rotation, robot.rotation, atol=1e-12)


def test_ik_reaches_static_targets_exactly():
    ik = make_ik()
    home = {side: ik.frame_pose(HOME_Q, side) for side in ("left", "right")}
    q = HOME_Q.copy()
    targets = {
        side: Pose(home[side].position + np.array([0.2, 0.0, 0.1]), home[side].rotation)
        for side in ("left", "right")
    }
    for _ in range(600):
        q = ik.step(q, targets)
    for side in ("left", "right"):
        error = np.linalg.norm(ik.frame_pose(q, side).position - targets[side].position)
        assert error < 0.005, f"{side}: {error * 1e3:.2f} mm"


def test_closed_ee_loops_do_not_drift_the_configuration():
    # The regression this locks: without the posture attractor, one loop drifted
    # the configuration by 0.88 rad while the end effector returned exactly.
    ik = make_ik()
    home = {side: ik.frame_pose(HOME_Q, side) for side in ("left", "right")}
    q = HOME_Q.copy()
    waypoints = (
        (np.array([0.15, 0.0, 0.0]), 0.0),
        (np.array([0.15, 0.0, 0.15]), np.radians(45)),
        (np.array([0.0, 0.0, 0.15]), 0.0),
        (np.zeros(3), 0.0),
    )
    for _ in range(3):
        for offset, angle in waypoints:
            rotation = pin.exp3(np.array([0.0, angle, 0.0]))
            targets = {
                side: Pose(home[side].position + offset, rotation @ home[side].rotation)
                for side in ("left", "right")
            }
            for _ in range(150):
                q = ik.step(q, targets)
    for _ in range(600):
        q = ik.step(q, home)
    drift = np.abs(q - HOME_Q).max()
    assert drift < 0.05, f"null-space drift {drift:.3f} rad"
    for side in ("left", "right"):
        error = np.linalg.norm(ik.frame_pose(q, side).position - home[side].position)
        assert error < 0.005


def test_configuration_evolves_continuously():
    # Differential IK integrates velocity under a hard clamp, so consecutive
    # configurations can never jump by more than max_joint_speed * dt.
    ik = make_ik()
    home = {side: ik.frame_pose(HOME_Q, side) for side in ("left", "right")}
    q = HOME_Q.copy()
    targets = {
        side: Pose(home[side].position + np.array([0.3, -0.2, 0.2]), home[side].rotation)
        for side in ("left", "right")
    }
    bound = 0.5 * 0.01 + 1e-9
    for _ in range(400):
        q_next = ik.step(q, targets)
        assert np.abs(q_next - q).max() <= bound
        q = q_next


def test_ik_diagnostics_report_ok_for_a_reached_target():
    ik = make_ik()
    home = {side: ik.frame_pose(HOME_Q, side) for side in ("left", "right")}
    q = HOME_Q.copy()
    targets = {
        side: Pose(home[side].position + np.array([0.05, 0.0, 0.05]), home[side].rotation)
        for side in ("left", "right")
    }
    for _ in range(600):
        q = ik.step(q, targets)
    for side in ("left", "right"):
        diagnostics = ik.last_diagnostics[side]
        assert classify_step(diagnostics) == "ok"
        assert diagnostics["position_error"] < 0.005
        assert not diagnostics["saturated_joints"]
        assert not diagnostics["limit_joints"]


def test_ik_diagnostics_flag_the_speed_clamp_while_catching_up():
    # A large instantaneous target step cannot be covered in one clamped tick;
    # the deficit is transient and must be attributed to the speed clamp.
    ik = make_ik()
    home = {side: ik.frame_pose(HOME_Q, side) for side in ("left", "right")}
    targets = {
        side: Pose(home[side].position + np.array([0.3, 0.0, 0.0]), home[side].rotation)
        for side in ("left", "right")
    }
    ik.step(HOME_Q.copy(), targets)
    for side in ("left", "right"):
        diagnostics = ik.last_diagnostics[side]
        assert diagnostics["position_error"] > 0.2
        assert diagnostics["saturated_joints"]
        assert classify_step(diagnostics) == "speed-clamp"


def test_ik_diagnostics_attribute_an_out_of_reach_target():
    # A target far outside the workspace: the configuration settles and the
    # residual must be attributed to a pinned joint or the workspace edge,
    # never reported as ok or still-catching-up.
    ik = make_ik()
    home = {side: ik.frame_pose(HOME_Q, side) for side in ("left", "right")}
    q = HOME_Q.copy()
    targets = {
        side: Pose(
            home[side].position + np.array([1.5, 0.0, 0.0]), home[side].rotation
        )
        for side in ("left", "right")
    }
    for _ in range(1200):
        q = ik.step(q, targets)
    for side in ("left", "right"):
        diagnostics = ik.last_diagnostics[side]
        assert diagnostics["position_error"] > 0.1
        assert classify_step(diagnostics) in ("joint-limit", "workspace")


def test_ik_diagnostics_clear_without_targets():
    ik = make_ik()
    ik.step(HOME_Q.copy(), {})
    assert ik.last_diagnostics == {}


def test_fk_pose_snapshots_do_not_change_after_a_later_fk_update():
    ik = make_ik()
    before = ik.frame_poses(HOME_Q)
    saved = {
        side: (pose.position.copy(), pose.rotation.copy())
        for side, pose in before.items()
    }
    moved = HOME_Q.copy()
    moved[0] += 0.1
    moved[7] -= 0.1
    ik.frame_poses(moved)
    for side, pose in before.items():
        np.testing.assert_array_equal(pose.position, saved[side][0])
        np.testing.assert_array_equal(pose.rotation, saved[side][1])


def test_uncommanded_side_never_moves():
    # Regression: with only one side engaged, the posture attractor used to
    # walk the other side's joints toward the reference at the speed clamp
    # (0.35 rad of phantom drift in 0.75 s, measured 2026-07-29), while the
    # real, uncommanded arm stayed put - deadlocking its re-engage against
    # the gateway's initial-delta check.
    ik = make_ik()
    home = {side: ik.frame_pose(HOME_Q, side) for side in ("left", "right")}
    q = HOME_Q.copy()
    # Drive both arms away from the posture reference first.
    targets = {
        side: Pose(home[side].position + np.array([0.2, 0.0, 0.1]), home[side].rotation)
        for side in ("left", "right")
    }
    for _ in range(400):
        q = ik.step(q, targets)
    # Now command only the left side for a while.
    left_only = {"left": targets["left"]}
    right_before = q[7:].copy()
    for _ in range(200):
        q = ik.step(q, left_only)
    np.testing.assert_array_equal(q[7:], right_before)
