"""ROS-free tests for the command safety gate.

Run with:
    PYTHONPATH=ros_ws/src/teleop_core python3 -m pytest -q \
        ros_ws/src/teleop_core/test/test_safety.py
"""

import numpy as np
import pytest

from teleop_core.contract import command_names
from teleop_core.joint_state import ordered_external_torques
from teleop_core.safety import CommandSafetyGate


# A valid FR3 ready pose (joint 4 only accepts [-3.077, -0.117] rad).
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785] * 2)
THRESHOLDS = [6.0, 6.0, 6.0, 6.0, 3.0, 3.0, 3.0]


def make_gate(**overrides):
    settings = dict(
        max_joint_speed=0.5,
        max_initial_delta=0.05,
        nominal_dt=0.01,
        contact_torque_thresholds=THRESHOLDS,
    )
    settings.update(overrides)
    return CommandSafetyGate(**settings)


def send(gate, target, measured, now, sides=("left",), torques=None):
    names = command_names(sides)
    positions = [target[i] for i in range(len(names))]
    return gate.validate(
        sides, names, positions, measured, now, external_torques=torques
    )


def test_slew_limits_each_step():
    gate = make_gate()
    first = send(gate, HOME[:7], HOME, now=0.0)
    assert np.allclose(first.positions, HOME[:7])
    target = HOME[:7] + 0.3
    second = send(gate, target, HOME, now=0.01)
    assert np.allclose(second.positions, HOME[:7] + 0.5 * 0.01)


def test_no_torque_data_leaves_commands_ungated():
    # Fail open: a missing estimate must not freeze teleoperation.
    gate = make_gate()
    target = HOME[:7] + 0.3
    send(gate, HOME[:7], HOME, now=0.0)
    now, out = 0.0, None
    for _ in range(100):
        now += 0.01
        out = send(gate, target, HOME, now=now, torques={"left": None})
    assert np.allclose(out.positions, target)


def test_pressing_against_external_torque_is_held():
    gate = make_gate()
    target = HOME[:7] + 0.3
    send(gate, HOME[:7], HOME, now=0.0)
    # The environment resists joint 1's positive advance. Measured hardware
    # convention (2026-07-29 probe): the broadcaster reports that resisting
    # torque with the sign of the robot's push, here positive - so a
    # positive step into a positive reported torque is pressing. Joint 2 is
    # loaded below threshold and advances.
    torques = {"left": np.array([10.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    now, out = 0.0, None
    for _ in range(100):
        now += 0.01
        out = send(gate, target, HOME, now=now, torques=torques)
    deviation = np.asarray(out.positions) - HOME[:7]
    assert deviation[0] == 0.0
    assert np.allclose(deviation[1:], 0.3)
    assert gate.pressing_joints["left"] == (1,)
    send(gate, target, HOME, now=now + 0.01, torques={"left": None})
    assert gate.pressing_joints["left"] == ()


def test_unloading_direction_always_passes():
    gate = make_gate()
    target = HOME[:7].copy()
    target[0] -= 0.2
    send(gate, HOME[:7], HOME, now=0.0)
    # Same reported torque on joint 1, but the command retreats negative -
    # unloading under the measured convention. Never held.
    torques = {"left": np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    now, out = 0.0, None
    for _ in range(100):
        now += 0.01
        out = send(gate, target, HOME, now=now, torques=torques)
    assert np.asarray(out.positions)[0] == pytest.approx(target[0])


def test_gating_releases_when_the_load_disappears():
    gate = make_gate()
    target = HOME[:7] + 0.1
    send(gate, HOME[:7], HOME, now=0.0)
    torques = {"left": np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    now = 0.0
    for _ in range(50):
        now += 0.01
        held = send(gate, target, HOME, now=now, torques=torques)
    assert np.asarray(held.positions)[0] == 0.0
    out = None
    for _ in range(100):
        now += 0.01
        out = send(gate, target, HOME, now=now, torques={"left": None})
    assert np.allclose(out.positions, target)


def test_rejects_invalid_thresholds():
    for bad in ([6.0, 6.0], [6.0] * 6 + [-1.0], [0.0] * 7):
        with pytest.raises(ValueError):
            make_gate(contact_torque_thresholds=bad)


def test_ordered_external_torques_by_name_and_position():
    named = ordered_external_torques(
        [f"fr3_joint{i}" for i in (3, 1, 2, 4, 5, 6, 7)],
        [3.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0],
    )
    assert named.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    positional = ordered_external_torques([], [1, 2, 3, 4, 5, 6, 7])
    assert positional.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    with pytest.raises(ValueError):
        ordered_external_torques(["fr3_joint1"], [1.0])
    with pytest.raises(ValueError):
        ordered_external_torques([], [1.0, 2.0])


def test_one_side_fault_does_not_block_the_other():
    # Regression 2026-07-29: a right re-engage rejected by the initial-delta
    # check silenced the whole message, freezing the healthy left arm too.
    gate = make_gate()
    names = command_names(("left", "right"))
    left_target = HOME[:7]
    far_right = HOME[7:] + 0.3
    out = gate.validate(
        ("left", "right"),
        names,
        list(left_target) + list(far_right),
        HOME,
        now=0.0,
    )
    assert out is not None
    assert out.names == command_names(("left",))
    assert gate.side_faults and "right" in gate.side_faults[0]
    assert gate.active["left"] and not gate.active["right"]
    # Once the command comes back within range, the right side re-engages.
    out = gate.validate(
        ("left", "right"),
        names,
        list(left_target) + list(HOME[7:]),
        HOME,
        now=0.01,
    )
    assert out.names == command_names(("left", "right"))
    assert not gate.side_faults
