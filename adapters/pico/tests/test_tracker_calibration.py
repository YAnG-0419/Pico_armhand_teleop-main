from pathlib import Path

import pytest

from pico_bimanual_franka_teleop.config import load_config
from pico_bimanual_franka_teleop.tracker_calibration import (
    CalibrationError,
    assign_sides,
    replace_device_env_serials,
    replace_serials,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PICO_YAML = REPO_ROOT / "config" / "modes" / "pico.yaml"
DEVICES_ENV = REPO_ROOT / "config" / "devices.env"

SERIAL_A = "PC0000000000000A"
SERIAL_B = "PC0000000000000B"


def test_assign_sides_from_clean_phases():
    mapping = assign_sides(
        {SERIAL_A: 0.42, SERIAL_B: 0.01},
        {SERIAL_A: 0.02, SERIAL_B: 0.38},
    )
    assert mapping == {"left": SERIAL_A, "right": SERIAL_B}


def test_assign_sides_rejects_too_little_motion():
    with pytest.raises(CalibrationError, match="nothing moved"):
        assign_sides(
            {SERIAL_A: 0.05, SERIAL_B: 0.01},
            {SERIAL_A: 0.02, SERIAL_B: 0.38},
        )


def test_assign_sides_rejects_both_hands_moving():
    with pytest.raises(CalibrationError, match="ambiguous"):
        assign_sides(
            {SERIAL_A: 0.42, SERIAL_B: 0.30},
            {SERIAL_A: 0.02, SERIAL_B: 0.38},
        )


def test_assign_sides_rejects_same_winner_twice():
    with pytest.raises(CalibrationError, match="same hand"):
        assign_sides(
            {SERIAL_A: 0.42, SERIAL_B: 0.01},
            {SERIAL_A: 0.40, SERIAL_B: 0.02},
        )


def test_assign_sides_needs_two_trackers():
    with pytest.raises(CalibrationError, match="two are required"):
        assign_sides({SERIAL_A: 0.42}, {SERIAL_A: 0.02})


def test_replace_serials_touches_only_the_serials_block(tmp_path):
    # The real config: rewriting it must keep every other line (comments,
    # the tracker_to_control left/right keys) untouched and stay loadable
    # by the strict parser.
    original = PICO_YAML.read_text(encoding="utf-8")
    updated = replace_serials(
        original, {"left": SERIAL_A, "right": SERIAL_B}
    )
    assert f"left: {SERIAL_A}\n" in updated
    assert f"right: {SERIAL_B}\n" in updated
    changed = [
        (old, new)
        for old, new in zip(original.splitlines(), updated.splitlines())
        if old != new
    ]
    assert len(changed) == 2
    target = tmp_path / "pico.yaml"
    target.write_text(updated, encoding="utf-8")
    config = load_config(target)
    assert config.input.motion_trackers.serials == {
        "left": SERIAL_A,
        "right": SERIAL_B,
    }


def test_replace_serials_rejects_unrecognized_layout():
    with pytest.raises(CalibrationError, match="serials"):
        replace_serials("input:\n  foo: 1\n", {"left": "A", "right": "B"})


def test_environment_can_override_tracker_identity(monkeypatch):
    monkeypatch.setenv("PICO_LEFT_TRACKER_SERIAL", SERIAL_A)
    monkeypatch.setenv("PICO_RIGHT_TRACKER_SERIAL", SERIAL_B)

    config = load_config(PICO_YAML)

    assert config.input.controllers.use_grip is False
    assert config.input.motion_trackers.serials == {
        "left": SERIAL_A,
        "right": SERIAL_B,
    }


def test_replace_device_defaults_preserves_environment_override_syntax():
    updated = replace_device_env_serials(
        DEVICES_ENV.read_text(encoding="utf-8"),
        {"left": SERIAL_A, "right": SERIAL_B},
    )

    assert f'PICO_LEFT_TRACKER_SERIAL="${{PICO_LEFT_TRACKER_SERIAL:-{SERIAL_A}}}"' in updated
    assert f'PICO_RIGHT_TRACKER_SERIAL="${{PICO_RIGHT_TRACKER_SERIAL:-{SERIAL_B}}}"' in updated
