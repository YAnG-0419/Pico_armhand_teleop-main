import json

import numpy as np

from pico_bimanual_franka_teleop.debug_log import (
    FollowDebugLogger,
    HandRetargetDebugLogger,
)
from pico_bimanual_franka_teleop.types import Pose


def test_follow_debug_log_records_raw_and_both_fk_streams(tmp_path):
    path = tmp_path / "follow.jsonl"
    identity = Pose(np.array([1.0, 2.0, 3.0]), np.eye(3))
    logger = FollowDebugLogger(path, flush_every=1)
    logger.record(
        12.5,
        np.arange(14),
        np.arange(14) + 0.5,
        {"left": identity},
        {"left": True},
        {"left": identity},
        {"left": identity},
        raw_tracker_poses={"left": identity},
        measured_ee_poses={"left": identity},
        ik_diagnostics={
            "left": {
                "position_error": 0.0123456789,
                "orientation_error": 0.05,
                "saturated_joints": (2, 4),
                "limit_joints": ((4, 0.012345),),
            }
        },
    )
    logger.close()

    header, row = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert header["schema"] == "follow-debug.v6"
    assert row["left"]["raw_tracker"]["p"] == [1.0, 2.0, 3.0]
    assert row["left"]["tracker"]["p"] == [1.0, 2.0, 3.0]
    assert row["left"]["ee_cmd"]["p"] == [1.0, 2.0, 3.0]
    # The v1 compatibility duplicate is gone; readers use ee_cmd.
    assert "ee" not in row["left"]
    assert row["left"]["ee_meas"]["p"] == [1.0, 2.0, 3.0]
    assert row["right"]["raw_tracker"] is None
    assert row["left"]["ik"] == {
        "ep": 0.012346,
        "eo": 0.05,
        "sat": [2, 4],
        "lim": [[4, 0.01235]],
    }
    assert "ik" not in row["right"]


def test_hand_debug_log_records_landmarks_commands_and_metrics(tmp_path):
    path = tmp_path / "hands.jsonl"
    logger = HandRetargetDebugLogger(
        path, flush_every=1, metadata={"source": "manus"}
    )
    logger.record(
        4.2,
        "right",
        np.zeros((21, 3)),
        np.arange(21) / 10,
        {"success": True, "iterations": 3, "thumb_tip_error": 0.012345678},
        joint_names=[f"joint_{index}" for index in range(21)],
        raw_qpos=np.arange(21) / 9,
        source={"sequence": 7, "timestamp_ns": 123},
        transport={"stream_id": "manus-right-o30i", "sequence": 4, "sent": True},
    )
    logger.close()

    header, row = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert header["schema"] == "hand-retarget-debug.v2"
    assert header["metadata"]["source"] == "manus"
    assert row["side"] == "right"
    assert isinstance(row["wall_time_ns"], int)
    assert row["source"]["sequence"] == 7
    assert row["transport"]["sequence"] == 4
    assert np.asarray(row["landmarks"]).shape == (21, 3)
    assert len(row["qpos"]) == 21
    assert len(row["raw_qpos"]) == 21
    assert len(row["joint_names"]) == 21
    assert row["stats"]["success"] is True
    assert row["stats"]["iterations"] == 3
    assert row["stats"]["thumb_tip_error"] == 0.0123457
