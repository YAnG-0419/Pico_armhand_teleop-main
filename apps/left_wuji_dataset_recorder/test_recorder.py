import json

import numpy as np

from pico_bimanual_franka_teleop.types import Pose

from .recorder import LeftWujiDatasetRecorder


def test_recorder_writes_aligned_npz_and_recovery_jsonl(tmp_path):
    output = tmp_path / "episode.npz"
    recorder = LeftWujiDatasetRecorder(
        output,
        hand_joint_names=[f"left_hand_joint_{index}" for index in range(20)],
        control_rate_hz=100.0,
    )
    recorder.record(
        wall_time_ns=123,
        monotonic_time_s=10.1,
        arm_q=np.arange(7, dtype=float),
        hand_feedback=(tuple(np.arange(20, dtype=float)), 10.0),
        ee_pose=Pose(np.array([1.0, 2.0, 3.0]), np.eye(3)),
        arm_active=True,
        hand_active=False,
    )
    recorder.close()

    with np.load(output) as data:
        assert data["arm_q"].shape == (1, 7)
        assert data["hand_q"].shape == (1, 20)
        assert data["ee_position"].tolist() == [[1.0, 2.0, 3.0]]
        assert data["ee_quaternion_xyzw"].tolist() == [[0.0, 0.0, 0.0, 1.0]]
        assert data["hand_valid"].tolist() == [True]
        assert data["arm_active"].tolist() == [True]
        assert data["hand_active"].tolist() == [False]

    raw = output.with_suffix(".raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(raw[0])["schema"] == "left-fr3-wuji-hand2.v1"
    assert json.loads(raw[1])["wall_time_ns"] == 123


def test_stale_or_missing_hand_feedback_is_explicit(tmp_path):
    recorder = LeftWujiDatasetRecorder(
        tmp_path / "episode.npz",
        hand_joint_names=[str(index) for index in range(20)],
        control_rate_hz=100.0,
        hand_stale_timeout_s=0.5,
    )
    identity = Pose(np.zeros(3), np.eye(3))
    for wall_time_ns, feedback in (
        (1, None),
        (2, (tuple(np.zeros(20)), 2.0)),
    ):
        recorder.record(
            wall_time_ns=wall_time_ns,
            monotonic_time_s=float(wall_time_ns + 1),
            arm_q=np.zeros(7),
            hand_feedback=feedback,
            ee_pose=identity,
            arm_active=False,
            hand_active=feedback is not None,
        )
    recorder.close()

    with np.load(tmp_path / "episode.npz") as data:
        assert data["hand_valid"].tolist() == [False, False]
        assert np.isnan(data["hand_q"][0]).all()
        assert np.isfinite(data["hand_q"][1]).all()
