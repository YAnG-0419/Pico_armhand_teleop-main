import ctypes

import numpy as np
import pytest

from manus_teleop.skeleton import (
    CANONICAL_FROM_MANUS,
    KEYPOINT_COUNT,
    ManusFrame,
    ManusPose,
    canonical_landmarks,
)


def test_native_frame_layout_has_all_keypoints():
    assert KEYPOINT_COUNT == 25
    assert len(CANONICAL_FROM_MANUS) == 21
    assert ctypes.sizeof(ManusFrame) > KEYPOINT_COUNT * ctypes.sizeof(ManusPose)


def test_canonical_landmarks_select_expected_nodes():
    frame = ManusFrame()
    frame.keypoint_count = KEYPOINT_COUNT
    for index, point in enumerate(frame.keypoints):
        point.position_x = index
        point.position_y = index + 100
        point.position_z = index + 200

    result = canonical_landmarks(frame)

    assert result.shape == (21, 3)
    np.testing.assert_array_equal(result[:, 0], CANONICAL_FROM_MANUS)


def test_canonical_landmarks_reject_non_finite_input():
    frame = ManusFrame()
    frame.keypoints[0].position_x = float("nan")
    with pytest.raises(ValueError, match="invalid landmarks"):
        canonical_landmarks(frame)
