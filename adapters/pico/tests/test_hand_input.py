import numpy as np
import pytest

from hand_fixtures import synthetic_skeleton

from pico_bimanual_franka_teleop.hand_input import HandSample, HandSkeletonReader


class FakeXrt:
    """Stand-in for the vendored SDK module, with no client of its own."""

    def __init__(self) -> None:
        self.left = synthetic_skeleton(mirror=True)
        self.right = synthetic_skeleton(mirror=False)
        self.left_active = 1
        self.right_active = 1
        self.raise_on_read = False

    def get_left_hand_tracking_state(self):
        if self.raise_on_read:
            raise RuntimeError("SDK exploded")
        return self.left

    def get_right_hand_tracking_state(self):
        if self.raise_on_read:
            raise RuntimeError("SDK exploded")
        return self.right

    def get_left_hand_is_active(self):
        return self.left_active

    def get_right_hand_is_active(self):
        return self.right_active

    def jitter(self, amount: float = 1e-4) -> None:
        """Move both skeletons slightly, as optical tracking always does."""
        self.left = self.left + np.array([amount, 0, 0, 0, 0, 0, 0])
        self.right = self.right + np.array([amount, 0, 0, 0, 0, 0, 0])


def advance(reader, xrt, start, steps, dt=0.02, jitter=True):
    """Poll repeatedly, returning the final per-side result."""
    result = {}
    for step in range(steps):
        if jitter:
            xrt.jitter()
        result = reader.sample(now=start + step * dt)
    return result


def test_reader_requires_a_module():
    with pytest.raises(ValueError):
        HandSkeletonReader(None)


def test_first_active_frame_is_withheld_then_becomes_usable():
    xrt = FakeXrt()
    reader = HandSkeletonReader(xrt)
    # A single sample cannot establish freshness, so the first is withheld.
    first = reader.sample(now=100.0)
    assert first["left"] is None and first["right"] is None
    assert "reacquiring" in reader.faults["left"]

    result = advance(reader, xrt, 100.02, 3)
    assert isinstance(result["left"], HandSample)
    assert isinstance(result["right"], HandSample)
    assert reader.faults["left"] is None
    assert result["left"].landmarks.shape == (21, 3)


def test_inactive_hand_is_rejected_even_with_a_plausible_pose():
    # Verified on hardware: complete, plausible pose arrays keep being served
    # after tracking is lost. isActive is the only trustworthy signal.
    xrt = FakeXrt()
    reader = HandSkeletonReader(xrt)
    advance(reader, xrt, 100.0, 4)

    xrt.left_active = 0
    result = reader.sample(now=100.10)
    assert result["left"] is None
    assert "inactive" in reader.faults["left"]
    # The other hand is unaffected: the sides are independent.
    assert isinstance(result["right"], HandSample)


def test_reacquisition_after_inactivity_withholds_one_frame():
    xrt = FakeXrt()
    reader = HandSkeletonReader(xrt)
    advance(reader, xrt, 100.0, 4)
    xrt.left_active = 0
    reader.sample(now=100.10)
    xrt.left_active = 1
    # Freshness history was discarded, so the first frame back is withheld.
    assert reader.sample(now=100.12)["left"] is None
    assert isinstance(advance(reader, xrt, 100.14, 3)["left"], HandSample)


def test_frozen_skeleton_is_reported_as_a_fault():
    # A hand held still still jitters, so bitwise-constant data means the
    # binding is serving a cached frame.
    xrt = FakeXrt()
    reader = HandSkeletonReader(xrt, frozen_timeout=0.5)
    advance(reader, xrt, 100.0, 4)
    result = advance(reader, xrt, 101.0, 40, dt=0.02, jitter=False)
    assert result["left"] is None
    assert "frozen" in reader.faults["left"]


def test_stale_skeleton_is_reported_before_frozen():
    xrt = FakeXrt()
    reader = HandSkeletonReader(xrt, stale_timeout=0.05, frozen_timeout=10.0)
    advance(reader, xrt, 100.0, 4)
    result = advance(reader, xrt, 101.0, 6, dt=0.02, jitter=False)
    assert result["left"] is None
    assert "stale" in reader.faults["left"]


def test_sdk_read_failure_is_contained():
    xrt = FakeXrt()
    reader = HandSkeletonReader(xrt)
    advance(reader, xrt, 100.0, 4)
    xrt.raise_on_read = True
    result = reader.sample(now=100.10)
    assert result["left"] is None and result["right"] is None
    assert "SDK read failed" in reader.faults["left"]


def test_wrong_shape_is_rejected():
    xrt = FakeXrt()
    xrt.left = np.zeros((26, 3))
    reader = HandSkeletonReader(xrt)
    reader.sample(now=100.0)
    result = reader.sample(now=100.02)
    assert result["left"] is None
    assert "shape" in reader.faults["left"]


def test_hand_sample_validates_its_own_contents():
    good = np.zeros((21, 3))
    HandSample("left", good, 1.0)
    with pytest.raises(ValueError):
        HandSample("middle", good, 1.0)
    with pytest.raises(ValueError):
        HandSample("left", np.zeros((26, 7)), 1.0)
    bad = good.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        HandSample("left", bad, 1.0)
