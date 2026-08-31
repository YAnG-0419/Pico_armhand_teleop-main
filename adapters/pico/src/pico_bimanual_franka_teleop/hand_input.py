"""Acquire PICO optical hand skeletons from an already-open SDK client.

This reader deliberately does not call `xrt.init()`. The XRoboToolkit PC Service
has not been shown to serve multiple simultaneous Python clients safely, and the
existing controller and motion-tracker inputs already own a client. Whoever owns
that client passes the module in here, so arm poses and hand skeletons arrive
through one connection.

Verified on real hardware: optical hand tracking and Object Motion Tracking do
coexist in one client. Motion timestamp advance was statistically identical
whether hands were active or not, and motion never went stale while a hand was
live. Hand tracking runs at about 52 Hz against an XR frame rate of about 69 Hz,
which is the native optical rate rather than contention.

Liveness rules, all established from measured data rather than assumed:

`isActive` is load-bearing. Complete, entirely plausible pose arrays continue to
be served after tracking is lost. In a 45 s recording, fully valid poses appeared
in 434 of 440 samples while `isActive` was 1 in only 321. Liveness therefore
cannot be inferred from array contents, and `isActive != 1` is treated as no
data at all.

There is no per-hand timestamp in the binding, so freshness is detected by the
pose array changing. That is sound here because optical jitter means a hand held
deliberately still still changes: zero consecutive active samples were bitwise
identical across the recording. A genuinely frozen cache would appear as exactly
constant data and is reported as a fault.

Loss of a hand skeleton is independent of loss of a wrist tracker and must never
be allowed to disengage the arms, nor the reverse.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .hand_landmarks import (
    OPENXR_JOINT_COUNT,
    to_canonical_landmarks,
    validate_skeleton,
)
from .types import SIDES


@dataclass(frozen=True)
class HandSample:
    """One usable optical hand observation for a single side."""

    side: str
    landmarks: np.ndarray
    timestamp: float

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise ValueError(f"side must be left or right, got {self.side!r}")
        array = np.asarray(self.landmarks, dtype=float)
        if array.shape != (21, 3):
            raise ValueError(
                f"Hand sample requires canonical (21, 3) landmarks, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("Hand sample contains a non-finite value")
        object.__setattr__(self, "landmarks", array.copy())


class SkeletonLiveness:
    """Decide whether one side's skeleton is a usable, current measurement.

    Extracted so the in-process reader and the out-of-process retargeting service
    apply exactly the same rules. Two copies of this logic would be free to drift,
    and the failure mode would be a hand that keeps being commanded from data that
    is no longer live.

    The rules come from measurement, not assumption. `isActive` must be 1, because
    complete and entirely plausible pose arrays keep being served after tracking is
    lost: in a 45 s recording, fully valid poses appeared in 434 of 440 samples
    while `isActive` was 1 in only 321. Freshness is then judged by the array
    changing, since the binding carries no per-hand timestamp; that is sound because
    optical jitter means a hand held deliberately still still moves, with zero
    consecutive active samples bitwise identical across that recording, whereas a
    frozen cache is exactly constant.
    """

    def __init__(self, stale_timeout: float = 0.25, frozen_timeout: float = 1.0) -> None:
        if stale_timeout <= 0 or frozen_timeout <= 0:
            raise ValueError("Hand stale and frozen timeouts must be positive")
        self.stale_timeout = float(stale_timeout)
        self.frozen_timeout = float(frozen_timeout)
        self.previous: np.ndarray | None = None
        self.changed_at: float | None = None
        self.fault: str | None = "hand tracking not yet seen"

    def forget(self) -> None:
        self.previous = None
        self.changed_at = None

    def accept(self, joints, is_active: int, now: float):
        """Return canonical landmarks, or None with `fault` explaining why not."""
        array = np.asarray(joints, dtype=float)
        if array.shape != (OPENXR_JOINT_COUNT, 7):
            self.fault = (
                f"unexpected hand array shape {array.shape}; "
                f"expected {(OPENXR_JOINT_COUNT, 7)}"
            )
            self.forget()
            return None
        if int(is_active) != 1:
            self.fault = f"hand tracking inactive (isActive={int(is_active)})"
            self.forget()
            return None

        previous = self.previous
        self.previous = array
        if previous is None:
            # First active frame after a dropout. Start the freshness clock rather
            # than trusting a single sample.
            self.changed_at = now
            self.fault = "hand tracking reacquiring"
            return None
        if not np.array_equal(previous, array):
            self.changed_at = now

        if self.changed_at is None or now - self.changed_at > self.frozen_timeout:
            self.fault = "hand skeleton is frozen"
            return None
        if now - self.changed_at > self.stale_timeout:
            self.fault = "hand skeleton is stale"
            return None
        try:
            validate_skeleton(array)
            landmarks = to_canonical_landmarks(array)
        except ValueError as error:
            self.fault = str(error)
            return None
        self.fault = None
        return landmarks


class HandSkeletonReader:
    """Read and validate both hand skeletons from a shared SDK client."""

    def __init__(
        self,
        xrt,
        *,
        stale_timeout: float = 0.25,
        frozen_timeout: float = 1.0,
    ) -> None:
        if xrt is None:
            raise ValueError("HandSkeletonReader requires an initialized SDK module")
        self.xrt = xrt
        self.liveness = {
            side: SkeletonLiveness(stale_timeout, frozen_timeout) for side in SIDES
        }

    @property
    def faults(self) -> dict[str, str | None]:
        return {side: state.fault for side, state in self.liveness.items()}

    def _raw(self, side: str):
        if side == "left":
            return (
                self.xrt.get_left_hand_tracking_state(),
                int(self.xrt.get_left_hand_is_active()),
            )
        return (
            self.xrt.get_right_hand_tracking_state(),
            int(self.xrt.get_right_hand_is_active()),
        )

    def sample(self, now: float | None = None) -> dict[str, HandSample | None]:
        """Return a usable sample per side, or None where the hand is unusable.

        Each side is evaluated independently. `self.faults` carries a short reason
        for every side that returned None.
        """
        moment = time.monotonic() if now is None else float(now)
        result: dict[str, HandSample | None] = {}
        for side in SIDES:
            state = self.liveness[side]
            try:
                raw, is_active = self._raw(side)
            except Exception as error:  # noqa: BLE001 - SDK raises bare exceptions
                state.fault = f"hand SDK read failed: {error}"
                state.forget()
                result[side] = None
                continue
            landmarks = state.accept(raw, is_active, moment)
            result[side] = (
                None if landmarks is None else HandSample(side, landmarks, moment)
            )
        return result
