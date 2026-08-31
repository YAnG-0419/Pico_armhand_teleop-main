"""PICO OpenXR 26-joint hand skeleton to canonical hand landmarks.

The L20 retargeter consumes a canonical 21-landmark array with the layout the
MANO/OpenPose hand model uses: a wrist followed by five fingers of four
landmarks each, ordered base, middle, distal, tip. This module converts PICO's
26-joint OpenXR skeleton into that layout so the retargeting objective stays
independent of where the skeleton came from.

The index table below is the whole substance of the conversion, and getting one
row of it wrong is not a small error.

OpenXR `XrHandJointEXT` order, as delivered by the vendored binding:

```text
 0 palm                     1 wrist
 2 thumb metacarpal         3 thumb proximal
 4 thumb distal             5 thumb tip
 6 index metacarpal         7 index proximal
 8 index intermediate       9 index distal
10 index tip
11..15 middle metacarpal, proximal, intermediate, distal, tip
16..20 ring   metacarpal, proximal, intermediate, distal, tip
21..25 little metacarpal, proximal, intermediate, distal, tip
```

Two properties of this mapping matter enough to state explicitly.

The MCP knuckle is `*_proximal`, not `*_metacarpal`. OpenXR places
`*_metacarpal` at the base of the metacarpal bone, near the carpus, while the
MANO model's finger base is the knuckle. Measured on real PICO output, the
wrist-to-middle-metacarpal distance is 0.0226 m but wrist-to-middle-proximal is
0.0819 m. Since the retargeter derives its human hand scale from the
wrist-to-middle-base distance, choosing `*_metacarpal` would make that scale
3.6 times too small, inflate the robot-to-human ratio by the same factor, and
saturate every finger joint. The correct choice is not cosmetic.

The OpenXR thumb has no intermediate joint, so its four joints map one to one
onto the four canonical thumb landmarks: metacarpal to the canonical base,
which corresponds to the URDF `thumb_metacarpals` link, then proximal, distal
and tip.

Unlike MANO, PICO reports a genuinely left-handed skeleton for the left hand,
verified by a reflection-signed volume that is consistently negative on the
left and positive on the right. No mirroring is applied here; each side is
retargeted against its own same-handed URDF.
"""

from __future__ import annotations

import numpy as np

OPENXR_JOINT_COUNT = 26
CANONICAL_LANDMARK_COUNT = 21

OPENXR_PALM = 0
OPENXR_WRIST = 1

# Canonical landmark index -> OpenXR joint index.
CANONICAL_FROM_OPENXR = (
    1,                # 0  wrist
    2, 3, 4, 5,       # 1..4   thumb  metacarpal, proximal, distal, tip
    7, 8, 9, 10,      # 5..8   index  proximal, intermediate, distal, tip
    12, 13, 14, 15,   # 9..12  middle proximal, intermediate, distal, tip
    17, 18, 19, 20,   # 13..16 ring   proximal, intermediate, distal, tip
    22, 23, 24, 25,   # 17..20 little proximal, intermediate, distal, tip
)

CANONICAL_WRIST = 0
CANONICAL_FINGER_BASES = (5, 9, 13, 17)
CANONICAL_INDEX_BASE = 5
CANONICAL_LITTLE_BASE = 17

# Joints that must carry a valid position for the conversion to be usable.
REQUIRED_OPENXR_JOINTS = tuple(sorted(set(CANONICAL_FROM_OPENXR)))


def _positions(skeleton: np.ndarray) -> np.ndarray:
    return np.asarray(skeleton, dtype=np.float64)[:, :3]


def validate_skeleton(skeleton: np.ndarray) -> None:
    """Raise if the raw PICO skeleton cannot be converted.

    Rejects the wrong shape, non-finite values, and any required joint the
    binding never filled in.

    Absence is detected from the quaternion, not the position. The binding
    zero-fills a joint it has no data for, which leaves an all-zero quaternion,
    while every joint carrying real data has a unit quaternion; measured PICO
    output held a norm of 1.0000 throughout. Testing the position instead would
    misclassify any joint legitimately at the frame origin, such as a wrist
    expressed in its own local frame.
    """
    array = np.asarray(skeleton, dtype=np.float64)
    if array.shape != (OPENXR_JOINT_COUNT, 7):
        raise ValueError(
            f"Expected PICO hand skeleton with shape {(OPENXR_JOINT_COUNT, 7)}, "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("PICO hand skeleton contains NaN or infinity")
    norms = np.linalg.norm(array[:, 3:7], axis=1)
    missing = [
        index for index in REQUIRED_OPENXR_JOINTS if abs(float(norms[index]) - 1.0) > 0.1
    ]
    if missing:
        raise ValueError(
            f"PICO hand skeleton is missing required joints at indices {missing}"
        )


def to_canonical_landmarks(skeleton: np.ndarray) -> np.ndarray:
    """Convert a (26, 7) PICO skeleton to (21, 3) canonical landmarks."""
    validate_skeleton(skeleton)
    positions = _positions(skeleton)
    return np.stack([positions[index] for index in CANONICAL_FROM_OPENXR])


def finger_base_centroid(landmarks: np.ndarray) -> np.ndarray:
    """Centroid of the four non-thumb finger bases.

    Used as the correspondence origin between human and robot. Unlike the wrist,
    this is the same identifiable anatomical feature on both, so it does not
    depend on where a URDF happens to put its base frame.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    return points[list(CANONICAL_FINGER_BASES)].mean(axis=0)


def palm_scale(landmarks: np.ndarray) -> float:
    """Palm width: RMS spread of the four finger bases about their centroid.

    This is the retargeting scale reference. It is a genuine anatomical width on
    both the human and the robot, it is measured from four points rather than
    two so sensor noise averages down, and it varies only with the mechanism's
    tightly bounded abduction joints.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    bases = points[list(CANONICAL_FINGER_BASES)]
    centroid = bases.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((bases - centroid) ** 2, axis=1))))


def chirality(landmarks: np.ndarray) -> float:
    """Reflection-signed volume distinguishing a left from a right skeleton.

    Negative for a left hand and positive for a right hand on measured PICO
    output. Independent of the global origin, so it is safe to use as a
    handedness check regardless of the XR frame.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    wrist = points[CANONICAL_WRIST]
    index = points[CANONICAL_INDEX_BASE] - wrist
    little = points[CANONICAL_LITTLE_BASE] - wrist
    thumb = points[1] - wrist
    return float(np.dot(np.cross(index, little), thumb))


def expected_chirality_sign(side: str) -> float:
    """Sign that `chirality` should return for a correctly identified side."""
    if side == "left":
        return -1.0
    if side == "right":
        return 1.0
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")
