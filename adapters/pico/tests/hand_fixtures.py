"""Shared synthetic hand geometry for the hand tests."""

import numpy as np

from pico_bimanual_franka_teleop import hand_landmarks as hl


def synthetic_skeleton(mirror: bool = False, flex: float = 0.0) -> np.ndarray:
    """A metrically plausible right hand; mirror=True makes it left.

    Segment lengths are chosen to sit in the range measured from real PICO
    output, so `palm_scale` and the retargeting gain stay realistic.
    """
    poses = np.zeros((26, 7))
    poses[:, 6] = 1.0
    poses[hl.OPENXR_WRIST, :3] = [0.0, 0.0, 0.0]
    poses[hl.OPENXR_PALM, :3] = [0.0, 0.0, 0.045]

    lateral = {"index": 0.031, "middle": 0.010, "ring": -0.010, "little": -0.031}
    metacarpal = {"index": 6, "middle": 11, "ring": 16, "little": 21}
    for finger, base in metacarpal.items():
        x = lateral[finger]
        poses[base, :3] = [x, 0.0, 0.024]
        z = 0.082
        poses[base + 1, :3] = [x, 0.0, z]
        for step, length in enumerate((0.042, 0.028, 0.021), start=2):
            z += length * np.cos(flex)
            poses[base + step, :3] = [x, -length * np.sin(flex) * step * 0.4, z]

    # The thumb sits on the same side of the palm plane that the fingers curl
    # toward, as it does anatomically. That also makes this a genuinely
    # right-handed skeleton under `chirality`.
    poses[2, :3] = [0.036, -0.012, 0.020]
    poses[3, :3] = [0.052, -0.020, 0.043]
    poses[4, :3] = [0.058, -0.026, 0.066]
    poses[5, :3] = [0.060, -0.030, 0.082]
    if mirror:
        poses[:, 0] *= -1.0
    return poses
