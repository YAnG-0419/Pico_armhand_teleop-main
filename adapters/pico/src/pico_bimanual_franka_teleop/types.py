from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


SIDES = ("left", "right")


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float)
        rotation = np.asarray(self.rotation, dtype=float)
        if position.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("Pose requires a 3-vector and a 3x3 rotation")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
            raise ValueError("Pose contains a non-finite value")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("Pose rotation is not orthonormal")
        object.__setattr__(self, "position", position.copy())
        object.__setattr__(self, "rotation", rotation.copy())


@dataclass(frozen=True)
class TeleopSample:
    poses: dict[str, Pose]
    activations: dict[str, bool]
    timestamp: float

    def __post_init__(self) -> None:
        if set(self.poses) != set(SIDES) or set(self.activations) != set(SIDES):
            raise ValueError("Teleop sample must contain left and right inputs")
        if not np.isfinite(self.timestamp):
            raise ValueError("Teleop timestamp is not finite")
        if any(not isinstance(active, bool) for active in self.activations.values()):
            raise ValueError("Teleop activations must be booleans")


@dataclass(frozen=True)
class JointTeleopSample:
    """One calibrated joint sample from a bimanual leader device."""

    positions: dict[str, np.ndarray]
    activations: dict[str, bool]
    timestamp: float

    def __post_init__(self) -> None:
        if set(self.positions) != set(SIDES) or set(self.activations) != set(SIDES):
            raise ValueError("Joint sample must contain left and right inputs")
        if not np.isfinite(self.timestamp):
            raise ValueError("Joint sample timestamp is not finite")
        if any(not isinstance(active, bool) for active in self.activations.values()):
            raise ValueError("Joint sample activations must be booleans")
        copied = {}
        for side in SIDES:
            values = np.asarray(self.positions[side], dtype=float)
            if values.shape != (7,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{side} joint sample must contain 7 finite values")
            copied[side] = values.copy()
        object.__setattr__(self, "positions", copied)


ArmSample: TypeAlias = TeleopSample | JointTeleopSample
