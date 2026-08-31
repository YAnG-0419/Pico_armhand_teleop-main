"""Minimal ctypes interface for calibrated MANUS skeleton frames."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


KEYPOINT_COUNT = 25
SIDE_CODES = {"left": 1, "right": 2}
CANONICAL_FROM_MANUS = (
    0,
    1, 2, 3, 4,
    6, 7, 8, 9,
    11, 12, 13, 14,
    16, 17, 18, 19,
    21, 22, 23, 24,
)


class ManusPose(ctypes.Structure):
    _fields_ = [
        ("position_x", ctypes.c_float),
        ("position_y", ctypes.c_float),
        ("position_z", ctypes.c_float),
        ("orientation_x", ctypes.c_float),
        ("orientation_y", ctypes.c_float),
        ("orientation_z", ctypes.c_float),
        ("orientation_w", ctypes.c_float),
    ]


class ManusFrame(ctypes.Structure):
    _fields_ = [
        ("side", ctypes.c_int32),
        ("keypoint_count", ctypes.c_uint32),
        ("timestamp_ns", ctypes.c_uint64),
        ("sequence", ctypes.c_uint64),
        ("keypoints", ManusPose * KEYPOINT_COUNT),
    ]


class ManusBridge:
    """Own one connection to the vendored MANUS native bridge."""

    def __init__(self, library: Path) -> None:
        self.library = ctypes.CDLL(str(library))
        self.library.litchi_manus_bridge_abi_version.restype = ctypes.c_uint32
        self.library.litchi_manus_frame_size.restype = ctypes.c_uint32
        self.library.litchi_manus_connect.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_char_p,
        ]
        self.library.litchi_manus_connect.restype = ctypes.c_int32
        self.library.litchi_manus_disconnect.restype = ctypes.c_int32
        self.library.litchi_manus_read_frame.argtypes = [
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.POINTER(ManusFrame),
        ]
        self.library.litchi_manus_read_frame.restype = ctypes.c_int32
        self.library.litchi_manus_available_sides.restype = ctypes.c_uint32
        self.library.litchi_manus_last_error.restype = ctypes.c_char_p
        if self.library.litchi_manus_bridge_abi_version() != 1:
            raise RuntimeError("unsupported MANUS skeleton bridge ABI")
        if self.library.litchi_manus_frame_size() != ctypes.sizeof(ManusFrame):
            raise RuntimeError("MANUS frame ABI size mismatch")
        self.connected = False

    def error(self) -> str:
        message = self.library.litchi_manus_last_error()
        return message.decode("utf-8", errors="replace") if message else "unknown error"

    def connect(self, calibration_dir: Path) -> None:
        result = self.library.litchi_manus_connect(
            1, 1, str(calibration_dir).encode("utf-8")
        )
        if result != 0:
            raise RuntimeError(self.error())
        self.connected = True

    def available_sides(self) -> tuple[str, ...]:
        mask = int(self.library.litchi_manus_available_sides())
        return tuple(
            side for side, code in SIDE_CODES.items() if mask & (1 << (code - 1))
        )

    def read(self, side: str, timeout_s: float = 0.0) -> ManusFrame | None:
        code = SIDE_CODES[side]
        frame = ManusFrame()
        result = self.library.litchi_manus_read_frame(
            code,
            max(0, round(float(timeout_s) * 1000)),
            ctypes.byref(frame),
        )
        if result < 0:
            raise RuntimeError(self.error())
        if result == 0:
            return None
        if frame.side != code or frame.keypoint_count != KEYPOINT_COUNT:
            raise RuntimeError(f"invalid {side} MANUS skeleton frame")
        return frame

    def close(self) -> None:
        if self.connected:
            self.library.litchi_manus_disconnect()
            self.connected = False


def canonical_landmarks(frame: ManusFrame) -> np.ndarray:
    points = np.asarray(
        [
            (point.position_x, point.position_y, point.position_z)
            for point in frame.keypoints
        ],
        dtype=np.float64,
    )
    selected = points[list(CANONICAL_FROM_MANUS)]
    if selected.shape != (21, 3) or not np.isfinite(selected).all():
        raise ValueError("MANUS skeleton contains invalid landmarks")
    return selected
