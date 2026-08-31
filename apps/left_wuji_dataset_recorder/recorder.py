"""Non-blocking, crash-recoverable recorder for one left-side teleop episode."""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

import numpy as np
import pinocchio as pin


SCHEMA = "left-fr3-wuji-hand2.v1"
ARM_JOINT_NAMES = tuple(f"left_fr3v2_joint{index}" for index in range(1, 8))
EE_FRAME = "left_fr3v2_link8"
BASE_FRAME = "lychee_root"


class LeftWujiDatasetRecorder:
    """Queue samples off the control thread and finalize them as one NPZ."""

    def __init__(
        self,
        output: str | Path,
        *,
        hand_joint_names,
        control_rate_hz: float,
        hand_stale_timeout_s: float = 0.5,
        queue_size: int = 4096,
    ) -> None:
        self.output = Path(output).expanduser().resolve()
        if self.output.suffix != ".npz":
            raise ValueError("left Wuji dataset output must end in .npz")
        if self.output.exists():
            raise FileExistsError(f"refusing to overwrite dataset: {self.output}")
        self.raw_output = self.output.with_suffix(".raw.jsonl")
        if self.raw_output.exists():
            raise FileExistsError(
                f"refusing to overwrite raw dataset: {self.raw_output}"
            )
        names = tuple(str(name) for name in hand_joint_names)
        if len(names) != 20 or len(set(names)) != 20:
            raise ValueError("Wuji Hand 2 must provide 20 unique joint names")
        if not np.isfinite(control_rate_hz) or control_rate_hz <= 0.0:
            raise ValueError("control rate must be positive")
        if hand_stale_timeout_s <= 0.0:
            raise ValueError("hand stale timeout must be positive")

        self.hand_joint_names = names
        self.control_rate_hz = float(control_rate_hz)
        self.hand_stale_timeout_s = float(hand_stale_timeout_s)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize=queue_size)
        self._rows: list[dict] = []
        self._dropped = 0
        self._failed: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._write_loop,
            name="left-wuji-dataset-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def sample_count(self) -> int:
        return len(self._rows)

    @property
    def dropped_samples(self) -> int:
        return self._dropped

    @staticmethod
    def _finite_vector(values, size: int, label: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (size,) or not np.isfinite(result).all():
            raise ValueError(f"{label} must contain {size} finite values")
        return result

    def record(
        self,
        *,
        wall_time_ns: int,
        monotonic_time_s: float,
        arm_q,
        ee_pose,
        hand_feedback: tuple[tuple[float, ...], float] | None,
        arm_active: bool,
        hand_active: bool,
    ) -> None:
        """Enqueue a sample without waiting for disk I/O."""
        if self._closed or self._failed is not None:
            return
        arm = self._finite_vector(arm_q, 7, "left arm q")
        rotation = np.asarray(ee_pose.rotation, dtype=np.float64)
        position = self._finite_vector(ee_pose.position, 3, "end-effector position")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("end-effector rotation must be a finite 3x3 matrix")
        quaternion_xyzw = np.asarray(pin.Quaternion(rotation).coeffs(), dtype=np.float64)

        hand_q = np.full(20, np.nan, dtype=np.float64)
        hand_sample_time_s = np.nan
        hand_age_s = np.nan
        hand_valid = False
        if hand_feedback is not None:
            positions, received_at = hand_feedback
            candidate = np.asarray(positions, dtype=np.float64)
            received_at = float(received_at)
            if candidate.shape == (20,) and np.isfinite(candidate).all():
                hand_q = candidate
                hand_sample_time_s = received_at
                hand_age_s = max(0.0, float(monotonic_time_s) - received_at)
                hand_valid = hand_age_s <= self.hand_stale_timeout_s

        row = {
            "wall_time_ns": int(wall_time_ns),
            "monotonic_time_s": float(monotonic_time_s),
            "arm_q": arm.tolist(),
            "hand_q": hand_q.tolist(),
            "hand_sample_time_s": hand_sample_time_s,
            "hand_age_s": hand_age_s,
            "hand_valid": bool(hand_valid),
            "ee_position": position.tolist(),
            "ee_quaternion_xyzw": quaternion_xyzw.tolist(),
            "arm_active": bool(arm_active),
            "hand_active": bool(hand_active),
        }
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self._dropped += 1

    def _write_loop(self) -> None:
        try:
            with self.raw_output.open("x", encoding="utf-8") as raw:
                raw.write(json.dumps({
                    "schema": SCHEMA,
                    "arm_joint_names": ARM_JOINT_NAMES,
                    "hand_joint_names": self.hand_joint_names,
                    "ee_frame": EE_FRAME,
                    "base_frame": BASE_FRAME,
                    "control_rate_hz": self.control_rate_hz,
                    "hand_stale_timeout_s": self.hand_stale_timeout_s,
                }, separators=(",", ":")) + "\n")
                while True:
                    row = self._queue.get()
                    if row is None:
                        break
                    self._rows.append(row)
                    raw.write(json.dumps(row, separators=(",", ":")) + "\n")
                    if len(self._rows) % 100 == 0:
                        raw.flush()
                raw.flush()
                os.fsync(raw.fileno())
        except BaseException as error:
            self._failed = error

    def _save_npz(self) -> None:
        rows = self._rows
        arrays = {
            "schema": np.asarray(SCHEMA),
            "arm_joint_names": np.asarray(ARM_JOINT_NAMES),
            "hand_joint_names": np.asarray(self.hand_joint_names),
            "ee_frame": np.asarray(EE_FRAME),
            "base_frame": np.asarray(BASE_FRAME),
            "control_rate_hz": np.asarray(self.control_rate_hz, dtype=np.float64),
            "hand_stale_timeout_s": np.asarray(self.hand_stale_timeout_s, dtype=np.float64),
            "dropped_samples": np.asarray(self._dropped, dtype=np.int64),
            "wall_time_ns": np.asarray([row["wall_time_ns"] for row in rows], dtype=np.int64),
            "monotonic_time_s": np.asarray([row["monotonic_time_s"] for row in rows], dtype=np.float64),
            "arm_q": np.asarray([row["arm_q"] for row in rows], dtype=np.float64).reshape(-1, 7),
            "hand_q": np.asarray([row["hand_q"] for row in rows], dtype=np.float64).reshape(-1, 20),
            "hand_sample_time_s": np.asarray([row["hand_sample_time_s"] for row in rows], dtype=np.float64),
            "hand_age_s": np.asarray([row["hand_age_s"] for row in rows], dtype=np.float64),
            "hand_valid": np.asarray([row["hand_valid"] for row in rows], dtype=np.bool_),
            "ee_position": np.asarray([row["ee_position"] for row in rows], dtype=np.float64).reshape(-1, 3),
            "ee_quaternion_xyzw": np.asarray([row["ee_quaternion_xyzw"] for row in rows], dtype=np.float64).reshape(-1, 4),
            "arm_active": np.asarray([row["arm_active"] for row in rows], dtype=np.bool_),
            "hand_active": np.asarray([row["hand_active"] for row in rows], dtype=np.bool_),
        }
        temporary = self.output.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(self.output)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while self._thread.is_alive() and self._failed is None:
            try:
                self._queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            raise RuntimeError("dataset writer did not stop within 30 seconds")
        if self._failed is not None:
            raise RuntimeError(f"dataset raw writer failed: {self._failed}") from self._failed
        self._save_npz()
