import threading
import time
import math

from .types import SIDES


class HandWorker:
    """Run a hand pipeline outside the deadline-critical arm loop."""

    def __init__(self, pipeline, tick_rate: float = 100.0) -> None:
        if tick_rate <= 0:
            raise ValueError("Hand worker tick rate must be positive")
        self.pipeline = pipeline
        self.sides = tuple(pipeline.sides)
        self.status = pipeline.status
        self.dt = 1.0 / float(tick_rate)
        self._active = {side: False for side in SIDES}
        self._open_requests: list[tuple[tuple[str, ...] | None, float]] = []
        self._feedback: dict[str, tuple[tuple[float, ...], float]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Hand worker is closed")
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="hand-teleop", daemon=True
        )
        self._thread.start()

    def set_active(self, active: dict[str, bool]) -> None:
        with self._lock:
            self._active = {
                side: bool(active.get(side, False)) for side in SIDES
            }

    def request_open(
        self, sides: tuple[str, ...] | None = None, duration: float = 2.0
    ) -> None:
        selected = None if sides is None else tuple(sides)
        if duration <= 0.0:
            raise ValueError("Open duration must be positive")
        if selected is not None and set(selected).difference(self.sides):
            raise ValueError(f"Invalid hand sides: {selected}")
        with self._lock:
            for side in self.sides if selected is None else selected:
                self._active[side] = False
            self._open_requests.append((selected, float(duration)))

    def feedback_snapshot(
        self, side: str
    ) -> tuple[tuple[float, ...], float] | None:
        """Return cached measured positions and their monotonic receipt time."""
        if side not in self.sides:
            raise ValueError(f"Invalid hand side: {side}")
        with self._lock:
            snapshot = self._feedback.get(side)
        if snapshot is None:
            return None
        positions, received_at = snapshot
        return tuple(positions), float(received_at)

    def _snapshot(self):
        with self._lock:
            active = dict(self._active)
            requests = self._open_requests
            self._open_requests = []
        return active, requests

    def _run(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            active, requests = self._snapshot()
            for sides, duration in requests:
                self.pipeline.request_open(sides=sides, duration=duration)
            self.pipeline.tick(active=active)
            feedback_reader = getattr(self.pipeline, "feedback_position", None)
            if feedback_reader is not None:
                for side in getattr(self.pipeline, "feedback_sides", self.sides):
                    try:
                        positions = feedback_reader(side)
                    except Exception as error:  # noqa: BLE001 - contain worker I/O
                        self.status.errors += 1
                        self.status.last_error = f"{side} hand feedback failed: {error}"
                        continue
                    if positions is None:
                        continue
                    values = tuple(float(value) for value in positions)
                    if len(values) == 20 and all(math.isfinite(value) for value in values):
                        with self._lock:
                            self._feedback[side] = (values, time.monotonic())
            deadline += self.dt
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                deadline = time.monotonic()
                continue
            self._stop.wait(remaining)

    def close(self) -> None:
        if self._closed:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 5.0 * self.dt))
            if self._thread.is_alive():
                raise RuntimeError("Hand worker did not stop")
        self.pipeline.close()
        self._closed = True
