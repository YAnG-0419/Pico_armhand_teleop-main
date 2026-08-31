"""UDP hand-command transport shared by every hand pipeline.

One sender owns the socket, the per-side sequence numbers, and the per-side
send deadlines; `emit` folds each send's outcome into the side's status so a
transport failure can never escape into the loop that owns the arms. Hand
sources (PICO optical, MANUS, ...) keep their own policy - what to send and
when a side follows - and delegate the how to this class.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field

from .hand_stream import build_hand_packet
from .types import SIDES


@dataclass
class HandSideStatus:
    sending: bool = False
    fault: str | None = None
    sent: int = 0
    solve_seconds: float = 0.0


@dataclass
class HandStatus:
    sides: dict[str, HandSideStatus] = field(
        default_factory=lambda: {side: HandSideStatus() for side in SIDES}
    )
    errors: int = 0
    last_error: str | None = None


class HandCommandSender:
    """Send hand qpos packets at a bounded per-side rate, never raising."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        rate: float,
        sides: tuple[str, ...],
        models: dict[str, str],
        status: HandStatus,
    ) -> None:
        if not 0.0 < rate <= 60.0:
            raise ValueError("Hand send rate must be in (0, 60] Hz")
        self.address = (str(host), int(port))
        self.interval = 1.0 / float(rate)
        self.status = status
        if set(models) != set(sides) or any(
            not str(model).strip() for model in models.values()
        ):
            raise ValueError("models must define one non-empty model for every side")
        self.models = {
            side: str(models[side]).strip().lower() for side in sides
        }
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sequence = {side: 0 for side in sides}
        self._next_due = {side: 0.0 for side in sides}

    def due(self, side: str, moment: float) -> bool:
        return moment >= self._next_due[side]

    def next_sequence(self, side: str) -> int:
        """Sequence that the next successful packet attempt will carry."""
        return self._sequence[side]

    def _advance_deadline(self, side: str, moment: float) -> None:
        """Advance a periodic deadline without drifting down to the loop grid."""
        deadline = self._next_due[side]
        if deadline <= 0.0 or moment - deadline >= self.interval:
            # First send or a long pause: do not create a catch-up burst.
            self._next_due[side] = moment + self.interval
        else:
            # Preserve fractional deadlines. With a 100 Hz owner loop, setting
            # `moment + interval` made nominal 30 Hz become 25 Hz because every
            # 33.3 ms period rounded up to four 10 ms ticks.
            self._next_due[side] = deadline + self.interval

    def emit(
        self,
        stream_id: str,
        side: str,
        joint_names,
        qpos,
        moment: float,
        error_label: str,
    ) -> bool:
        """Send one packet and fold the outcome into the side's status."""
        status = self.status.sides[side]
        try:
            self._socket.sendto(
                build_hand_packet(
                    stream_id,
                    self._sequence[side],
                    time.time(),
                    side,
                    joint_names,
                    qpos,
                    model=self.models[side],
                ),
                self.address,
            )
        except Exception as error:  # noqa: BLE001 - never break arm control
            self.status.errors += 1
            self.status.last_error = f"{error_label}: {error}"
            status.sending = False
            status.fault = str(error)
            return False
        self._sequence[side] += 1
        self._advance_deadline(side, moment)
        status.sending = True
        status.fault = None
        status.sent += 1
        return True

    def close(self) -> None:
        self._socket.close()
