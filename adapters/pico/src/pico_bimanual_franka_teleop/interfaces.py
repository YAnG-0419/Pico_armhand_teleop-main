from typing import Protocol

from .types import ArmSample


class ArmPoseSource(Protocol):
    """Device adapter consumed by the arm coordinator."""

    def sample(self) -> ArmSample | None: ...

    def close(self) -> None: ...


class OperatorState(Protocol):
    """Shared operator intent, independent of any pose device."""

    def poll(self) -> dict[str, bool]: ...

    def poll_hands(self) -> dict[str, bool]: ...

    def take_requests(self) -> dict[str, bool]: ...

    def disable_all(self, reason: str) -> None: ...

    def deny(self, side: str, reason: str) -> None: ...

    def show(self, message: str) -> None: ...

    def set_status(self, status: str) -> None: ...

    def set_dataset_recording(self, **status) -> None: ...


class HandController(Protocol):
    sides: tuple[str, ...]
    status: object

    def set_active(self, active: dict[str, bool]) -> None: ...

    def request_open(
        self, sides: tuple[str, ...] | None = None, duration: float = 2.0
    ) -> None: ...

    def close(self) -> None: ...
