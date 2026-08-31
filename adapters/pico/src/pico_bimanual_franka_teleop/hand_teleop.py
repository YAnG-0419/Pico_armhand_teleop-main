"""Retarget PICO hand skeletons and send hand commands to the bridge.

The pipeline is a synchronous tickable object. The unified teleop wraps it in a
local worker so SDK reads and retargeting cannot delay the arm loop; the
hands-only command ticks it directly. It solves at most one side per tick.

Failure is contained in both directions. `tick` never raises, and losing an
optical skeleton never disengages an arm: they are independent signals, and a
wrist tracker can be perfectly healthy while the cameras lose sight of the
fingers. When a side becomes unusable this simply stops sending for it, the
bridge's watchdog stops publishing, the hand holds position, and the retargeter's
filter history is dropped so reacquisition cannot jump.
"""

from __future__ import annotations

import time
from pathlib import Path

from .hand_input import HandSkeletonReader
from .hand_sender import HandCommandSender, HandStatus
from .types import SIDES

__all__ = ["HandPipeline"]


class HandPipeline:
    """Read, retarget and send both hands, at most one solve per tick."""

    def __init__(
        self,
        xrt,
        *,
        assets_root: Path,
        host: str,
        port: int,
        rate: float = 30.0,
        sides: tuple[str, ...] = SIDES,
        models: dict[str, str] | None = None,
        stale_timeout: float = 0.25,
        frozen_timeout: float = 1.0,
        max_iterations: int = 20,
        debug_log: str | Path | None = None,
    ) -> None:
        if xrt is None:
            raise ValueError(
                "PICO optical hands need an initialized PICO SDK client; "
                "the selected arm source does not own one"
            )
        if not sides or set(sides).difference(SIDES):
            raise ValueError(f"Invalid hand sides: {sides}")
        self.sides = tuple(sides)
        if models is not None and set(models) != set(self.sides):
            raise ValueError("models must define exactly the active hand sides")
        self.models = (
            {side: "g20" for side in self.sides}
            if models is None
            else {
                side: str(models[side]).strip().lower() for side in self.sides
            }
        )
        from .hand_profiles import create_hand_retargeter

        self.retargeters = {}
        try:
            for side in sides:
                self.retargeters[side] = create_hand_retargeter(
                    self.models[side],
                    side=side,
                    assets_root=Path(assets_root),
                    max_iterations=max_iterations,
                )
        except BaseException:
            for retargeter in self.retargeters.values():
                retargeter.close()
            raise
        self.status = HandStatus()
        try:
            self.sender = HandCommandSender(
                host=host,
                port=port,
                rate=rate,
                sides=self.sides,
                models=self.models,
                status=self.status,
            )
        except BaseException:
            for retargeter in self.retargeters.values():
                retargeter.close()
            raise

        self.reader = HandSkeletonReader(
            xrt, stale_timeout=stale_timeout, frozen_timeout=frozen_timeout
        )
        self._open_until = {side: 0.0 for side in sides}
        self._was_following = {side: False for side in sides}
        # Round-robin start point, so one side cannot starve the other when both
        # come due on the same tick.
        self._preferred = 0
        self.debug_logger = None
        if debug_log is not None:
            from .debug_log import HandRetargetDebugLogger

            self.debug_logger = HandRetargetDebugLogger(debug_log)

    def request_open(
        self,
        now: float | None = None,
        duration: float = 2.0,
        sides: tuple[str, ...] | None = None,
    ) -> None:
        """Stream the open-hand pose to the selected (default: every)
        configured side that is not following.

        URDF zeros are the bridge's own home() pose: fingers straight, abduction
        centred. The stream lasts `duration` seconds because the bridge's 250 ms
        watchdog needs a continuous feed, not one packet. A side that is
        actively following the operator's hand ignores the request: live
        tracking always supersedes a parked-hand command.
        """
        if duration <= 0.0:
            raise ValueError("Open duration must be positive")
        moment = time.monotonic() if now is None else float(now)
        selected = self.sides if sides is None else sides
        for side in selected:
            if side in self._open_until:
                self._open_until[side] = moment + float(duration)

    def tick(
        self,
        now: float | None = None,
        active: dict[str, bool] | None = None,
    ) -> None:
        """Advance the hand pipeline by at most one solve. Never raises.

        `active` optionally gates following per side: a side whose flag is
        False stops sending, exactly as if its skeleton were lost, so the
        bridge watchdog holds that hand. The coordinator passes the shared
        operator engagement here, making one per-side switch control arm and hand.
        Passing None (the standalone hands-only path) keeps every side
        following whenever its skeleton is live.
        """
        moment = time.monotonic() if now is None else float(now)
        try:
            samples = self.reader.sample(now=moment)
        except Exception as error:  # noqa: BLE001 - must not reach the arm loop
            self.status.errors += 1
            self.status.last_error = f"hand sample failed: {error}"
            return

        following = {}
        for side in self.sides:
            allowed = active is None or bool(active.get(side, False))
            follows = allowed and samples.get(side) is not None
            if follows:
                # Live tracking on an engaged side supersedes a pending open.
                self._open_until[side] = 0.0
            elif self._was_following[side]:
                # Dropping filter history keeps reacquisition smooth, whether
                # the side went on to stream the open pose or to hold.
                self.retargeters[side].reset()
            following[side] = follows
            self._was_following[side] = follows

        # Open-pose sends carry a fixed qpos and no solve, so they run outside
        # the one-solve-per-tick rotation; both sides may send in one tick.
        for side in self.sides:
            if (
                following[side]
                or moment >= self._open_until[side]
                or not self.sender.due(side, moment)
            ):
                continue
            names = self.retargeters[side].joint_names
            self.sender.emit(
                f"pico-hand-{side}-open",
                side,
                names,
                [0.0] * len(names),
                moment,
                f"{side} open command failed",
            )

        for side in self.sides:
            if following[side] or moment < self._open_until[side]:
                continue
            # Send nothing; the bridge watchdog holds the hand.
            status = self.status.sides[side]
            status.sending = False
            status.fault = (
                self.reader.faults[side]
                if samples.get(side) is None
                else "disengaged by operator"
            )

        # Solve at most one side per tick so a tick never costs two solves.
        order = [
            self.sides[(self._preferred + offset) % len(self.sides)]
            for offset in range(len(self.sides))
        ]
        for side in order:
            sample = samples.get(side)
            if (
                not following[side]
                or sample is None
                or not self.sender.due(side, moment)
            ):
                continue
            status = self.status.sides[side]
            try:
                started = time.monotonic()
                qpos, stats = self.retargeters[side].retarget(sample.landmarks)
                elapsed = time.monotonic() - started
                if self.debug_logger is not None:
                    self.debug_logger.record(
                        moment, side, sample.landmarks, qpos, stats
                    )
            except Exception as error:  # noqa: BLE001 - contain per side
                self.status.errors += 1
                self.status.last_error = f"{side} hand retargeting failed: {error}"
                status.sending = False
                status.fault = str(error)
                return
            if self.sender.emit(
                f"pico-hand-{side}",
                side,
                self.retargeters[side].joint_names,
                qpos,
                moment,
                f"{side} hand send failed",
            ):
                self._preferred = (self.sides.index(side) + 1) % len(self.sides)
                status.solve_seconds = elapsed
            return

    def close(self) -> None:
        try:
            if self.debug_logger is not None:
                self.debug_logger.close()
        finally:
            self.sender.close()
            for retargeter in self.retargeters.values():
                retargeter.close()
