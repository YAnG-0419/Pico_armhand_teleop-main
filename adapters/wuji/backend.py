"""Hardware backends copied and adapted from wuji-retargeting."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def _hand2_joint_index(node_id: int) -> int:
    """Map a Hand 2 firmware node id to its command-vector index."""
    finger, joint = divmod(int(node_id) - 1, 5)
    if node_id <= 0 or finger >= 5 or joint >= 4:
        raise ValueError(f"Wuji Hand 2 reported invalid joint id {node_id}")
    return finger * 4 + joint


def _hand2_feedback_positions(frame: Any) -> np.ndarray:
    """Return one complete feedback frame in firmware command order."""
    entries = tuple(frame.joints)
    if int(frame.num_joints) != len(entries):
        raise ValueError("Wuji Hand 2 feedback count does not match its entries")
    positions: dict[int, float] = {}
    for entry in entries:
        node_id = int(entry.nid)
        index = _hand2_joint_index(node_id)
        if index in positions:
            raise ValueError(f"Wuji Hand 2 reported duplicate joint id {node_id}")
        positions[index] = float(entry.position)
    if set(positions) != set(range(20)):
        missing = sorted(set(range(20)).difference(positions))
        raise RuntimeError(f"Wuji Hand 2 feedback is missing joints {missing}")
    values = np.asarray([positions[index] for index in range(20)], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Wuji Hand 2 feedback contains non-finite positions")
    return values


class WujiHandBackend:
    """Original USB Wuji Hand backend."""

    def __init__(self, serial: str = "") -> None:
        try:
            import wujihandpy
        except ImportError as error:
            raise RuntimeError(
                "wujihandpy is required for model=wuji_hand"
            ) from error
        self._sdk = wujihandpy
        self._hand = (
            wujihandpy.Hand(serial_number=serial)
            if serial
            else wujihandpy.Hand()
        )
        self._hand.write_joint_enabled(True)
        self._controller = self._hand.realtime_controller(
            enable_upstream=False,
            filter=wujihandpy.filter.LowPass(cutoff_freq=5.0),
        )
        time.sleep(0.5)

    def send(self, qpos: np.ndarray) -> None:
        values = np.asarray(qpos, dtype=np.float64)
        if values.shape != (20,) or not np.isfinite(values).all():
            raise ValueError("Wuji Hand command must be 20 finite positions")
        self._controller.set_joint_target_position(values.reshape(5, 4))

    def close(self) -> None:
        if getattr(self, "_hand", None) is not None:
            self._hand.write_joint_enabled(False)


def _set_with_retry(description: str, operation, attempts: int = 3) -> None:
    for attempt in range(attempts):
        try:
            operation()
            return
        except (AttributeError, TypeError):
            raise
        except Exception:
            if attempt + 1 == attempts:
                raise
            print(
                f"Wuji Hand 2 {description} timed out; retrying "
                f"({attempt + 1}/{attempts})"
            )
            time.sleep(0.6)


class WujiHand2Backend:
    """Network Wuji Hand 2 backend using the vendor SDK."""

    def __init__(
        self,
        *,
        side: str,
        address: str,
        kp: float,
        kd: float,
        current_limit: float,
    ) -> None:
        try:
            import wuji_sdk
            from wuji_sdk import SdkManager
        except ImportError as error:
            raise RuntimeError(
                "wuji-sdk>=0.10 is required for model=wuji_hand_2"
            ) from error
        if not address:
            raise ValueError(
                f"{side} Wuji Hand 2 requires an explicit address; "
                f"pass --wuji-{side}-address IP:PORT"
            )
        self._sdk = wuji_sdk
        self._manager = SdkManager.instance()
        # device_name is a local SdkManager alias, not the product type; it must
        # be unique when both hands are connected in one process.
        self._hand: Any = self._manager.connect(
            address=address, device_name=f"wuji_hand_2_{side}"
        )
        self._publisher: Any = None
        self._state_subscription: Any = None
        try:
            reported_side = str(self._hand.handedness().get()).lower()
            if reported_side != side:
                raise RuntimeError(
                    f"Wuji hand at {address} reports {reported_side}, expected {side}"
                )
            online = int(self._hand.online_joints_count().get())
            if online == 0:
                raise RuntimeError(f"{side} Wuji Hand 2 has no online joints")
            time.sleep(0.5)
            _set_with_retry(
                "effort_limit", lambda: self._hand.effort_limit().set(current_limit)
            )
            _set_with_retry(
                "mit_params", lambda: self._hand.mit_params().set((kp, kd))
            )
            self._hand.enable()
            self._wait_until_enabled()
            self._state_subscription = self._hand.joint_states().subscribe()
            self._publisher = self._hand.joint_command().publish()
            self._JointCommand = wuji_sdk.JointCommand
            print(
                f"{side} Wuji Hand 2 enabled at {address}: "
                f"kp={kp:g}, kd={kd:g}, current_limit={current_limit:g}A"
            )
        except BaseException:
            self.close()
            raise

    def _wait_until_enabled(self) -> None:
        deadline = time.monotonic() + 5.0
        subscription = self._hand.joint_diagnostics().subscribe()
        try:
            while time.monotonic() < deadline:
                time.sleep(0.2)
                frame = subscription.recv()
                if frame is None:
                    continue
                live = [entry for entry in frame.joints if entry.vbus_v_fb > 0.5]
                if live and all(entry.status_word.ext_state == 2 for entry in live):
                    return
        finally:
            subscription.close()
        self._hand.disable()
        raise RuntimeError("Wuji Hand 2 did not reach Enabled state within 5 s")

    def send(self, qpos: np.ndarray) -> None:
        values = np.asarray(qpos, dtype=np.float64)
        if values.shape != (20,) or not np.isfinite(values).all():
            raise ValueError("Wuji Hand 2 command must be 20 finite positions")
        commands = [
            self._JointCommand(float(position), 0.0, 0.0)
            for position in values
        ]
        self._publisher.send(commands)

    def read_position(self) -> np.ndarray | None:
        """Drain state feedback and return the newest complete position frame."""
        subscription = self._state_subscription
        if subscription is None:
            return None
        latest = None
        while (frame := subscription.recv()) is not None:
            latest = _hand2_feedback_positions(frame)
        return latest

    def close(self) -> None:
        subscription = getattr(self, "_state_subscription", None)
        if subscription is not None:
            try:
                subscription.close()
            except Exception:
                pass
            self._state_subscription = None
        publisher = getattr(self, "_publisher", None)
        if publisher is not None:
            try:
                publisher.close()
            except Exception:
                pass
            self._publisher = None
        hand = getattr(self, "_hand", None)
        if hand is not None:
            try:
                hand.disable()
            except Exception:
                pass
            try:
                hand.disconnect()
            except Exception:
                pass
            self._hand = None
