import socket
import time
import uuid

import numpy as np
from teleop_core.contract import COMMAND_JOINT_NAMES, command_names
from teleop_core.protocol import JointPacket, decode_packet, encode_packet


class UdpRobotBackend:
    def __init__(
        self,
        command_host: str,
        command_port: int,
        state_host: str,
        state_port: int,
        state_timeout: float,
    ) -> None:
        self.command_address = (command_host, command_port)
        self.state_timeout = float(state_timeout)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((state_host, state_port))
        self.socket.setblocking(False)
        self.stream_id = uuid.uuid4().hex
        self.sequence = 0
        self.last_state_sequence = -1
        self.last_state_at: float | None = None
        self.last_gateway_status_sequence = -1
        self.gateway_faults: tuple[str, ...] = ()
        self.q: np.ndarray | None = None

    def receive_state(self) -> np.ndarray | None:
        while True:
            try:
                payload, _ = self.socket.recvfrom(16_385)
            except BlockingIOError:
                break
            packet = decode_packet(payload)
            if packet.kind != "state" or packet.sequence <= self.last_state_sequence:
                continue
            if packet.names != COMMAND_JOINT_NAMES:
                raise ValueError("State packet does not contain all 14 joints in order")
            self.q = np.asarray(packet.positions, dtype=float)
            if (
                packet.command_stream_id == self.stream_id
                and packet.command_sequence > self.last_gateway_status_sequence
            ):
                self.last_gateway_status_sequence = packet.command_sequence
                self.gateway_faults += packet.faults
            self.last_state_sequence = packet.sequence
            self.last_state_at = time.monotonic()
        if self.last_state_at is None:
            return None
        if time.monotonic() - self.last_state_at > self.state_timeout:
            return None
        assert self.q is not None
        return self.q.copy()

    def take_gateway_faults(self) -> tuple[str, ...]:
        faults = self.gateway_faults
        self.gateway_faults = ()
        return faults

    def wait_for_state(self, timeout: float) -> np.ndarray:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            q = self.receive_state()
            if q is not None:
                return q
            time.sleep(0.01)
        raise TimeoutError("No fresh dual-FR3 state was received")

    def send_command(self, q: np.ndarray, active_sides: tuple[str, ...]) -> None:
        values = np.asarray(q, dtype=float)
        if values.shape != (14,) or not np.all(np.isfinite(values)):
            raise ValueError("Command must contain 14 finite joint positions")
        names = command_names(active_sides)
        positions = tuple(
            float(values[COMMAND_JOINT_NAMES.index(name)]) for name in names
        )
        packet = JointPacket(
            kind="command",
            stream_id=self.stream_id,
            sequence=self.sequence,
            timestamp=time.monotonic(),
            active_sides=active_sides,
            names=names,
            positions=positions,
        )
        self.socket.sendto(encode_packet(packet), self.command_address)
        self.sequence += 1

    def close(self) -> None:
        self.socket.close()
