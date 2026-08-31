import json
import math
from dataclasses import dataclass
from typing import Literal

from .contract import COMMAND_JOINT_NAMES, SIDES, command_names


PROTOCOL_VERSION = 2
MAX_PACKET_BYTES = 16_384


@dataclass(frozen=True)
class JointPacket:
    kind: Literal["command", "state"]
    stream_id: str
    sequence: int
    timestamp: float
    active_sides: tuple[str, ...]
    names: tuple[str, ...]
    positions: tuple[float, ...]
    command_sequence: int = -1
    command_stream_id: str = ""
    faults: tuple[str, ...] = ()


def encode_packet(packet: JointPacket) -> bytes:
    payload = {
        "version": PROTOCOL_VERSION,
        "kind": packet.kind,
        "stream_id": packet.stream_id,
        "sequence": packet.sequence,
        "timestamp": packet.timestamp,
        "active_sides": list(packet.active_sides),
        "names": list(packet.names),
        "positions": list(packet.positions),
        "command_sequence": packet.command_sequence,
        "command_stream_id": packet.command_stream_id,
        "faults": list(packet.faults),
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_PACKET_BYTES:
        raise ValueError("Joint packet exceeds the datagram limit.")
    return encoded


def decode_packet(payload: bytes) -> JointPacket:
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError("Joint packet exceeds the datagram limit.")
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Joint packet is not valid JSON.") from exc
    required = {
        "version",
        "kind",
        "stream_id",
        "sequence",
        "timestamp",
        "active_sides",
        "names",
        "positions",
        "command_sequence",
        "command_stream_id",
        "faults",
    }
    if not isinstance(message, dict) or set(message) != required:
        raise ValueError("Joint packet has an invalid schema.")
    if message["version"] != PROTOCOL_VERSION:
        raise ValueError("Unsupported joint protocol version.")
    if message["kind"] not in {"command", "state"}:
        raise ValueError("Joint packet kind must be command or state.")
    if not isinstance(message["stream_id"], str) or not message["stream_id"]:
        raise ValueError("Joint packet stream ID is invalid.")
    try:
        sequence = int(message["sequence"])
        timestamp = float(message["timestamp"])
        positions = tuple(float(value) for value in message["positions"])
        command_sequence = int(message["command_sequence"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Joint packet contains a non-numeric field.") from exc
    if (
        not isinstance(message["command_stream_id"], str)
        or not isinstance(message["faults"], list)
        or any(not isinstance(fault, str) for fault in message["faults"])
    ):
        raise ValueError("Joint packet gateway feedback is invalid.")
    names = tuple(str(name) for name in message["names"])
    active_sides = tuple(str(side) for side in message["active_sides"])
    command_stream_id = message["command_stream_id"]
    faults = tuple(message["faults"])
    if sequence < 0 or command_sequence < -1 or not math.isfinite(timestamp):
        raise ValueError("Joint packet sequence or timestamp is invalid.")
    if set(active_sides).difference(SIDES):
        raise ValueError("Joint packet contains an invalid active side.")
    if len(active_sides) != len(set(active_sides)):
        raise ValueError("Joint packet contains duplicate active sides.")
    if len(names) != len(positions) or len(names) != len(set(names)):
        raise ValueError("Joint packet names and positions do not match.")
    if not all(math.isfinite(value) for value in positions):
        raise ValueError("Joint packet contains a non-finite position.")
    if set(names).difference(COMMAND_JOINT_NAMES):
        raise ValueError("Joint packet contains an unknown joint name.")
    if message["kind"] == "command" and (
        command_sequence != -1 or command_stream_id or faults
    ):
        raise ValueError("Command packet contains gateway feedback.")
    if message["kind"] == "state" and command_sequence >= 0 and not command_stream_id:
        raise ValueError("State packet gateway feedback has no command stream.")
    expected = COMMAND_JOINT_NAMES if message["kind"] == "state" else command_names(active_sides)
    if names != expected:
        raise ValueError("Joint packet names are not in canonical order.")
    return JointPacket(
        kind=message["kind"],
        stream_id=message["stream_id"],
        sequence=sequence,
        timestamp=timestamp,
        active_sides=active_sides,
        names=names,
        positions=positions,
        command_sequence=command_sequence,
        command_stream_id=command_stream_id,
        faults=faults,
    )
