import pytest

from teleop_core.contract import COMMAND_JOINT_NAMES
from teleop_core.protocol import JointPacket, decode_packet, encode_packet


def test_state_packet_round_trips_gateway_feedback():
    packet = JointPacket(
        kind="state",
        stream_id="bridge",
        sequence=12,
        timestamp=3.5,
        active_sides=("left",),
        names=COMMAND_JOINT_NAMES,
        positions=(0.0,) * 14,
        command_sequence=99,
        command_stream_id="operator",
        faults=("right first target is too far from measured state.",),
    )
    assert decode_packet(encode_packet(packet)) == packet


def test_old_protocol_packets_are_rejected_instead_of_misread():
    payload = (
        b'{"version":1,"kind":"state","stream_id":"old","sequence":0,'
        b'"timestamp":0,"active_sides":["left","right"],"names":[],'
        b'"positions":[]}'
    )
    with pytest.raises(ValueError, match="schema|version"):
        decode_packet(payload)
