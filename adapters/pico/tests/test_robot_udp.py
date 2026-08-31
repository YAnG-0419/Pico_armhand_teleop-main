import socket

from pico_bimanual_franka_teleop.robot_udp import UdpRobotBackend
from teleop_core.contract import COMMAND_JOINT_NAMES
from teleop_core.protocol import JointPacket, encode_packet


def state_packet(sequence, command_stream_id, command_sequence, faults=()):
    return JointPacket(
        kind="state",
        stream_id="bridge",
        sequence=sequence,
        timestamp=float(sequence),
        active_sides=(),
        names=COMMAND_JOINT_NAMES,
        positions=(0.0,) * 14,
        command_sequence=command_sequence,
        command_stream_id=command_stream_id,
        faults=tuple(faults),
    )


def test_gateway_faults_are_stream_scoped_and_not_lost_while_draining():
    backend = UdpRobotBackend(
        command_host="127.0.0.1",
        command_port=9,
        state_host="127.0.0.1",
        state_port=0,
        state_timeout=1.0,
    )
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    address = backend.socket.getsockname()
    try:
        packets = (
            state_packet(0, "previous-host", 50, ("stale fault",)),
            state_packet(
                1,
                backend.stream_id,
                0,
                ("right first target is too far from measured state.",),
            ),
            # A later healthy acknowledgement must not erase a fault that the
            # control loop has not consumed yet.
            state_packet(2, backend.stream_id, 1),
        )
        for packet in packets:
            sender.sendto(encode_packet(packet), address)
        assert backend.receive_state() is not None
        assert backend.take_gateway_faults() == (
            "right first target is too far from measured state.",
        )
        assert backend.take_gateway_faults() == ()
    finally:
        sender.close()
        backend.close()
