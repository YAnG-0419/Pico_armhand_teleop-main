import socket
import time
import uuid

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from teleop_core.contract import (
    ARM_STATE_TOPIC,
    COMMAND_STATUS_TOPIC,
    COMMAND_JOINT_NAMES,
    SOURCE_COMMAND_TOPIC,
)
from teleop_core.joint_state import ordered_arm_positions
from teleop_core.protocol import (
    MAX_PACKET_BYTES,
    JointPacket,
    decode_packet,
    encode_packet,
)
from teleop_interfaces.msg import ArmCommand, ArmCommandStatus


class PicoTeleopBridge(Node):
    def __init__(self):
        super().__init__("pico_teleop_bridge")
        listen_host = self._required_parameter(
            "listen_host", Parameter.Type.STRING
        )
        command_port = self._required_parameter(
            "command_port", Parameter.Type.INTEGER
        )
        feedback_host = self._required_parameter(
            "feedback_host", Parameter.Type.STRING
        )
        feedback_port = self._required_parameter(
            "feedback_port", Parameter.Type.INTEGER
        )
        self.state_timeout = self._required_parameter(
            "state_timeout", Parameter.Type.DOUBLE
        )
        self.source_id = self._required_parameter(
            "source_id", Parameter.Type.STRING
        ).strip()
        if self.state_timeout <= 0:
            raise ValueError("State timeout must be positive.")
        if not self.source_id:
            raise ValueError("Source ID must be non-empty.")

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((listen_host, command_port))
        self.socket.setblocking(False)
        self.feedback_address = (feedback_host, feedback_port)
        self.state = {"left": None, "right": None}
        self.state_at = {"left": None, "right": None}
        self.state_stream_id = uuid.uuid4().hex
        self.state_sequence = 0
        self.command_stream_id = None
        self.command_sequence = -1
        self.gateway_status_sequence = -1
        self.gateway_active_sides = ()
        self.gateway_faults = ()
        self.rejected = 0

        for side in ("left", "right"):
            self.create_subscription(
                JointState,
                ARM_STATE_TOPIC.format(side=side),
                lambda message, selected=side: self._state(selected, message),
                qos_profile_sensor_data,
            )
        self.publisher = self.create_publisher(
            ArmCommand, SOURCE_COMMAND_TOPIC, 10
        )
        self.create_subscription(
            ArmCommandStatus, COMMAND_STATUS_TOPIC, self._gateway_status, 10
        )
        self.create_timer(0.01, self._tick)
        self.get_logger().info(
            f"{self.source_id} adapter listening on "
            f"udp://{listen_host}:{command_port}."
        )

    def _required_parameter(self, name, parameter_type):
        parameter = self.declare_parameter(name, parameter_type)
        if parameter.type_ == Parameter.Type.NOT_SET:
            raise ValueError(f"Required parameter '{name}' is missing")
        return parameter.value

    def _state(self, side, message):
        try:
            self.state[side] = ordered_arm_positions(
                message.name, message.position, side
            )
            self.state_at[side] = time.monotonic()
        except ValueError as exc:
            self._reject(str(exc))

    def _measured(self, now):
        if any(self.state[side] is None for side in ("left", "right")):
            return None
        if any(now - self.state_at[side] > self.state_timeout for side in ("left", "right")):
            return None
        return np.concatenate((self.state["left"], self.state["right"]))

    def _reject(self, reason):
        self.rejected += 1
        if self.rejected <= 3 or self.rejected % 100 == 0:
            self.get_logger().warn(
                f"Rejected {self.source_id} packet: {reason}"
            )

    def _send_state(self, measured, now):
        packet = JointPacket(
            kind="state",
            stream_id=self.state_stream_id,
            sequence=self.state_sequence,
            timestamp=now,
            active_sides=self.gateway_active_sides,
            names=COMMAND_JOINT_NAMES,
            positions=tuple(float(value) for value in measured),
            command_sequence=self.gateway_status_sequence,
            command_stream_id=self.command_stream_id or "",
            faults=self.gateway_faults,
        )
        self.socket.sendto(encode_packet(packet), self.feedback_address)
        self.state_sequence += 1

    def _receive_latest(self):
        latest = None
        while True:
            try:
                payload, _ = self.socket.recvfrom(MAX_PACKET_BYTES + 1)
            except BlockingIOError:
                break
            try:
                packet = decode_packet(payload)
                if packet.kind != "command":
                    continue
                if packet.stream_id != self.command_stream_id:
                    self.command_stream_id = packet.stream_id
                    self.command_sequence = -1
                    self.gateway_status_sequence = -1
                    self.gateway_active_sides = ()
                    self.gateway_faults = ()
                if packet.sequence <= self.command_sequence:
                    continue
                self.command_sequence = packet.sequence
                latest = packet
            except ValueError as exc:
                self._reject(str(exc))
        return latest

    def _gateway_status(self, message):
        if (
            message.source != self.source_id
            or message.session_id != self.command_stream_id
            or int(message.sequence) <= self.gateway_status_sequence
        ):
            return
        self.gateway_status_sequence = int(message.sequence)
        self.gateway_active_sides = tuple(message.accepted_sides)
        self.gateway_faults = tuple(message.faults)

    def _tick(self):
        now = time.monotonic()
        measured = self._measured(now)
        if measured is not None:
            self._send_state(measured, now)
        packet = self._receive_latest()
        if packet is None:
            return
        output = ArmCommand()
        output.header.stamp = self.get_clock().now().to_msg()
        output.source = self.source_id
        output.session_id = packet.stream_id
        output.sequence = packet.sequence
        output.active_sides = list(packet.active_sides)
        output.joint_names = list(packet.names)
        output.positions = list(packet.positions)
        self.publisher.publish(output)

    def destroy_node(self):
        self.socket.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PicoTeleopBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
