import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from teleop_interfaces.msg import ArmCommand, ArmCommandStatus

from .arbitration import SourceArbiter
from .contract import (
    ARM_COMMAND_TOPIC,
    ARM_STATE_TOPIC,
    COMMAND_STATUS_TOPIC,
    EXTERNAL_TORQUES_TOPIC,
    RESET_ACTIVE_TOPIC,
    SOURCE_COMMAND_TOPIC,
    VALIDATED_COMMAND_TOPIC,
)
from .joint_state import ordered_arm_positions, ordered_external_torques
from .safety import CommandSafetyGate


class SafetyGateway(Node):
    def __init__(self):
        super().__init__("teleop_safety_gateway")
        allowed_sources = self._required_parameter("allowed_sources")
        self.state_timeout = float(self._required_parameter("state_timeout"))
        command_timeout = float(self._required_parameter("command_timeout"))
        if self.state_timeout <= 0 or command_timeout <= 0:
            raise ValueError("State and command timeouts must be positive.")
        self.arbiter = SourceArbiter(
            allowed_sources, command_timeout
        )
        self.gate = CommandSafetyGate(
            max_joint_speed=float(self._required_parameter("max_joint_speed")),
            max_initial_delta=float(self._required_parameter("max_initial_delta")),
            nominal_dt=float(self._required_parameter("nominal_dt")),
            contact_torque_thresholds=self._required_parameter(
                "contact_torque_thresholds"
            ),
        )
        self.get_logger().info(
            "Contact torque gating active (Nm per joint): "
            + ", ".join(
                f"{value:.1f}" for value in self.gate.contact_torque_thresholds
            )
        )
        self.state = {"left": None, "right": None}
        self.state_at = {"left": None, "right": None}
        self.tau_ext = {"left": None, "right": None}
        self.tau_ext_at = {"left": None, "right": None}
        self.gated = {"left": (), "right": ()}
        self.rejected = 0
        self.reset_active = False

        for side in ("left", "right"):
            self.create_subscription(
                JointState,
                ARM_STATE_TOPIC.format(side=side),
                lambda message, selected=side: self._state(selected, message),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                JointState,
                EXTERNAL_TORQUES_TOPIC.format(side=side),
                lambda message, selected=side: self._external_torques(
                    selected, message
                ),
                qos_profile_sensor_data,
            )
        self.create_subscription(ArmCommand, SOURCE_COMMAND_TOPIC, self._command, 10)
        reset_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Bool, RESET_ACTIVE_TOPIC, self._reset_state, reset_qos)
        self.validated_publisher = self.create_publisher(
            JointState, VALIDATED_COMMAND_TOPIC, 10
        )
        self.hardware_publisher = self.create_publisher(
            JointState, ARM_COMMAND_TOPIC, 10
        )
        self.status_publisher = self.create_publisher(
            ArmCommandStatus, COMMAND_STATUS_TOPIC, 10
        )
        self.get_logger().info(
            "Teleoperation safety gateway is forwarding validated commands."
        )

    def _required_parameter(self, name):
        parameter = self.declare_parameter(name)
        if parameter.type_ == Parameter.Type.NOT_SET:
            raise ValueError(f"Required parameter '{name}' is missing")
        return parameter.value

    def _reset_state(self, message):
        self.reset_active = bool(message.data)
        self.arbiter.reset()
        self.gate.reset()

    def _state(self, side, message):
        try:
            self.state[side] = ordered_arm_positions(
                message.name, message.position, side
            )
            self.state_at[side] = time.monotonic()
        except ValueError as exc:
            self._reject(str(exc))

    def _external_torques(self, side, message):
        try:
            self.tau_ext[side] = ordered_external_torques(
                message.name, message.effort
            )
            self.tau_ext_at[side] = time.monotonic()
        except ValueError as exc:
            self.get_logger().warning(
                f"Ignoring {side} external torques: {exc}",
                throttle_duration_sec=5.0,
            )

    def _fresh_torques(self, now):
        """Per-side external torques, or None where missing or stale.

        Gating fails open on a missing estimate: the reflex thresholds still
        protect, while failing closed would freeze teleoperation whenever the
        broadcaster topic drops.
        """
        torques = {}
        for side in ("left", "right"):
            fresh = (
                self.tau_ext[side] is not None
                and now - self.tau_ext_at[side] <= self.state_timeout
            )
            if not fresh:
                self.get_logger().warning(
                    f"No fresh {side} external torques; contact gating is off "
                    "for that side.",
                    throttle_duration_sec=5.0,
                )
            torques[side] = self.tau_ext[side] if fresh else None
        return torques

    def _measured(self, now):
        if any(self.state[side] is None for side in ("left", "right")):
            return None
        if any(now - self.state_at[side] > self.state_timeout for side in ("left", "right")):
            return None
        return np.concatenate((self.state["left"], self.state["right"]))

    def _report_gating(self):
        """Log contact-gate transitions: direct evidence for contact trials
        (2026-07-29 the gate's engagement had to be inferred from EE
        geometry after a reflex). Transition-only, so it cannot spam."""
        for side in ("left", "right"):
            now_held = self.gate.pressing_joints[side]
            if now_held == self.gated[side]:
                continue
            if now_held:
                torques = self.tau_ext[side]
                detail = ", ".join(
                    f"j{joint}"
                    + (
                        f" {torques[joint - 1]:+.1f}Nm"
                        if torques is not None
                        else ""
                    )
                    for joint in now_held
                )
                self.get_logger().info(f"contact gate holds {side}: {detail}")
            else:
                self.get_logger().info(f"contact gate released {side}")
            self.gated[side] = now_held

    def _reject(self, reason):
        self.rejected += 1
        if self.rejected <= 3 or self.rejected % 100 == 0:
            self.get_logger().warn(f"Rejected command: {reason}")

    def _publish_status(self, message, accepted_sides=(), faults=()):
        status = ArmCommandStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.source = message.source
        status.session_id = message.session_id
        status.sequence = message.sequence
        status.accepted_sides = list(accepted_sides)
        status.faults = list(faults)
        self.status_publisher.publish(status)

    def _command(self, message):
        if self.reset_active:
            self._publish_status(message, faults=("reset is active",))
            return
        now = time.monotonic()
        measured = self._measured(now)
        if measured is None:
            self.arbiter.reset()
            self.gate.reset()
            reason = "dual-arm state is missing or stale"
            self._reject(reason)
            self._publish_status(message, faults=(reason,))
            return
        try:
            new_session = self.arbiter.accept(
                message.source,
                message.session_id,
                int(message.sequence),
                now,
            )
            if new_session:
                self.gate.reset()
            validated = self.gate.validate(
                message.active_sides,
                message.joint_names,
                message.positions,
                measured,
                now,
                external_torques=self._fresh_torques(now),
            )
        except ValueError as exc:
            self._reject(str(exc))
            self._publish_status(message, faults=(str(exc),))
            return
        for fault in self.gate.side_faults:
            self._reject(fault)
        self._report_gating()
        if validated is None:
            self.gate.reset()
            self._publish_status(message)
            return
        accepted_sides = tuple(
            side
            for side in ("left", "right")
            if any(side in name for name in validated.names)
        )
        self._publish_status(
            message,
            accepted_sides=accepted_sides,
            faults=self.gate.side_faults,
        )
        output = JointState()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = message.source
        output.name = list(validated.names)
        output.position = list(validated.positions)
        self.validated_publisher.publish(output)
        self.hardware_publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyGateway()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
