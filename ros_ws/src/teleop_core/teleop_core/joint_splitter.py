import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .contract import (
    ARM_COMMAND_TOPIC,
    CONTROLLER_COMMAND_TOPIC,
    CONTROLLER_JOINT_NAMES,
    LEFT_COMMAND_JOINT_NAMES,
    RIGHT_COMMAND_JOINT_NAMES,
)
from .joint_state import ordered_joint_positions


class JointSplitterNode(Node):
    def __init__(self):
        super().__init__("joint_splitter")
        self.subscription = self.create_subscription(
            JointState, ARM_COMMAND_TOPIC, self._command, 10
        )
        self.command_publishers = {
            side: self.create_publisher(
                JointState, CONTROLLER_COMMAND_TOPIC.format(side=side), 10
            )
            for side in ("left", "right")
        }

    def _command(self, message):
        names_by_side = {
            "left": LEFT_COMMAND_JOINT_NAMES,
            "right": RIGHT_COMMAND_JOINT_NAMES,
        }
        for side, required_names in names_by_side.items():
            positions = ordered_joint_positions(
                message.name, message.position, required_names
            )
            if positions is None:
                continue
            output = JointState()
            output.header.stamp = message.header.stamp
            output.header.frame_id = "fr3_link0"
            output.name = list(CONTROLLER_JOINT_NAMES)
            output.position = positions
            self.command_publishers[side].publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = JointSplitterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
