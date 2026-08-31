#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from franka_msgs.srv import SetFullCollisionBehavior


# The srv uses fixed-size arrays, so every unset field is transmitted as
# zeros. An earlier version of this script filled only the *_nominal fields;
# had it ever run, the zero acceleration thresholds would have tripped the
# reflex on any accelerating motion. All eight fields must be set together.
#
# Teleop streams commands continuously, so the robot rarely reports a steady
# "nominal" phase; the acceleration thresholds are the ones that gate most of
# a session. Using one set of values for both phases makes the behavior
# uniform and predictable.
LOWER_TORQUE = [40.0, 40.0, 36.0, 36.0, 32.0, 28.0, 24.0]
UPPER_TORQUE = [60.0, 60.0, 50.0, 50.0, 45.0, 40.0, 35.0]
LOWER_FORCE = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
UPPER_FORCE = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]


class CollisionBehaviorSetter(Node):
    def __init__(self):
        super().__init__('collision_behavior_setter')
        self.left_client = self.create_client(
            SetFullCollisionBehavior,
            '/left/service_server/set_full_collision_behavior')
        self.right_client = self.create_client(
            SetFullCollisionBehavior,
            '/right/service_server/set_full_collision_behavior')
        self.set_behaviors()

    def call_service(self, client, arm_name):
        # Launched concurrently with the hardware bringup, so the service can
        # take a while to appear while the arms connect.
        if not client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError(f'{arm_name} collision-behavior service is unavailable.')

        request = SetFullCollisionBehavior.Request()
        request.lower_torque_thresholds_acceleration = LOWER_TORQUE
        request.upper_torque_thresholds_acceleration = UPPER_TORQUE
        request.lower_torque_thresholds_nominal = LOWER_TORQUE
        request.upper_torque_thresholds_nominal = UPPER_TORQUE
        request.lower_force_thresholds_acceleration = LOWER_FORCE
        request.upper_force_thresholds_acceleration = UPPER_FORCE
        request.lower_force_thresholds_nominal = LOWER_FORCE
        request.upper_force_thresholds_nominal = UPPER_FORCE

        self.get_logger().info(
            f'Setting {arm_name} collision thresholds: '
            f'torque {LOWER_TORQUE}/{UPPER_TORQUE} Nm, '
            f'force {LOWER_FORCE}/{UPPER_FORCE} N (acceleration = nominal).')
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            raise RuntimeError(f'Setting {arm_name} collision thresholds failed.')
        result = future.result()
        # The service reports libfranka command rejections (for example while
        # a control loop is active) in the response instead of raising.
        if not result.success:
            raise RuntimeError(
                f'{arm_name} arm rejected the collision thresholds: {result.error}')
        self.get_logger().info(f'{arm_name} arm accepted the collision thresholds.')

    def set_behaviors(self):
        self.call_service(self.left_client, 'left')
        self.call_service(self.right_client, 'right')


def main():
    rclpy.init()
    node = CollisionBehaviorSetter()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
