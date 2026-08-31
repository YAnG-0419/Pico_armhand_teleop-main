from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = DeclareLaunchArgument(
        "config", description="Required teleoperation-control configuration file"
    )
    gateway = Node(
        package="teleop_core",
        executable="safety_gateway",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )
    splitter = Node(
        package="teleop_core",
        executable="joint_splitter",
        output="screen",
    )
    return LaunchDescription([config, gateway, splitter])
