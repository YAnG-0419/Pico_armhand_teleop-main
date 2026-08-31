#  Copyright (c) 2025 Franka Robotics GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import xacro
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Generates the "default" nodes (controller_manager, robot_state_publisher, etc.)
# for the Franka robot. This function is called by the main launch file.
# It uses the xacro library to process the URDF file and generate the robot description.


def generate_robot_nodes(context):
    urdf_path = PathJoinSubstitution(
        [FindPackageShare("franka_description"), "robots", LaunchConfiguration("urdf_file")]
    ).perform(context)
    robot_description = xacro.process_file(
        urdf_path,
        mappings={
            "ros2_control": "true",
            "arm_id": LaunchConfiguration("arm_id").perform(context),
            "arm_prefix": LaunchConfiguration("arm_prefix").perform(context),
            "robot_ip": LaunchConfiguration("robot_ip").perform(context),
            "hand": "false",
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware").perform(context),
            "fake_sensor_commands": LaunchConfiguration("fake_sensor_commands").perform(context),
        },
    ).toprettyxml(indent="  ")

    namespace = LaunchConfiguration("namespace").perform(context)
    controller_cpus = LaunchConfiguration("controller_cpus").perform(context)
    if not controller_cpus:
        raise ValueError(f"controller_cpus is required for namespace {namespace}")
    arm_id = LaunchConfiguration("arm_id").perform(context)
    controllers_yaml = PathJoinSubstitution(
        [FindPackageShare("franka_fr3_arm_controllers"), "config", "controllers.yaml"]
    ).perform(context)

    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=namespace,
            parameters=[{"robot_description": robot_description}],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            prefix=f"taskset -c {controller_cpus}",
            namespace=namespace,
            parameters=[
                controllers_yaml,
                {"robot_description": robot_description},
                {"arm_id": arm_id},
                {"namespace": namespace},
            ],
            remappings=[("joint_states", "franka/joint_states")],
            output={
                "stdout": "screen",
                "stderr": "screen",
            },
            on_exit=Shutdown(),
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            namespace=namespace,
            arguments=["joint_state_broadcaster"],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            namespace=namespace,
            arguments=["franka_robot_state_broadcaster"],
            parameters=[{"arm_id": LaunchConfiguration("arm_id").perform(context)}],
            condition=UnlessCondition(LaunchConfiguration("use_fake_hardware")),
            output="screen",
        ),
    ]

    return nodes


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            "arm_id", description="Required robot arm type"
        ),
        DeclareLaunchArgument("arm_prefix", description="Required arm topic prefix"),
        DeclareLaunchArgument(
            "namespace", description="Required robot namespace"
        ),
        DeclareLaunchArgument(
            "controller_cpus",
            description="Host CPU set dedicated to this arm's ros2_control process",
        ),
        DeclareLaunchArgument(
            "urdf_file", description="Required path to the robot URDF"
        ),
        DeclareLaunchArgument(
            "robot_ip",
            description="Required robot hostname or IP address",
        ),
        DeclareLaunchArgument(
            "use_fake_hardware", description="Required fake-hardware mode"
        ),
        DeclareLaunchArgument(
            "fake_sensor_commands", description="Required fake-sensor command mode"
        ),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=generate_robot_nodes)])
