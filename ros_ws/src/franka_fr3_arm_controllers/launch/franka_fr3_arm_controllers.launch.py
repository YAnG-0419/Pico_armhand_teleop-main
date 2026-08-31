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

import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Opens the specified YAML file and loads its contents into a Python dictionary.


def load_yaml(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


WORKCELL_SIDES = {"LEFT", "RIGHT"}
WORKCELL_FIELDS = {
    "arm_id",
    "arm_prefix",
    "controller_cpus",
    "fake_sensor_commands",
    "namespace",
    "robot_ip",
    "urdf_file",
    "use_fake_hardware",
}


def validate_workcell(configs, path):
    if not isinstance(configs, dict):
        raise ValueError(f"{path}: expected a mapping")
    sides = set(configs)
    if sides != WORKCELL_SIDES:
        missing = sorted(WORKCELL_SIDES - sides)
        unknown = sorted(sides - WORKCELL_SIDES)
        details = []
        if missing:
            details.append(f"missing arms: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown arms: {', '.join(unknown)}")
        raise ValueError(f"{path}: {'; '.join(details)}")
    for side in sorted(WORKCELL_SIDES):
        config = configs[side]
        if not isinstance(config, dict):
            raise ValueError(f"{path}: {side} must be a mapping")
        fields = set(config)
        missing = sorted(WORKCELL_FIELDS - fields)
        unknown = sorted(fields - WORKCELL_FIELDS)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing keys: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown keys: {', '.join(unknown)}")
            raise ValueError(f"{path}: {side}: {'; '.join(details)}")
        for field in WORKCELL_FIELDS:
            if not isinstance(config[field], str) or not config[field].strip():
                raise ValueError(f"{path}: {side}.{field} must be a non-empty string")
        for field in ("use_fake_hardware", "fake_sensor_commands"):
            if config[field] not in {"true", "false"}:
                raise ValueError(f"{path}: {side}.{field} must be 'true' or 'false'")
    return configs


def generate_robot_nodes(context):
    config_file_name = LaunchConfiguration("robot_config_file").perform(context)
    if os.path.isabs(config_file_name):
        config_file = config_file_name
    else:
        package_config_dir = FindPackageShare(
            "franka_fr3_arm_controllers"
        ).perform(context)
        config_file = os.path.join(package_config_dir, "config", config_file_name)
    configs = validate_workcell(load_yaml(config_file), config_file)
    nodes = []
    for item_name, config in configs.items():
        namespace = config["namespace"]
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("franka_fr3_arm_controllers"),
                            "launch",
                            "franka.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "arm_id": str(config["arm_id"]),
                    "arm_prefix": str(config["arm_prefix"]),
                    "controller_cpus": str(config["controller_cpus"]),
                    "namespace": str(namespace),
                    "urdf_file": str(config["urdf_file"]),
                    "robot_ip": str(config["robot_ip"]),
                    "use_fake_hardware": str(config["use_fake_hardware"]),
                    "fake_sensor_commands": str(config["fake_sensor_commands"]),
                }.items(),
            )
        )
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                namespace=namespace,
                arguments=["joint_impedance_controller", "--controller-manager-timeout", "30"],
                parameters=[
                    PathJoinSubstitution(
                        [
                            FindPackageShare("franka_fr3_arm_controllers"),
                            "config",
                            "controllers.yaml",
                        ]
                    )
                ],
                output="screen",
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_config_file",
                description="Absolute path or package-relative robot configuration file.",
            ),
            OpaqueFunction(function=generate_robot_nodes),
        ]
    )
