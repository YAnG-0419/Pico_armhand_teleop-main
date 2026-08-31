import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_bridge(context):
    path = LaunchConfiguration("config").perform(context)
    with open(path, "r", encoding="utf-8") as config_file:
        root = yaml.safe_load(config_file)
    if not isinstance(root, dict):
        raise ValueError(f"{path}: expected a mapping")
    expected_sections = {"udp", "host", "input"}
    sections = set(root)
    if sections != expected_sections:
        missing = sorted(expected_sections - sections)
        unknown = sorted(sections - expected_sections)
        details = []
        if missing:
            details.append(f"missing sections: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown sections: {', '.join(unknown)}")
        raise ValueError(f"{path}: {'; '.join(details)}")
    if any(not isinstance(root[section], dict) for section in expected_sections):
        raise ValueError(f"{path}: udp, host, and input must be mappings")
    udp = root["udp"]
    required = {
        "command_host",
        "command_port",
        "state_host",
        "state_port",
        "state_timeout",
    }
    fields = set(udp)
    missing = sorted(required - fields)
    unknown = sorted(fields - required)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing udp keys: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown udp keys: {', '.join(unknown)}")
        raise ValueError(f"{path}: {'; '.join(details)}")
    required_host = {
        "translation_scale",
        "rotation_scale",
        "control_rate",
        "max_joint_speed",
        "robot_state_wait_timeout",
    }
    host_fields = set(root["host"])
    missing_host = sorted(required_host - host_fields)
    unknown_host = sorted(host_fields - required_host)
    if missing_host or unknown_host:
        details = []
        if missing_host:
            details.append(f"missing host keys: {', '.join(missing_host)}")
        if unknown_host:
            details.append(f"unknown host keys: {', '.join(unknown_host)}")
        raise ValueError(f"{path}: {'; '.join(details)}")
    required_input = {
        "controllers",
        "motion_trackers",
        "hand_roots",
    }
    input_fields = set(root["input"])
    if input_fields != required_input:
        raise ValueError(
            f"{path}: input fields differ: "
            f"missing={sorted(required_input - input_fields)}, "
            f"unknown={sorted(input_fields - required_input)}"
        )
    input_config = root["input"]
    controller_fields = {
        "grip_threshold",
        "use_grip",
        "ready_timeout",
        "stale_timeout",
    }
    controllers = input_config["controllers"]
    if not isinstance(controllers, dict) or set(controllers) != controller_fields:
        actual = set(controllers) if isinstance(controllers, dict) else set()
        raise ValueError(
            f"{path}: input.controllers fields differ: "
            f"missing={sorted(controller_fields - actual)}, "
            f"unknown={sorted(actual - controller_fields)}"
        )
    motion_trackers = input_config["motion_trackers"]
    motion_fields = {
        "serials",
        "ready_timeout",
        "stale_timeout",
        "frozen_timeout",
        "max_position_jump",
        "max_rotation_jump",
        "max_linear_speed",
        "max_angular_speed",
        "tracker_to_control",
    }
    if not isinstance(motion_trackers, dict) or set(motion_trackers) != motion_fields:
        actual = set(motion_trackers) if isinstance(motion_trackers, dict) else set()
        raise ValueError(
            f"{path}: input.motion_trackers fields differ: "
            f"missing={sorted(motion_fields - actual)}, "
            f"unknown={sorted(actual - motion_fields)}"
        )
    nested_fields = {
        "serials": {"left", "right"},
        "tracker_to_control": {"left", "right"},
    }
    for name, expected in nested_fields.items():
        value = motion_trackers[name]
        if not isinstance(value, dict) or set(value) != expected:
            actual = set(value) if isinstance(value, dict) else set()
            raise ValueError(
                f"{path}: input.motion_trackers.{name} fields differ: "
                f"missing={sorted(expected - actual)}, "
                f"unknown={sorted(actual - expected)}"
            )
    transform_fields = {"translation_xyz", "quaternion_xyzw"}
    for side in ("left", "right"):
        transform = motion_trackers["tracker_to_control"][side]
        if not isinstance(transform, dict) or set(transform) != transform_fields:
            actual = set(transform) if isinstance(transform, dict) else set()
            raise ValueError(
                f"{path}: input.motion_trackers.tracker_to_control.{side} "
                "fields differ: "
                f"missing={sorted(transform_fields - actual)}, "
                f"unknown={sorted(actual - transform_fields)}"
            )
    hand_roots = input_config["hand_roots"]
    hand_root_fields = {
        "ready_timeout",
        "stale_timeout",
        "frozen_timeout",
        "max_position_jump",
        "max_rotation_jump",
        "smoothing_time_constant",
        "rotation_slow_time_constant",
        "rotation_fast_time_constant",
        "rotation_error_low",
        "rotation_error_high",
    }
    if not isinstance(hand_roots, dict) or set(hand_roots) != hand_root_fields:
        actual = set(hand_roots) if isinstance(hand_roots, dict) else set()
        raise ValueError(
            f"{path}: input.hand_roots fields differ: "
            f"missing={sorted(hand_root_fields - actual)}, "
            f"unknown={sorted(actual - hand_root_fields)}"
        )
    return [
        Node(
            package="pico_teleop_bridge",
            executable="bridge",
            output="screen",
            parameters=[
                {
                    "listen_host": udp["command_host"],
                    "command_port": udp["command_port"],
                    "feedback_host": udp["state_host"],
                    "feedback_port": udp["state_port"],
                    "state_timeout": udp["state_timeout"],
                    "source_id": LaunchConfiguration("source_id").perform(context),
                }
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config", description="Required host teleoperation configuration file"
            ),
            DeclareLaunchArgument(
                "source_id",
                default_value="pico",
                description="ArmCommand source identifier",
            ),
            OpaqueFunction(function=launch_bridge),
        ]
    )
