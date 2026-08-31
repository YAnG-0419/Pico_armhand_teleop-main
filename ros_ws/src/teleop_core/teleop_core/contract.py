SIDES = ("left", "right")

LEFT_COMMAND_JOINT_NAMES = tuple(
    f"left_fr3v2_joint{index}" for index in range(1, 8)
)
RIGHT_COMMAND_JOINT_NAMES = tuple(
    f"right_fr3v2_joint{index}" for index in range(1, 8)
)
COMMAND_JOINT_NAMES = LEFT_COMMAND_JOINT_NAMES + RIGHT_COMMAND_JOINT_NAMES
CONTROLLER_JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))

SOURCE_COMMAND_TOPIC = "/teleop/arm_commands"
COMMAND_STATUS_TOPIC = "/teleop/arm_command_status"
VALIDATED_COMMAND_TOPIC = "/teleop/validated_arm_commands"
ARM_COMMAND_TOPIC = "/target_robot/joint_commands"
ARM_STATE_TOPIC = "/{side}/franka/joint_states"
EXTERNAL_TORQUES_TOPIC = (
    "/{side}/franka_robot_state_broadcaster/external_joint_torques"
)
CONTROLLER_COMMAND_TOPIC = "/{side}/gello/joint_states"
RESET_ACTIVE_TOPIC = "/reset_to_initial_pose/active"


def command_names(active_sides):
    unknown = set(active_sides).difference(SIDES)
    if unknown or len(active_sides) != len(set(active_sides)):
        raise ValueError("Active sides must be unique values from left and right.")
    return tuple(
        name
        for side in active_sides
        for name in (
            LEFT_COMMAND_JOINT_NAMES if side == "left" else RIGHT_COMMAND_JOINT_NAMES
        )
    )
