from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pinocchio as pin
import yaml


PICO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PICO_ROOT.parents[1]
URDF_PATH = PICO_ROOT / "assets" / "dual_fr3" / "dual_fr3_kinematics.urdf"
MJCF_PATH = PICO_ROOT / "assets" / "dual_fr3" / "scene.xml"
HARDWARE_HOME_PATH = (
    REPO_ROOT
    / "ros_ws"
    / "src"
    / "franka_fr3_arm_controllers"
    / "config"
    / "initial_pose.yaml"
)

CALIBRATED_MOUNTS = {
    "left": {
        "position": np.array([0.013079894, 0.100675805, -0.005974552]),
        "rpy": np.array([-0.78850572, 0.02334617, 0.01037711]),
    },
    "right": {
        "position": np.array([-0.013079894, -0.100675805, 0.005974552]),
        "rpy": np.array([0.76133216, 0.03792674, -0.01038175]),
    },
}


def _numbers(value: str) -> np.ndarray:
    return np.fromstring(value, sep=" ")


def test_urdf_uses_calibrated_mounts() -> None:
    root = ET.parse(URDF_PATH).getroot()
    joints = {element.attrib["name"]: element for element in root.findall("joint")}

    for side, expected in CALIBRATED_MOUNTS.items():
        origin = joints[f"{side}_mount"].find("origin")
        assert origin is not None
        np.testing.assert_allclose(
            _numbers(origin.attrib["xyz"]), expected["position"], atol=1e-12
        )
        np.testing.assert_allclose(
            _numbers(origin.attrib["rpy"]), expected["rpy"], atol=1e-12
        )


def test_mujoco_and_pinocchio_mount_frames_match() -> None:
    mujoco_model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    pinocchio_model = pin.buildModelFromUrdf(str(URDF_PATH))
    pinocchio_data = pinocchio_model.createData()
    pin.forwardKinematics(
        pinocchio_model,
        pinocchio_data,
        pin.neutral(pinocchio_model),
    )
    pin.updateFramePlacements(pinocchio_model, pinocchio_data)

    for side, expected in CALIBRATED_MOUNTS.items():
        body_id = mujoco.mj_name2id(
            mujoco_model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{side}_fr3v2_link0",
        )
        np.testing.assert_allclose(
            mujoco_model.body_pos[body_id], expected["position"], atol=1e-12
        )

        frame_id = pinocchio_model.getFrameId(f"{side}_fr3v2_link0")
        pinocchio_pose = pinocchio_data.oMf[frame_id]
        mujoco_rotation = mujoco_model.body_quat[body_id]
        mujoco_rotation = pin.Quaternion(
            mujoco_rotation[0],
            mujoco_rotation[1],
            mujoco_rotation[2],
            mujoco_rotation[3],
        ).toRotationMatrix()
        np.testing.assert_allclose(
            mujoco_rotation,
            pinocchio_pose.rotation,
            atol=1e-9,
        )


def test_mujoco_home_matches_captured_hardware_pose() -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    with HARDWARE_HOME_PATH.open(encoding="utf-8") as stream:
        captured = yaml.safe_load(stream)["initial_pose"]
    expected = np.array(
        captured["left"]["positions"] + captured["right"]["positions"]
    )
    np.testing.assert_allclose(model.key("home").qpos, expected, atol=1e-12)
