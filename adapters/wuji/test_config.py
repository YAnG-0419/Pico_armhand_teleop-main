from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np
import pytest
import yaml

import adapters.wuji.pipeline as pipeline_module
from adapters.wuji.pipeline import _config_path, _device_permutation
from adapters.wuji.wuji_retargeting import Retargeter


ROOT = Path(__file__).resolve().parent
HAND2_DEVICE_PERMUTATION = [
    16, 17, 18, 19,
    0, 1, 2, 3,
    4, 5, 6, 7,
    12, 13, 14, 15,
    8, 9, 10, 11,
]


def test_hand2_configs_resolve_twenty_joint_models():
    for side, prefix in (("left", "l_"), ("right", "r_")):
        config_path = ROOT / "config" / f"retarget_manus_wuji_hand_2_{side}.yaml"
        optimizer = yaml.safe_load(config_path.read_text())["optimizer"]
        urdf_path = (config_path.parent / optimizer["urdf_path"]).resolve()
        mjcf_path = (config_path.parent / optimizer["mjcf_path"]).resolve()
        assert urdf_path.is_file()
        assert mjcf_path.is_file()
        assert "models/hand2_beta" in urdf_path.as_posix()

        urdf = ElementTree.parse(urdf_path).getroot()
        urdf_joints = {
            joint.attrib["name"]
            for joint in urdf.findall("joint")
            if joint.attrib.get("type") in {"revolute", "continuous"}
        }
        mjcf_joints = {
            joint.attrib["name"]
            for joint in ElementTree.parse(mjcf_path).getroot().findall(".//joint")
            if joint.attrib.get("name")
        }
        assert len(urdf_joints) == 20
        assert urdf_joints == mjcf_joints
        assert all(name.startswith(prefix) for name in urdf_joints)

        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        assert model.nu == 20
        assert model.njnt == 20


def test_real_hand2_entry_uses_beta_model_and_device_joint_order():
    for side in ("left", "right"):
        config_path = _config_path(side, "wuji_hand_2")
        optimizer = yaml.safe_load(config_path.read_text())["optimizer"]
        assert "models/hand2_beta" in optimizer["urdf_path"]
        assert "models/hand2_beta" in optimizer["mjcf_path"]

        retargeter = Retargeter.from_yaml(str(config_path), side)
        permutation = _device_permutation(retargeter, config_path)
        assert permutation.tolist() == HAND2_DEVICE_PERMUTATION


@pytest.mark.parametrize("side", ("left", "right"))
def test_real_hand2_tick_sends_reordered_command(monkeypatch, side):
    config_path = _config_path(side, "wuji_hand_2")
    real_retargeter = Retargeter.from_yaml(str(config_path), side)
    source_qpos = np.arange(20, dtype=np.float64)
    sent = []

    class FakeRetargeter:
        optimizer = real_retargeter.optimizer

        def retarget(self, _landmarks):
            return source_qpos.copy()

    class FakeBridge:
        def __init__(self, _library):
            pass

        def connect(self, _calibrations):
            pass

        def read(self, _side, _timeout):
            return object()

        def close(self):
            pass

    class FakeBackend:
        def __init__(self, **_kwargs):
            pass

        def send(self, command):
            sent.append(np.asarray(command).copy())

        def close(self):
            pass

    monkeypatch.setattr(
        pipeline_module.Retargeter,
        "from_yaml",
        lambda _path, _side: FakeRetargeter(),
    )
    monkeypatch.setattr(pipeline_module, "ManusBridge", FakeBridge)
    monkeypatch.setattr(
        pipeline_module, "canonical_landmarks", lambda _frame: object()
    )
    monkeypatch.setattr(pipeline_module, "WujiHand2Backend", FakeBackend)

    pipeline = pipeline_module.WujiHandPipeline(
        sides=(side,),
        addresses={side: "test-address"},
    )
    try:
        pipeline.tick(now=1.0, active={side: True})
    finally:
        pipeline.close()

    assert len(sent) == 1
    np.testing.assert_array_equal(
        sent[0], source_qpos[HAND2_DEVICE_PERMUTATION]
    )
