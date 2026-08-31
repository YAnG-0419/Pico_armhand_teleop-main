from pathlib import Path


# adapters/pico, NOT the repository root (the hand URDFs live there).
PICO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PICO_ROOT / "assets" / "dual_fr3"
URDF_PATH = ASSET_ROOT / "dual_fr3_kinematics.urdf"
MJCF_PATH = ASSET_ROOT / "scene.xml"
