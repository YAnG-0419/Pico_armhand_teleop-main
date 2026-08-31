"""Host-side retargeting profiles selected independently for each hand."""

from __future__ import annotations

from pathlib import Path


REGISTERED_RETARGETING_MODELS = ("g20",)


def g20_urdf_path(assets_root: Path, side: str) -> Path:
    """Return the verified kinematic model for one G20 side.

    The physical left hand is L20 V10.1.  The right V10.1 delivery has not
    been checked against the installed right device, so keep its established
    model until that verification happens.
    """
    root = Path(assets_root)
    if side == "left":
        return (
            root
            / "linkerhand_l20_v101"
            / "linkerhand_L20_V10.1_left.urdf"
            / "linkerhand_L20v10.1_left.urdf"
        )
    if side == "right":
        return root / "linkerhand_l20" / "right" / "linkerhand_l20_right.urdf"
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def create_hand_retargeter(
    model: str,
    *,
    side: str,
    assets_root: Path,
    max_iterations: int,
):
    """Create the kinematic retargeter for one configured device model."""
    normalized = str(model).strip().lower()
    if normalized != "g20":
        raise ValueError(
            f"no retargeting profile registered for {model!r}; "
            f"registered models: {list(REGISTERED_RETARGETING_MODELS)}"
        )

    # Import lazily so arm-only runs never pay for pinocchio.
    from .hand_retarget import L20Retargeter, THUMB_OPPOSITION_YAW_ROLL

    urdf = g20_urdf_path(Path(assets_root), side)
    if not urdf.is_file():
        raise FileNotFoundError(f"Hand URDF not found: {urdf}")
    return L20Retargeter(
        urdf,
        side,
        max_iterations=max_iterations,
        thumb_opposition_fixed=THUMB_OPPOSITION_YAW_ROLL[side],
    )
