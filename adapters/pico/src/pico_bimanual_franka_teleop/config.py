import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


def _exact_mapping(value, fields: set[str], label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        raise ValueError(f"{label} fields differ: missing={missing}, unknown={unknown}")
    return value


def _positive(value, label: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


@dataclass(frozen=True)
class UdpConfig:
    command_host: str
    command_port: int
    state_host: str
    state_port: int
    state_timeout: float


@dataclass(frozen=True)
class HostConfig:
    translation_scale: float
    rotation_scale: float
    control_rate: float
    max_joint_speed: float
    robot_state_wait_timeout: float


@dataclass(frozen=True)
class ControllerConfig:
    grip_threshold: float
    use_grip: bool
    ready_timeout: float
    stale_timeout: float


@dataclass(frozen=True)
class MotionTrackerConfig:
    serials: dict[str, str]
    tracker_to_control: dict[str, dict]
    ready_timeout: float
    stale_timeout: float
    frozen_timeout: float
    max_position_jump: float
    max_rotation_jump: float
    max_linear_speed: float
    max_angular_speed: float


@dataclass(frozen=True)
class HandRootConfig:
    ready_timeout: float
    stale_timeout: float
    frozen_timeout: float
    max_position_jump: float
    max_rotation_jump: float
    smoothing_time_constant: float
    rotation_slow_time_constant: float
    rotation_fast_time_constant: float
    rotation_error_low: float
    rotation_error_high: float


@dataclass(frozen=True)
class InputConfig:
    controllers: ControllerConfig
    motion_trackers: MotionTrackerConfig
    hand_roots: HandRootConfig


@dataclass(frozen=True)
class PicoConfig:
    udp: UdpConfig
    host: HostConfig
    input: InputConfig


def _load_udp(raw) -> UdpConfig:
    udp = _exact_mapping(
        raw,
        {
            "command_host",
            "command_port",
            "state_host",
            "state_port",
            "state_timeout",
        },
        "udp",
    )
    command_port = int(udp["command_port"])
    state_port = int(udp["state_port"])
    if not 1 <= command_port <= 65535 or not 1 <= state_port <= 65535:
        raise ValueError("UDP ports must be between 1 and 65535")
    command_host = str(udp["command_host"]).strip()
    state_host = str(udp["state_host"]).strip()
    if not command_host or not state_host:
        raise ValueError("UDP hosts must be non-empty")
    return UdpConfig(
        command_host=command_host,
        command_port=command_port,
        state_host=state_host,
        state_port=state_port,
        state_timeout=_positive(udp["state_timeout"], "udp.state_timeout"),
    )


def _load_host(raw) -> HostConfig:
    host = _exact_mapping(
        raw,
        {
            "translation_scale",
            "rotation_scale",
            "control_rate",
            "max_joint_speed",
            "robot_state_wait_timeout",
        },
        "host",
    )
    return HostConfig(
        translation_scale=_positive(
            host["translation_scale"], "host.translation_scale"
        ),
        rotation_scale=_positive(host["rotation_scale"], "host.rotation_scale"),
        control_rate=_positive(host["control_rate"], "host.control_rate"),
        max_joint_speed=_positive(host["max_joint_speed"], "host.max_joint_speed"),
        robot_state_wait_timeout=_positive(
            host["robot_state_wait_timeout"],
            "host.robot_state_wait_timeout",
        ),
    )


def _load_controllers(raw) -> ControllerConfig:
    controllers = _exact_mapping(
        raw,
        {"grip_threshold", "use_grip", "ready_timeout", "stale_timeout"},
        "input.controllers",
    )
    grip_threshold = float(controllers["grip_threshold"])
    if not 0 < grip_threshold <= 1:
        raise ValueError("input.controllers.grip_threshold must be in (0, 1]")
    use_grip = controllers["use_grip"]
    if not isinstance(use_grip, bool):
        raise ValueError("input.controllers.use_grip must be a boolean")
    return ControllerConfig(
        grip_threshold=grip_threshold,
        use_grip=use_grip,
        ready_timeout=_positive(
            controllers["ready_timeout"], "input.controllers.ready_timeout"
        ),
        stale_timeout=_positive(
            controllers["stale_timeout"], "input.controllers.stale_timeout"
        ),
    )


def _load_tracker_transforms(raw) -> dict[str, dict]:
    section = "input.motion_trackers"
    transforms = _exact_mapping(
        raw,
        {"left", "right"},
        f"{section}.tracker_to_control",
    )
    for side in ("left", "right"):
        transform = _exact_mapping(
            transforms[side],
            {"translation_xyz", "quaternion_xyzw"},
            f"{section}.tracker_to_control.{side}",
        )
        translation = np.asarray(transform["translation_xyz"], dtype=float)
        quaternion = np.asarray(transform["quaternion_xyzw"], dtype=float)
        if translation.shape != (3,) or quaternion.shape != (4,):
            raise ValueError(f"{side} tracker transform dimensions must be 3 and 4")
        if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)):
            raise ValueError(f"{side} tracker transform must be finite")
        if np.linalg.norm(quaternion) <= 1e-8:
            raise ValueError(f"{side} tracker quaternion must be non-zero")
    return transforms


def _load_motion_trackers(raw) -> MotionTrackerConfig:
    section = "input.motion_trackers"
    trackers = _exact_mapping(
        raw,
        {
            "serials",
            "ready_timeout",
            "stale_timeout",
            "frozen_timeout",
            "max_position_jump",
            "max_rotation_jump",
            "max_linear_speed",
            "max_angular_speed",
            "tracker_to_control",
        },
        section,
    )
    serials = _exact_mapping(
        trackers["serials"],
        {"left", "right"},
        f"{section}.serials",
    )
    serials = {side: str(serials[side]).strip() for side in ("left", "right")}
    if any(not serial for serial in serials.values()):
        raise ValueError("Both motion tracker serials must be non-empty")
    if serials["left"] == serials["right"]:
        raise ValueError("Left and right motion tracker serials must differ")

    return MotionTrackerConfig(
        serials=serials,
        tracker_to_control=_load_tracker_transforms(
            trackers["tracker_to_control"]
        ),
        ready_timeout=_positive(
            trackers["ready_timeout"], f"{section}.ready_timeout"
        ),
        stale_timeout=_positive(
            trackers["stale_timeout"], f"{section}.stale_timeout"
        ),
        frozen_timeout=_positive(
            trackers["frozen_timeout"],
            f"{section}.frozen_timeout",
        ),
        max_position_jump=_positive(
            trackers["max_position_jump"],
            f"{section}.max_position_jump",
        ),
        max_rotation_jump=_positive(
            trackers["max_rotation_jump"],
            f"{section}.max_rotation_jump",
        ),
        max_linear_speed=_positive(
            trackers["max_linear_speed"],
            f"{section}.max_linear_speed",
        ),
        max_angular_speed=_positive(
            trackers["max_angular_speed"],
            f"{section}.max_angular_speed",
        ),
    )


def _load_hand_roots(raw) -> HandRootConfig:
    section = "input.hand_roots"
    hand_roots = _exact_mapping(
        raw,
        {
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
        },
        section,
    )
    rotation_slow = _positive(
        hand_roots["rotation_slow_time_constant"],
        f"{section}.rotation_slow_time_constant",
    )
    rotation_fast = _positive(
        hand_roots["rotation_fast_time_constant"],
        f"{section}.rotation_fast_time_constant",
    )
    rotation_error_low = _positive(
        hand_roots["rotation_error_low"],
        f"{section}.rotation_error_low",
    )
    rotation_error_high = _positive(
        hand_roots["rotation_error_high"],
        f"{section}.rotation_error_high",
    )
    if rotation_slow < rotation_fast:
        raise ValueError(
            f"{section}.rotation_slow_time_constant must be at least "
            "rotation_fast_time_constant"
        )
    if rotation_error_high <= rotation_error_low:
        raise ValueError(
            f"{section}.rotation_error_high must exceed rotation_error_low"
        )
    return HandRootConfig(
        ready_timeout=_positive(
            hand_roots["ready_timeout"], f"{section}.ready_timeout"
        ),
        stale_timeout=_positive(
            hand_roots["stale_timeout"], f"{section}.stale_timeout"
        ),
        frozen_timeout=_positive(
            hand_roots["frozen_timeout"], f"{section}.frozen_timeout"
        ),
        max_position_jump=_positive(
            hand_roots["max_position_jump"], f"{section}.max_position_jump"
        ),
        max_rotation_jump=_positive(
            hand_roots["max_rotation_jump"], f"{section}.max_rotation_jump"
        ),
        smoothing_time_constant=_positive(
            hand_roots["smoothing_time_constant"],
            f"{section}.smoothing_time_constant",
        ),
        rotation_slow_time_constant=rotation_slow,
        rotation_fast_time_constant=rotation_fast,
        rotation_error_low=rotation_error_low,
        rotation_error_high=rotation_error_high,
    )


def load_config(path) -> PicoConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        root = yaml.safe_load(stream)
    root = _exact_mapping(root, {"udp", "host", "input"}, str(config_path))
    input_raw = _exact_mapping(
        root["input"],
        {"controllers", "motion_trackers", "hand_roots"},
        "input",
    )
    tracker_serials = input_raw["motion_trackers"]["serials"]
    for side in ("left", "right"):
        override = os.environ.get(f"PICO_{side.upper()}_TRACKER_SERIAL", "").strip()
        if override:
            tracker_serials[side] = override
    return PicoConfig(
        udp=_load_udp(root["udp"]),
        host=_load_host(root["host"]),
        input=InputConfig(
            controllers=_load_controllers(input_raw["controllers"]),
            motion_trackers=_load_motion_trackers(input_raw["motion_trackers"]),
            hand_roots=_load_hand_roots(input_raw["hand_roots"]),
        ),
    )
