"""MuJoCo-only Wuji Hand 2 simulation.

This entry point never connects to a Wuji hand or publishes a hardware command.
It accepts the bundled replay by default and can optionally read one live MANUS
glove through the existing native bridge.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Iterator

import mujoco
import numpy as np
import yaml

from adapters.wuji.wuji_retargeting import Retargeter


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
DEFAULT_REPLAY = ROOT / "sim_data" / "avp1.pkl"


def config_path(side: str) -> Path:
    return CONFIG_DIR / f"retarget_manus_wuji_hand_2_{side}.yaml"


def model_path(config: Path) -> Path:
    content = yaml.safe_load(config.read_text()) or {}
    relative = (content.get("optimizer") or {}).get("mjcf_path")
    if not relative:
        raise ValueError(f"optimizer.mjcf_path is missing from {config}")
    resolved = (config.parent / relative).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"MuJoCo model is missing: {resolved}")
    return resolved


def actuator_permutation(
    retargeter: Retargeter, model: mujoco.MjModel
) -> np.ndarray:
    source_names = list(retargeter.optimizer.robot.dof_joint_names)
    destination_names = [
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            int(model.actuator_trnid[index, 0]),
        )
        for index in range(model.nu)
    ]
    source_index = {name: index for index, name in enumerate(source_names)}
    if None in destination_names or set(destination_names) != set(source_names):
        raise ValueError("URDF and MJCF joint names do not describe the same hand")
    return np.asarray(
        [source_index[name] for name in destination_names], dtype=int
    )


def replay_frames(path: Path, side: str) -> Iterator[np.ndarray]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"replay is missing: {path}")
    with path.open("rb") as stream:
        rows = pickle.load(stream)  # trusted, repository-bundled fixture
    key = f"{side}_fingers"
    valid = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        frame = np.asarray(value, dtype=np.float64)
        if frame.shape == (21, 3) and np.isfinite(frame).all():
            valid.append(frame)
    if not valid:
        raise ValueError(f"replay contains no valid {side} hand frames")
    while True:
        yield from valid


def manus_frames(side: str) -> Iterator[np.ndarray]:
    from manus_teleop.skeleton import ManusBridge, canonical_landmarks

    library = ROOT.parent / "manus" / "build" / "libmanus_skeleton_bridge.so"
    calibrations = ROOT.parent / "manus" / "config"
    if not library.is_file():
        raise FileNotFoundError(
            f"MANUS bridge is missing: {library}; run adapters/manus/scripts/build.sh"
        )
    bridge = ManusBridge(library)
    bridge.connect(calibrations)
    try:
        while True:
            frame = bridge.read(side, 0.1)
            if frame is not None:
                yield canonical_landmarks(frame)
    finally:
        bridge.close()


def run(args: argparse.Namespace) -> None:
    config = config_path(args.side)
    retargeter = Retargeter.from_yaml(str(config), args.side)
    mjcf_path = model_path(config)
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    permutation = actuator_permutation(retargeter, model)

    frames = (
        replay_frames(args.replay, args.side)
        if args.input == "replay"
        else manus_frames(args.side)
    )
    maximum_frames = args.frames or (300 if args.headless else None)
    steps_per_frame = max(1, round(1.0 / (args.fps * model.opt.timestep)))

    viewer = None
    if not args.headless:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20
        viewer.cam.distance = 0.5
        viewer.cam.lookat[:] = [0.0, 0.0, 0.05]

    print("Wuji simulation (hardware output disabled)")
    print(f"  model: hand2_beta ({mjcf_path})")
    print(f"  input: {args.input}")
    print(f"  side: {args.side}")
    started = time.monotonic()
    processed = 0
    try:
        for landmarks in frames:
            tick_started = time.monotonic()
            qpos = retargeter.retarget(landmarks)
            data.ctrl[:] = qpos[permutation]
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            processed += 1
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                remaining = 1.0 / args.fps - (time.monotonic() - tick_started)
                if remaining > 0.0:
                    time.sleep(remaining)
            if maximum_frames is not None and processed >= maximum_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        close_frames = getattr(frames, "close", None)
        if callable(close_frames):
            close_frames()

    elapsed = max(time.monotonic() - started, 1e-9)
    print(f"processed {processed} frames at {processed / elapsed:.1f} frame/s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--input", choices=("replay", "manus"), default="replay"
    )
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="stop after N frames; 0 means viewer-controlled (300 headless)",
    )
    parser.add_argument(
        "--headless", action="store_true", help="run without opening a viewer"
    )
    args = parser.parse_args()
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.frames < 0:
        parser.error("--frames cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
