#!/usr/bin/env python3
"""Verify PICO optical hand tracking and whether it coexists with Object Motion Tracking.

This is a read-only diagnostic. It opens one short-lived SDK client, sends
nothing to any robot, and exits on its own. Run it with the desktop
RobotLinuxDemo GUI closed and no teleoperation running.

It answers, in a single headset session, every live question the handover
leaves open:

  1. does optical hand tracking stream at all, and at what rate;
  2. do hand tracking and Object Motion Tracking stream SIMULTANEOUSLY, or
     does enabling one starve the other;
  3. what coordinate frame and unit the 26 joint poses use;
  4. whether the left and right skeletons are genuinely mirrored;
  5. what isActive actually reports, and whether a stationary hand can be
     told apart from a stale cached one.

There is no per-hand timestamp in the current binding, so hand freshness is
inferred by detecting a change in the pose array. A perfectly stationary hand
is therefore indistinguishable from a frozen one; that limitation is the point
of measurement 5 and is reported rather than hidden.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SIDES = ("left", "right")

JOINT_NAMES = (
    "palm", "wrist",
    "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip",
    "index_metacarpal", "index_proximal", "index_intermediate",
    "index_distal", "index_tip",
    "middle_metacarpal", "middle_proximal", "middle_intermediate",
    "middle_distal", "middle_tip",
    "ring_metacarpal", "ring_proximal", "ring_intermediate",
    "ring_distal", "ring_tip",
    "little_metacarpal", "little_proximal", "little_intermediate",
    "little_distal", "little_tip",
)

PALM = 0
WRIST = 1
FINGER_CHAINS = {
    "thumb": (2, 3, 4, 5),
    "index": (6, 7, 8, 9, 10),
    "middle": (11, 12, 13, 14, 15),
    "ring": (16, 17, 18, 19, 20),
    "little": (21, 22, 23, 24, 25),
}
METACARPALS = {name: chain[0] for name, chain in FINGER_CHAINS.items()}
TIPS = {name: chain[-1] for name, chain in FINGER_CHAINS.items()}


def _desktop_gui_pids() -> list[int]:
    pids = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"RobotLinuxDemo.x86_64" in command:
            pids.append(int(process.name))
    return sorted(pids)


def _positions(poses: np.ndarray) -> np.ndarray:
    return poses[:, :3]


def _has_pose(poses: np.ndarray) -> bool:
    return bool(np.any(np.linalg.norm(_positions(poses), axis=1) > 1e-9))


def _chirality(poses: np.ndarray) -> float | None:
    """Signed volume that flips sign under reflection.

    Distinguishes a left skeleton from a right one regardless of where the
    global origin sits. Left and right hands must report opposite signs.
    """
    positions = _positions(poses)
    wrist = positions[WRIST]
    index = positions[METACARPALS["index"]] - wrist
    little = positions[METACARPALS["little"]] - wrist
    thumb = positions[METACARPALS["thumb"]] - wrist
    if min(np.linalg.norm(v) for v in (index, little, thumb)) < 1e-6:
        return None
    return float(np.dot(np.cross(index, little), thumb))


def _bone_lengths(poses: np.ndarray) -> dict[str, float]:
    positions = _positions(poses)
    lengths = {}
    for finger, chain in FINGER_CHAINS.items():
        total = 0.0
        for start, end in zip(chain[:-1], chain[1:]):
            total += float(np.linalg.norm(positions[end] - positions[start]))
        lengths[finger] = total
    lengths["wrist_to_middle_metacarpal"] = float(
        np.linalg.norm(positions[METACARPALS["middle"]] - positions[WRIST])
    )
    lengths["palm_to_wrist"] = float(
        np.linalg.norm(positions[PALM] - positions[WRIST])
    )
    return lengths


def _chain_bend(poses: np.ndarray, chain: tuple[int, ...]) -> float:
    """Total unsigned angle between consecutive finger segments."""
    points = _positions(poses)[list(chain)]
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths < 1e-8):
        return float("nan")
    directions = segments / lengths[:, None]
    cosines = np.clip(
        np.sum(directions[:-1] * directions[1:], axis=1), -1.0, 1.0
    )
    return float(np.sum(np.arccos(cosines)))


class SideStats:
    def __init__(self) -> None:
        self.updates = 0
        self.last_update_at: float | None = None
        self.previous: np.ndarray | None = None
        self.active_values: dict[int, int] = {}
        self.live_seconds = 0.0
        self.longest_stall = 0.0
        self.last_valid: np.ndarray | None = None
        self.grip_extent: list[float] = []
        self.wrist_positions: list[np.ndarray] = []
        self.chirality_values: list[float] = []
        self.quaternion_norms: list[float] = []
        self.finger_bends: dict[str, list[float]] = {
            finger: [] for finger in FINGER_CHAINS
        }
        self.frames_with_pose = 0
        self.polls = 0

    def observe(self, poses: np.ndarray, is_active: int, now: float) -> bool:
        self.polls += 1
        self.active_values[is_active] = self.active_values.get(is_active, 0) + 1
        changed = self.previous is not None and not np.array_equal(self.previous, poses)
        self.previous = poses
        if changed:
            self.updates += 1
            self.last_update_at = now
        if not _has_pose(poses):
            return changed
        self.frames_with_pose += 1
        self.last_valid = poses
        positions = _positions(poses)
        self.wrist_positions.append(positions[WRIST].copy())
        extent = float(
            np.linalg.norm(positions[TIPS["index"]] - positions[WRIST])
        )
        self.grip_extent.append(extent)
        chirality = _chirality(poses)
        if chirality is not None:
            self.chirality_values.append(chirality)
        self.quaternion_norms.append(
            float(np.mean(np.linalg.norm(poses[:, 3:7], axis=1)))
        )
        for finger, chain in FINGER_CHAINS.items():
            bend = _chain_bend(poses, chain)
            if np.isfinite(bend):
                self.finger_bends[finger].append(bend)
        return changed


def _summarize_side(side: str, stats: SideStats, duration: float) -> None:
    print(f"  {side} hand")
    if stats.polls == 0:
        print("    no polls recorded")
        return
    rate = stats.updates / duration if duration > 0 else 0.0
    active = ", ".join(
        f"{value}x{count}" for value, count in sorted(stats.active_values.items())
    )
    print(f"    pose updates          : {stats.updates}  (~{rate:.1f} Hz)")
    print(f"    frames with a pose    : {stats.frames_with_pose}/{stats.polls}")
    print(f"    isActive values seen  : {active}")
    print(f"    longest stall         : {stats.longest_stall:.3f} s")
    if stats.last_valid is None:
        print("    never produced a non-zero pose")
        return

    wrists = np.asarray(stats.wrist_positions)
    print(
        f"    wrist position range  : min={np.round(wrists.min(axis=0), 3).tolist()} "
        f"max={np.round(wrists.max(axis=0), 3).tolist()}"
    )
    print(f"    wrist travel (bbox)   : {np.round(np.ptp(wrists, axis=0), 3).tolist()} m")
    lengths = _bone_lengths(stats.last_valid)
    print(
        "    bone lengths (m)      : "
        + "  ".join(f"{k}={v:.3f}" for k, v in lengths.items())
    )
    extents = np.asarray(stats.grip_extent)
    print(
        f"    index tip to wrist    : min={extents.min():.3f} max={extents.max():.3f} "
        f"span={np.ptp(extents):.3f} m"
    )
    quats = np.asarray(stats.quaternion_norms)
    print(f"    mean quaternion norm  : min={quats.min():.4f} max={quats.max():.4f}")
    for finger, samples in stats.finger_bends.items():
        if not samples:
            continue
        bends = np.asarray(samples)
        percentiles = np.percentile(bends, [5, 50, 95])
        print(
            f"    {finger:6s} bend (rad) : "
            f"p05={percentiles[0]:.3f} median={percentiles[1]:.3f} "
            f"p95={percentiles[2]:.3f} max={bends.max():.3f}"
        )
    if stats.chirality_values:
        chirality = np.asarray(stats.chirality_values)
        sign = "positive" if chirality.mean() > 0 else "negative"
        consistent = bool(np.all(chirality > 0) or np.all(chirality < 0))
        print(
            f"    chirality             : {sign} "
            f"(mean={chirality.mean():.3e}, sign consistent={consistent})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify PICO hand tracking and its coexistence with Object Motion "
            "Tracking using one SDK client. Sends nothing to any robot."
        )
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=45.0,
        help="seconds to observe before exiting (default: 45)",
    )
    parser.add_argument(
        "--stale-timeout",
        type=float,
        default=0.25,
        help=(
            "seconds without an update after which a signal counts as stale; "
            "matches input.motion_trackers.stale_timeout (default: 0.25)"
        ),
    )
    parser.add_argument(
        "--poll-rate",
        type=float,
        default=100.0,
        help="polls per second (default: 100)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="optional JSONL path for raw samples, for offline frame analysis",
    )
    parser.add_argument(
        "--log-rate",
        type=float,
        default=10.0,
        help="raw samples written per second when --log is set (default: 10)",
    )
    args = parser.parse_args()
    for name, value in (
        ("--duration", args.duration),
        ("--stale-timeout", args.stale_timeout),
        ("--poll-rate", args.poll_rate),
        ("--log-rate", args.log_rate),
    ):
        if value <= 0:
            parser.error(f"{name} must be positive")

    gui_pids = _desktop_gui_pids()
    if gui_pids:
        print(
            "Refusing to start while the desktop RobotLinuxDemo GUI is running "
            f"(PID(s): {gui_pids}). Close only the desktop GUI; keep "
            "RoboticsService and the headset app running.",
            file=sys.stderr,
        )
        return 2

    import xrobotoolkit_sdk as xrt

    print(__doc__.strip().splitlines()[0], flush=True)
    print()
    print("Operator checklist before the measurement window:", flush=True)
    print("  - headset on, app streaming", flush=True)
    print("  - Motion Tracker Mode set to Object, both wrist trackers worn", flush=True)
    print("  - hand tracking enabled, both hands inside the headset cameras", flush=True)
    print(
        "  - during the window: open and close both hands slowly, then hold "
        "both hands perfectly still for about five seconds",
        flush=True,
    )
    print()

    hands = {side: SideStats() for side in SIDES}
    motion_updates = 0
    motion_last_update_at: float | None = None
    motion_live_seconds = 0.0
    motion_longest_stall = 0.0
    motion_serials: tuple[str, ...] = ()
    motion_max_count = 0
    previous_motion_timestamp: int | None = None
    frame_timestamps: set[int] = set()

    both_live_seconds = 0.0
    only_motion_seconds = 0.0
    only_hand_seconds = 0.0
    neither_seconds = 0.0
    hand_available_seconds = 0.0
    hand_unavailable_seconds = 0.0
    both_live_when_available = 0.0

    log_file = None
    next_log_at = 0.0
    log_interval = 1.0 / args.log_rate
    poll_interval = 1.0 / args.poll_rate
    started = time.monotonic()
    elapsed = 0.0

    try:
        xrt.init()
        if args.log is not None:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log_file = args.log.open("w", encoding="utf-8")
        print("SDK initialized. Observing...", flush=True)
        started = time.monotonic()
        deadline = started + args.duration
        previous_poll = started
        next_report = started + 3.0

        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            dt = now - previous_poll
            previous_poll = now

            frame_timestamp = int(xrt.get_time_stamp_ns())
            frame_timestamps.add(frame_timestamp)

            motion_timestamp = int(xrt.get_motion_timestamp_ns())
            count = int(xrt.num_motion_data_available())
            serials = tuple(str(v) for v in xrt.get_motion_tracker_serial_numbers())
            if count > motion_max_count:
                motion_max_count = count
            if serials:
                motion_serials = serials
            if (
                motion_timestamp > 0
                and motion_timestamp != previous_motion_timestamp
            ):
                motion_updates += 1
                motion_last_update_at = now
            previous_motion_timestamp = motion_timestamp

            samples = {}
            for side in SIDES:
                getter = (
                    xrt.get_left_hand_tracking_state
                    if side == "left"
                    else xrt.get_right_hand_tracking_state
                )
                active_getter = (
                    xrt.get_left_hand_is_active
                    if side == "left"
                    else xrt.get_right_hand_is_active
                )
                poses = np.asarray(getter(), dtype=float)
                is_active = int(active_getter())
                hands[side].observe(poses, is_active, now)
                samples[side] = (poses, is_active)

            motion_live = (
                motion_last_update_at is not None
                and now - motion_last_update_at <= args.stale_timeout
            )
            hand_live = any(
                hands[side].last_update_at is not None
                and now - hands[side].last_update_at <= args.stale_timeout
                for side in SIDES
            )
            hand_available = any(samples[side][1] == 1 for side in SIDES)
            if hand_available:
                hand_available_seconds += dt
                if motion_live and hand_live:
                    both_live_when_available += dt
            else:
                hand_unavailable_seconds += dt
            if motion_live and hand_live:
                both_live_seconds += dt
            elif motion_live:
                only_motion_seconds += dt
            elif hand_live:
                only_hand_seconds += dt
            else:
                neither_seconds += dt
            if motion_live:
                motion_live_seconds += dt
            for side in SIDES:
                stats = hands[side]
                if (
                    stats.last_update_at is not None
                    and now - stats.last_update_at <= args.stale_timeout
                ):
                    stats.live_seconds += dt
                if stats.last_update_at is not None:
                    stats.longest_stall = max(
                        stats.longest_stall, now - stats.last_update_at
                    )
            if motion_last_update_at is not None:
                motion_longest_stall = max(
                    motion_longest_stall, now - motion_last_update_at
                )

            if log_file is not None and now >= next_log_at:
                next_log_at = now + log_interval
                record = {
                    "elapsed": round(now - started, 4),
                    "frame_timestamp_ns": frame_timestamp,
                    "motion_timestamp_ns": motion_timestamp,
                    "motion_count": count,
                    "motion_serials": list(serials),
                    "motion_poses": [
                        [float(v) for v in pose]
                        for pose in np.asarray(
                            xrt.get_motion_tracker_pose(), dtype=float
                        ).reshape(-1, 7)
                    ]
                    if count
                    else [],
                }
                for side in SIDES:
                    poses, is_active = samples[side]
                    record[f"{side}_is_active"] = is_active
                    record[f"{side}_joints"] = [
                        [float(v) for v in row] for row in poses
                    ]
                log_file.write(json.dumps(record, separators=(",", ":")) + "\n")

            if now >= next_report:
                next_report = now + 3.0
                remaining = deadline - now
                flags = []
                for side in SIDES:
                    live = (
                        hands[side].last_update_at is not None
                        and now - hands[side].last_update_at <= args.stale_timeout
                    )
                    flags.append(f"{side}={'LIVE' if live else 'stale'}")
                print(
                    f"  t-{remaining:4.1f}s  trackers={count} "
                    f"motion={'LIVE' if motion_live else 'stale'}  "
                    + "  ".join(flags),
                    flush=True,
                )

            time.sleep(poll_interval)

        elapsed = time.monotonic() - started
    except KeyboardInterrupt:
        print("\nStopped by operator.", flush=True)
        elapsed = time.monotonic() - started
    finally:
        if log_file is not None:
            log_file.close()
        print("\nClosing the XRoboToolkit SDK.", flush=True)
        xrt.close()
        print("SDK closed.", flush=True)

    print()
    print("=" * 70)
    print(f"OBSERVATION WINDOW: {elapsed:.1f} s")
    print("=" * 70)
    print()
    print("XR frame")
    print(f"  distinct frame timestamps : {len(frame_timestamps)}")
    print()
    print("Object Motion Tracking")
    motion_rate = motion_updates / elapsed if elapsed > 0 else 0.0
    print(f"  timestamp updates         : {motion_updates}  (~{motion_rate:.1f} Hz)")
    print(f"  trackers seen (max)       : {motion_max_count}")
    print(f"  serials                   : {list(motion_serials)}")
    print(f"  live time                 : {motion_live_seconds:.1f} s")
    print(f"  longest stall             : {motion_longest_stall:.3f} s")
    print()
    print("Optical hand tracking")
    for side in SIDES:
        _summarize_side(side, hands[side], elapsed)
    print()
    print("-" * 70)
    print("COEXISTENCE")
    print("-" * 70)
    total = both_live_seconds + only_motion_seconds + only_hand_seconds + neither_seconds

    def share(value: float) -> str:
        return f"{value:6.1f} s ({100.0 * value / total:5.1f}%)" if total > 0 else "n/a"

    print(f"  both live                 : {share(both_live_seconds)}")
    print(f"  only motion trackers live : {share(only_motion_seconds)}")
    print(f"  only hand tracking live   : {share(only_hand_seconds)}")
    print(f"  neither live              : {share(neither_seconds)}")
    print()
    print("  Hands out of the headset cameras is NOT a coexistence failure, so the")
    print("  verdict below conditions on availability (isActive==1) instead of")
    print("  counting raw wall-clock time.")
    print(f"  hand available (isActive) : {share(hand_available_seconds)}")
    print(f"  hand unavailable          : {share(hand_unavailable_seconds)}")
    conditioned = (
        both_live_when_available / hand_available_seconds
        if hand_available_seconds > 0
        else 0.0
    )
    print(
        f"  both live GIVEN available : {100.0 * conditioned:5.1f}%"
        f"   <- the real coexistence number"
    )
    print()
    print("  Competition test: motion tracking must not degrade while a hand is live.")
    print(
        f"  time hand live but motion stale : {only_hand_seconds:.2f} s"
        "   <- competition would make this large"
    )
    print()

    hand_seen = any(hands[side].updates > 0 for side in SIDES)
    motion_seen = motion_updates > 0
    both_share = conditioned

    print("VERDICT")
    if not hand_seen and not motion_seen:
        print("  NO DATA. Neither signal streamed. The headset app is not sending,")
        print("  or the PC Service connection is not established. Re-check the app")
        print("  before drawing any conclusion about coexistence.")
        verdict = 1
    elif not hand_seen:
        print("  HAND TRACKING ABSENT. Object Motion Tracking streamed but the hand")
        print("  skeleton never changed. Enable hand tracking in the headset app and")
        print("  keep both hands inside the headset cameras.")
        verdict = 1
    elif not motion_seen:
        print("  MOTION TRACKING ABSENT. Hand tracking streamed but no tracker")
        print("  timestamps arrived. Set Motion Tracker Mode to Object and confirm")
        print("  both wrist trackers are shown in the app.")
        verdict = 1
    elif hand_available_seconds < 1.0:
        print("  HAND TRACKING NEVER LOCKED. Motion tracking streamed and the hand")
        print("  arrays changed, but isActive was never 1 for a meaningful stretch, so")
        print("  there is no window in which coexistence could be judged. Confirm hand")
        print("  tracking is enabled in the headset app and that the operator keeps")
        print("  both hands inside the headset cameras during the window.")
        verdict = 1
    elif both_share >= 0.80:
        print("  COEXIST: YES. Both signals were simultaneously fresh for")
        print(f"  {100.0 * both_share:.1f}% of the window. The planned architecture holds:")
        print("  wrist trackers can drive the arms while the optical skeleton drives")
        print("  the hands, through this single SDK client.")
        verdict = 0
    elif both_share >= 0.20:
        print(
            f"  COEXIST: PARTIAL ({100.0 * both_share:.1f}% both live while "
            "a hand was available)."
        )
        print("  Out-of-view time is already excluded, so this is not simply the")
        print("  operator's hands leaving the cameras. Check the competition number")
        print("  above: if 'hand live but motion stale' is near zero the two streams")
        print("  are independent and the shortfall is hand-tracking dropout, not")
        print("  contention. Re-run holding both hands steadily in view throughout.")
        verdict = 3
    else:
        print(
            f"  COEXIST: NO ({100.0 * both_share:.1f}% both live while "
            "a hand was available)."
        )
        print("  Each signal appeared, but almost never together even after excluding")
        print("  out-of-view time, so enabling one appears to starve the other. The")
        print("  two-signal design is not viable as planned; fall back to deriving the")
        print("  arm wrist pose from the hand skeleton itself.")
        verdict = 3
    print()
    print("Handedness check: left and right chirality must have OPPOSITE signs.")
    print("Same sign means one skeleton is a mirrored copy, and the left/right")
    print("handling in retargeting must compensate.")
    if args.log is not None:
        print(f"\nRaw samples written to {args.log}")
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
