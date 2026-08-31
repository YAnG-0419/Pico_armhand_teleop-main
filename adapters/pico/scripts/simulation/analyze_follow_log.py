#!/usr/bin/env python3
import sys

# A sourced ROS environment leaves /opt/ros/... on PYTHONPATH; see env_guard.
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "adapters" / "pico" / "src"))

from pico_bimanual_franka_teleop.env_guard import ensure_ros_free_process  # noqa: E402

ensure_ros_free_process()

"""Attribute following deficits to the stage that loses them.

Reads a --debug-log recording and reports, per side and per engaged segment:

  raw -> tracker      optical wrist noise removed by the EMA
  tracker -> target   the mapper; must be exactly one-to-one
  target  -> EE(cmd)  IK and its joint-speed clamp; lag and amplitude loss here
                      mean the commanded motion never asked the arm to go
  EE(cmd) -> EE(meas) real-arm following through the gateway slew limit and
                      impedance controller

plus the operator's actual hand speeds, how often the commanded configuration
moves at the clamp, and how close joints come to their limits.
"""

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

SIDES = ("left", "right")
CLAMP_DEFAULT = 0.5  # keep in sync with host.max_joint_speed in config/modes/pico.yaml

# FR3 position limits in command order (left j1-7 then right j1-7); keep in
# sync with teleop_core/safety.py. The analyzer stays ROS- and URDF-free.
LOWER_LIMITS = np.array(
    [-2.9007400167, -1.8360900167, -2.9007400167, -3.0770200167,
     -2.87630335, 0.43982265, -3.05083335] * 2
)
UPPER_LIMITS = np.array(
    [2.9007400167, 1.8360900167, 2.9007400167, -0.1169370833,
     2.87630335, 4.62163335, 3.05083335] * 2
)
LIMIT_MARGIN = 0.05  # rad; keep in sync with pico_bimanual_franka_teleop.ik
IK_DEFICIT = 0.010  # m of target->EE(cmd) error that deserves a cause
PROGRESS_EPSILON = 0.0002  # m/tick; keep in sync with pico_bimanual_franka_teleop.ik


def load(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "schema" in row:
                continue
            rows.append(row)
    return rows


def segments(rows, side):
    """Contiguous stretches where the side is engaged with a target."""
    out = []
    current = []
    for row in rows:
        entry = row.get(side) or {}
        if entry.get("engaged") and entry.get("target") and entry.get("tracker"):
            current.append(row)
        elif current:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def positions(seg, side, key):
    return np.array([row[side][key]["p"] for row in seg], dtype=float)


def rotations(seg, side, key):
    return np.array([row[side][key]["r"] for row in seg], dtype=float)


def has_pose(seg, side, key):
    return all((row.get(side) or {}).get(key) is not None for row in seg)


def detrend(values):
    """Remove constant velocity so residuals expose tremor, not intended motion."""
    values = np.asarray(values, dtype=float)
    samples = np.arange(len(values), dtype=float)
    design = np.column_stack((samples, np.ones(len(samples))))
    trend = design @ np.linalg.lstsq(design, values, rcond=None)[0]
    return values - trend


def quiet_windows(seg, side, count=10, duration=1.0):
    """Select non-overlapping seconds with the least filtered-hand movement."""
    times = np.array([row["t"] for row in seg], dtype=float)
    dt = np.median(np.diff(times))
    size = max(10, int(round(duration / dt)))
    if len(seg) < size:
        return []
    tracker = positions(seg, side, "tracker")
    spans = np.array(
        [
            np.linalg.norm(np.ptp(tracker[start : start + size], axis=0))
            for start in range(len(seg) - size + 1)
        ]
    )
    selected = []
    for start in np.argsort(spans):
        if all(abs(int(start) - previous) >= size for previous in selected):
            selected.append(int(start))
        if len(selected) >= count:
            break
    return [(start, size, spans[start]) for start in sorted(selected)]


def report_quiet_jitter(seg, side):
    windows = quiet_windows(seg, side)
    if not windows:
        return
    print(
        "        quiet-window filtered-hand span range "
        f"{min(item[2] for item in windows)*1e3:.1f}-"
        f"{max(item[2] for item in windows)*1e3:.1f} mm"
    )
    for key in ("raw_tracker", "tracker", "target", "ee_cmd", "ee_meas"):
        if not has_pose(seg, side, key):
            continue
        rms = []
        span = []
        for start, size, _ in windows:
            residual = detrend(positions(seg[start : start + size], side, key))
            rms.append(np.sqrt(np.mean(np.sum(residual**2, axis=1))) * 1e3)
            span.append(np.linalg.norm(np.ptp(residual, axis=0)) * 1e3)
        print(
            f"        quiet {key:<11} detrended RMS p50/p95 "
            f"{np.median(rms):.2f}/{np.percentile(rms, 95):.2f} mm; "
            f"span {np.median(span):.2f}/{np.percentile(span, 95):.2f} mm"
        )
        angular_rms = []
        angular_span = []
        for start, size, _ in windows:
            # Within a quiet one-second window, detrending the logged rotation
            # vectors is a stable small-angle tremor estimate.
            residual = detrend(rotations(seg[start : start + size], side, key))
            angular_rms.append(
                np.sqrt(np.mean(np.sum(residual**2, axis=1))) * 1e3
            )
            angular_span.append(
                np.linalg.norm(np.ptp(residual, axis=0)) * 1e3
            )
        print(
            f"        quiet {key:<11} angular RMS p50/p95 "
            f"{np.median(angular_rms):.2f}/"
            f"{np.percentile(angular_rms, 95):.2f} mrad; "
            f"span {np.median(angular_span):.2f}/"
            f"{np.percentile(angular_span, 95):.2f} mrad"
        )
    for key in ("q_cmd", "q_meas"):
        rms = []
        span = []
        for start, size, _ in windows:
            values = np.array(
                [row[key] for row in seg[start : start + size]], dtype=float
            )
            residual = detrend(values)
            rms.append(
                np.max(np.sqrt(np.mean(residual**2, axis=0))) * 1e3
            )
            span.append(np.max(np.ptp(residual, axis=0)) * 1e3)
        print(
            f"        quiet {key:<11} worst-joint RMS p50/p95 "
            f"{np.median(rms):.2f}/{np.percentile(rms, 95):.2f} mrad; "
            f"span {np.median(span):.2f}/{np.percentile(span, 95):.2f} mrad"
        )


def attribute_ik_deficits(seg, side, clamp):
    """Classify each deficit tick by its binding constraint.

    A deficit tick has more than IK_DEFICIT of target->EE(cmd) error. The
    cause falls out of the logged series alone, so v2 logs classify the same
    way as v3: a side joint inside LIMIT_MARGIN of a position limit is a
    limit hit; the command moving at the clamp while the error shrinks is
    still catching up; anything else means the QP settled short of the
    target - out of reach. The smoothed-progress test mirrors classify_step
    in pico_bimanual_franka_teleop.ik: at a workspace edge the clamp chatters
    without net progress, so saturation alone would misread stuck as lag.
    """
    t = np.array([row["t"] for row in seg])
    target = positions(seg, side, "target")
    ee_cmd_key = "ee_cmd" if has_pose(seg, side, "ee_cmd") else "ee"
    ee_cmd = positions(seg, side, ee_cmd_key)
    q_cmd = np.array([row["q_cmd"] for row in seg], dtype=float)
    offset = 0 if side == "left" else 7
    side_q = q_cmd[:, offset:offset + 7]
    margins = np.minimum(
        side_q - LOWER_LIMITS[offset:offset + 7],
        UPPER_LIMITS[offset:offset + 7] - side_q,
    )
    near_limit = (margins < LIMIT_MARGIN).any(axis=1)
    speed = np.abs(np.diff(side_q, axis=0)) / np.diff(t)[:, None]
    saturated = np.concatenate(([False], (speed > clamp * 0.98).any(axis=1)))
    error = np.linalg.norm(ee_cmd - target, axis=1)
    progress = np.empty(len(error))
    progress[0] = np.inf
    average = None
    for index, sample in enumerate(-np.diff(error), start=1):
        average = sample if average is None else 0.8 * average + 0.2 * sample
        progress[index] = average
    deficit = error > IK_DEFICIT
    total = int(deficit.sum())
    if not total:
        print(f"        IK deficit ticks (> {IK_DEFICIT*1e3:.0f} mm): none")
        return
    limit_ticks = int((deficit & near_limit).sum())
    clamp_ticks = int(
        (deficit & ~near_limit & saturated & (progress > PROGRESS_EPSILON)).sum()
    )
    unreachable = total - limit_ticks - clamp_ticks
    print(
        f"        IK deficit ticks (> {IK_DEFICIT*1e3:.0f} mm): "
        f"{total} ({100 * total / len(seg):.1f}% of segment) -> "
        f"joint-limit {limit_ticks}, speed-clamp {clamp_ticks}, "
        f"unreachable {unreachable}"
    )
    if limit_ticks:
        worst = margins[deficit & near_limit]
        joints = ", ".join(
            f"j{index + 1} ({int((worst[:, index] < LIMIT_MARGIN).sum())} ticks)"
            for index in range(7)
            if (worst[:, index] < LIMIT_MARGIN).any()
        )
        print(f"        joints at their limit during deficits: {joints}")


def report_limit_proximity(q_cmd):
    margins = np.minimum(q_cmd - LOWER_LIMITS, UPPER_LIMITS - q_cmd)
    closest = margins.min(axis=0)
    flagged = [
        (index, closest[index], float(np.mean(margins[:, index] < LIMIT_MARGIN) * 100))
        for index in np.argsort(closest)
        if closest[index] < 2 * LIMIT_MARGIN
    ]
    if not flagged:
        print(
            "commanded joints stay clear of position limits "
            f"(closest {closest.min():.3f} rad)"
        )
        return
    for index, margin, share in flagged:
        side = "left" if index < 7 else "right"
        print(
            f"commanded {side} j{index % 7 + 1} came within {margin:.3f} rad of "
            f"its limit ({share:.1f}% of ticks inside {LIMIT_MARGIN:g} rad)"
        )


def analyze(path, clamp):
    rows = load(path)
    if not rows:
        print("empty log")
        return 1
    times = np.array([row["t"] for row in rows])
    dt = np.median(np.diff(times)) if len(times) > 1 else 0.01
    print(f"{len(rows)} ticks over {times[-1] - times[0]:.1f} s (median dt {dt*1e3:.1f} ms)")

    q_cmd = np.array([row["q_cmd"] for row in rows], dtype=float)
    q_meas = np.array([row["q_meas"] for row in rows], dtype=float)
    speed = np.abs(np.diff(q_cmd, axis=0)) / np.diff(times)[:, None]
    saturated = float(np.mean(speed.max(axis=1) > clamp * 0.98) * 100)
    print(f"commanded joint speed at the {clamp:g} rad/s clamp on {saturated:.1f}% of ticks")
    cmd_vs_meas = np.abs(q_cmd - q_meas).max(axis=1)
    print(
        f"command minus measured joints: median "
        f"{np.median(cmd_vs_meas):.3f} rad, p95 {np.percentile(cmd_vs_meas, 95):.3f} rad"
    )
    # Per-joint p99 over engaged ticks: the real arm's following lag behind
    # the command, one joint at a time. Disengaged ticks are excluded -
    # resets legitimately move the arm away from the held command.
    deviation = np.abs(q_cmd - q_meas)
    for side, offset in (("left", 0), ("right", 7)):
        engaged = np.array(
            [bool((row.get(side) or {}).get("engaged")) for row in rows]
        )
        if not engaged.any():
            print(f"per-joint |cmd - meas| p99 {side} j1-7: never engaged")
            continue
        per_joint = np.percentile(deviation[engaged, offset:offset + 7], 99, axis=0)
        joints = "/".join(f"{value:.3f}" for value in per_joint)
        print(f"per-joint engaged |cmd - meas| p99 {side} j1-7: {joints} rad")
    report_limit_proximity(q_cmd)

    for side in SIDES:
        segs = [s for s in segments(rows, side) if len(s) >= 100]
        print()
        print(f"=== {side}: {len(segs)} engaged segment(s) >= 1 s ===")
        for index, seg in enumerate(segs):
            t = np.array([row["t"] for row in seg])
            tracker = positions(seg, side, "tracker")
            target = positions(seg, side, "target")
            ee_cmd_key = "ee_cmd" if has_pose(seg, side, "ee_cmd") else "ee"
            ee_cmd = positions(seg, side, ee_cmd_key)

            hand_v = np.linalg.norm(np.diff(tracker, axis=0), axis=1) / np.diff(t)
            hand_v = hand_v[np.isfinite(hand_v)]

            # tracker -> target: subtract each stream's own start (anchors)
            tracker_delta = tracker - tracker[0]
            target_delta = target - target[0]
            mapper_err = np.linalg.norm(tracker_delta - target_delta, axis=1)

            ik_err = np.linalg.norm(ee_cmd - target, axis=1)

            tracker_span = tracker_delta.max(axis=0) - tracker_delta.min(axis=0)
            ee_span = (
                (ee_cmd - ee_cmd[0]).max(axis=0)
                - (ee_cmd - ee_cmd[0]).min(axis=0)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                transfer = np.where(tracker_span > 0.02, ee_span / tracker_span, np.nan)

            print(
                f"  seg{index}: {t[-1] - t[0]:5.1f} s  hand speed p50/p90 "
                f"{np.median(hand_v):.2f}/{np.percentile(hand_v, 90):.2f} m/s"
            )
            print(
                f"        mapper (tracker->target) error max {mapper_err.max()*1e3:.1f} mm"
                f"   <- must be ~0"
            )
            print(
                f"        IK lag (target->EE cmd)  median {np.median(ik_err)*1e3:.0f} mm"
                f"  p95 {np.percentile(ik_err, 95)*1e3:.0f} mm  max {ik_err.max()*1e3:.0f} mm"
            )
            attribute_ik_deficits(seg, side, clamp)
            print(
                f"        amplitude transfer x/y/z: "
                + "/".join("-" if not np.isfinite(r) else f"{r*100:.0f}%" for r in transfer)
            )
            if has_pose(seg, side, "raw_tracker"):
                raw = positions(seg, side, "raw_tracker")
                filter_delta = np.linalg.norm(raw - tracker, axis=1)
                raw_step = np.linalg.norm(np.diff(raw, axis=0), axis=1)
                filtered_step = np.linalg.norm(np.diff(tracker, axis=0), axis=1)
                print(
                    f"        EMA raw->filtered offset median "
                    f"{np.median(filter_delta)*1e3:.1f} mm  "
                    f"p95 {np.percentile(filter_delta, 95)*1e3:.1f} mm"
                )
                print(
                    f"        per-tick wrist step p95 raw/filtered "
                    f"{np.percentile(raw_step, 95)*1e3:.1f}/"
                    f"{np.percentile(filtered_step, 95)*1e3:.1f} mm"
                )
            if has_pose(seg, side, "ee_meas"):
                ee_meas = positions(seg, side, "ee_meas")
                following_err = np.linalg.norm(ee_meas - ee_cmd, axis=1)
                print(
                    f"        EE commanded->measured error median "
                    f"{np.median(following_err)*1e3:.1f} mm  "
                    f"p95 {np.percentile(following_err, 95)*1e3:.1f} mm  "
                    f"max {following_err.max()*1e3:.1f} mm"
                )
            report_quiet_jitter(seg, side)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--clamp", type=float, default=CLAMP_DEFAULT)
    args = parser.parse_args()
    return analyze(args.log, args.clamp)


if __name__ == "__main__":
    raise SystemExit(main())
