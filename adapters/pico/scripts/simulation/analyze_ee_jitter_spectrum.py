#!/usr/bin/env python3
import sys

# A sourced ROS environment leaves /opt/ros/... on PYTHONPATH; see env_guard.
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "adapters" / "pico" / "src"))

from pico_bimanual_franka_teleop.env_guard import ensure_ros_free_process  # noqa: E402

ensure_ros_free_process()

"""Wrap-safe spectral and stick-slip analysis of a --debug-log recording.

Complements analyze_follow_log.py (aggregate following metrics) with the two
analyses that localized the 2026-07-26 jitter:

1. Per-stage rotation and position band RMS over the quietest windows, so
   tremor can be attributed to raw input, EMA, mapper, IK, or the real arm.
   The logged rotations are world-frame rotation vectors; near |r| ~ pi the
   log map wraps and naive detrended statistics explode. Windows are therefore
   rebuilt as cumulative per-tick geodesic increments log3(R_{t-1}^T R_t)
   before detrending.
2. Per-joint stick-slip statistics on deduplicated q_meas updates: tracking
   error percentiles, build-then-collapse slip events, and stuck/overshoot
   fractions during slow commanded ramps. The measured/commanded speed
   variability ratio near 1.0 means the arm adds no jerk beyond its command.

Reference quiet-window rotation levels from the 2026-07-26 recordings: raw
optical wrist noise ranged 7-31 mrad RMS (0.5-3 Hz) across sessions, and the
measured EE follows it with gain 0.6-0.8; treat much larger stage-to-stage
jumps as pipeline regressions, and expect session-to-session input variation.
"""

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

SIDES = ("left", "right")
BANDS = ((0.5, 3.0), (3.0, 8.0), (8.0, 25.0))
STAGES = ("raw_tracker", "tracker", "target", "ee_cmd", "ee_meas")


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
    out, current = [], []
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


def rotvec_to_matrix(rotvec):
    rotvec = np.asarray(rotvec, dtype=float)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.eye(3)
    axis = rotvec / angle
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def matrix_to_rotvec(matrix):
    angle = np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.zeros(3)
    return axis / norm * angle


def geodesic_trajectory(rotvecs):
    """Cumulative per-tick geodesic increments; wrap-free by construction."""
    matrices = [rotvec_to_matrix(value) for value in rotvecs]
    out = np.zeros((len(matrices), 3))
    for index in range(1, len(matrices)):
        out[index] = out[index - 1] + matrix_to_rotvec(
            matrices[index - 1].T @ matrices[index]
        )
    return out


def detrend(values):
    values = np.asarray(values, dtype=float)
    samples = np.arange(len(values), dtype=float)
    design = np.column_stack((samples, np.ones(len(samples))))
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def band_rms(data, rate, low, high):
    """RMS of the detrended signal within [low, high) Hz, over all columns."""
    size = len(data)
    window = np.hanning(size)
    total = 0.0
    for column in range(data.shape[1]):
        tapered = detrend(data[:, column]) * window
        power = (np.abs(np.fft.rfft(tapered)) ** 2) / (rate * (window**2).sum())
        power[1:-1] *= 2.0
        freqs = np.fft.rfftfreq(size, 1.0 / rate)
        mask = (freqs >= low) & (freqs < high)
        total += np.trapezoid(power[mask], freqs[mask])
    return np.sqrt(total)


def quiet_starts(seg, side, size, count=8):
    tracker = np.array([row[side]["tracker"]["p"] for row in seg])
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
    return sorted(selected)


def report_spectra(seg, side, rate):
    size = int(2 * rate)
    if len(seg) < size:
        print("        segment too short for spectral windows")
        return
    starts = quiet_starts(seg, side, size)
    print(
        "        quiet-window band RMS, wrap-safe rotation (mrad) "
        "and position (mm):"
    )
    header = "        stage        " + "  ".join(
        f"{low:g}-{high:g}Hz" for low, high in BANDS
    )
    print(header + "   |   " + "  ".join(f"{low:g}-{high:g}Hz" for low, high in BANDS))
    gains = {}
    for stage in STAGES:
        if any((row[side].get(stage) is None) for row in seg):
            continue
        rotation = []
        position = []
        for low, high in BANDS:
            rot = np.mean(
                [
                    band_rms(
                        geodesic_trajectory(
                            [row[side][stage]["r"] for row in seg[s : s + size]]
                        ),
                        rate,
                        low,
                        high,
                    )
                    for s in starts
                ]
            )
            pos = np.mean(
                [
                    band_rms(
                        np.array(
                            [row[side][stage]["p"] for row in seg[s : s + size]]
                        ),
                        rate,
                        low,
                        high,
                    )
                    for s in starts
                ]
            )
            rotation.append(rot * 1e3)
            position.append(pos * 1e3)
        gains[stage] = rotation
        print(
            f"        {stage:<11}"
            + "  ".join(f"{value:7.2f}" for value in rotation)
            + "   |   "
            + "  ".join(f"{value:7.2f}" for value in position)
        )
    if "ee_cmd" in gains and "ee_meas" in gains:
        print(
            "        arm rotation gain (ee_meas/ee_cmd): "
            + "  ".join(
                f"{gains['ee_meas'][i] / max(gains['ee_cmd'][i], 1e-9):.2f}"
                for i in range(len(BANDS))
            )
        )


def report_stick_slip(seg, side, rate):
    q_cmd = np.array([row["q_cmd"] for row in seg], dtype=float)
    q_meas = np.array([row["q_meas"] for row in seg], dtype=float)
    times = np.array([row["t"] for row in seg], dtype=float)
    offset = 0 if side == "left" else 7
    changed = np.concatenate(
        ([True], np.any(np.diff(q_meas, axis=0) != 0, axis=1))
    )
    index = np.where(changed)[0]
    if len(index) < 20:
        return
    measured_times = times[index]
    measured = q_meas[index][:, offset : offset + 7]
    commanded = q_cmd[index][:, offset : offset + 7]
    steps = np.diff(measured_times)
    print(
        f"        q_meas updates at ~{1.0 / np.median(steps):.0f} Hz "
        "(below ~60 Hz the high-frequency columns above are aliased)"
    )
    print(
        "        joint |q_cmd-q_meas| mrad p50/p95/max, slip events, "
        "ramp stuck/overshoot:"
    )
    for joint in range(7):
        error = commanded[:, joint] - measured[:, joint]
        magnitude = np.abs(error)
        events = 0
        k = 0
        while k < len(error) - 1:
            if magnitude[k] > 0.015:
                horizon = np.searchsorted(
                    measured_times, measured_times[k] + 0.09
                )
                if (
                    horizon > k + 1
                    and magnitude[k] - magnitude[k + 1 : horizon].min() > 0.008
                ):
                    events += 1
                    k = horizon
                    continue
            k += 1
        measured_speed = np.abs(np.diff(measured[:, joint])) / steps
        commanded_speed = np.abs(np.diff(commanded[:, joint])) / steps
        ramp = (commanded_speed > 0.02) & (commanded_speed < 0.3)
        stuck = (
            float(np.mean(measured_speed[ramp] < 0.25 * commanded_speed[ramp]))
            if ramp.any()
            else float("nan")
        )
        overshoot = (
            float(np.mean(measured_speed[ramp] > 1.5 * commanded_speed[ramp]))
            if ramp.any()
            else float("nan")
        )
        print(
            f"          j{joint + 1}: "
            f"{np.median(magnitude) * 1e3:5.1f}/"
            f"{np.percentile(magnitude, 95) * 1e3:5.1f}/"
            f"{magnitude.max() * 1e3:6.1f}   "
            f"slips {events:3d}   stuck {stuck * 100:3.0f}%  "
            f"overshoot {overshoot * 100:3.0f}%"
        )
    ee_cmd = np.array([row[side]["ee_cmd"]["p"] for row in seg])
    ee_meas = np.array([row[side]["ee_meas"]["p"] for row in seg])[index]
    commanded_speed = np.linalg.norm(np.diff(ee_cmd, axis=0), axis=1) * rate
    moving = commanded_speed > 0.03
    if moving.any():
        measured_speed = np.linalg.norm(np.diff(ee_meas, axis=0), axis=1) / steps
        moving_at_updates = moving[np.clip(index[1:] - 1, 0, len(moving) - 1)]
        vm = measured_speed[moving_at_updates]
        vc = commanded_speed[moving]
        if len(vm) and np.mean(vm) > 0:
            print(
                "        speed variability (std/mean) measured "
                f"{np.std(vm) / np.mean(vm):.2f} vs commanded "
                f"{np.std(vc) / np.mean(vc):.2f} (parity = arm adds no jerk)"
            )


def analyze(path):
    rows = load(path)
    if not rows:
        print("empty log")
        return 1
    times = np.array([row["t"] for row in rows])
    rate = 1.0 / np.median(np.diff(times)) if len(times) > 1 else 100.0
    print(f"{len(rows)} ticks over {times[-1] - times[0]:.1f} s ({rate:.1f} Hz)")
    for side in SIDES:
        usable = [
            seg
            for seg in segments(rows, side)
            if len(seg) >= 300
            and all(
                (row[side].get(key) is not None)
                for row in seg
                for key in ("raw_tracker", "ee_cmd", "ee_meas")
            )
        ]
        print(f"\n=== {side}: {len(usable)} engaged segment(s) >= 3 s ===")
        for number, seg in enumerate(usable):
            duration = seg[-1]["t"] - seg[0]["t"]
            print(f"  seg{number}: {duration:.1f} s")
            report_spectra(seg, side, rate)
            report_stick_slip(seg, side, rate)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    args = parser.parse_args()
    return analyze(args.log)


if __name__ == "__main__":
    raise SystemExit(main())
