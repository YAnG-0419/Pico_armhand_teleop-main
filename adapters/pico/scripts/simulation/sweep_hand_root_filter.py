#!/usr/bin/env python3
"""Replay raw hand-root poses through candidate EMA time constants.

The follow-debug v2 log contains the accepted pre-EMA wrist pose on every
owner-loop tick. Repeated poses are deliberately ignored, matching HandRootInput
which advances its filter only when the optical skeleton publishes a new frame.
"""

import argparse
import json

import numpy as np
import pinocchio as pin


def load(path, side):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("schema") or not (row.get(side) or {}).get("raw_tracker"):
                continue
            rows.append(row)
    return rows


def replay(
    rows,
    side,
    translation_tau,
    rotation_tau,
    adaptive_rotation=None,
):
    positions = []
    rotations = []
    previous_raw = None
    previous_observed_at = None
    filtered_position = None
    filtered_rotation = None
    for row in rows:
        record = row[side]["raw_tracker"]
        raw_position = np.asarray(record["p"], dtype=float)
        raw_rotation = pin.exp3(np.asarray(record["r"], dtype=float))
        raw_key = tuple(record["p"]) + tuple(record["r"])
        if raw_key != previous_raw:
            now = float(row["t"])
            if filtered_position is None:
                filtered_position = raw_position.copy()
                filtered_rotation = raw_rotation.copy()
            else:
                elapsed = now - previous_observed_at
                position_alpha = 1.0 - np.exp(-elapsed / translation_tau)
                effective_rotation_tau = rotation_tau
                if adaptive_rotation is not None:
                    slow_tau, fast_tau, low_error, high_error = adaptive_rotation
                    angular_error = np.linalg.norm(
                        pin.log3(raw_rotation @ filtered_rotation.T)
                    )
                    activation = np.clip(
                        (angular_error - low_error) / (high_error - low_error),
                        0.0,
                        1.0,
                    )
                    effective_rotation_tau = (
                        slow_tau + activation * (fast_tau - slow_tau)
                    )
                rotation_alpha = 1.0 - np.exp(
                    -elapsed / effective_rotation_tau
                )
                filtered_position += position_alpha * (
                    raw_position - filtered_position
                )
                filtered_rotation = (
                    pin.exp3(
                        rotation_alpha
                        * pin.log3(raw_rotation @ filtered_rotation.T)
                    )
                    @ filtered_rotation
                )
            previous_raw = raw_key
            previous_observed_at = now
        positions.append(filtered_position.copy())
        rotations.append(filtered_rotation.copy())
    return np.asarray(positions), rotations


def detrend(values):
    values = np.asarray(values, dtype=float)
    samples = np.arange(len(values), dtype=float)
    design = np.column_stack((samples, np.ones(len(samples))))
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def quiet_windows(rows, side, count=10):
    times = np.asarray([row["t"] for row in rows], dtype=float)
    size = int(round(1.0 / np.median(np.diff(times))))
    logged = np.asarray([row[side]["tracker"]["p"] for row in rows], dtype=float)
    engaged = np.asarray([row[side]["engaged"] for row in rows], dtype=bool)
    candidates = []
    for start in range(len(rows) - size + 1):
        if not engaged[start : start + size].all():
            continue
        span = np.linalg.norm(np.ptp(logged[start : start + size], axis=0))
        candidates.append((span, start))
    selected = []
    for span, start in sorted(candidates):
        if all(abs(start - prior) >= size for _, prior in selected):
            selected.append((span, start))
        if len(selected) >= count:
            break
    return [(start, size) for _, start in sorted(selected, key=lambda item: item[1])]


def angular_residual(rotations):
    reference = rotations[0]
    vectors = np.asarray(
        [pin.log3(rotation @ reference.T) for rotation in rotations]
    )
    return detrend(vectors)


def summarize(rows, side, positions, rotations, windows):
    position_rms = []
    angular_rms = []
    for start, size in windows:
        position_delta = detrend(positions[start : start + size])
        angle_delta = angular_residual(rotations[start : start + size])
        position_rms.append(
            np.sqrt(np.mean(np.sum(position_delta**2, axis=1))) * 1e3
        )
        angular_rms.append(
            np.sqrt(np.mean(np.sum(angle_delta**2, axis=1))) * 1e3
        )

    raw_rotations = [
        pin.exp3(np.asarray(row[side]["raw_tracker"]["r"], dtype=float))
        for row in rows
    ]
    changed = [
        index
        for index in range(1, len(rows))
        if rows[index][side]["raw_tracker"]
        != rows[index - 1][side]["raw_tracker"]
    ]
    angular_offset = np.asarray(
        [
            np.linalg.norm(
                pin.log3(raw_rotations[index] @ rotations[index].T)
            )
            for index in changed
        ]
    )
    position_offset = np.asarray(
        [
            np.linalg.norm(
                np.asarray(rows[index][side]["raw_tracker"]["p"])
                - positions[index]
            )
            for index in changed
        ]
    )
    return {
        "position_rms_p50": np.median(position_rms),
        "position_rms_p95": np.percentile(position_rms, 95),
        "angular_rms_p50": np.median(angular_rms),
        "angular_rms_p95": np.percentile(angular_rms, 95),
        "position_offset_p95": np.percentile(position_offset, 95) * 1e3,
        "angular_offset_p95": np.percentile(angular_offset, 95) * 1e3,
    }


def analyze(path, side, taus):
    rows = load(path, side)
    if len(rows) < 100:
        raise ValueError("Log has fewer than 100 usable raw-pose rows")
    windows = quiet_windows(rows, side)
    if not windows:
        raise ValueError(f"No engaged one-second windows for {side}")
    print(f"{len(rows)} ticks; {len(windows)} quiet one-second windows")
    print(
        "rotation tau | quiet angular RMS p50/p95 | raw-filter angle p95 | "
        "quiet position RMS p50/p95 | raw-filter position p95"
    )
    for tau in taus:
        positions, rotations = replay(rows, side, 0.10, tau)
        stats = summarize(rows, side, positions, rotations, windows)
        print(
            f"{tau:11.3f} | "
            f"{stats['angular_rms_p50']:6.2f}/{stats['angular_rms_p95']:6.2f} mrad | "
            f"{stats['angular_offset_p95']:7.2f} mrad | "
            f"{stats['position_rms_p50']:5.2f}/{stats['position_rms_p95']:5.2f} mm | "
            f"{stats['position_offset_p95']:6.2f} mm"
        )
    print("\nadaptive: slow/fast tau, low/high angular tracking error")
    profiles = (
        (0.30, 0.075, 0.015, 0.080),
        (0.40, 0.075, 0.015, 0.080),
        (0.30, 0.050, 0.010, 0.060),
    )
    for profile in profiles:
        positions, rotations = replay(
            rows,
            side,
            0.10,
            profile[0],
            adaptive_rotation=profile,
        )
        stats = summarize(rows, side, positions, rotations, windows)
        print(
            f"{profile[0]:.3f}/{profile[1]:.3f} s, "
            f"{profile[2]*1e3:.0f}/{profile[3]*1e3:.0f} mrad | "
            f"quiet angular {stats['angular_rms_p50']:.2f}/"
            f"{stats['angular_rms_p95']:.2f} mrad | "
            f"raw-filter p95 {stats['angular_offset_p95']:.2f} mrad"
        )
    print("\nStep-response reference: 63% response=tau; 90% response=2.30*tau.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument(
        "--rotation-taus",
        type=float,
        nargs="+",
        default=(0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30),
    )
    args = parser.parse_args()
    if any(tau <= 0.0 for tau in args.rotation_taus):
        parser.error("rotation time constants must be positive")
    return analyze(args.log, args.side, args.rotation_taus)


if __name__ == "__main__":
    raise SystemExit(main())
