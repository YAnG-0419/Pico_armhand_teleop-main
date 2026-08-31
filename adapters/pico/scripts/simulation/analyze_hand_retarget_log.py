#!/usr/bin/env python3
"""Summarize live PICO-to-G20 thumb fidelity from a hand debug JSONL log."""

import argparse
import json

import numpy as np

SIDES = ("left", "right")


def load(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if "schema" not in row:
                rows.append(row)
    return rows


def percentile_line(label, values, scale=1.0, unit=""):
    values = np.abs(np.asarray(values, dtype=float)) * scale
    return (
        f"  {label:<31} p50 {np.percentile(values, 50):6.2f}{unit}  "
        f"p95 {np.percentile(values, 95):6.2f}{unit}  "
        f"max {values.max():6.2f}{unit}"
    )


def analyze(path):
    rows = load(path)
    if not rows:
        print("empty log")
        return 1
    print(f"{len(rows)} solved hand frames")
    for side in SIDES:
        side_rows = [row for row in rows if row["side"] == side]
        if not side_rows:
            continue
        stats = [row["stats"] for row in side_rows]
        failed = sum(not item["success"] for item in stats)
        print(f"\n=== {side}: {len(stats)} frames; optimizer failures {failed} ===")
        print(
            percentile_line(
                "thumb landmark position RMSE",
                [item["thumb_position_rmse"] for item in stats],
                1000.0,
                " mm",
            )
        )
        print(
            percentile_line(
                "thumb tip position error",
                [item["thumb_tip_error"] for item in stats],
                1000.0,
                " mm",
            )
        )
        print(
            percentile_line(
                "thumb segment direction error",
                [item["thumb_direction_error_deg"] for item in stats],
                1.0,
                " deg",
            )
        )
        print(
            percentile_line(
                "thumb bend error",
                [item["thumb_bend_error"] for item in stats],
                180.0 / np.pi,
                " deg",
            )
        )
        print(
            percentile_line(
                "thumb flex filter lag",
                [
                    item["thumb_flex_emitted"] - item["thumb_flex_target"]
                    for item in stats
                ],
                1.0,
                " rad",
            )
        )
        for finger in ("index", "middle", "ring", "pinky"):
            print(
                percentile_line(
                    f"thumb-{finger} distance error",
                    [
                        item[f"thumb_{finger}_distance_error"]
                        for item in stats
                    ],
                    1000.0,
                    " mm",
                )
            )
        activation = np.asarray(
            [item["thumb_orientation_activation"] for item in stats],
            dtype=float,
        )
        print(
            f"  orientation objectives active     "
            f"{np.mean(activation > 0.05) * 100:6.1f}% of frames"
        )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    args = parser.parse_args()
    return analyze(args.log)


if __name__ == "__main__":
    raise SystemExit(main())
