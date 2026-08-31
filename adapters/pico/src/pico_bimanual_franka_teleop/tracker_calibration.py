"""Assign motion-tracker serials to sides from a move-one-hand protocol.

The operator moves only the left hand while the right stays still, then
only the right hand. Whichever tracker dominates the motion in a phase is
that phase's side - no serial numbers to read off labels or remember.

Pure logic only; the SDK-facing recording loop lives in
scripts/hardware/calibrate_tracker_sides.py.
"""

from __future__ import annotations

import re
from collections.abc import Mapping


# A deliberate hand wave travels tens of centimetres; a "still" hand
# resting on a knee wobbles millimetres. The dominance ratio guards
# against the operator moving both hands.
MIN_MOTION = 0.10  # m of peak displacement required from the moving hand
MIN_DOMINANCE = 3.0  # moving hand must exceed every other tracker by this


class CalibrationError(ValueError):
    pass


def _phase_winner(displacements: Mapping[str, float], phase: str) -> str:
    if len(displacements) < 2:
        raise CalibrationError(
            f"{phase} phase saw {len(displacements)} tracker(s); "
            "two are required."
        )
    ranked = sorted(
        displacements.items(), key=lambda item: item[1], reverse=True
    )
    (winner, moved), (_, runner_up) = ranked[0], ranked[1]
    if moved < MIN_MOTION:
        raise CalibrationError(
            f"{phase} phase: nothing moved enough "
            f"(best {moved * 100:.0f} cm, need {MIN_MOTION * 100:.0f} cm) - "
            "wave the hand more deliberately."
        )
    if moved < MIN_DOMINANCE * max(runner_up, 1e-9):
        raise CalibrationError(
            f"{phase} phase: motion is ambiguous "
            f"({moved * 100:.0f} cm vs {runner_up * 100:.0f} cm) - "
            "keep the other hand still and repeat."
        )
    return winner


def assign_sides(
    left_phase: Mapping[str, float], right_phase: Mapping[str, float]
) -> dict[str, str]:
    """Map each side to a serial from per-phase peak displacements (m)."""
    left = _phase_winner(left_phase, "left")
    right = _phase_winner(right_phase, "right")
    if left == right:
        raise CalibrationError(
            f"Both phases were won by the same tracker ({left}) - "
            "the same hand moved twice; repeat the protocol."
        )
    return {"left": left, "right": right}


def replace_serials(config_text: str, serials: Mapping[str, str]) -> str:
    """Rewrite the motion_trackers serial entries in a pico.yaml text.

    Textual, not a YAML round-trip, so every comment and the formatting
    survive. Only the two lines directly inside the `serials:` block are
    touched; `left:`/`right:` keys elsewhere (tracker_to_control) are not.
    """
    lines = config_text.splitlines(keepends=True)
    output = []
    in_serials = False
    serials_indent = None
    replaced = set()
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if in_serials:
            if stripped and indent <= serials_indent:
                in_serials = False
            else:
                match = re.match(r"^(\s*)(left|right):\s*\S+\s*$", line)
                if match:
                    side = match.group(2)
                    output.append(f"{match.group(1)}{side}: {serials[side]}\n")
                    replaced.add(side)
                    continue
        if stripped == "serials:":
            in_serials = True
            serials_indent = indent
        output.append(line)
    if replaced != {"left", "right"}:
        raise CalibrationError(
            "Could not find both serial entries under a 'serials:' block; "
            "config layout changed?"
        )
    return "".join(output)


def replace_device_env_serials(config_text: str, serials: Mapping[str, str]) -> str:
    """Rewrite only the tracked defaults in ``config/devices.env``."""
    output = config_text
    replaced = set()
    for side in ("left", "right"):
        variable = f"PICO_{side.upper()}_TRACKER_SERIAL"
        pattern = re.compile(
            rf'^({variable}="\$\{{{variable}:-)[^}}]+(\}}")$',
            re.MULTILINE,
        )
        output, count = pattern.subn(
            rf"\g<1>{serials[side]}\g<2>", output, count=1
        )
        if count == 1:
            replaced.add(side)
    if replaced != {"left", "right"}:
        raise CalibrationError(
            "Could not find both PICO tracker defaults in devices.env; "
            "config layout changed?"
        )
    return output
