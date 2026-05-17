"""Per-rock monotonicity rules for HUD field values.

While the user is mining a single rock the panel shows three numeric
fields, each with a known evolution model:

  * ``mass``         — strictly DECREASES (the rock loses material as
    it is mined; it can never gain mass). Floor: 0.
  * ``resistance``   — CONSTANT (a rock property, not a state).
    Once locked it must not change for the lifetime of the rock.
  * ``instability``  — fluctuates freely (rises during fracturing,
    can fall back when the laser pauses). NO monotonicity constraint.

A new candidate read that violates these constraints is almost
certainly an OCR error — typical failure modes are leading-digit
gain (e.g. ``15683`` → ``215683``), a stray glyph splice
(``46`` → ``146``), or a misidentified digit shifting the magnitude.
Rejecting at this layer is cheaper than letting the bad value
contaminate the consensus buffer.

The rules below are intentionally generous (10% noise band on mass,
±2% on resistance) so that legitimate per-frame OCR jitter on the
SAME rock still passes through. The downstream consensus / lock
machinery handles the fine-grained agreement check.

These rules only fire AFTER a value is already locked AND we have at
least two locked reads — first reads always pass so the system can
warm up on a fresh rock. The just-spawned-rock case (``v_locked == 0``)
is also exempt because the previous lock was for the empty-panel
state; the freshly-scanned rock legitimately produces a non-zero
read on its first frame.

Wired into ``api._ocr_value_crop`` (and the consensus gate after it)
so the only effect is on the DISPLAYED value — OCR text itself is
untouched.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

# Tolerances. Keep these conservative — the goal is to catch the
# obvious OCR explosions (e.g. mass jumping from 15683 → 21449
# between frames), not to police every percent of jitter.
_MASS_UPWARD_TOLERANCE = 0.10   # 10% above locked is the upper bound
_RESISTANCE_ABS_TOLERANCE = 2   # ±2 (rock-property invariant)
_RESISTANCE_MIN_LOCK_COUNT = 3  # need this many locked reads first
_MIN_LOCK_COUNT = 2             # generic floor — see is_monotonically_plausible


def is_monotonically_plausible(
    field: str,
    v_new: float,
    v_locked: Optional[float],
    locked_count: int,
) -> tuple[bool, str]:
    """Decide whether ``v_new`` is compatible with the rock's history.

    Parameters
    ----------
    field:
        ``"mass"``, ``"resistance"``, or ``"instability"``. Unknown
        field names are treated as exempt (return ``True, "unknown
        field"``) so future fields don't have to register here.
    v_new:
        The fresh OCR candidate to validate.
    v_locked:
        The currently-locked / stable value for this field, or ``None``
        when no lock has been established yet. ``None`` always passes.
    locked_count:
        How many times the locked value has been confirmed (i.e. the
        size of the agreement window that produced it). The check is
        skipped on the very first read so the system can warm up;
        ``< _MIN_LOCK_COUNT`` always passes.

    Returns
    -------
    (is_valid, reason)
        ``is_valid`` is ``True`` when ``v_new`` is compatible with the
        field's monotonicity model and should propagate to the lock /
        consensus layer, ``False`` when it should be rejected. ``reason``
        is a short human-readable string used only for logging — empty
        on the success path.
    """
    # Cold-start guards. With nothing locked or only one prior read
    # we have no monotonicity baseline yet, so accept everything and
    # let the downstream consensus / lock machinery do its job.
    if v_locked is None:
        return True, ""
    if locked_count < _MIN_LOCK_COUNT:
        return True, ""

    # Numerical safety: v_locked == 0 happens when the prior rock's
    # panel showed an empty/zero state and a fresh rock has just been
    # scanned. The "old" lock should not block the legitimate first
    # non-zero read of the new rock.
    try:
        v_locked_f = float(v_locked)
        v_new_f = float(v_new)
    except (TypeError, ValueError):
        # Non-numeric input — treat as exempt so we never wedge the
        # pipeline. Callers upstream should already have validated.
        return True, ""

    if v_locked_f == 0.0:
        return True, ""

    if field == "mass":
        # Mass monotonically DECREASES while mining. A new value at or
        # below the lock is always fine (the rock is shrinking).
        # Above the lock is OCR noise unless it's within a generous
        # tolerance band (10%) — typical OCR jitter on a 5-digit mass
        # produces ±1-2 units, well inside 10%.
        ceiling = v_locked_f * (1.0 + _MASS_UPWARD_TOLERANCE)
        if v_new_f <= ceiling:
            return True, ""
        return (
            False,
            f"mass increase >{int(_MASS_UPWARD_TOLERANCE * 100)}% "
            f"(v_new={v_new_f:g} > {ceiling:g})",
        )

    if field == "resistance":
        # Resistance is a rock property — it should never change for
        # the same rock. Allow ±2 absolute units (OCR confidence noise
        # on the last digit) once we have enough confirmations that
        # the lock represents the true value.
        if locked_count < _RESISTANCE_MIN_LOCK_COUNT:
            return True, ""
        if abs(v_new_f - v_locked_f) <= _RESISTANCE_ABS_TOLERANCE:
            return True, ""
        return (
            False,
            f"resistance change >±{_RESISTANCE_ABS_TOLERANCE} "
            f"(v_new={v_new_f:g} vs v_locked={v_locked_f:g})",
        )

    if field == "instability":
        # No monotonicity constraint — it fluctuates during fracturing.
        # The existing plausibility / priors gate handles the bounds.
        return True, ""

    # Unknown field name — fail open.
    return True, "unknown field"


def _smoke_test() -> None:
    """Inline assertions exercising the rule matrix.

    Run with ``python -m ocr.sc_ocr.monotonicity`` (the ``__main__``
    block below invokes this) to verify the constants and logic stay
    in sync with the spec.
    """
    # ── mass ──
    ok, _ = is_monotonically_plausible("mass", 15500, 15683, 5)
    assert ok, "mass decrease should be valid"

    ok, _ = is_monotonically_plausible("mass", 15800, 15683, 5)
    assert ok, "mass within 10% noise should be valid (delta ~0.7%)"

    ok, why = is_monotonically_plausible("mass", 21449, 15683, 5)
    assert not ok, "mass +37% should be rejected"
    assert "mass increase" in why

    ok, _ = is_monotonically_plausible("mass", 3384, 0, 5)
    assert ok, "fresh rock from 0 should accept any value"

    # ── resistance ──
    ok, _ = is_monotonically_plausible("resistance", 46, 46, 5)
    assert ok, "resistance equal to lock should be valid"

    ok, _ = is_monotonically_plausible("resistance", 48, 46, 5)
    assert ok, "resistance within ±2 should be valid"

    ok, why = is_monotonically_plausible("resistance", 80, 46, 5)
    assert not ok, "resistance jumping by 34 should be rejected"
    assert "resistance change" in why

    # ── instability — exempt ──
    ok, _ = is_monotonically_plausible("instability", 22.58, 1.31, 5)
    assert ok, "instability is exempt from monotonicity"

    ok, _ = is_monotonically_plausible("instability", 0.0, 150.0, 5)
    assert ok, "instability dropping to 0 is also legal"

    # ── cold-start guards ──
    ok, _ = is_monotonically_plausible("mass", 21449, None, 0)
    assert ok, "no lock means accept"

    ok, _ = is_monotonically_plausible("mass", 21449, 15683, 1)
    assert ok, "locked_count<2 means accept (warmup)"

    # ── numerical safety ──
    ok, _ = is_monotonically_plausible("mass", 3384, 0.0, 5)
    assert ok, "v_locked == 0 must always pass"

    # ── unknown field ──
    ok, _ = is_monotonically_plausible("density", 999.0, 1.0, 5)
    assert ok, "unknown fields fail open"

    print("monotonicity._smoke_test: all assertions passed")


if __name__ == "__main__":
    _smoke_test()
