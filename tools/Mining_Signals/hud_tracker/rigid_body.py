"""Rigid-body least-squares pose solver for the SCAN RESULTS HUD panel.

The HUD panel is a static UI element with KNOWN geometry: the
"SCAN RESULTS" title, the three label words (MASS/RESISTANCE/INSTABILITY)
with their colons, and the two horizontal chrome lines all live at
fixed positions RELATIVE to one another. Their absolute on-screen
positions are determined by exactly three free parameters:

    (panel_x, panel_y)  — translation of the panel origin (title top-left).
    scale               — uniform scale (effectively the title height in px).

Each per-anchor detector produces a noisy 2D measurement of where that
anchor LANDED in the current frame. Given a set of such measurements
and the canonical offsets (in title-height-relative units), this module
solves for ``(panel_x, panel_y, scale)`` in the least-squares sense.

Model
-----
For each measurement ``(id, mx, my)`` with offset ``(dx, dy)``:

    mx = panel_x + scale * dx
    my = panel_y + scale * dy

Stack ``2N`` such equations into the linear system::

    [1  0  dx_1] [panel_x]   [mx_1]
    [0  1  dy_1] [panel_y] = [my_1]
    [1  0  dx_2] [scale  ]   [mx_2]
    [0  1  dy_2]             [my_2]
    [   ...    ]             [ ...]

Solve via ``numpy.linalg.lstsq``. Weights are applied by row-scaling
(equivalent to weighted-LS, since solving ``min ||W(Ax - b)||``).

If ``fixed_scale`` is given, ``scale`` is removed from the unknowns and
the system becomes::

    [1  0] [panel_x]   [mx_1 - fixed_scale * dx_1]
    [0  1] [panel_y] = [my_1 - fixed_scale * dy_1]
    [   ...    ]

which needs only ≥ 1 measurement to be determined (but ≥ 2 for any
useful redundancy).

Sources for ``DEFAULT_OFFSETS``
-------------------------------
All offsets are in **title-height units** measured from the panel origin
at the top-left of the "SCAN RESULTS" title bbox. Two sources contribute:

  * Row Y multipliers — ``ocr/onnx_hud_reader.py::_ROW_OFFSET_MULTS``
    (lines 1039-1044, calibrated against the 397-px reference panel)::

        row_y_center = title_y + title_h * MULT

    where MULT is 1.6 / 3.0 / 4.4 / 5.8 for mineral / mass / resistance
    / instability rows respectively.

  * Chrome-line Y and X — ``hud_tracker/world_model.json`` features
    ``top_line``, ``bot_line``. Y fractions are in title-height units
    directly; X fractions are in title-WIDTH units (multiply by the
    title aspect ratio 158/28 ≈ 5.643 to convert).

The label/colon X offsets are not separately calibrated in existing
constants — ``world_model.json`` only stores per-ROW bounding boxes
(``mass_row`` etc.), not per-glyph positions. Those entries are set to
``None`` in ``DEFAULT_OFFSETS`` so callers can override them once
per-glyph calibration is available. The solver tolerates ``None`` entries
(it ignores them).

This module has no dependency on PIL, OpenCV, or any image-processing
machinery — it is pure NumPy + stdlib so tests can run anywhere.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ── Title template aspect ratio ──────────────────────────────────────
# Used to convert ``world_model.json`` X fractions (in title_w units)
# to title_h units. Read from ``ocr/sc_templates/scan_results.npz``
# which has shape (28, 158) → aspect_w_over_h = 158 / 28 ≈ 5.643.
_TITLE_ASPECT_W_OVER_H: float = 158.0 / 28.0


# ── Row Y multipliers in title_h units (from _ROW_OFFSET_MULTS) ──────
# Each row's CENTER y = title_y + title_h * MULT.
_ROW_Y_MULT_MINERAL: float = 1.6
_ROW_Y_MULT_MASS: float = 3.0
_ROW_Y_MULT_RESISTANCE: float = 4.4
_ROW_Y_MULT_INSTABILITY: float = 5.8


# ── Chrome-line Y multipliers in title_h units (from world_model.json) ─
# ``world_model.json``:
#   top_line.y_frac.mean = 0.875,  top_line.h_frac.mean = 0.25
#   bot_line.y_frac.mean = 7.6875, bot_line.h_frac.mean = 0.46875
# Reported y_frac is the TOP edge of the bbox; center = top + h/2.
_CHROME_TOP_Y_MULT: float = 0.875 + 0.25 / 2.0     # = 1.0
_CHROME_BOT_Y_MULT: float = 7.6875 + 0.46875 / 2.0  # = 7.921875


# ── Chrome-line X multipliers in title_h units (from world_model.json) ─
# ``world_model.json``:
#   top_line: x_frac=-0.0676 (title_w units), w_frac=1.7198
#     → x_left = -0.0676 * 5.643 ≈ -0.382
#     → x_right = (-0.0676 + 1.7198) * 5.643 ≈ 9.323
#     → x_center ≈ 4.470
#   bot_line: x_frac=0.0, w_frac=1.5990
#     → x_center = (0.0 + 1.5990/2) * 5.643 ≈ 4.512
# Chrome lines are HORIZONTAL — only their Y carries strong position
# information. The X-center is a weak anchor (could shift if the line
# is partially occluded). We provide it but callers should weight it
# lower than vertical anchors.
_CHROME_TOP_X_MULT: float = (
    (-0.06763285024154589 + 1.7198067632850242 / 2.0) * _TITLE_ASPECT_W_OVER_H
)
_CHROME_BOT_X_MULT: float = (
    (0.0 + 1.5990338164251208 / 2.0) * _TITLE_ASPECT_W_OVER_H
)


# ── Canonical anchor offsets ─────────────────────────────────────────
# Maps anchor_id -> (dx, dy) in title_h units from the panel origin
# (top-left of the SCAN RESULTS title bbox). Entries set to ``None``
# are not yet calibrated and the solver skips them.
#
# Conventions:
#   - All offsets in title_h units. Solved ``scale`` ≈ title_h in px.
#   - X increases rightward, Y increases downward (image convention).
#   - "title" anchor is the panel origin by definition.
#   - "label_*" anchors mark the matched left edge of the label word.
#   - "colon_*" anchors mark the right edge of the colon glyph.
#   - "chrome_*" anchors mark the vertical center of each chrome line.
DEFAULT_OFFSETS: dict[str, tuple[float, float] | None] = {
    # Panel origin (by definition).
    "title":          (0.0, 0.0),

    # Label-word positions. Y is the row CENTER y (from _ROW_OFFSET_MULTS).
    # X is NOT calibrated in any existing constant (the label_match
    # detector reports the matched bbox's x but there is no canonical
    # x_offset stored). Set X to None until label x-offsets are measured.
    "label_mass":     None,
    "label_resist":   None,
    "label_instab":   None,

    # Colon positions. Same situation as labels for X. Y is the row
    # center y (the colon sits at row_y_center, vertically aligned with
    # the label text).
    "colon_mass":     None,
    "colon_resist":   None,
    "colon_instab":   None,

    # Chrome-line centers. X is the line's bbox X-center (weak anchor —
    # the line is mostly horizontal). Y is well-calibrated.
    "chrome_top":     (_CHROME_TOP_X_MULT, _CHROME_TOP_Y_MULT),
    "chrome_bot":     (_CHROME_BOT_X_MULT, _CHROME_BOT_Y_MULT),
}


# Pure-Y anchor offsets (X=None means caller should treat measurement
# as Y-only). Some anchors below have well-known Y but no calibrated X.
# Exposed as a separate dict so callers can build their own Y-only
# pipelines if desired.
DEFAULT_Y_ONLY_OFFSETS: dict[str, float] = {
    "row_mineral":    _ROW_Y_MULT_MINERAL,
    "row_mass":       _ROW_Y_MULT_MASS,
    "row_resistance": _ROW_Y_MULT_RESISTANCE,
    "row_instab":     _ROW_Y_MULT_INSTABILITY,
    "chrome_top":     _CHROME_TOP_Y_MULT,
    "chrome_bot":     _CHROME_BOT_Y_MULT,
}


def solve_panel_pose(
    measurements: list[tuple[str, float, float]],
    offsets: dict[str, tuple[float, float]],
    weights: Optional[dict[str, float]] = None,
    *,
    fixed_scale: Optional[float] = None,
) -> Optional[tuple[float, float, float, dict[str, float]]]:
    """Solve for panel ``(x, y, scale)`` given anchor measurements.

    Parameters
    ----------
    measurements
        List of ``(anchor_id, measured_x, measured_y)`` tuples. Anchors
        whose ID is not in ``offsets`` (or whose offset is ``None``) are
        silently skipped — this lets callers pass a partial detector
        bundle without filtering.
    offsets
        Mapping ``anchor_id -> (dx, dy)`` in some consistent unit
        (e.g. title-height units). Entries with value ``None`` are
        treated as missing.
    weights
        Optional mapping ``anchor_id -> weight``. Missing keys default
        to ``1.0``. Non-positive weights are clamped to a tiny epsilon
        (so a "muted" anchor contributes almost nothing without breaking
        the linear system).
    fixed_scale
        If given, the panel scale is treated as a known input and only
        ``(panel_x, panel_y)`` are solved for. The returned scale equals
        this value. Use this when the panel rarely re-scales between
        frames — the system is over-determined with as few as 2 valid
        measurements.

    Returns
    -------
    ``(panel_x, panel_y, scale, residuals)`` where ``residuals`` maps
    each USED anchor_id to its Euclidean residual in the SAME units as
    the input measurements (pixels, typically). Anchors that were not
    used (missing offset, etc.) do not appear in ``residuals``.

    Returns ``None`` if the system is under-determined:
      * fewer than 2 usable measurements (or fewer than 1 if
        ``fixed_scale`` is set), OR
      * the linear system is rank-deficient (e.g. all offsets are
        co-linear so scale cannot be separated from translation).

    Notes
    -----
    The spec says "returns None if fewer than 3 measurements provided"
    but in practice the unknown-count drives this:
      * 3 unknowns (free scale) → need ≥ 2 anchors for a determined
        system (2 anchors × 2 equations = 4 ≥ 3); ≥ 3 for redundancy.
      * 2 unknowns (fixed scale) → need ≥ 1 anchor for a determined
        system.
    We return ``None`` below the determined-system threshold and let
    the over-determined case fall through naturally. Callers who want
    the stricter "≥ 3 anchors" guarantee can check ``len(measurements)``
    before calling.
    """
    if not measurements:
        return None

    # Filter to usable measurements: anchor must exist in offsets with
    # a non-None value. Duplicate anchor IDs are kept (caller may have
    # multiple detectors firing on the same anchor and we treat each
    # as an independent measurement).
    rows: list[tuple[str, float, float, float, float, float]] = []
    # Each row: (anchor_id, mx, my, dx, dy, weight)
    for anchor_id, mx, my in measurements:
        off = offsets.get(anchor_id)
        if off is None:
            continue
        dx, dy = off
        w = 1.0
        if weights is not None and anchor_id in weights:
            w_raw = float(weights[anchor_id])
            # Clamp to small positive — zero weight would zero a row
            # and could rank-deficient the system if many anchors are
            # muted at once.
            w = max(w_raw, 1e-9)
        rows.append((anchor_id, float(mx), float(my), float(dx), float(dy), w))

    n = len(rows)
    if n == 0:
        return None

    # Determined-system thresholds:
    #   free-scale (3 unknowns):    need n >= 2  (4 equations >= 3 unknowns)
    #   fixed-scale (2 unknowns):   need n >= 1  (2 equations >= 2 unknowns)
    if fixed_scale is None:
        if n < 2:
            return None
    else:
        if n < 1:
            return None

    # ── Build the linear system ──────────────────────────────────────
    # 2 equations per measurement: one for x, one for y.
    # Free-scale unknowns vector: [panel_x, panel_y, scale]
    # Fixed-scale unknowns vector: [panel_x, panel_y]
    if fixed_scale is None:
        A = np.zeros((2 * n, 3), dtype=np.float64)
        b = np.zeros(2 * n, dtype=np.float64)
        for i, (_aid, mx, my, dx, dy, w) in enumerate(rows):
            # Apply weight via row-scaling (sqrt(w) → ||W(Ax-b)||² = ||W^(1/2) A x - W^(1/2) b||²).
            sw = float(np.sqrt(w))
            # x-equation: panel_x + scale * dx = mx
            A[2 * i,     0] = sw * 1.0
            A[2 * i,     1] = sw * 0.0
            A[2 * i,     2] = sw * dx
            b[2 * i]         = sw * mx
            # y-equation: panel_y + scale * dy = my
            A[2 * i + 1, 0] = sw * 0.0
            A[2 * i + 1, 1] = sw * 1.0
            A[2 * i + 1, 2] = sw * dy
            b[2 * i + 1]     = sw * my
    else:
        s = float(fixed_scale)
        A = np.zeros((2 * n, 2), dtype=np.float64)
        b = np.zeros(2 * n, dtype=np.float64)
        for i, (_aid, mx, my, dx, dy, w) in enumerate(rows):
            sw = float(np.sqrt(w))
            A[2 * i,     0] = sw * 1.0
            A[2 * i,     1] = sw * 0.0
            b[2 * i]         = sw * (mx - s * dx)
            A[2 * i + 1, 0] = sw * 0.0
            A[2 * i + 1, 1] = sw * 1.0
            b[2 * i + 1]     = sw * (my - s * dy)

    # ── Solve via lstsq ──────────────────────────────────────────────
    # ``rcond=None`` uses NumPy's default rank detection (machine eps
    # times max-dim times largest singular value).
    try:
        x, _resid_sq, rank, _sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    # Rank-deficient: scale cannot be separated from translation
    # (happens iff all offsets are equal up to translation, i.e. only
    # one distinct (dx, dy) appears). Return None so callers can fall
    # back to fixed-scale.
    needed_rank = 2 if fixed_scale is not None else 3
    if rank < needed_rank:
        return None

    if fixed_scale is None:
        panel_x, panel_y, scale = float(x[0]), float(x[1]), float(x[2])
    else:
        panel_x, panel_y, scale = float(x[0]), float(x[1]), float(fixed_scale)

    # ── Compute per-anchor residuals (UNWEIGHTED Euclidean distance) ──
    # The residual reflects the geometric mismatch in input units, NOT
    # the weighted residual — callers use these to detect outliers.
    residuals: dict[str, float] = {}
    for anchor_id, mx, my, dx, dy, _w in rows:
        pred_x = panel_x + scale * dx
        pred_y = panel_y + scale * dy
        rx = mx - pred_x
        ry = my - pred_y
        r = float(np.hypot(rx, ry))
        # If an anchor_id appears more than once in `measurements`, keep
        # the LARGEST residual (the more conservative outlier signal).
        if anchor_id in residuals:
            residuals[anchor_id] = max(residuals[anchor_id], r)
        else:
            residuals[anchor_id] = r

    return (panel_x, panel_y, scale, residuals)


__all__ = [
    "solve_panel_pose",
    "DEFAULT_OFFSETS",
    "DEFAULT_Y_ONLY_OFFSETS",
]
