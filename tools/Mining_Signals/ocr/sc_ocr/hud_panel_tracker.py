"""Predict-and-verify rigid-body tracker for the SCAN RESULTS panel.

The SC mining HUD panel is a rigid body: it has multiple known anchor
points (the SCAN RESULTS title, the MASS/RESIST/INSTAB labels, the top
and bottom chrome separator lines, etc.) whose pixel offsets relative
to a "panel origin" are fixed by the HUD design. Between consecutive
frames the panel moves smoothly (capped by gameplay constraints — ship
pitch/yaw is bounded, the rock-scan UI is statically docked, etc.).

The existing detection pipeline in ``onnx_hud_reader._find_label_rows``
treats every frame as a cold start — full-image multi-scale NCC search
for the SCAN RESULTS title plus the label templates. The fallback
tiers (PRIMARY through TESSERACT) each search independently, so the
"final" panel position can quietly jump between tiers as one detector
gives up and another picks up. The user sees this as cyan-band jitter
in the overlay even when the in-game panel is stationary.

This tracker fixes the jitter by enforcing rigid-body consistency
across frames:

* **Frame 0 (cold start)** — call every available anchor detector with
  full-image search. Pool the (anchor_name → measured_xy) pairs and
  solve for the panel pose (panel_x, panel_y, scale) via least-squares
  against the canonical offset model. Accept the lock if ≥3 anchors
  agree within ``max_residual_px``.
* **Frame N (tracking)** — predict each anchor's pixel position from
  ``last_pose + offset``. Call each detector in LOCAL-SEARCH mode
  (small radius around the prediction). Solve again. Accept the new
  pose iff the residuals are still bounded AND the panel didn't jump
  more than ``max_motion_px``. On rejection, increment a counter; after
  ``cold_start_after_rejections`` consecutive rejections, drop the
  lock and re-run full-image search next frame.

The tracker is a thin wrapper around three external APIs:

* ``hud_tracker.rigid_body.solve_panel_pose`` — weighted least-squares
  pose solver. Returns ``(panel_x, panel_y, scale, residuals)`` from a
  set of (anchor_id, measured_xy) tuples and a canonical offset table.
* ``ocr.sc_ocr.scan_results_match.find_scan_results_anchor`` — the
  existing NCC title detector, extended with optional ``search_center``
  / ``search_radius`` kwargs for local-window search (a ~100× speedup
  over full-frame scan and structurally immune to COMPOSITION-row false
  matches because we never look there).
* ``ocr.sc_ocr.label_match.find_label_positions`` — per-row label NCC
  with the same ``search_centers`` / ``search_radius`` kwargs.

The module-level adapters ``_solve_panel_pose`` and
``_find_scan_results_anchor`` are thin wrappers that handle the
shape conversions between Agent A's flat tuple return and the tracker's
nested-tuple expectation, and forward kwargs to the local-search-aware
detectors.
"""
from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from PIL import Image

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Adapters around the external solver + detector APIs.
# Kept module-level (rather than methods) so tests can monkeypatch them
# without instantiating the tracker.
# ─────────────────────────────────────────────────────────────────────


def _solve_panel_pose(
    measurements: dict[str, tuple[float, float]],
    offsets: dict[str, tuple[float, float]],
    weights: Optional[dict[str, float]] = None,
    *,
    fixed_scale: Optional[float] = None,
) -> tuple[tuple[float, float, float], dict[str, float]]:
    """Solve for panel pose via Agent A's least-squares solver.

    Returns ``((panel_x, panel_y, scale), residuals_by_anchor)``.

    Thin adapter around ``hud_tracker.rigid_body.solve_panel_pose``:
      * Converts the tracker's dict-form measurements into Agent A's
        list-of-tuples form.
      * Re-shapes Agent A's flat ``(x, y, scale, residuals)`` return
        into the tracker's nested ``((x, y, scale), residuals)`` shape.
      * Returns the identity pose ``((0, 0, 1), {})`` when the system
        is under-determined (Agent A returns None) so the caller's
        ``_residuals_ok`` check uniformly rejects the pose without
        needing a separate None handler.

    ``fixed_scale`` defaults to ``None`` (let scale float). With the
    title-h-unit offset convention in ``DEFAULT_OFFSETS``, the solver
    returns scale ≈ actual title height in pixels — which adapts to
    whatever resolution the game is rendering at. An earlier version
    pinned scale=1.0 with pixel-unit offsets; that broke lock because
    the canonical 28-px title doesn't match in-game ~45-px renders,
    inflating residuals past tolerance. With dy values that vary
    across anchors (0 / 1.5 / 3.33 / 4.98 / 6.64) the LSQ system is
    over-determined for (panel_y, scale) even with all dx=0, so the
    rank-deficiency-on-X concern that motivated fixed_scale=1.0 is
    moot — only the panel_x estimate is rank-1 (pulled entirely by
    the title anchor), which is correct.
    """
    try:
        from hud_tracker.rigid_body import solve_panel_pose
    except ImportError:  # pragma: no cover — should never happen post-merge
        log.error(
            "hud_tracker.rigid_body unavailable — cannot solve panel pose"
        )
        return ((0.0, 0.0, 1.0), {})

    if not measurements:
        return ((0.0, 0.0, 1.0), {})

    meas_list: list[tuple[str, float, float]] = [
        (str(name), float(xy[0]), float(xy[1]))
        for name, xy in measurements.items()
    ]
    result = solve_panel_pose(
        meas_list, offsets, weights=weights, fixed_scale=fixed_scale,
    )
    if result is None:
        # Under-determined: return identity pose + empty residuals so
        # the caller's residual-check uniformly rejects.
        return ((0.0, 0.0, 1.0), {})

    panel_x, panel_y, scale, residuals = result
    return (
        (float(panel_x), float(panel_y), float(scale)),
        {str(k): float(v) for k, v in residuals.items()},
    )


def _find_scan_results_anchor(
    img: "Image.Image",
    *,
    search_center: Optional[tuple[float, float]] = None,
    search_radius: int = 60,
) -> Optional[dict]:
    """Locate the SCAN RESULTS title via Agent B's extended NCC detector.

    Forwards ``search_center`` and ``search_radius`` directly to
    ``ocr.sc_ocr.scan_results_match.find_scan_results_anchor``. When
    ``search_center`` is ``None`` the underlying detector does its
    classic full-frame multi-scale sweep; when a center is given it
    restricts the sweep to a ±``search_radius`` window — typically
    50-100× faster and structurally immune to COMPOSITION-row false
    matches because we never look there.

    The detector returns image-absolute coordinates either way.
    """
    try:
        from ocr.sc_ocr.scan_results_match import find_scan_results_anchor
    except ImportError:
        # Relative import for the in-package layout.
        from .scan_results_match import find_scan_results_anchor  # type: ignore
    # Convert tuple[float, float] → tuple[int, int] for the detector
    # (NCC works on pixel grid; sub-pixel centers add no value).
    sc: Optional[tuple[int, int]] = None
    if search_center is not None:
        sc = (int(search_center[0]), int(search_center[1]))
    return find_scan_results_anchor(
        img, search_center=sc, search_radius=int(search_radius),
    )


# ─────────────────────────────────────────────────────────────────────
# Anchor offsets used by the tracker.
#
# These offsets are in **title-height units** (not pixels). The solver
# returns ``scale`` ≈ the actual rendered title height in pixels, and
# the predicted anchor pixel is ``panel_origin + scale * offset``.
# This lets the tracker discover the panel's actual scale per session
# (game-resolution sensitive) instead of assuming a fixed canonical
# pixel size — which was the bug that prevented the tracker from
# locking: the canonical title is 28 px but in-game capture often
# renders the title at 45-50 px, so pixel-unit offsets at scale=1.0
# overshot the actual label positions by 50+ px and the LSQ residuals
# blew past the 5 px tolerance every frame.
#
# Multipliers were empirically calibrated from production HUD scans
# (title @ y=18 w=253 h=45 with labels at y=168/242/317 → 3.33/4.98/
# 6.64 in title-h units). They differ from `_ROW_OFFSET_MULTS` in
# onnx_hud_reader (3.0/4.4/5.8) because those are ROW-CENTER mults
# while label_match returns the LABEL-TOP position; the gap is roughly
# 0.33 title-h units (about half the label glyph height).
#
# X offsets are pinned to 0.0 — the label text starts at roughly the
# panel's left edge, same as the title. We treat each anchor as a
# Y-only constraint; the title anchor alone pins panel_x.
#
# Why not use ``hud_tracker.rigid_body.DEFAULT_OFFSETS`` directly?
# That table is the canonical authority but its label_* and colon_*
# X entries are ``None`` because the existing codebase only stores
# per-ROW bbox geometry, not per-glyph X. Plumbing ``None`` X through
# the solver makes it skip the label anchors entirely, leaving the
# tracker with too few measurements to lock. The local table here
# uses empirically-calibrated Y multipliers with X=0 so the tracker
# has 4 usable anchors per cold start.
# ─────────────────────────────────────────────────────────────────────

_CANONICAL_TITLE_H_PX: float = 28.0

#: Tracker offsets in title-height units from the panel origin
#: (top-left of the SCAN RESULTS title). Values are ``(dx, dy)``.
#: The solver multiplies by the discovered ``scale`` (≈ actual title
#: height in pixels) to produce predicted pixel positions.
DEFAULT_OFFSETS: dict[str, tuple[float, float]] = {
    # Panel origin (by definition).
    "scan_results":      (0.0, 0.0),
    # Label TOP-LEFT positions (label_match returns the matched
    # template's top-left, not center). Multipliers are empirical
    # from production captures — see docstring above for derivation.
    "label_mineral":     (0.0, 1.50),  # estimate, not directly observed
    "label_mass":        (0.0, 3.33),
    "label_resistance":  (0.0, 4.98),
    "label_instability": (0.0, 6.64),
}

# Multiplier table mirroring _ROW_OFFSET_MULTS in onnx_hud_reader, kept
# here so the tracker can compute label_rows output without importing
# from onnx_hud_reader (which would create a circular dependency).
# These are ROW-CENTER multipliers (different from DEFAULT_OFFSETS,
# which are LABEL-TOP multipliers as label_match reports).
_ROW_OFFSET_MULTS: dict[str, float] = {
    "_mineral_row": 1.6,
    "mass":         3.0,
    "resistance":   4.4,
    "instability":  5.8,
}
_ROW_HEIGHT_MULT: float = 0.9
_VALUE_COL_LEFT_FRAC: float = 0.52


# ─────────────────────────────────────────────────────────────────────
# Auto-calibration of label multipliers
#
# The hardcoded ``DEFAULT_OFFSETS`` (label_mass=3.33, label_resistance=
# 4.98, label_instability=6.64 in title-h units) were measured from one
# reference panel. Real captures show meaningful per-user variation —
# the user's specific HUD reads label-top ratios ~4.5/6.2/7.8 instead.
# When the LSQ solver tries to fit observations with wrong canonical
# offsets it inflates ``scale`` to compensate; the result is a "lock"
# whose scale doesn't match the rendered title height, and the
# downstream ``_pose_to_label_rows`` produces bands offset from reality.
#
# Fix: every time the legacy PRIMARY tier succeeds (title anchor +
# ≥2 label matches), record the observed (label_y - title_y) / title_h
# ratios as a calibration sample. Once we have enough samples whose
# per-field median is stable (low relative std), publish the medians as
# *learned* offsets/row-mults that override the defaults.
#
# Why median + relative-std rather than mean + abs-std? The label
# matcher occasionally produces tiny vertical mis-localisations (±2px)
# from sub-pixel resampling; medians are robust to those, and using
# relative std (vs the median) gives a unit-free "is this consistent?"
# gate that works across resolutions.
#
# The samples buffer is bounded to ``_CAL_MAX_SAMPLES`` so a long-
# running session doesn't accrete unbounded history; once we've locked
# in stable values, additional samples confirm rather than re-derive
# (we still re-compute on every new sample so a HUD change is picked
# up within ``_CAL_MIN_SAMPLES`` frames).
# ─────────────────────────────────────────────────────────────────────

# Minimum samples before we can publish learned values.
_CAL_MIN_SAMPLES: int = 5
# Bounded ring buffer so a long-running session can't grow unbounded.
_CAL_MAX_SAMPLES: int = 20
# Relative-std-to-median tolerance for "stable enough to publish".
# 10% is permissive enough to accept normal label-matcher jitter while
# tight enough to reject samples taken across very different captures
# (e.g. some at title_h=45 and some at title_h=50 would blow this).
_CAL_REL_TOL: float = 0.10

# Per-field plausibility bounds for label-TOP-to-title ratios. The
# production reference panel has ratios 3.33 / 4.98 / 6.64; a user's
# specific HUD can deviate ±30% but anything outside these wide bounds
# is structurally impossible (the labels can't be BETWEEN the title and
# its bottom edge, and they can't be 12+ title-heights below it).
# Pre-anchor full-frame label_match has been observed to false-match
# the MASS template against parts of the SCAN RESULTS title itself or
# the mineral-name row above the data rows; ratios below 2.0 catch
# those false matches before they pollute the median.
_CAL_RATIO_BOUNDS: dict[str, tuple[float, float]] = {
    "mass":        (2.0, 7.5),
    "resistance":  (3.0, 9.0),
    "instability": (4.0, 11.0),
}

# Minimum gap between consecutive label-TOP ratios. The HUD design
# pitch is ~1.4 title-heights between rows; allowing as little as 1.0
# gives us margin for slight scale-mismatch in the NCC sweep without
# admitting samples where two labels accidentally landed at the same
# Y (which means at least one false-matched).
_CAL_MIN_INTER_LABEL_GAP: float = 1.0

# Sample buffer — list of dicts, each shaped:
#   {
#       "title_y": float, "title_h": float,
#       "mass":        {"y": float, "h": float, "ratio_top": float},
#       "resistance":  {...},
#       "instability": {...},
#   }
# Missing fields mean that label wasn't matched in that scan; the
# sample is still recorded if at least two of the three numeric labels
# matched.
_calibration_samples: list[dict] = []

# Learned state — published once samples are stable. Shape:
#   {
#       "n_samples": int,
#       "offsets":   dict[str, tuple[float, float]] (solver offsets,
#                                                    label-TOP mults),
#       "row_mults": dict[str, float] (row-CENTER mults for
#                                      _pose_to_label_rows),
#   }
# ``None`` until enough stable samples accrue. After publication,
# additional samples can update the values (e.g. if the HUD subtly
# changes between sessions).
_learned_state: Optional[dict] = None

# Monotonic counter bumped every time ``_learned_state`` changes value.
# Consumers (the stabilizer, in particular) snapshot this when they
# cold-start and check it on every subsequent call; if the version has
# moved, their stored pose was computed with stale offsets and they
# should reset to pick up the new geometry. Starts at 0 and ticks up
# by 1 per published change.
_calibration_version: int = 0


def _default_delta(field: str) -> float:
    """Default offset between label-TOP mult and row-CENTER mult.

    Returns ``DEFAULT_OFFSETS[label_<field>].dy - _ROW_OFFSET_MULTS[
    field]``. The label sits a small distance BELOW the row's vertical
    center in the production geometry; preserving this delta when we
    derive learned row-center mults from observed label-top mults
    keeps the row bands positioned the same way relative to the
    actual label (just at the user's actual scale).
    """
    label_key = f"label_{field}"
    if label_key not in DEFAULT_OFFSETS or field not in _ROW_OFFSET_MULTS:
        return 0.0
    return float(DEFAULT_OFFSETS[label_key][1]) - float(
        _ROW_OFFSET_MULTS[field]
    )


def check_calibration_consistency(
    *,
    title_y: float,
    title_h: float,
    label_matches: dict,
) -> tuple[bool, str]:
    """Decide whether a ``(title, labels)`` sample is geometrically sane.

    Runs the same per-field ratio bounds, y-order, and inter-label gap
    checks that :func:`observe_calibration_sample` applies internally —
    just exposed so callers (e.g. ``onnx_hud_reader``'s EARLY-DIRECT
    path) can refuse to BUILD row bands from a sample they wouldn't
    trust enough to LEARN from.

    Failure mode this is meant to catch: the title detector sometimes
    matches only the short "SCAN" sub-string of "SCAN RESULTS" — title
    comes back at ``h=17`` (1× scale of the sub-string) — while
    label_match independently matches the labels at the proper 2×
    scale (``h=56``). The label-to-title ratio explodes (e.g. 10.8 for
    mass) and lands well outside the 2.0-7.5 plausibility band. The
    calibration learner correctly rejects each field, but EARLY-DIRECT
    used to build row bands anyway, producing 16-px-tall value crops
    that cascade into garbage OCR (``mass='39939'``,
    ``instability='0701%70'``, HUD-RGB reading ``'0000660686660'``).

    Returns
    -------
    ``(passed, reason)``. ``passed`` is True when every provided label
    is in-bounds AND the labels are in correct y-order with sufficient
    inter-label gap. False otherwise; ``reason`` is a short
    human-readable explanation suitable for logging.
    """
    try:
        title_y = float(title_y)
        title_h = float(title_h)
    except (TypeError, ValueError):
        return (False, "title coords not numeric")
    if title_h <= 0:
        return (False, f"title_h={title_h} not positive")
    ratios: dict[str, float] = {}
    for field in ("mass", "resistance", "instability"):
        info = (label_matches or {}).get(field)
        if not info:
            continue
        try:
            ly = float(info["y"])
        except (KeyError, TypeError, ValueError):
            continue
        ratio = (ly - title_y) / title_h
        bounds = _CAL_RATIO_BOUNDS.get(field)
        if bounds is not None and not (bounds[0] <= ratio <= bounds[1]):
            return (
                False,
                f"{field} ratio={ratio:.2f} outside bounds {bounds}",
            )
        ratios[field] = ratio
    if len(ratios) < 2:
        return (False, f"only {len(ratios)} labels in-bounds (need >=2)")
    _ordered = sorted(ratios.items(), key=lambda kv: kv[1])
    _expected = [
        f for f in ("mass", "resistance", "instability") if f in ratios
    ]
    if [kv[0] for kv in _ordered] != _expected:
        return (False, "labels out of expected y-order")
    for i in range(len(_ordered) - 1):
        gap = _ordered[i + 1][1] - _ordered[i][1]
        if gap < _CAL_MIN_INTER_LABEL_GAP:
            return (
                False,
                f"gap {gap:.2f} between {_ordered[i][0]} and "
                f"{_ordered[i + 1][0]} below min "
                f"{_CAL_MIN_INTER_LABEL_GAP}",
            )
    return (True, "")


def observe_calibration_sample(
    *,
    title_y: float,
    title_h: float,
    label_matches: dict,
) -> None:
    """Record a calibration sample from a successful detection.

    Call this whenever the legacy PRIMARY tier (title anchor + label
    template matching) succeeds with at least 2 label matches. Each
    observation records the per-field (label_y - title_y) / title_h
    ratio; once we have ≥``_CAL_MIN_SAMPLES`` ratios whose relative
    std is below ``_CAL_REL_TOL``, the learned medians are published
    via ``get_learned_offsets`` / ``get_learned_row_mults``.

    Parameters
    ----------
    title_y, title_h:
        Image-absolute Y position and height of the SCAN RESULTS title.
    label_matches:
        ``label_match.find_label_positions`` return shape — a dict
        mapping ``"mass" / "resistance" / "instability"`` to a dict
        with at least ``"y"`` and ``"h"`` (image-absolute label-top
        and label-height in pixels). Missing fields are tolerated;
        the sample is recorded if at least 2 of the 3 are present.
    """
    global _calibration_samples
    try:
        title_y = float(title_y)
        title_h = float(title_h)
    except (TypeError, ValueError):
        return
    if title_h <= 0:
        return
    sample: dict = {"title_y": title_y, "title_h": title_h}
    for field in ("mass", "resistance", "instability"):
        info = (label_matches or {}).get(field)
        if not info:
            continue
        try:
            ly = float(info["y"])
            lh = float(info["h"])
        except (KeyError, TypeError, ValueError):
            continue
        ratio = (ly - title_y) / title_h
        # Plausibility gate: reject ratios outside the per-field bounds.
        # Pre-anchor full-frame label_match has been observed to
        # false-match templates against parts of the SCAN RESULTS title
        # or the mineral-name row above the data rows; those false
        # matches produce ratios outside the bounds. Without this gate
        # the polluted median pulls learned offsets way off the truth,
        # which then puts row bands in the wrong place (a worse failure
        # than just falling back to defaults).
        bounds = _CAL_RATIO_BOUNDS.get(field)
        if bounds is not None and not (bounds[0] <= ratio <= bounds[1]):
            log.debug(
                "calibration sample REJECT field=%s ratio=%.3f outside "
                "bounds %s — likely a false-match label position",
                field, ratio, bounds,
            )
            continue
        sample[field] = {
            "y": ly,
            "h": lh,
            "ratio_top": ratio,
        }
    # Require ≥2 of the 3 numeric labels — calibrating off a single
    # observation is too noisy and the LSQ already handles single-
    # anchor cases gracefully via the title alone.
    n_labels = sum(
        1 for f in ("mass", "resistance", "instability") if f in sample
    )
    if n_labels < 2:
        return
    # Ordering + pitch gate: when we have multiple labels, they must
    # appear in strictly-increasing Y AND the gap between consecutive
    # ratios must be at least ``_CAL_MIN_INTER_LABEL_GAP``. Either
    # condition failing means at least one label false-matched (e.g.
    # MASS template hit the mineral-row, then "resistance" got placed
    # exactly there too because its anchor logic is MASS-relative).
    _ordered = [
        (f, sample[f]["ratio_top"])
        for f in ("mass", "resistance", "instability") if f in sample
    ]
    _ordered.sort(key=lambda kv: kv[1])
    _ordered_fields = [kv[0] for kv in _ordered]
    _expected_order = [
        f for f in ("mass", "resistance", "instability") if f in sample
    ]
    if _ordered_fields != _expected_order:
        log.debug(
            "calibration sample REJECT — labels out of order: %s "
            "(expected %s)", _ordered_fields, _expected_order,
        )
        return
    for i in range(len(_ordered) - 1):
        gap = _ordered[i + 1][1] - _ordered[i][1]
        if gap < _CAL_MIN_INTER_LABEL_GAP:
            log.debug(
                "calibration sample REJECT — gap %.3f between %s and "
                "%s below min %s (likely a false-match collision)",
                gap, _ordered[i][0], _ordered[i + 1][0],
                _CAL_MIN_INTER_LABEL_GAP,
            )
            return
    _calibration_samples.append(sample)
    # Bounded buffer — drop the oldest sample when we exceed the cap.
    if len(_calibration_samples) > _CAL_MAX_SAMPLES:
        _calibration_samples = _calibration_samples[-_CAL_MAX_SAMPLES:]
    _maybe_update_learned()


def _maybe_update_learned() -> None:
    """Recompute the learned state from the sample buffer.

    Per-field: if we have ≥``_CAL_MIN_SAMPLES`` ratios AND their
    relative std is below ``_CAL_REL_TOL``, take the median. Publish
    the learned offsets (label-TOP mults for the solver) plus derived
    row-center mults (label-TOP minus default-delta per field).

    No-op if no field is stable; preserves the previous learned state
    if some fields are stable and others aren't (a partial update is
    still better than reverting to defaults for those fields).
    """
    global _learned_state
    if len(_calibration_samples) < _CAL_MIN_SAMPLES:
        return

    new_offsets: dict[str, tuple[float, float]] = {
        "scan_results": (0.0, 0.0),
    }
    new_row_mults: dict[str, float] = {}
    locked_fields: list[str] = []
    rejected_fields: list[tuple[str, str]] = []

    for field, off_key in (
        ("mass", "label_mass"),
        ("resistance", "label_resistance"),
        ("instability", "label_instability"),
    ):
        # label-TOP ratios (for the solver offset table) and
        # label-CENTER ratios (for the row-center band placement)
        # are derived from the same samples but computed
        # independently — the band needs to sit ON the label
        # glyphs, which means row_center = label_center, NOT
        # row_center = label_top - some-constant. Earlier versions
        # used (label_top - default_delta) which baked in the
        # reference panel's specific label-below-row geometry, and
        # produced bands above the actual data rows on HUDs that
        # don't share that geometry.
        ratios_top = [
            s[field]["ratio_top"]
            for s in _calibration_samples
            if field in s
        ]
        ratios_center = [
            (
                (s[field]["y"] + s[field]["h"] / 2.0 - s["title_y"])
                / s["title_h"]
            )
            for s in _calibration_samples
            if field in s
        ]
        if len(ratios_top) < _CAL_MIN_SAMPLES:
            rejected_fields.append(
                (field, f"only {len(ratios_top)} ratios (need {_CAL_MIN_SAMPLES})")
            )
            continue
        try:
            med_top = float(statistics.median(ratios_top))
            med_center = float(statistics.median(ratios_center))
            sd = float(statistics.pstdev(ratios_top))
        except statistics.StatisticsError:
            rejected_fields.append((field, "stats raised StatisticsError"))
            continue
        rel = sd / max(abs(med_top), 1e-6)
        if rel > _CAL_REL_TOL:
            rejected_fields.append(
                (field, f"rel_std={rel:.3f} > tol={_CAL_REL_TOL:.3f}")
            )
            continue
        new_offsets[off_key] = (0.0, med_top)
        new_row_mults[field] = med_center
        locked_fields.append(field)

    if not locked_fields:
        return

    # Mineral row / label_mineral: not directly observed. Scale them
    # proportionally with the mass row's learned mult so the relative
    # spacing matches the production design. Falls back to defaults if
    # the mass row hasn't locked yet.
    if "mass" in new_row_mults:
        mass_scale_ratio = (
            new_row_mults["mass"] / float(_ROW_OFFSET_MULTS["mass"])
        )
        new_row_mults["_mineral_row"] = (
            float(_ROW_OFFSET_MULTS["_mineral_row"]) * mass_scale_ratio
        )
        new_offsets["label_mineral"] = (
            0.0,
            float(DEFAULT_OFFSETS["label_mineral"][1]) * mass_scale_ratio,
        )
    else:
        # Mass not learned yet — preserve defaults so the solver still
        # has a mineral anchor candidate even before mass stabilises.
        new_offsets["label_mineral"] = DEFAULT_OFFSETS["label_mineral"]
        new_row_mults["_mineral_row"] = float(
            _ROW_OFFSET_MULTS["_mineral_row"]
        )

    # Preserve any unlearned per-field defaults so the solver always
    # has a complete offset table (a missing key would make the LSQ
    # silently drop that anchor).
    for fallback_key in ("label_mass", "label_resistance", "label_instability"):
        if fallback_key not in new_offsets:
            new_offsets[fallback_key] = DEFAULT_OFFSETS[fallback_key]
    for fallback_field in ("mass", "resistance", "instability"):
        if fallback_field not in new_row_mults:
            new_row_mults[fallback_field] = float(
                _ROW_OFFSET_MULTS[fallback_field]
            )

    new_state = {
        "n_samples": len(_calibration_samples),
        "offsets": new_offsets,
        "row_mults": new_row_mults,
    }
    # Only bump version + log on actual change to avoid spamming the
    # log every scan once we've settled. Compare by value, not identity.
    global _calibration_version
    if (
        _learned_state is None
        or _learned_state.get("offsets") != new_offsets
        or _learned_state.get("row_mults") != new_row_mults
    ):
        _calibration_version += 1
        log.info(
            "HudPanelTracker AUTO-CAL: published learned values v=%d "
            "(n_samples=%d locked=%s rejected=%s); offsets=%s row_mults=%s",
            _calibration_version,
            new_state["n_samples"], locked_fields, rejected_fields,
            {k: (round(v[0], 3), round(v[1], 3)) for k, v in new_offsets.items()},
            {k: round(v, 3) for k, v in new_row_mults.items()},
        )
    _learned_state = new_state


def get_learned_offsets() -> Optional[dict]:
    """Return the published learned offsets dict (solver format), or
    ``None`` if calibration hasn't locked yet.

    The returned dict is safe to mutate by the caller (it's a shallow
    copy). Values are ``(dx, dy)`` in title-h units, suitable for
    passing directly to ``solve_panel_pose``.
    """
    if _learned_state is None:
        return None
    return dict(_learned_state["offsets"])


def get_learned_row_mults() -> Optional[dict]:
    """Return the published learned row-center multipliers, or ``None``
    if calibration hasn't locked yet.

    The returned dict is safe to mutate by the caller. Keys are
    ``"_mineral_row" / "mass" / "resistance" / "instability"`` and
    values are title-h-unit multipliers from the title TOP to each
    row's CENTER.
    """
    if _learned_state is None:
        return None
    return dict(_learned_state["row_mults"])


def get_calibration_state() -> dict:
    """Diagnostic snapshot of the current calibration state. Returns a
    dict with sample count, learned values, and the per-field ratio
    history. Used by tests and the panel-finder UI.
    """
    return {
        "n_samples": len(_calibration_samples),
        "learned": (
            None if _learned_state is None else dict(_learned_state)
        ),
        "samples": list(_calibration_samples),
        "version": _calibration_version,
    }


def get_calibration_version() -> int:
    """Monotonically-increasing version that bumps each time the
    learned offsets/row-mults change value. Consumers that cache a
    pose computed from these values (e.g. ``HudPanelStabilizer``)
    can snapshot this at cold-start and compare on subsequent calls
    to detect that their cached pose is stale.
    """
    return _calibration_version


def reset_calibration() -> None:
    """Wipe the sample buffer and learned state. For tests."""
    global _calibration_samples, _learned_state, _calibration_version
    _calibration_samples = []
    _learned_state = None
    _calibration_version = 0


class HudPanelTracker:
    """Predict-and-verify rigid-body tracker for the SCAN RESULTS panel.

    Frame 0 (cold start):
        Full-frame search via existing detectors. Find all anchors that
        respond. Solve for panel pose via ``solve_panel_pose``.
        Lock if >=3 anchors agree within residual tolerance.

    Frame N (tracking):
        Predict each anchor's position from ``last_pose + offset``.
        Call local-search detectors with each predicted center.
        Solve for new pose. Accept if:
          - >=3 anchors found
          - max residual <= ``max_residual_px`` (rigid-body fit good)
          - panel motion <= ``max_motion_px`` (bounded velocity)
        Else: increment rejection counter. After
        ``cold_start_after_rejections`` rejections, cold-start.
    """

    # Minimum number of anchor measurements that must agree before we
    # call the LSQ fit a "lock". Below this, the system is too
    # under-constrained to trust scale.
    _MIN_ANCHORS_FOR_LOCK: int = 3

    def __init__(
        self,
        offsets: Optional[dict[str, tuple[float, float]]] = None,
        *,
        search_radius: int = 60,
        max_residual_px: float = 25.0,
        max_motion_px: float = 40.0,
        cold_start_after_rejections: int = 3,
        min_anchors_for_lock: int = 3,
    ) -> None:
        self._offsets: dict[str, tuple[float, float]] = dict(
            offsets if offsets is not None else DEFAULT_OFFSETS
        )
        self.search_radius = int(search_radius)
        self.max_residual_px = float(max_residual_px)
        self.max_motion_px = float(max_motion_px)
        self.cold_start_after_rejections = int(cold_start_after_rejections)
        self._min_anchors_for_lock = int(min_anchors_for_lock)

        self._last_pose: Optional[tuple[float, float, float]] = None
        self._rejection_count: int = 0
        self._is_locked: bool = False
        # Human-readable explanation of the most recent track() outcome.
        # Set by every code path that returns None (and cleared when a
        # lock succeeds). Surfaced to callers via the ``last_failure_reason``
        # property so the integration layer can log diagnostic info even
        # when this module's logger is filtered out of the viewer.
        self._last_failure_reason: Optional[str] = None

    # ─── Public API ────────────────────────────────────────────────

    @property
    def is_locked(self) -> bool:
        return self._is_locked

    @property
    def last_pose(self) -> Optional[tuple[float, float, float]]:
        """``(panel_x, panel_y, scale)`` or ``None`` if no lock."""
        return self._last_pose

    @property
    def last_failure_reason(self) -> Optional[str]:
        """Human-readable explanation of the most recent ``track()``
        outcome that returned ``None``. Cleared when a lock succeeds.
        Useful for the integration layer to surface diagnostic info
        even when this module's logger is filtered out of the viewer.
        """
        return self._last_failure_reason

    def reset(self) -> None:
        """Clear last pose. Forces a cold start on the next track()."""
        self._last_pose = None
        self._rejection_count = 0
        self._is_locked = False

    def track(self, img: "Image.Image") -> Optional[dict]:
        """Return ``label_rows`` dict or ``None`` if not locked.

        ``label_rows`` matches the shape produced by
        ``onnx_hud_reader._find_label_rows`` — a dict mapping field
        names (``"mass"``, ``"resistance"``, ``"instability"`` plus the
        ``"_mineral_row"`` sentinel) to ``(y1, y2, label_right)``
        tuples in img coordinates.

        Side effects:
          * Updates ``self._last_pose`` on a successful lock or
            tracking step.
          * Updates ``self._rejection_count`` on a failed tracking
            step. After ``self.cold_start_after_rejections`` failures
            in a row, drops the lock so the next call cold-starts.
        """
        if self._last_pose is None or not self._is_locked:
            return self._cold_start(img)
        return self._track_step(img)

    # ─── Internal: cold start ──────────────────────────────────────

    def _cold_start(self, img: "Image.Image") -> Optional[dict]:
        """Full-frame detector sweep + LSQ lock attempt."""
        # ── Panel-presence pre-filter ──
        # The mining HUD's three value rows each end with a colon (":");
        # any visible mining panel has 3+ colons.  If the panel has
        # essentially no colons, this frame is NOT a mining HUD (could
        # be a scanner transition, the inventory screen, a tooltip,
        # the game in non-mining mode, etc.) and there is nothing to
        # lock onto.
        #
        # Bailing early here is the difference between "no panel here,
        # move on" (correct) and "failed to lock 3+ anchors with garbage
        # log spam every frame" (the v2.2.12 behaviour).  Measured on
        # the 241-panel test set: 30 of the 38 NOT_LOCKED panels have
        # <=1 colon and are non-HUD frames; suppressing them turns the
        # effective lock rate from 84.2% into 96.2% on actual mining
        # panels and quiets the log noise that confused us into
        # chasing imaginary regressions.
        try:
            try:
                from ocr.sc_ocr.colon_anchor import find_colons
            except ImportError:
                from .colon_anchor import find_colons  # type: ignore
            _presence_colons = find_colons(
                img,
                y_band=(0, img.height),
                x_range=(0, int(img.width * 0.75)),
            )
            if len(_presence_colons) < 2:
                reason = (
                    f"panel-presence check: only {len(_presence_colons)} "
                    f"colons in frame -- not a mining HUD"
                )
                log.debug("HudPanelTracker._cold_start: %s", reason)
                self._is_locked = False
                self._last_pose = None
                self._last_failure_reason = reason
                return None
        except Exception as exc:
            # If the pre-filter raises, just continue to the normal
            # path -- we'd rather attempt a lock than skip a real panel
            # because find_colons hit an edge case.
            log.debug(
                "HudPanelTracker._cold_start: presence check raised "
                "(%s); proceeding with full anchor collection", exc,
            )

        measurements = self._collect_anchors(img, search_centers=None)
        log.info(
            "HudPanelTracker._cold_start: collected %d anchors: %s",
            len(measurements), sorted(measurements.keys()),
        )
        if len(measurements) < self._min_anchors_for_lock:
            reason = (
                f"only {len(measurements)} anchors found "
                f"(need {self._min_anchors_for_lock}); "
                f"keys={sorted(measurements.keys())}"
            )
            log.warning("HudPanelTracker._cold_start: %s — no lock", reason)
            self._is_locked = False
            self._last_pose = None
            self._last_failure_reason = reason
            return None

        pose, residuals = self._solve(measurements)
        dropped_anchor: Optional[str] = None
        if not self._residuals_ok(residuals):
            # ─── Outlier rejection ─────────────────────────────────
            # Sometimes a single anchor matches at a geometrically
            # inconsistent position -- most commonly ``scan_results``
            # locking onto a stale/sibling SCAN RESULTS title far
            # from the actual data panel.  When that happens the
            # LSQ fit gets dragged into a compromise pose that's
            # wrong for ALL anchors (every residual blows past
            # threshold) even though the *other* anchors agree
            # among themselves.
            #
            # If we have >= (min + 1) anchors, drop the single
            # worst-residual one and re-solve.  Validated on 241
            # captured panels (May 2026): rescues 9/72 ANNOTATED
            # panels (+12.5pp -> 93.1% lock rate) with zero
            # regressions on previously-locking panels.  The
            # dropped anchor was scan_results in 22 of 26 saves
            # (85%) -- confirming it's the dominant outlier.
            if len(measurements) > self._min_anchors_for_lock and residuals:
                worst_key = max(residuals.items(), key=lambda kv: kv[1])[0]
                reduced = {k: v for k, v in measurements.items() if k != worst_key}
                pose2, residuals2 = self._solve(reduced)
                if self._residuals_ok(residuals2):
                    pose = pose2
                    residuals = residuals2
                    measurements = reduced
                    dropped_anchor = worst_key
                    log.warning(
                        "HudPanelTracker._cold_start: dropped outlier "
                        "anchor '%s' (residual was too high); re-solved "
                        "with %d remaining anchors -> lock",
                        worst_key, len(reduced),
                    )
        if not self._residuals_ok(residuals):
            max_resid = max(residuals.values()) if residuals else 0.0
            reason = (
                f"residuals too high (max={max_resid:.2f}px > "
                f"{self.max_residual_px:.2f}px threshold); "
                f"per-anchor={residuals}"
            )
            log.warning("HudPanelTracker._cold_start: %s", reason)
            self._is_locked = False
            self._last_pose = None
            self._last_failure_reason = reason
            return None

        self._last_pose = pose
        self._is_locked = True
        self._rejection_count = 0
        self._last_failure_reason = None
        log.info(
            "HudPanelTracker: COLD-START lock @ pose=(%.1f,%.1f,scale=%.3f) "
            "from %d anchors (max_residual=%.2fpx)%s",
            pose[0], pose[1], pose[2], len(measurements),
            max(residuals.values()) if residuals else 0.0,
            f" [outlier dropped: {dropped_anchor}]" if dropped_anchor else "",
        )
        return self._pose_to_label_rows(pose, img.width, img.height)

    # ─── Internal: tracking step ───────────────────────────────────

    def _track_step(self, img: "Image.Image") -> Optional[dict]:
        """Predict anchors from last_pose, verify, update."""
        assert self._last_pose is not None
        prev_pose = self._last_pose
        prev_x, prev_y, prev_scale = prev_pose

        # Predict each anchor's pixel position from the last pose.
        predicted_centers: dict[str, tuple[float, float]] = {}
        for name, (dx, dy) in self._offsets.items():
            predicted_centers[name] = (
                prev_x + prev_scale * dx,
                prev_y + prev_scale * dy,
            )

        # Run detectors in local-search mode.
        measurements = self._collect_anchors(
            img, search_centers=predicted_centers,
        )
        log.debug(
            "HudPanelTracker._track_step: predicted=%d, measured=%d",
            len(predicted_centers), len(measurements),
        )

        # Validate the result against the rigid-body model.
        reject_reason: Optional[str] = None
        new_pose: Optional[tuple[float, float, float]] = None
        residuals: dict[str, float] = {}

        if len(measurements) < self._min_anchors_for_lock:
            reject_reason = (
                f"only {len(measurements)} anchors found "
                f"(need {self._min_anchors_for_lock})"
            )
        else:
            new_pose, residuals = self._solve(measurements)
            if not self._residuals_ok(residuals):
                reject_reason = (
                    f"max residual {max(residuals.values()):.2f}px > "
                    f"{self.max_residual_px:.2f}px"
                )
            else:
                motion = (
                    (new_pose[0] - prev_x) ** 2
                    + (new_pose[1] - prev_y) ** 2
                ) ** 0.5
                if motion > self.max_motion_px:
                    reject_reason = (
                        f"panel motion {motion:.1f}px > "
                        f"{self.max_motion_px:.1f}px"
                    )

        if reject_reason is not None or new_pose is None:
            self._rejection_count += 1
            reason = reject_reason or "no pose computed"
            log.warning(
                "HudPanelTracker: REJECT (%d/%d) — %s",
                self._rejection_count,
                self.cold_start_after_rejections,
                reason,
            )
            self._last_failure_reason = (
                f"track-step reject {self._rejection_count}/"
                f"{self.cold_start_after_rejections}: {reason}"
            )
            if self._rejection_count >= self.cold_start_after_rejections:
                log.warning(
                    "HudPanelTracker: too many rejections — dropping lock "
                    "(next track() will cold-start)"
                )
                self._is_locked = False
                self._last_pose = None
                self._rejection_count = 0
            return None

        # Accept.
        self._last_pose = new_pose
        self._rejection_count = 0
        self._is_locked = True
        self._last_failure_reason = None
        log.debug(
            "HudPanelTracker: TRACK pose=(%.1f,%.1f,scale=%.3f), "
            "max_resid=%.2fpx, motion=%.1fpx",
            new_pose[0], new_pose[1], new_pose[2],
            max(residuals.values()) if residuals else 0.0,
            (
                (new_pose[0] - prev_x) ** 2
                + (new_pose[1] - prev_y) ** 2
            ) ** 0.5,
        )
        return self._pose_to_label_rows(new_pose, img.width, img.height)

    # ─── Internal: anchor collection ───────────────────────────────

    def _collect_anchors(
        self,
        img: "Image.Image",
        search_centers: Optional[dict[str, tuple[float, float]]],
    ) -> dict[str, tuple[float, float]]:
        """Run each available detector and collect the anchor positions.

        ``search_centers`` is a dict mapping anchor name → predicted
        ``(x, y)`` in img coordinates. ``None`` means cold-start (full
        image search for every detector).

        Currently the tracker only consumes the SCAN RESULTS title
        detector (Agent B's modified ``find_scan_results_anchor``).
        Additional detectors (label_match.find_label_positions,
        chrome_lines, etc.) can be wired in by extending this method
        once Agent B has retrofitted them with the same
        ``search_center`` / ``search_radius`` / ``y_range`` kwargs.

        # TODO: when Agent B finishes, add per-row label-match calls
        # here so the tracker has 3+ measurements every frame:
        #     from ocr.sc_ocr.label_match import find_label_positions
        #     matches = find_label_positions(
        #         img, search_centers={...}, search_radius=...,
        #     )
        """
        measurements: dict[str, tuple[float, float]] = {}

        # --- Anchor: SCAN RESULTS title (Agent B's detector) ---
        sc_center = (
            search_centers.get("scan_results")
            if search_centers is not None else None
        )
        try:
            sc_anchor = _find_scan_results_anchor(
                img,
                search_center=sc_center,
                search_radius=self.search_radius,
            )
        except Exception as exc:
            log.debug(
                "HudPanelTracker: scan_results detector raised: %s",
                exc,
            )
            sc_anchor = None
        if sc_anchor is not None:
            measurements["scan_results"] = (
                float(sc_anchor["title_x"]),
                float(sc_anchor["title_y"]),
            )

        # --- Anchor: per-row labels (Agent B's modified label_match) ---
        # Build a label_match-shaped search_centers dict by mapping the
        # tracker's "label_<field>" keys back to find_label_positions'
        # field-only keys ("mass", "resistance", "instability").
        #
        # In COLD-START mode (search_centers=None) we pass None so the
        # detector does its classic full-frame multi-scale sweep.
        # In TRACKING mode we forward the predicted centers so the
        # detector restricts each label's NCC search to a small window
        # — typically a ~100× speedup and structurally immune to
        # COMPOSITION-row false matches because we never look there.
        label_search_centers: Optional[dict[str, tuple[float, float]]] = None
        if search_centers is not None:
            label_search_centers = {}
            for tracker_key, ctr in search_centers.items():
                if not tracker_key.startswith("label_"):
                    continue
                field = tracker_key[len("label_"):]
                # find_label_positions only knows mass/resistance/instability.
                if field in ("mass", "resistance", "instability"):
                    label_search_centers[field] = (
                        float(ctr[0]), float(ctr[1]),
                    )
            if not label_search_centers:
                # Empty dict means "search nothing" to find_label_positions
                # — match that intent explicitly.
                label_search_centers = None
        try:
            # Try absolute import first (production layout).
            try:
                from ocr.sc_ocr.label_match import find_label_positions
            except ImportError:
                from .label_match import find_label_positions  # type: ignore
            if label_search_centers is None:
                label_matches = find_label_positions(img)
            else:
                label_matches = find_label_positions(
                    img,
                    search_centers=label_search_centers,
                    search_radius=self.search_radius,
                )
        except Exception as exc:
            log.debug(
                "HudPanelTracker: label_match raised: %s", exc,
            )
            label_matches = {}

        for key, info in (label_matches or {}).items():
            try:
                # The offset table key is "label_<field>"; map the
                # label_match result keys ("mass", "resistance",
                # "instability") into that namespace.
                offset_key = f"label_{key}"
                if offset_key not in self._offsets:
                    continue
                measurements[offset_key] = (
                    float(info["x"]),
                    float(info["y"]),
                )
            except Exception:
                continue

        # ── MASS y-position sanity check ──
        # label_mass is geometrically constrained to sit
        # ``scale * 3.33`` pixels below scan_results (per the offset
        # table -- "title-height units").  Even at the largest
        # observed in-game HUD scale (~60 px title), that puts MASS
        # at most ~200 pixels below SCAN RESULTS.  Detecting MASS
        # FAR below that distance means the NCC matched a spurious
        # text fragment elsewhere on screen (UNKNOWN footer, mineral
        # name, COMPOSITION section, etc.).
        #
        # Reject such mass measurements -- this lets the colon-anchor
        # fallback (scan_results-anchored mode) try to synthesize
        # MASS from the colon glyphs at the actual row positions.
        # Without this gate, the wrong-mass anchor at e.g. y=503 in a
        # 670-tall image gates the colon fallback into the wrong
        # y-band (mass_y - 20 .. mass_y + 200) and finds no colons.
        # Validated on cap_20260418_155746_172.png: MASS detected at
        # (88, 503) when scan_results was at (174, 130) -- delta_y =
        # 373 px, far above the 300 px plausibility ceiling.
        MAX_MASS_BELOW_SCAN_PX = 300
        if (
            "label_mass" in measurements
            and "scan_results" in measurements
        ):
            _sx, _sy = measurements["scan_results"]
            _mx, _my = measurements["label_mass"]
            _delta_y = _my - _sy
            if _delta_y > MAX_MASS_BELOW_SCAN_PX or _delta_y < -20:
                log.warning(
                    "HudPanelTracker: rejecting MASS at (%.0f, %.0f) -- "
                    "delta_y=%.0fpx from scan_results at (%.0f, %.0f) "
                    "is implausible (max %d / min -20). Likely NCC "
                    "false match; dropping so colon fallback can use "
                    "scan_results-anchored mode.",
                    _mx, _my, _delta_y, _sx, _sy,
                    MAX_MASS_BELOW_SCAN_PX,
                )
                del measurements["label_mass"]
                # Also clear synthesized resistance/instability if
                # they came from the wrong-mass position -- they'd be
                # geometrically stale.
                for synth_key in ("label_resistance", "label_instability"):
                    if (
                        synth_key in measurements
                        and abs(measurements[synth_key][0] - _mx) < 2
                    ):
                        del measurements[synth_key]

        # ── Fallback: colon-glyph NCC for missing row anchors ────────
        #
        # When label_match misses RESISTANCE or INSTABILITY (cropped
        # HUD, dim text, chromatic aberration, non-standard scaling)
        # the tracker would otherwise fall one anchor short of the
        # 3-required cold-start quorum -- the
        # "only 2 anchors found (need 3); keys=[label_mass,
        # scan_results]" state seen in the v2.2.12 user crash log.
        #
        # Recover by running an INDEPENDENT colon-glyph NCC over the
        # expected y-band: the colons after MASS / RESISTANCE /
        # INSTABILITY are visually identical across all three rows
        # and at all HUD scales (~3-6 px x ~10-14 px), so they
        # survive the conditions that defeat the full-word label
        # NCC.  Colons are returned top-to-bottom by the API, so
        # positional assignment binds them to the missing
        # label_<field> keys without needing to predict exact row
        # pitch.
        #
        # Requires label_mass to be present (anchors the search
        # y-band and shares the x-column with the missing labels --
        # per the offset table all three label_<field> have
        # offset_x = 0.0, so synthesizing the missing labels at
        # mass_x is geometrically consistent with the rigid-body
        # solver).
        #
        # Mirrors the parallel fallback in
        # ``ocr.onnx_hud_reader._find_label_rows`` (lines 1379+)
        # but inside the tracker's anchor-collection path so the
        # cold-start quorum can be reached -- previously colon
        # recovery only existed in a sibling code path that never
        # ran for the failing user.
        _NEEDED_LABEL_KEYS = ("label_mass", "label_resistance", "label_instability")
        _missing = [k for k in _NEEDED_LABEL_KEYS if k not in measurements]
        # Pick a column anchor: prefer label_mass (most precisely
        # aligned with the other label rows), fall back to scan_results
        # when MASS itself failed.  Both share x=0 in the offset table,
        # so either gives a geometrically valid column for the LSQ.
        _column_anchor: Optional[tuple[str, float, float]] = None
        if "label_mass" in measurements:
            _mx, _my = measurements["label_mass"]
            _column_anchor = ("label_mass", float(_mx), float(_my))
        elif "scan_results" in measurements:
            # scan_results case: y-band starts well below the title
            # since label_mass would sit ~3.33 * scale pixels below it.
            # Without knowing the scale a priori we use a generous
            # band that covers all observed HUD sizes (30..500 px
            # below the title).  The colon-finder is cheap and
            # discriminative enough to handle the wider band.
            _sx, _sy = measurements["scan_results"]
            _column_anchor = ("scan_results", float(_sx), float(_sy))
        if _missing and _column_anchor is not None:
            anchor_kind, _ax, _ay = _column_anchor
            _colons: list[dict] = []
            _ca_y_top = 0
            _ca_y_bot = 0
            _ca_x_left = 0
            _ca_x_right = 0
            try:
                try:
                    from ocr.sc_ocr.colon_anchor import find_colons
                except ImportError:
                    from .colon_anchor import find_colons  # type: ignore
                if anchor_kind == "label_mass":
                    # Search from slightly above mass downward enough
                    # to cover mass + resistance + instability rows.
                    _ca_y_top = max(0, int(_ay) - 20)
                    _ca_y_bot = min(img.height, int(_ay) + 200)
                else:
                    # Search well below the SCAN RESULTS title; label
                    # rows start ~3.33 * scale px below at the
                    # smallest observed scale -> ~50 px minimum.
                    _ca_y_top = max(0, int(_ay) + 30)
                    _ca_y_bot = min(img.height, int(_ay) + 500)
                _ca_x_left = max(0, int(_ax) - 20)
                _ca_x_right = min(img.width, int(img.width * 0.65))
                _colons = find_colons(
                    img,
                    y_band=(_ca_y_top, _ca_y_bot),
                    x_range=(_ca_x_left, _ca_x_right),
                )
            except Exception as exc:
                log.debug(
                    "HudPanelTracker: colon fallback raised: %s", exc,
                )

            if _colons:
                # MASS-anchored mode: drop colons above mass row (they
                # would be header colons), then drop the first
                # remaining colon (mass's own, since mass is already
                # matched).  Remaining colons map sequentially to the
                # missing RESISTANCE/INSTABILITY rows.
                #
                # SCAN_RESULTS-anchored mode: ALL detected colons are
                # candidates for the 3 label rows in order (mass,
                # resistance, instability).
                if anchor_kind == "label_mass":
                    _candidate_colons = [
                        c for c in _colons if c["y"] >= _ay - 10
                    ]
                    _candidate_colons = _candidate_colons[1:]
                else:
                    _candidate_colons = list(_colons)
                _added: list[str] = []
                for label_key in _NEEDED_LABEL_KEYS:
                    if label_key in measurements:
                        continue  # already matched, leave alone
                    if not _candidate_colons:
                        break
                    _c = _candidate_colons.pop(0)
                    # All three label_<field> have offset_x = 0.0
                    # in the rigid-body offset table, so using the
                    # column anchor's x is geometrically consistent
                    # with the solver.
                    measurements[label_key] = (
                        _ax, float(_c["y"]),
                    )
                    _added.append(label_key)
                if _added:
                    log.warning(
                        "HudPanelTracker: colon fallback (%s-anchored) "
                        "synthesized %d missing anchor(s) from %d "
                        "colon detection(s): %s",
                        anchor_kind, len(_added), len(_colons), _added,
                    )
            elif _missing:
                log.warning(
                    "HudPanelTracker: colon fallback (%s-anchored) ran "
                    "but found 0 colons in y_band=(%d, %d) x_range=(%d, %d) "
                    "-- missing %s remain unmeasured",
                    anchor_kind, _ca_y_top, _ca_y_bot,
                    _ca_x_left, _ca_x_right, _missing,
                )

        return measurements

    # ─── Internal: LSQ solver and validation ───────────────────────

    def _solve(
        self,
        measurements: dict[str, tuple[float, float]],
    ) -> tuple[tuple[float, float, float], dict[str, float]]:
        # Auto-calibration: if the calibration learner has published a
        # stable set of label-top multipliers for this user's HUD,
        # use those instead of the defaults. The published table
        # always contains every key the solver needs (mineral / mass /
        # resistance / instability) — the publisher backfills any
        # unlearned fields with defaults — so we can pass it straight
        # to the solver without merging.
        offsets = get_learned_offsets() or self._offsets
        return _solve_panel_pose(measurements, offsets)

    def _residuals_ok(self, residuals: dict[str, float]) -> bool:
        if not residuals:
            return False
        return max(residuals.values()) <= self.max_residual_px

    # ─── Internal: pose -> label_rows conversion ───────────────────

    def _pose_to_label_rows(
        self,
        pose: tuple[float, float, float],
        img_width: int,
        img_height: int,
    ) -> dict[str, tuple[int, int, int]]:
        """Convert a panel pose to the ``label_rows`` dict shape that
        ``_find_label_rows`` returns to its caller.

        Output shape::

            {
                "_mineral_row": (y1, y2, label_right),
                "mass":         (y1, y2, label_right),
                "resistance":   (y1, y2, label_right),
                "instability":  (y1, y2, label_right),
            }

        ``y1`` / ``y2`` bracket the row's vertical band; ``label_right``
        is the x past which the value column begins (used by
        downstream value-crop code).

        With the title-h-unit offset convention, the solver returns
        ``scale`` ≈ actual title height in pixels. Row centers come
        from ``_ROW_OFFSET_MULTS`` (title-h multipliers from the row
        TOP to each row's CENTER) and band height comes from
        ``_ROW_HEIGHT_MULT`` × title_h_px.
        """
        panel_x, panel_y, scale = pose
        # scale ≈ actual title height in pixels (title-h-unit
        # convention). Treat it as title_h directly.
        title_h_px = max(1.0, float(scale))
        title_bottom = panel_y + title_h_px
        # Band half-height: 0.5 × title_h gives a band ~= title_h tall.
        # This stays comfortably under the downstream CRNN's 50-px
        # value-crop ceiling for typical HUD resolutions (title_h≈45 →
        # band 44px). Earlier 0.75 × title_h produced 66-72px bands
        # that broke CRNN's read; rejection cascaded into wrong
        # COUNT-ORACLE digit counts and the proportional segmenter
        # fabricated extra digits (e.g. 23694 → 236944) which then
        # got frozen.
        half_h = max(8, int(title_h_px * 0.5))
        label_right = int(img_width * _VALUE_COL_LEFT_FRAC)

        # Auto-calibration: use the learned row-center mults if the
        # calibration learner has published them, otherwise fall back
        # to the production defaults. The published table always
        # contains every row key so iteration shape is unchanged.
        row_mults = get_learned_row_mults() or _ROW_OFFSET_MULTS
        rows: dict[str, tuple[int, int, int]] = {}
        for key, mult in row_mults.items():
            # mult is title-h units from title TOP to row CENTER.
            center_y = panel_y + title_h_px * mult
            y1 = max(0, int(center_y - half_h))
            y2 = min(int(img_height), int(center_y + half_h))
            if y2 - y1 < 4:
                # Too thin to be useful; skip this field.
                continue
            rows[key] = (y1, y2, label_right)
        return rows
