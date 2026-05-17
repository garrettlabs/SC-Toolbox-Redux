"""RGB comma detector for SC mining-signature value crops.

The signature value crop is rendered in one of two layouts:

    4-digit:  D , D D D
    5-digit:  D D , D D D

The COMMA is the single most distinctive structural feature of the
crop — a small, narrow, low-only-ink blob that sits in the bottom
~30% of the line height between digits. Once we know where the comma
is, every digit's column is determined by simple offset arithmetic
from the comma's center.

This module provides the dedicated "find the comma" detector — the
signature HUD's equivalent of the mining HUD's ``MASS:`` label NCC.
Unlike the ``_detect_comma_extent`` helper inside
``signal_proportional_segmenter.py`` (a quick column-projection
heuristic for hypothesis tiebreaking), this is a structurally-checked
finder that returns:

  * a precise column AND bbox
  * a polarity-aware variant (primary + inverted)
  * a "voted" combiner that runs both polarities and reports
    agreement
  * a 6-check structural validator with per-check telemetry

Decorrelating the two polarities is exactly the same trick that
``model_signal_inv_cnn`` plays with respect to ``model_signal_cnn``:
the same shape from a different polarity tells us whether the
detection survives the contrast inversion (real comma) or only
exists in one polarity (chromatic-aberration artefact).

Public API
----------
``find_comma(rgb_image, *, polarity="auto", median_digit_width=None)
    → dict | None``

``find_comma_inv(rgb_image, *, median_digit_width=None) → dict | None``

``find_comma_voted(rgb_image, *, median_digit_width=None) → dict | None``

The dict shape:

    {
        "bbox": (x, y, w, h),       # comma bbox in input crop coords
        "x_center": int,            # x_center of bbox
        "confidence": float,        # 0..1 (n_checks_passed / 6)
        "polarity_used": str,       # "bright_on_dark" | "dark_on_light"
        "details": {
            "checks": {             # per-check booleans + supporting numbers
                "bottom_heavy_ink": bool,
                "top_empty": bool,
                "narrow_width": bool,
                "small_mass": bool,
                "aspect_compact": bool,
                "isolated_horizontally": bool,
            },
            "numbers": {            # per-check supporting metrics
                "bottom_frac": float,
                "top_frac": float,
                "width_px": int,
                "height_px": int,
                "mass_px": int,
                "median_digit_width_used": float,
                "median_digit_mass_used": float,
                "left_gap_px": int,
                "right_gap_px": int,
            },
            "n_checks_passed": int,
            "n_checks_total": int,  # always 6
        },
    }

For ``find_comma_voted`` the dict additionally contains:

    "voted":   True,
    "agreed":  bool,
    "primary": dict | None,         # raw find_comma() result
    "inverted": dict | None,        # raw find_comma_inv() result
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np
from PIL import Image
from scipy import ndimage  # type: ignore[import]

log = logging.getLogger(__name__)

ImageLike = Union[Image.Image, np.ndarray]

__all__ = [
    "find_comma",
    "find_comma_inv",
    "find_comma_voted",
]


# ── Tunable constants ────────────────────────────────────────────────
# All thresholds are derived from empirical measurements of the SC
# signature font on real region2 captures (e.g. cap_20260418_160452_795
# GT 16,960). See the docstring of each check for the supporting
# numbers.

# Total number of structural checks (denominator for confidence).
_N_CHECKS = 6

# Pass threshold: ≥4 of 6 checks must vote yes.
_MIN_CHECKS_TO_PASS = 4

# Bottom-heavy: the candidate's ENTIRE ink mass should sit in the
# bottom band of the WHOLE-CROP row range (not within its own bbox).
# A comma's bbox lives at the bottom of the crop; its top y is
# already > 60% of the crop height. We require >= 50% of the
# candidate's ink mass to be in the bottom 35% of crop height.
# Digits are full-height (ink spans top → bottom of the crop), so
# their bottom-band fraction is ~30-40% of total mass, well below 50%.
_BOTTOM_HEAVY_FRAC_THR = 0.50
_BOTTOM_BAND_FRAC = 0.35

# Top-empty: the candidate has < 10% of its ink in the top 30% of
# the WHOLE-CROP row range. Digits ink the top of the crop heavily
# (~30-40% of their mass); commas have zero ink there.
_TOP_EMPTY_FRAC_THR = 0.10
_TOP_BAND_FRAC = 0.30

# Narrow-width: comma bbox width must be < median_digit_width × 0.6
# OR < 7 px when median_digit_width is unknown. SC signature commas
# are 3-6 px wide vs digits 8-14 px. Empirical measurement:
# cap_20260418_160452_795 (16,960) renders the comma at 5 px wide
# next to digits at 7-12 px wide; ratio-based threshold of 0.6
# gives a narrow_width threshold of 7 px when digits are 12 px,
# reliably accepting 5-px commas while rejecting any 7-8 px digit.
_NARROW_RATIO_THR = 0.6
_NARROW_ABS_THR_PX = 7

# Small-mass: comma total ink mass must be < median_digit_mass × 0.4
# OR < 30 px² when median_digit_mass is unknown. Commas are ~15-25 px²
# of ink; digits 100-300 px² depending on the glyph.
_SMALL_MASS_RATIO_THR = 0.4
_SMALL_MASS_ABS_THR_PX2 = 30

# Aspect-compact: the comma's bbox h/w ratio sits in [0.7, 1.6] — a
# squarish small blob, not a tall thin stroke (a `1` digit) or a wide
# flat bar (the pill outline).
_ASPECT_LO = 0.7
_ASPECT_HI = 1.6

# Isolation: the COLUMN range of the comma must have no ink in the
# UPPER ~60% of the crop (a real comma is bottom-only, so the columns
# directly above it are background). Threshold: < 5% of the comma's
# column-range pixels in the top 60% of crop rows are ink. Rationale:
# a real digit occupies its full column with ink top-to-bottom, so
# even a narrow `1` would have >> 5% of its column's top-60% pixels
# inked. The comma's columns have ZERO top-60% ink (with 5% buffer
# for chromatic-aberration speckle).
_ISOLATION_TOP_BAND_FRAC = 0.60
_ISOLATION_TOP_INK_FRAC_THR = 0.05

# Position prior: prefer comma candidates in the middle 60% of the
# crop (commas are between digits, never at the very edges). This
# resolves multi-candidate ties when several blobs all pass the
# checks (rare, but happens on aberration-fused crops).
_MIDDLE_REGION_FRAC_LO = 0.20
_MIDDLE_REGION_FRAC_HI = 0.80

# Polarity detection (auto mode): if the border-median is brighter
# than the center-median by this much, the input is dark-on-light
# (need to invert before processing).
_POLARITY_BORDER_MARGIN = 5

# Threshold-relative-to-otsu fallback. Some captures have very low
# dynamic range and Otsu returns a degenerate threshold. We add a
# small offset above the median to ensure we get SOME ink mask, even
# if it's noisy.
_FALLBACK_OFFSET_OVER_MEDIAN = 30

# Threshold escalation: we try Otsu first, then a sequence of
# percentile-based thresholds (``min + (max - min) × frac``). The
# escalation is necessary because the SC signature font's bright-
# text-on-dark renders have heavy chromatic-aberration halos that
# Otsu lumps in with the digit cores — and those halos visually
# fuse adjacent digits into one mega-component. The escalating
# fractional thresholds peel back to the digit cores so the comma
# (which has no halo of its own — it's already a small dim blob)
# emerges as a separate component. Order matters: Otsu first because
# it's right on captures with crisp polarity-matched lighting; the
# higher fractions handle the chromatic-aberration captures. The
# 0.85 ceiling stays below the visible-saturation regime (>240) so
# we don't accidentally lose the dimmer comma's pixels.
_THRESHOLD_FRACTIONS = (0.55, 0.65, 0.75, 0.80, 0.85, 0.90, 0.93)

# Voted agreement: two polarities count as "agreeing" if their bboxes
# either (a) overlap with IoU > 0.4, OR (b) their x-centers are
# within 3 px of each other AND their bbox y-ranges overlap. Pure
# IoU is too strict because the inv-polarity threshold often picks
# a tighter mask (just the densest part of the comma blob), giving
# a smaller bbox even when both clearly point at the same feature.
# X-center proximity catches this case while still rejecting
# unrelated blobs at different parts of the crop.
_VOTED_IOU_AGREEMENT_THR = 0.4
_VOTED_X_CENTER_AGREEMENT_PX = 3

# Confidence boost when both polarities agree (added to the higher-
# confidence result, capped at 1.0).
_VOTED_AGREEMENT_BOOST = 0.10


# ── Helpers ──────────────────────────────────────────────────────────


def _to_rgb_array(image: ImageLike) -> Optional[np.ndarray]:
    """Coerce input to a (H, W, 3) uint8 numpy array."""
    if image is None:
        return None
    arr: Optional[np.ndarray]
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        return None

    if arr is None or arr.size == 0:
        return None

    if arr.ndim == 2:
        # Gray → fake RGB by stacking.
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim != 3:
        return None
    if arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        # Single-channel as RGB.
        arr = np.stack([arr[:, :, 0]] * 3, axis=-1)

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[0] < 4 or arr.shape[1] < 4:
        return None
    return arr


def _luma(rgb: np.ndarray) -> np.ndarray:
    """Max-of-channels grayscale (matches the rest of the SC OCR
    pipeline which uses per-pixel max to preserve chromatic-aberration
    coloured fringes that luma weighting would average away)."""
    return rgb.max(axis=2).astype(np.uint8)


def _detect_polarity(gray: np.ndarray) -> str:
    """Return ``"bright_on_dark"`` or ``"dark_on_light"`` from
    border-median vs center-median.

    Border-median rule: the value-bbox crop has padding on the top
    and bottom edges (digit baselines and ascender heights don't
    reach the bbox corners — there's always 1-3 px of background).
    Border > center ⇒ background bright ⇒ ink dark ⇒ dark_on_light.
    """
    h, w = gray.shape[:2]
    if h < 4 or w < 4:
        return "bright_on_dark"
    # Border strip — top + bottom rows, skipping corner columns.
    margin = max(1, w // 8)
    border = np.concatenate([
        gray[0, margin:w - margin],
        gray[h - 1, margin:w - margin],
    ])
    bg_med = float(np.median(border))
    # Center strip — middle 60% in both dimensions.
    cx0 = max(1, w // 5)
    cx1 = max(cx0 + 1, w - w // 5)
    cy0 = max(1, h // 5)
    cy1 = max(cy0 + 1, h - h // 5)
    center_med = float(np.median(gray[cy0:cy1, cx0:cx1]))
    if bg_med > center_med + _POLARITY_BORDER_MARGIN:
        return "dark_on_light"
    return "bright_on_dark"


def _normalize_to_canonical(
    rgb: np.ndarray, polarity: str,
) -> tuple[np.ndarray, str]:
    """Force the input to bright-text-on-dark polarity (canonical).

    Returns ``(canonical_gray, polarity_used)`` where ``polarity_used``
    is the polarity DETECTED (or supplied) — it is not flipped, even
    when we inverted the array. Callers report the detected polarity
    so the user can see what the auto-detector decided.
    """
    gray = _luma(rgb)
    if polarity == "auto":
        polarity_used = _detect_polarity(gray)
    elif polarity == "dark_on_light":
        polarity_used = "dark_on_light"
    elif polarity == "bright_on_dark":
        polarity_used = "bright_on_dark"
    else:
        polarity_used = "bright_on_dark"

    if polarity_used == "dark_on_light":
        gray = (255 - gray).astype(np.uint8)
    return gray, polarity_used


def _otsu_threshold(gray: np.ndarray) -> int:
    """Standard Otsu's method via the histogram. Returns a uint8
    threshold value in [0, 255].

    No SciPy/OpenCV required — straight numpy.
    """
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = hist.sum()
    if total <= 0:
        return 128
    sum_total = float((np.arange(256) * hist).sum())
    sum_bg = 0.0
    weight_bg = 0.0
    var_max = -1.0
    threshold = 128
    for t in range(256):
        weight_bg += float(hist[t])
        if weight_bg == 0:
            continue
        weight_fg = float(total) - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * float(hist[t])
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t
    return int(threshold)


def _build_ink_mask(gray_canon: np.ndarray, thr: int) -> np.ndarray:
    """Threshold the canonicalized gray to a boolean ink mask at
    a SPECIFIC threshold value. Returns an all-False mask on
    degenerate input (low dynamic range).
    """
    g_min = int(gray_canon.min())
    g_max = int(gray_canon.max())
    if g_max - g_min < 16:
        return np.zeros_like(gray_canon, dtype=bool)
    return gray_canon > thr


def _candidate_thresholds(gray_canon: np.ndarray) -> list[int]:
    """Return the ordered list of thresholds the finder will try.

    Otsu first (right on crisp captures), followed by percentile-
    fraction thresholds at progressively higher cuts. Each higher
    threshold peels chromatic-aberration halos off the digit cores
    so adjacent digits separate into their own connected components,
    revealing the comma as its own small bottom-only blob.
    """
    g_min = int(gray_canon.min())
    g_max = int(gray_canon.max())
    if g_max - g_min < 16:
        return []
    thresholds: list[int] = []
    # Otsu (first try).
    otsu = _otsu_threshold(gray_canon)
    if otsu <= g_min + 4:
        # Degenerate Otsu — fall back to median + offset.
        otsu = max(
            g_min + 8,
            int(np.median(gray_canon)) + _FALLBACK_OFFSET_OVER_MEDIAN,
        )
    otsu = max(g_min + 1, min(g_max - 1, otsu))
    thresholds.append(int(otsu))
    # Fractional thresholds.
    for frac in _THRESHOLD_FRACTIONS:
        t = int(g_min + (g_max - g_min) * frac)
        t = max(g_min + 1, min(g_max - 1, t))
        if t not in thresholds:
            thresholds.append(t)
    return thresholds


def _component_bbox(component_mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight bbox of a boolean component mask. Returns ``(x, y, w, h)``."""
    ys, xs = np.where(component_mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    x = int(xs.min())
    y = int(ys.min())
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    return (x, y, w, h)


def _measure_isolation_top_ink(
    ink_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[float, int, int]:
    """Measure how empty the columns directly above the comma are.

    A real comma's column range has NO ink in the top 60% of the
    crop (the comma is bottom-only, so above it is background).
    Returns ``(top_ink_frac, left_gap_px, right_gap_px)`` where:
      * ``top_ink_frac`` is the fraction of pixels in the comma's
        column range AND the top 60% of crop rows that are inked.
      * ``left_gap_px`` and ``right_gap_px`` are kept for telemetry —
        the count of empty (no-top-ink) columns immediately
        adjacent to the comma's bbox in the top 60% of rows.
    """
    x, y, w, h = bbox
    H, W = ink_mask.shape[:2]
    top_band_h = max(1, int(round(H * _ISOLATION_TOP_BAND_FRAC)))
    top_band = ink_mask[:top_band_h, :]
    col_has_top_ink = top_band.any(axis=0)
    # Fraction of the comma's column range that has ink in the top band.
    if w > 0:
        top_ink_in_comma_cols = int(top_band[:, x:x + w].sum())
        total_pixels = top_band_h * w
        top_ink_frac = top_ink_in_comma_cols / max(1, total_pixels)
    else:
        top_ink_frac = 1.0

    # Telemetry: count adjacent empty columns above the comma.
    left = 0
    cur = x - 1
    while cur >= 0 and not col_has_top_ink[cur]:
        left += 1
        cur -= 1
    right = 0
    cur = x + w
    while cur < W and not col_has_top_ink[cur]:
        right += 1
        cur += 1
    return float(top_ink_frac), int(left), int(right)


def _evaluate_candidate(
    ink_mask: np.ndarray,
    component_mask: np.ndarray,
    *,
    median_digit_width: Optional[float],
    median_digit_mass: Optional[float],
) -> dict[str, Any]:
    """Run the 6 structural checks on one connected component.

    Returns a dict with the bbox, mass, per-check booleans, and the
    n_passed count. The caller decides whether to accept (≥
    ``_MIN_CHECKS_TO_PASS``) and converts to confidence.
    """
    H, W = ink_mask.shape[:2]
    bbox = _component_bbox(component_mask)
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return {
            "bbox": bbox,
            "x_center": 0,
            "mass_px": 0,
            "checks": {k: False for k in (
                "bottom_heavy_ink", "top_empty", "narrow_width",
                "small_mass", "aspect_compact", "isolated_horizontally",
            )},
            "numbers": {},
            "n_passed": 0,
        }
    sub = component_mask[y:y + h, x:x + w]
    mass = int(sub.sum())
    x_center = int(x + w / 2)

    # ── Check 1: bottom_heavy_ink ────────────────────────────────────
    # >= 50% of ink mass in the bottom 35% of WHOLE CROP height.
    crop_band_h = max(1, int(round(H * _BOTTOM_BAND_FRAC)))
    crop_bottom_y0 = H - crop_band_h
    # mass-in-bottom-band-of-crop = mass of component pixels with y >= crop_bottom_y0
    if y + h <= crop_bottom_y0:
        bottom_in_crop = 0
    else:
        bottom_in_crop = int(
            component_mask[max(y, crop_bottom_y0):y + h, x:x + w].sum()
        )
    bottom_frac = bottom_in_crop / max(1, mass)
    bottom_heavy_ink = bottom_frac >= _BOTTOM_HEAVY_FRAC_THR

    # ── Check 2: top_empty ───────────────────────────────────────────
    # < 10% of ink mass in the top 30% of WHOLE CROP height.
    crop_top_band_h = max(1, int(round(H * _TOP_BAND_FRAC)))
    if y >= crop_top_band_h:
        top_in_crop = 0
    else:
        top_in_crop = int(
            component_mask[y:min(y + h, crop_top_band_h), x:x + w].sum()
        )
    top_frac = top_in_crop / max(1, mass)
    top_empty = top_frac <= _TOP_EMPTY_FRAC_THR

    # ── Check 3: narrow_width ────────────────────────────────────────
    # bbox width below threshold.
    if median_digit_width is not None and median_digit_width > 0:
        narrow_width = w < median_digit_width * _NARROW_RATIO_THR
    else:
        narrow_width = w < _NARROW_ABS_THR_PX

    # ── Check 4: small_mass ──────────────────────────────────────────
    # total ink mass below threshold.
    if median_digit_mass is not None and median_digit_mass > 0:
        small_mass = mass < median_digit_mass * _SMALL_MASS_RATIO_THR
    else:
        small_mass = mass < _SMALL_MASS_ABS_THR_PX2

    # ── Check 5: aspect_compact ──────────────────────────────────────
    # h/w ratio in [0.7, 1.6] — squarish, not tall-thin or flat-wide.
    aspect = h / max(1, w)
    aspect_compact = (_ASPECT_LO <= aspect <= _ASPECT_HI)

    # ── Check 6: isolated_horizontally ───────────────────────────────
    # The comma's column range has < 5% ink in the top 60% of crop
    # rows. Real digits ink their full column top-to-bottom, so this
    # is a strong shape-based isolation signal that doesn't depend
    # on the noise sensitivity of column-gap counting.
    top_ink_frac, left_gap, right_gap = _measure_isolation_top_ink(
        ink_mask, bbox,
    )
    isolated_horizontally = top_ink_frac <= _ISOLATION_TOP_INK_FRAC_THR

    checks = {
        "bottom_heavy_ink": bool(bottom_heavy_ink),
        "top_empty": bool(top_empty),
        "narrow_width": bool(narrow_width),
        "small_mass": bool(small_mass),
        "aspect_compact": bool(aspect_compact),
        "isolated_horizontally": bool(isolated_horizontally),
    }
    n_passed = sum(1 for v in checks.values() if v)

    return {
        "bbox": bbox,
        "x_center": x_center,
        "mass_px": mass,
        "checks": checks,
        "numbers": {
            "bottom_frac": float(bottom_frac),
            "top_frac": float(top_frac),
            "width_px": int(w),
            "height_px": int(h),
            "mass_px": int(mass),
            "median_digit_width_used": (
                float(median_digit_width)
                if median_digit_width is not None
                else float(_NARROW_ABS_THR_PX / _NARROW_RATIO_THR)
            ),
            "median_digit_mass_used": (
                float(median_digit_mass)
                if median_digit_mass is not None
                else float(_SMALL_MASS_ABS_THR_PX2 / _SMALL_MASS_RATIO_THR)
            ),
            "left_gap_px": int(left_gap),
            "right_gap_px": int(right_gap),
            "top_ink_frac_in_comma_cols": float(top_ink_frac),
        },
        "n_passed": int(n_passed),
    }


def _estimate_digit_metrics(
    components: list[np.ndarray],
    ink_mask: np.ndarray,
) -> tuple[float, float]:
    """Estimate median digit width + mass from the connected components.

    The comma is one of the components and would skew a naive median,
    so we discard the smallest component (likely the comma) and the
    largest (could be a fused digit-pair from chromatic aberration).
    Returns ``(median_digit_width, median_digit_mass)``.

    On degenerate inputs (< 3 components) returns conservative defaults
    that keep the absolute thresholds in play.
    """
    if len(components) < 3:
        # Not enough info — return defaults that keep the absolute
        # thresholds active.
        return float(_NARROW_ABS_THR_PX / _NARROW_RATIO_THR), float(
            _SMALL_MASS_ABS_THR_PX2 / _SMALL_MASS_RATIO_THR
        )
    widths = []
    masses = []
    for comp in components:
        bbox = _component_bbox(comp)
        widths.append(bbox[2])
        masses.append(int(comp.sum()))
    widths_arr = np.array(widths, dtype=np.float32)
    masses_arr = np.array(masses, dtype=np.float32)
    # Drop the smallest (likely the comma) and the largest (likely
    # fused-digits) components.
    if widths_arr.size >= 3:
        order = np.argsort(widths_arr)
        keep = order[1:-1] if widths_arr.size > 3 else order[1:]
        widths_arr = widths_arr[keep]
        masses_arr = masses_arr[keep]
    median_w = float(np.median(widths_arr)) if widths_arr.size else float(
        _NARROW_ABS_THR_PX / _NARROW_RATIO_THR
    )
    median_m = float(np.median(masses_arr)) if masses_arr.size else float(
        _SMALL_MASS_ABS_THR_PX2 / _SMALL_MASS_RATIO_THR
    )
    return median_w, median_m


# ── Core single-polarity finder ──────────────────────────────────────


def _find_comma_internal(
    rgb_image: ImageLike,
    *,
    polarity: str,
    median_digit_width: Optional[float],
) -> Optional[dict[str, Any]]:
    """Implementation backbone shared by ``find_comma`` and
    ``find_comma_inv``.

    The two polarity variants differ only in the input passed at the
    top: the inverted variant pre-flips the gray channel so commas
    that print as DARK pixels on a LIGHT pill (the inverted polarity)
    are surfaced the same way commas that print as BRIGHT pixels on a
    DARK background do (the canonical polarity).
    """
    rgb = _to_rgb_array(rgb_image)
    if rgb is None:
        return None

    gray, polarity_used = _normalize_to_canonical(rgb, polarity)
    H, W = gray.shape[:2]

    thresholds = _candidate_thresholds(gray)
    if not thresholds:
        return None

    # Walk the threshold ladder until we find at least one candidate
    # that passes ≥ _MIN_CHECKS_TO_PASS checks. Higher thresholds are
    # more aggressive at peeling chromatic-aberration halos that
    # otherwise fuse adjacent digits into one mega-component (and
    # bury the comma as a single-row protrusion at the bottom of
    # that mega-component, where the structural checks can't see
    # it as a standalone blob).
    candidates: list[dict[str, Any]] = []
    for thr in thresholds:
        ink_mask = _build_ink_mask(gray, thr)
        if not ink_mask.any():
            continue
        labels, n_components = ndimage.label(ink_mask)
        if n_components <= 0:
            continue
        components: list[np.ndarray] = []
        for label_idx in range(1, n_components + 1):
            comp = (labels == label_idx)
            if comp.sum() < 3:
                continue
            components.append(comp)
        if not components:
            continue
        if median_digit_width is None:
            med_w, med_m = _estimate_digit_metrics(components, ink_mask)
        else:
            med_w = float(median_digit_width)
            med_m = float(median_digit_width) * float(H) * 0.6
        candidates_at_thr: list[dict[str, Any]] = []
        for comp in components:
            cand = _evaluate_candidate(
                ink_mask, comp,
                median_digit_width=med_w,
                median_digit_mass=med_m,
            )
            if cand["n_passed"] >= _MIN_CHECKS_TO_PASS:
                candidates_at_thr.append(cand)
        if candidates_at_thr:
            candidates = candidates_at_thr
            break

    if not candidates:
        return None

    # Resolve ties by preferring candidates whose x_center is in the
    # middle 60% of the crop. Outside that band we still consider
    # them, but with a penalty.
    middle_lo = W * _MIDDLE_REGION_FRAC_LO
    middle_hi = W * _MIDDLE_REGION_FRAC_HI

    def _candidate_priority(c: dict[str, Any]) -> tuple[float, float]:
        in_middle = middle_lo <= c["x_center"] <= middle_hi
        # Higher n_passed first, in-middle preferred, smaller x distance
        # from center as tiebreak.
        x_distance_from_center = abs(c["x_center"] - W * 0.5)
        priority = c["n_passed"] + (0.5 if in_middle else 0.0)
        return priority, -x_distance_from_center

    best = max(candidates, key=_candidate_priority)

    return {
        "bbox": (
            int(best["bbox"][0]),
            int(best["bbox"][1]),
            int(best["bbox"][2]),
            int(best["bbox"][3]),
        ),
        "x_center": int(best["x_center"]),
        "confidence": float(best["n_passed"] / float(_N_CHECKS)),
        "polarity_used": polarity_used,
        "details": {
            "checks": dict(best["checks"]),
            "numbers": dict(best["numbers"]),
            "n_checks_passed": int(best["n_passed"]),
            "n_checks_total": int(_N_CHECKS),
        },
    }


# ── Public API ──────────────────────────────────────────────────────


def find_comma(
    rgb_image: ImageLike,
    *,
    polarity: str = "auto",
    median_digit_width: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Locate the comma in a signature value crop.

    Parameters
    ----------
    rgb_image
        PIL ``Image.Image`` (RGB), 2-D gray array, or 3-D RGB array.
    polarity
        ``"auto"`` (default) — sample border vs center to detect the
        polarity. ``"bright_on_dark"`` — treat input as canonical (no
        inversion). ``"dark_on_light"`` — pre-invert before processing.
    median_digit_width
        If known, supplies a tighter width threshold (comma must be
        narrower than ``median_digit_width × 0.5``). When ``None``,
        absolute thresholds are used (comma < 6 px, mass < 30 px²).

    Returns
    -------
    dict | None
        See module docstring for shape. ``None`` when no candidate
        passed at least ``_MIN_CHECKS_TO_PASS`` (4) of the 6
        structural checks.

    Notes
    -----
    Never raises on bad input. Logs a debug message and returns
    ``None`` on any unexpected error.
    """
    try:
        return _find_comma_internal(
            rgb_image,
            polarity=polarity,
            median_digit_width=median_digit_width,
        )
    except Exception as exc:
        log.debug("find_comma: detector raised %r", exc)
        return None


def find_comma_inv(
    rgb_image: ImageLike,
    *,
    median_digit_width: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Decorrelated peer of :func:`find_comma`.

    Pre-inverts the RGB input pixel-wise (channel-wise 255-x) and
    then runs the same detection pipeline with auto-polarity
    detection on the FLIPPED image. This is symmetric to
    ``model_signal_inv_cnn``'s relationship to ``model_signal_cnn``:

    * For a canonical bright-on-dark crop, the pre-flip turns it
      into dark-on-light. Auto-polarity detection on the flipped
      copy says "dark_on_light" and inverts AGAIN, so the comma
      ends up at the same position the canonical detector would
      find — agreement.
    * For an already-inverted dark-on-light crop, the pre-flip
      makes it bright-on-dark, and auto-polarity detection takes
      no action. We get a comma position from a different starting
      point (the originally-inverted polarity), giving the voted
      combiner a decorrelated check.

    The two detectors disagree ONLY when one detected a spurious
    blob from chromatic-aberration artefacts that survives one
    polarity but not the other — which is exactly the noise the
    voted combiner exists to filter out.

    Returns the same dict shape as :func:`find_comma`. The
    ``polarity_used`` field reports the polarity detected on the
    pre-flipped image, prefixed with ``"inv:"`` so callers can
    distinguish it from a primary-polarity result.
    """
    try:
        rgb = _to_rgb_array(rgb_image)
        if rgb is None:
            return None
        flipped = (255 - rgb).astype(np.uint8)
        result = _find_comma_internal(
            flipped,
            polarity="auto",
            median_digit_width=median_digit_width,
        )
        if result is None:
            return None
        # Tag the polarity so callers can distinguish primary vs inv.
        result["polarity_used"] = "inv:" + result["polarity_used"]
        return result
    except Exception as exc:
        log.debug("find_comma_inv: detector raised %r", exc)
        return None


def _bbox_iou(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int],
) -> float:
    """IoU between two ``(x, y, w, h)`` tuples."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def find_comma_voted(
    rgb_image: ImageLike,
    *,
    median_digit_width: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Run both :func:`find_comma` and :func:`find_comma_inv` and
    combine their results.

    Combination rules:

    * Both return a result and bboxes overlap (IoU > 0.4) → return the
      higher-confidence one with a small confidence boost
      (``+0.10``, capped at 1.0). ``agreed=True`` in the result.
    * Both return a result but disagree → return whichever has the
      higher confidence. ``agreed=False``.
    * Only one returns a result → return that one. ``agreed=False``.
    * Both return None → return None.

    Returns a dict with the same shape as :func:`find_comma` plus:

        ``voted: True`` always
        ``agreed: bool`` whether the two polarities agreed
        ``primary``: raw ``find_comma()`` result (or None)
        ``inverted``: raw ``find_comma_inv()`` result (or None)
    """
    primary = find_comma(
        rgb_image,
        polarity="auto",
        median_digit_width=median_digit_width,
    )
    inverted = find_comma_inv(
        rgb_image,
        median_digit_width=median_digit_width,
    )

    if primary is None and inverted is None:
        return None

    if primary is not None and inverted is not None:
        iou = _bbox_iou(primary["bbox"], inverted["bbox"])
        # Y-range overlap (so a "comma found" at top of crop and
        # another at bottom can't accidentally cross the x-center
        # proximity test even when their x's happen to coincide).
        py0, py1 = primary["bbox"][1], primary["bbox"][1] + primary["bbox"][3]
        iy0, iy1 = inverted["bbox"][1], inverted["bbox"][1] + inverted["bbox"][3]
        y_overlap = max(0, min(py1, iy1) - max(py0, iy0))
        x_proximity = abs(primary["x_center"] - inverted["x_center"])
        agreed = (
            iou > _VOTED_IOU_AGREEMENT_THR
            or (
                x_proximity <= _VOTED_X_CENTER_AGREEMENT_PX
                and y_overlap > 0
            )
        )
        # Pick the higher-confidence result as the base.
        if primary["confidence"] >= inverted["confidence"]:
            base = primary
        else:
            base = inverted
        if agreed:
            boosted = min(1.0, base["confidence"] + _VOTED_AGREEMENT_BOOST)
        else:
            boosted = base["confidence"]
        out = dict(base)
        out["confidence"] = float(boosted)
        out["voted"] = True
        out["agreed"] = bool(agreed)
        out["primary"] = primary
        out["inverted"] = inverted
        return out

    only = primary if primary is not None else inverted
    out = dict(only)  # type: ignore[arg-type]
    out["voted"] = True
    out["agreed"] = False
    out["primary"] = primary
    out["inverted"] = inverted
    return out
