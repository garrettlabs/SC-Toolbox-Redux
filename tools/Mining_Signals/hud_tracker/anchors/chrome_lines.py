"""Chrome-line anchor for the SC scan-results HUD.

Detects the pair of horizontal HUD chrome lines that bracket the
scan-results data area:

    top_line  — under "SCAN RESULTS"  (above the mineral name)
    bot_line  — above COMPOSITION     (below the difficulty bar)

These are the FATTEST pieces of fixed HUD geometry on the panel
(~330 px wide, 1-3 px central stroke, plus 3-8 px end-bracket
margins captured in the labelled GT boxes) and survive when the
panel title or labels are occluded by particles. They are the
single highest-leverage anchor for the multi-anchor HUD tracker
(see ``hud_tracker/detector_inventory.md`` section 3.1).

Algorithm (rev2 — isolation discriminator + bracket extension)
--------------------------------------------------------------
The legacy ``ocr/onnx_hud_reader.py::_find_panel_lines`` heuristic
(80% column-fill on a polarity-canonicalized text mask) suffers
from two failures on real captures:

1.  Bot_line is dim (peak intensity ~150, sometimes ~140) and on a
    handful of frames the stroke is only 2 px tall with imperfect
    column fill. The strict ≥ 80% rule sometimes culled it.
2.  When picking ``lines[-1]`` (bottommost run) the legacy code
    happily selected the COMPOSITION underline rows below the
    bot_line, mislabelling them as ``bot_line`` for downstream use.

The fix here uses two new ideas:

A.  *Vertical isolation*: a real chrome line has ≤ ~3 nearby rows
    with > 50% column fill within a ±15 px band. The COMPOSITION
    bracket lines come in pairs ~20–25 px apart with the green
    progress fill or empty bracket box between them, putting many
    high-density rows in their bands.

B.  *Inter-candidate gap*: across the entire labelled + unlabelled
    set, the top→bot gap is consistently ~220 px while the
    bot→COMPOSITION gap is ~50 px and COMPOSITION lines are paired
    ~20 px apart. So after sorting candidates top-to-bottom we
    take the topmost as ``top_line`` and the *first candidate
    below it that has a large gap (≥30 px) to the next candidate
    or no next candidate at all* as ``bot_line``. This rejects
    the COMPOSITION pair (which is the densely-packed tail).

The detector also widens the returned bbox to include the
characteristic ``[ … ]`` end-bracket pixels at the line endpoints
(measured per frame from local pixel evidence with a hard ±8 px
cap from the stroke), so IoU against GT boxes (which span the
end-brackets, not just the central stroke) comes out reasonable.

Public API
----------
``find_chrome_lines(image, *, min_width_frac=0.18, max_thickness=5,
                    fill_threshold=0.85) -> dict``

Always returns a dict shaped::

    {
        "top_line": {"x", "y", "w", "h", "score"} | None,
        "bot_line": {"x", "y", "w", "h", "score"} | None,
    }

Either field is ``None`` if that line wasn't found. The function
never raises on bad input — it logs a warning and returns the
``None``/``None`` shape instead.

The ``score`` field combines the column-fill ratio and the
isolation strength (0..1 each) into a single number in [0, 1].
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageFilter

log = logging.getLogger(__name__)

ImageLike = Union[Image.Image, np.ndarray]

__all__ = ["find_chrome_lines"]


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _build_text_mask(gray: np.ndarray) -> np.ndarray:
    """Polarity-aware boolean mask where True ≈ "likely text/chrome pixel".

    Mirrors ``ocr/onnx_hud_reader.py::_build_text_mask`` but with
    the threshold loosened from > 150 to > 90 on dark backgrounds.
    The chrome lines on dim panels (no minerals scanned, low signal)
    can peak at only ~140 in luma, which the strict > 150 rule
    silently dropped. The end-bracket pixels are even dimmer
    (peaks ~80–100), and we want them in the mask so the bbox
    can include the bracket extent.
    """
    median = float(np.median(gray))
    if median < 130:
        return gray > 90
    blurred = np.asarray(
        Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=5)),
        dtype=np.float32,
    )
    local_contrast = np.abs(gray.astype(np.float32) - blurred)
    return local_contrast > 15


def _to_gray_array(image: ImageLike) -> Optional[np.ndarray]:
    """Coerce the input to a 2-D uint8 grayscale numpy array."""
    if image is None:
        return None

    arr: Optional[np.ndarray]
    if isinstance(image, Image.Image):
        if image.mode != "L":
            arr = np.asarray(image.convert("L"))
        else:
            arr = np.asarray(image)
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        return None

    if arr is None or arr.size == 0:
        return None

    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            r = arr[:, :, 0].astype(np.float32)
            g = arr[:, :, 1].astype(np.float32)
            b = arr[:, :, 2].astype(np.float32)
            arr = (0.299 * r + 0.587 * g + 0.114 * b)
        else:
            arr = arr[:, :, 0]
    elif arr.ndim != 2:
        return None

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return None

    return arr


def _find_runs(
    row_density: np.ndarray, min_width: int, max_thickness: int
) -> list[tuple[int, int]]:
    """Find consecutive row-runs whose density passes ``min_width``.

    Returns ``[(y_start, y_end), ...]`` filtered to thickness in
    ``1..max_thickness`` (inclusive on both ends).
    """
    h = int(row_density.shape[0])
    runs: list[tuple[int, int]] = []
    in_run = False
    run_start = 0
    for y in range(h + 1):
        d = int(row_density[y]) if y < h else 0
        is_hot = d >= min_width
        if is_hot and not in_run:
            in_run = True
            run_start = y
        elif not is_hot and in_run:
            in_run = False
            thickness = y - run_start
            if 1 <= thickness <= max_thickness:
                runs.append((run_start, y))
    return runs


def _measure_bracket_extent(
    mask: np.ndarray,
    stroke_ys: int,
    stroke_ye: int,
    x_left: int,
    x_right: int,
    *,
    cap_px: int = 4,
) -> tuple[int, int]:
    """Probe the line's left and right endpoints for ``[`` / ``]`` brackets.

    Looks at the leftmost ``probe_w`` columns and rightmost
    ``probe_w`` columns (~5% of span, min 4, max 10 px) and finds
    the y-extent of bright mask pixels within ``±cap_px`` of the
    stroke. Returns ``(y_top, y_bot)`` — the union extent across
    both endpoints, clamped to the mask bounds.

    The cap prevents the bracket extent from extending into
    SCAN RESULTS title text (which lives well above the top_line)
    or unrelated rows below the panel. We also reject candidate
    rows where the MIDDLE columns of the line are bright too —
    that's text or a fill bar, not an end-bracket.
    """
    h_img = mask.shape[0]
    span = max(1, x_right - x_left)
    probe_w = int(np.clip(round(span * 0.05), 4, 10))

    y_top_cap = max(0, stroke_ys - cap_px)
    y_bot_cap = min(h_img, stroke_ye + cap_px)

    if y_top_cap >= y_bot_cap:
        return stroke_ys, stroke_ye

    left_cols = mask[y_top_cap:y_bot_cap, x_left:x_left + probe_w]
    right_cols = mask[y_top_cap:y_bot_cap, max(0, x_right - probe_w):x_right]

    # Middle columns: 30% from each edge, the inner 40% of the span.
    mid_x0 = x_left + int(span * 0.30)
    mid_x1 = x_left + int(span * 0.70)
    if mid_x1 <= mid_x0:
        mid_x0, mid_x1 = x_left, x_right
    mid_cols = mask[y_top_cap:y_bot_cap, mid_x0:mid_x1]

    left_bright = left_cols.any(axis=1) if left_cols.size else np.zeros(
        y_bot_cap - y_top_cap, dtype=bool
    )
    right_bright = right_cols.any(axis=1) if right_cols.size else np.zeros(
        y_bot_cap - y_top_cap, dtype=bool
    )
    # Reject rows where the middle columns are mostly bright — that's
    # text or a fill bar above/below, not an end-bracket pixel.
    if mid_cols.size:
        mid_density = mid_cols.sum(axis=1) / max(1, mid_cols.shape[1])
        # Inside the stroke rows the mid columns ARE bright (correctly).
        # Anywhere else, if mid > 30% it's a bar/text row, not bracket.
        bracket_candidate = (left_bright | right_bright) & (mid_density < 0.30)
    else:
        bracket_candidate = left_bright | right_bright

    # Force the stroke rows themselves into the extent.
    stroke_lo = max(0, stroke_ys - y_top_cap)
    stroke_hi = min(bracket_candidate.shape[0], stroke_ye - y_top_cap)
    bracket_candidate[stroke_lo:stroke_hi] = True

    if not bracket_candidate.any():
        return stroke_ys, stroke_ye

    bright_idx = np.where(bracket_candidate)[0]
    y_top_extent = y_top_cap + int(bright_idx[0])
    y_bot_extent = y_top_cap + int(bright_idx[-1]) + 1

    return y_top_extent, y_bot_extent


def _isolation_score(
    gray: np.ndarray,
    stroke_ys: int,
    stroke_ye: int,
    x_left: int,
    x_right: int,
    *,
    band_px: int = 15,
    fill_thresh: int = 90,
    high_density_frac: float = 0.5,
) -> tuple[int, float]:
    """Count bright rows in a ±band_px neighbourhood of the stroke.

    A real chrome line is vertically isolated. The COMPOSITION
    bracket lines have a paired structure within ~20 px and the
    progress fill bar between them adds many high-density rows
    inside their band.

    Returns ``(num_high_density_rows, max_below_density)`` where:
      * ``num_high_density_rows`` is the count of rows in
        ``[stroke_ys - band_px, stroke_ye + band_px]`` whose
        column-fill within ``[x_left, x_right]`` exceeds
        ``high_density_frac``.  Excludes the stroke rows themselves.
      * ``max_below_density`` is the maximum row-fill ratio in the
        12-row band immediately below the stroke (used as a
        secondary discriminator: COMPOSITION top-line has a
        progress bar below = high density).
    """
    h, w = gray.shape
    y0 = max(0, stroke_ys - band_px)
    y1 = min(h, stroke_ye + band_px)
    if y1 <= y0:
        return 0, 0.0
    band = gray[y0:y1, x_left:x_right]
    if band.size == 0:
        return 0, 0.0
    width = max(1, band.shape[1])
    row_density = (band > fill_thresh).sum(axis=1) / float(width)
    # mask out the stroke rows
    stroke_offset_lo = max(0, stroke_ys - y0)
    stroke_offset_hi = min(band.shape[0], stroke_ye - y0)
    keep = np.ones(band.shape[0], dtype=bool)
    keep[stroke_offset_lo:stroke_offset_hi] = False
    high = (row_density[keep] > high_density_frac).sum()

    # 12-row band immediately below the stroke
    yb0 = min(h, stroke_ye + 1)
    yb1 = min(h, stroke_ye + 13)
    if yb1 > yb0:
        below = gray[yb0:yb1, x_left:x_right]
        if below.size:
            below_density = (below > fill_thresh).sum(axis=1) / float(width)
            below_max = float(below_density.max())
        else:
            below_max = 0.0
    else:
        below_max = 0.0

    return int(high), below_max


def _detect_candidates(
    gray: np.ndarray,
    *,
    min_width_frac: float,
    max_thickness: int,
    fill_threshold: float,
) -> list[dict]:
    """Find every horizontal-stroke candidate.

    Returns dicts with the stroke geometry plus the bracket-extended
    y range and an isolation score, sorted top to bottom.
    """
    h, w = gray.shape
    if h == 0 or w == 0:
        return []

    mask = _build_text_mask(gray)
    row_density = mask.sum(axis=1)
    min_width = int(w * min_width_frac)

    runs = _find_runs(row_density, min_width, max_thickness)
    if not runs:
        return []

    out: list[dict] = []
    for y_start, y_end in runs:
        line_mask = mask[y_start:y_end, :].any(axis=0)
        xs = np.where(line_mask)[0]
        if xs.size == 0:
            continue
        x_left = int(xs[0])
        x_right = int(xs[-1]) + 1
        span = x_right - x_left
        if span < min_width:
            continue
        fill = int(line_mask[x_left:x_right].sum())
        fill_ratio = fill / float(span) if span > 0 else 0.0
        if fill_ratio < fill_threshold:
            continue
        bracket_top, bracket_bot = _measure_bracket_extent(
            mask, y_start, y_end, x_left, x_right
        )
        nearby_rows, below_max = _isolation_score(
            gray, y_start, y_end, x_left, x_right
        )
        out.append({
            "stroke_y0": int(y_start),
            "stroke_y1": int(y_end),
            "bracket_y0": int(bracket_top),
            "bracket_y1": int(bracket_bot),
            "x_left": int(x_left),
            "x_right": int(x_right),
            "fill_ratio": float(fill_ratio),
            "nearby_rows": int(nearby_rows),
            "below_max": float(below_max),
        })

    out.sort(key=lambda c: c["stroke_y0"])
    return out


def _candidate_to_bbox(c: dict, *, min_h_px: int = 12, max_h_px: int = 18) -> dict:
    """Convert a candidate dict to the public bbox shape.

    The bracket-pixel extent measured from the image is usually
    smaller than the GT box (the manual annotator pads by a few
    extra pixels around the bracket). We expand the y range
    symmetrically around the stroke center so the final bbox
    height lands in ``[min_h_px, max_h_px]``. This drives IoU
    against the GT boxes up without compromising localization.
    """
    bracket_y0 = c["bracket_y0"]
    bracket_y1 = c["bracket_y1"]
    stroke_center = (c["stroke_y0"] + c["stroke_y1"]) / 2.0

    # Start from the measured bracket extent.
    y0 = bracket_y0
    y1 = bracket_y1
    h = max(1, y1 - y0)

    # Pad symmetrically around stroke center until h >= min_h_px.
    if h < min_h_px:
        half = min_h_px / 2.0
        y0 = int(round(stroke_center - half))
        y1 = int(round(stroke_center + half))
        # Re-include the measured bracket extent if it pokes past
        # the symmetric pad (the bracket is real evidence, not just
        # padding).
        y0 = min(y0, bracket_y0)
        y1 = max(y1, bracket_y1)
        h = y1 - y0

    # Cap at max_h_px symmetrically.
    if h > max_h_px:
        half = max_h_px / 2.0
        y0 = int(round(stroke_center - half))
        y1 = int(round(stroke_center + half))
        h = y1 - y0

    # Score combines fill ratio (0..1) with an isolation factor:
    # the more nearby high-density rows, the worse — clamp at 8.
    iso = max(0.0, 1.0 - min(1.0, c["nearby_rows"] / 8.0))
    score = 0.5 * c["fill_ratio"] + 0.5 * iso
    return {
        "x": int(c["x_left"]),
        "y": int(max(0, y0)),
        "w": int(c["x_right"] - c["x_left"]),
        "h": int(max(1, h)),
        "score": float(score),
    }


def _pick_top_and_bot(
    candidates: list[dict],
    *,
    isolation_max_nearby: int = 6,
    bot_min_gap_px: int = 30,
) -> tuple[Optional[dict], Optional[dict]]:
    """Disambiguate which candidates are top_line vs bot_line.

    Strategy:
      * Topmost candidate that passes a basic isolation check
        becomes ``top_line``. Falls back to the literal topmost
        if none pass.
      * For ``bot_line`` we walk the remaining candidates in
        order. Skip COMPOSITION-pair candidates: those have either
        (a) a high ``below_max`` (progress fill bar below) or
        (b) a near-neighbour ≤ ``bot_min_gap_px`` away (the paired
        underline). The first candidate that fails BOTH "skip"
        conditions is bot_line.
    """
    if not candidates:
        return None, None

    # top_line: topmost reasonably-isolated candidate.
    top_idx = 0
    for i, c in enumerate(candidates):
        if c["nearby_rows"] <= isolation_max_nearby and c["below_max"] < 0.6:
            top_idx = i
            break
    top = candidates[top_idx]

    # bot_line: first candidate strictly below top whose isolation
    # checks pass and which is not in a tightly-packed pair.
    bot: Optional[dict] = None
    for i in range(top_idx + 1, len(candidates)):
        c = candidates[i]
        if c["nearby_rows"] > isolation_max_nearby:
            continue
        if c["below_max"] >= 0.6:
            # Something dense (progress bar) immediately below — skip.
            continue
        # Reject if there's a near-neighbour above (already handled)
        # or below (COMPOSITION's own pair).
        nxt = candidates[i + 1] if i + 1 < len(candidates) else None
        prev = candidates[i - 1] if i - 1 >= 0 else None
        gap_below = (nxt["stroke_y0"] - c["stroke_y1"]) if nxt else 9999
        gap_above = (c["stroke_y0"] - prev["stroke_y1"]) if (
            prev and i - 1 != top_idx
        ) else 9999
        if gap_below < bot_min_gap_px and gap_above < bot_min_gap_px:
            # Tight on both sides → middle of a triple → unlikely.
            continue
        if gap_below < bot_min_gap_px and gap_above == 9999:
            # Tight pair near top of remaining list → COMPOSITION pair.
            continue
        bot = c
        break

    return top, bot


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def find_chrome_lines(
    image: ImageLike,
    *,
    min_width_frac: float = 0.18,
    max_thickness: int = 5,
    fill_threshold: float = 0.85,
) -> dict:
    """Locate the top and bottom HUD chrome lines on a scan-results panel.

    Parameters
    ----------
    image
        PIL ``Image.Image`` (any mode) or numpy array (2-D gray
        or 3-D RGB/RGBA, uint8 or float).
    min_width_frac
        Minimum line span as a fraction of image width. Default
        0.18 matches the legacy detector.
    max_thickness
        Maximum line thickness (consecutive bright rows) in pixels.
        Default 5; HUD chrome strokes are 1-3 px tall but the
        relaxed mask sometimes yields 4-5 px runs by including
        anti-aliased edge rows. Anything thicker is text or a
        continuous bar.
    fill_threshold
        Minimum column-fill ratio within the detected horizontal
        span. Default 0.85.

    Returns
    -------
    dict
        ``{"top_line": bbox|None, "bot_line": bbox|None}`` where
        each bbox is ``{"x", "y", "w", "h", "score"}``. The ``y``
        / ``h`` fields include the end-bracket margins (probed from
        local pixels with an ±8 px cap from the stroke), so IoU
        against ground-truth boxes (which span the brackets) is
        meaningful.

    Notes
    -----
    Never raises on bad input. On any unexpected error the
    function logs a warning and returns
    ``{"top_line": None, "bot_line": None}``.
    """
    empty = {"top_line": None, "bot_line": None}

    try:
        gray = _to_gray_array(image)
        if gray is None:
            log.debug("find_chrome_lines: input could not be coerced to gray array")
            return empty

        candidates = _detect_candidates(
            gray,
            min_width_frac=min_width_frac,
            max_thickness=max_thickness,
            fill_threshold=fill_threshold,
        )
        if not candidates:
            return empty

        top, bot = _pick_top_and_bot(candidates)

        return {
            "top_line": _candidate_to_bbox(top) if top is not None else None,
            "bot_line": _candidate_to_bbox(bot) if bot is not None else None,
        }

    except Exception as exc:
        log.warning("find_chrome_lines: detector raised %r", exc)
        return empty
