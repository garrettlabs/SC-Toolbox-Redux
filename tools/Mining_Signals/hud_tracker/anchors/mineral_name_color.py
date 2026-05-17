"""Color-aware mineral-name row detector for the SC scan-results HUD.

The mineral name (e.g. ``ALUMINUM``, ``IRON``, ``BERYL``, ``HADANITE``)
on the SCAN RESULTS panel renders in *mineral-specific* warm or cool
colors. Across captures we see warm orange/yellow, cyan/teal, and
the occasional purple/magenta. The text color is consistently
distinct from:

  * the cyan/teal HUD chrome label text (MASS:, RESISTANCE:, ...),
  * the green chrome bracket lines,
  * and any background (asteroid, sky, dark space).

So a multi-band HSV mask covering the known mineral palette finds the
mineral row even when the legacy luma-projection-band detector
``ocr/sc_ocr/api.py::_find_mineral_row_universal`` is fooled by a
bright icy/glaring asteroid background — its known failure mode (3 of
20 unlabeled captures sampled return y_frac 0.70-0.94 instead of the
correct ~0.10-0.13).

Same architectural pattern as the icon RGB NCC: replace a luma-only
heuristic with color-aware detection where color is a discriminating
feature.

Algorithm
---------
1.  RGB → HSV (PIL convention; uint8, H/S/V each on 0-255).
2.  Build a multi-band "mineral palette" mask. Hue bands cover:

    * Warm   (H ≈ 0..50  on the 0-360 wheel) → orange / yellow / red
    * Cyan   (H ≈ 85..110)                   → teal / cyan
    * Purple (H ≈ 200..240)                  → magenta / violet

    Plus a saturation floor (excludes desaturated background) and a
    value floor (excludes dim noise).

3.  Morphological close with a horizontal kernel (~7×1) to fuse text
    strokes within a mineral name into a single horizontal blob.

4.  Connected components with ``scipy.ndimage.label``.

5.  Filter components by:
      * Width  ≥ 50 px       (mineral names span ~40-60% of panel)
      * Height ∈ [10, 35] px (single text line)
      * Aspect ratio > 2.0   (horizontal)

6.  Position prior: the blob's vertical center must be plausible
    (between the SCAN RESULTS title and the chrome lines). When a
    ``panel_bbox`` is supplied the bounds are derived from it; without
    one we use coarse image-fraction defaults.

7.  Score by area × extent × position-plausibility. Return the
    highest-scoring blob.

Calibration
-----------
``hud_tracker/mineral_color_calibration.json`` (alongside the
``world_model_*.json`` siblings of this package) contains the actual
hue-band edges sampled from the labeled ``resource`` boxes in the 72
labeled region1 captures under
``training_data_panels/user_20260418_154408/region1/``. If the file
is missing, baked-in defaults below are used.

Public API
----------
``find_mineral_name_row(rgb_image, panel_bbox=None) -> dict | None``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
from scipy import ndimage

log = logging.getLogger(__name__)

ImageLike = Union[Image.Image, np.ndarray]

__all__ = [
    "find_mineral_name_row",
    "load_calibration",
    "DEFAULT_CALIBRATION",
]


# ─────────────────────────────────────────────────────────────────────
# Defaults — used when no calibration JSON is present.
# Hue bounds in PIL HSV space (0-255 scale, where 255 ≡ 360°).
# Conversion: pil_h = round(deg * 255 / 360).
# ─────────────────────────────────────────────────────────────────────
#   warm   (0..50°)    →  pil  (  0,  35)
#   cyan   (85..110°)  →  pil  ( 60,  78)
#   purple (200..240°) →  pil  (141, 170)
#
# These are the palette hue bands. The mineral-text actual sat/val are
# very high in real captures, but we leave the floors where they let
# weakly-saturated bright text in too — the mineral name is always
# fully-saturated so this is just a safety net for sub-pixel rendering.
DEFAULT_CALIBRATION: dict = {
    "version": 1,
    "source": "fallback-defaults",
    "n_captures": 0,
    # PIL-HSV bands. Each is inclusive on both ends.
    "warm_band":   {"h_min":   0, "h_max":  35},
    "cyan_band":   {"h_min":  60, "h_max":  78},
    "purple_band": {"h_min": 141, "h_max": 170},
    "sat_min": 60,
    "val_min": 80,
    # Geometry filters for the connected-component blob.
    "min_width_px":  50,
    "min_height_px": 10,
    "max_height_px": 35,
    "min_aspect":     2.0,
    # Morphology kernel for bridging text-stroke gaps within the name.
    # 11 is wide enough to bridge the inter-character gap of the SC HUD
    # font at typical render scale (~6-9 px between letters) without
    # bleeding across vertical row boundaries (which are ~10+ px apart).
    "morph_horiz_close_px": 11,
    # Position prior — fractions of image height when no panel_bbox is
    # supplied. The mineral row sits just below the SCAN RESULTS title
    # and well above the chrome bot_line; conservative bounds:
    "position_y_min_frac": 0.05,
    "position_y_max_frac": 0.55,
    # Position prior when panel_bbox is supplied — fractions of the
    # panel height from the panel's top edge:
    "panel_y_min_frac": 0.10,
    "panel_y_max_frac": 0.55,
}


# ─────────────────────────────────────────────────────────────────────
# Calibration I/O
# ─────────────────────────────────────────────────────────────────────


def _calibration_path() -> Path:
    """Where the calibration JSON lives (one directory up from this
    module, alongside ``hud_color_calibration.json`` and the
    ``world_model_*.json`` siblings)."""
    return Path(__file__).resolve().parent.parent / "mineral_color_calibration.json"


def load_calibration(path: Optional[Path] = None) -> dict:
    """Load calibration JSON or fall back to baked-in defaults.

    Never raises — bad/missing files just return the defaults with a
    log warning, so the finder can always run.
    """
    target = path if path is not None else _calibration_path()
    if not target.is_file():
        log.info(
            "mineral_name_color: no calibration at %s, using defaults",
            target,
        )
        return dict(DEFAULT_CALIBRATION)
    try:
        data = json.loads(Path(target).read_text())
    except Exception as exc:
        log.warning(
            "mineral_name_color: failed to read %s (%s); using defaults",
            target, exc,
        )
        return dict(DEFAULT_CALIBRATION)
    merged = dict(DEFAULT_CALIBRATION)
    merged.update({k: v for k, v in data.items() if v is not None})
    return merged


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────


def _to_rgb_array(image: ImageLike) -> Optional[np.ndarray]:
    """Coerce input to a (H, W, 3) uint8 RGB array. None on bad input."""
    if image is None:
        return None
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.asarray(image)
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
    else:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.size == 0 or arr.shape[0] < 8 or arr.shape[1] < 8:
        return None
    return arr


def _rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    """RGB → HSV via PIL (uint8 H/S/V with H on the 0-255 scale)."""
    pil = Image.fromarray(arr, mode="RGB").convert("HSV")
    return np.asarray(pil)


def _build_palette_mask(hsv: np.ndarray, calib: dict) -> np.ndarray:
    """Boolean (H, W) mask of plausible mineral-name pixels.

    True where:
      hue ∈ warm_band ∪ cyan_band ∪ purple_band
      AND  sat ≥ sat_min
      AND  val ≥ val_min
    """
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    warm = (
        (H >= int(calib["warm_band"]["h_min"]))
        & (H <= int(calib["warm_band"]["h_max"]))
    )
    cyan = (
        (H >= int(calib["cyan_band"]["h_min"]))
        & (H <= int(calib["cyan_band"]["h_max"]))
    )
    purple = (
        (H >= int(calib["purple_band"]["h_min"]))
        & (H <= int(calib["purple_band"]["h_max"]))
    )
    bright = (S >= int(calib["sat_min"])) & (V >= int(calib["val_min"]))
    return (warm | cyan | purple) & bright


def _morph_close_horizontal(mask: np.ndarray, kernel_px: int) -> np.ndarray:
    """Bridge horizontal text-stroke gaps within the mineral name.

    A horizontal closing kernel fuses adjacent character glyphs into a
    single connected blob without merging across vertical row gaps
    (which would pull in the MASS row label below).
    """
    if kernel_px < 2:
        return mask
    hor = np.ones((1, int(kernel_px)), dtype=bool)
    return ndimage.binary_closing(mask, structure=hor)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected labeling. Returns (labels, n)."""
    structure = ndimage.generate_binary_structure(2, 2)
    labels, n = ndimage.label(mask, structure=structure)
    return labels, int(n)


def _component_stats(labels: np.ndarray, n: int) -> list[dict]:
    """For each component, compute bbox + area + extent."""
    if n == 0:
        return []
    objs = ndimage.find_objects(labels)
    out: list[dict] = []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        y_sl, x_sl = sl
        y0, y1 = int(y_sl.start), int(y_sl.stop)
        x0, x1 = int(x_sl.start), int(x_sl.stop)
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        sub = labels[y_sl, x_sl] == i
        area = int(sub.sum())
        if area == 0:
            continue
        out.append({
            "label": int(i),
            "x": x0, "y": y0, "w": bw, "h": bh,
            "area": area,
            "extent": float(area) / float(bw * bh),
            "aspect": float(bw) / float(bh),
        })
    return out


def _position_score(
    cy: float,
    img_h: int,
    panel_bbox: Optional[tuple[int, int, int, int]],
    calib: dict,
    *,
    top_line_y: Optional[int] = None,
    bot_line_y: Optional[int] = None,
) -> float:
    """Triangle window over the y-center.

    Without a panel_bbox: clip cy/img_h to ``[position_y_min_frac,
    position_y_max_frac]`` and reward proximity to the centre of that
    window. The mineral row sits in the upper-middle of the captured
    region.

    With a panel_bbox: same idea, but the bounds are derived from the
    panel's vertical span. This is much tighter and a better signal.

    With top_line_y / bot_line_y from chrome_lines: the tightest
    bracket — the mineral row is structurally between these two
    lines, with a generous +5 px above and the bottom rolls off
    before bot_line.
    """
    # Chrome-line bracket is the tightest, most reliable y window.
    # When supplied, use it directly with a slight pad above (mineral
    # name occasionally rides up onto top_line's bracket pixels) and
    # well above the bot_line (mineral row sits in the upper half of
    # the data area, never near the bottom).
    if top_line_y is not None and bot_line_y is not None and bot_line_y > top_line_y:
        span = float(bot_line_y - top_line_y)
        y_min = max(0.0, float(top_line_y) - 4.0)
        # Cap at 25% of the bracket span. Past that we're squarely on
        # the MASS row — even if MASS's blob outscores everything else
        # by a hair, the bracket should keep us off it.
        y_max = float(top_line_y) + 0.25 * span
        # Peak at 13% of span — the canonical mineral-row fraction
        # (matches ``_find_label_rows_by_hud_grid``'s _ROW_FRACTIONS
        # constant in onnx_hud_reader.py).
        peak = float(top_line_y) + 0.13 * span
        if cy < y_min or cy > y_max:
            return 0.0
        if cy <= peak:
            return float((cy - y_min) / max(1.0, peak - y_min))
        return float((y_max - cy) / max(1.0, y_max - peak))

    if panel_bbox is not None:
        px, py, pw, ph = panel_bbox
        if ph <= 0:
            return 0.0
        y_min = py + ph * float(calib["panel_y_min_frac"])
        y_max = py + ph * float(calib["panel_y_max_frac"])
    else:
        y_min = img_h * float(calib["position_y_min_frac"])
        y_max = img_h * float(calib["position_y_max_frac"])

    if cy < y_min or cy > y_max:
        return 0.0
    # Triangle window peaked 33% of the way through the band — mineral
    # name is closer to the top of the data area than the bottom.
    span = max(1.0, y_max - y_min)
    peak = y_min + 0.33 * span
    if cy <= peak:
        return float((cy - y_min) / max(1.0, peak - y_min))
    return float((y_max - cy) / max(1.0, y_max - peak))


def _score_blob(
    c: dict,
    img_h: int,
    panel_bbox,
    calib: dict,
    *,
    top_line_y: Optional[int] = None,
    bot_line_y: Optional[int] = None,
) -> float:
    """Combine area × extent × position_score into a single rank score."""
    cy = c["y"] + c["h"] / 2.0
    pos = _position_score(
        cy, img_h, panel_bbox, calib,
        top_line_y=top_line_y,
        bot_line_y=bot_line_y,
    )
    if pos <= 0.0:
        return 0.0
    # Saturating area score — anything ≥ 600 px is "definitely big enough".
    area_score = float(np.clip(c["area"] / 600.0, 0.0, 1.0))
    # Extent score: prefer 0.30..0.85. Below 0.10 is sparse noise; above
    # 0.95 is a solid bar (like a chrome line) — mineral text has gaps.
    extent = c["extent"]
    if extent < 0.10 or extent > 0.95:
        extent_score = 0.0
    elif extent <= 0.30:
        extent_score = (extent - 0.10) / 0.20
    elif extent <= 0.85:
        extent_score = 1.0
    else:
        extent_score = (0.95 - extent) / 0.10
    return float(0.40 * area_score + 0.30 * extent_score + 0.30 * pos)


def _dominant_hue_deg(hsv: np.ndarray, c: dict) -> float:
    """Mean PIL-hue (in degrees, 0-360) of the bright-saturated pixels
    inside the candidate's bbox. Diagnostic only."""
    sub = hsv[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"], :]
    if sub.size == 0:
        return float("nan")
    H = sub[:, :, 0].astype(np.float32)
    S = sub[:, :, 1]
    V = sub[:, :, 2]
    keep = (S >= 60) & (V >= 80)
    if not keep.any():
        return float("nan")
    h_pil = float(np.mean(H[keep]))
    return round(h_pil * 360.0 / 255.0, 1)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def find_mineral_name_row(
    rgb_image: ImageLike,
    panel_bbox: Optional[tuple[int, int, int, int]] = None,
    *,
    calibration: Optional[dict] = None,
    top_line_y: Optional[int] = None,
    bot_line_y: Optional[int] = None,
    use_chrome_lines: bool = True,
) -> Optional[dict]:
    """Find the mineral name row via color-aware blob detection.

    The mineral name renders in distinctive colors (warm orange,
    yellow, cyan, purple, etc. — varies per mineral type). Use a
    multi-color HSV mask covering the known mineral palette, find
    horizontal blobs in that mask above a minimum width threshold,
    and return the topmost one whose vertical position is plausible
    (just below the SCAN RESULTS title region).

    Parameters
    ----------
    rgb_image
        PIL ``Image.Image`` (RGB) or numpy array (RGB or RGBA).
    panel_bbox
        Optional ``(x, y, w, h)`` bbox of the SCAN RESULTS panel,
        produced by ``hud_color_finder.find_hud_panel`` upstream.
        When supplied the position prior is much tighter.
    calibration
        Optional calibration dict (same shape as
        ``mineral_color_calibration.json``). When None, the JSON is
        loaded from disk with a fallback to baked-in defaults.

    Returns
    -------
    dict | None
        ``None`` if no plausible mineral row found. Otherwise::

            {
                "bbox": (x, y, w, h),
                "confidence": float,
                "details": {
                    "dominant_hue": float,        # degrees, 0..360
                    "n_blobs_considered": int,
                },
            }
    """
    arr = _to_rgb_array(rgb_image)
    if arr is None:
        return None

    calib = calibration if calibration is not None else load_calibration()

    # Auto-bracket via chrome_lines when caller didn't supply explicit
    # top_line/bot_line and didn't opt out. Cheap (~1 ms on a 448x670)
    # so always-on by default. The chrome lines, when both detected,
    # bracket the data area very tightly and rule out SCAN RESULTS
    # title pixels (above top_line) and the difficulty bar (below
    # bot_line) without any luma/color heuristic.
    if use_chrome_lines and top_line_y is None and bot_line_y is None:
        try:
            from .chrome_lines import find_chrome_lines  # noqa: PLC0415
            cl = find_chrome_lines(rgb_image)
            tl = cl.get("top_line")
            bl = cl.get("bot_line")
            if tl and bl:
                top_line_y = int(tl["y"]) + int(tl["h"]) // 2
                bot_line_y = int(bl["y"]) + int(bl["h"]) // 2
        except Exception as _exc:
            log.debug("mineral_name_color: chrome_lines bracket disabled: %s", _exc)

    try:
        hsv = _rgb_to_hsv(arr)
        raw_mask = _build_palette_mask(hsv, calib)
        if not raw_mask.any():
            return None

        mask = _morph_close_horizontal(
            raw_mask,
            kernel_px=int(calib.get("morph_horiz_close_px", 7)),
        )
        labels, n_components = _label_components(mask)
        if n_components == 0:
            return None

        stats = _component_stats(labels, n_components)
        if not stats:
            return None

        min_w = int(calib["min_width_px"])
        min_h = int(calib["min_height_px"])
        max_h = int(calib["max_height_px"])
        min_aspect = float(calib["min_aspect"])

        survivors: list[dict] = []
        for c in stats:
            if c["w"] < min_w:
                continue
            if c["h"] < min_h or c["h"] > max_h:
                continue
            if c["aspect"] < min_aspect:
                continue
            survivors.append(c)

        if not survivors:
            return None

        img_h = int(arr.shape[0])
        scored: list[tuple[float, dict]] = []
        for c in survivors:
            s = _score_blob(
                c, img_h, panel_bbox, calib,
                top_line_y=top_line_y,
                bot_line_y=bot_line_y,
            )
            if s > 0.0:
                scored.append((s, c))

        if not scored:
            return None

        # Topmost-among-near-best: if two blobs score within 5% of each
        # other, prefer the one with the smaller y. Keeps us from
        # accidentally picking a near-equal-score blob that lives in
        # the COMPOSITION area (the original failure mode).
        scored.sort(key=lambda t: (-t[0], t[1]["y"]))
        best_score, best = scored[0]
        for s, c in scored[1:]:
            if s >= 0.95 * best_score and c["y"] < best["y"] - 8:
                best_score, best = s, c

        bbox = (
            int(best["x"]), int(best["y"]),
            int(best["w"]), int(best["h"]),
        )
        return {
            "bbox": bbox,
            "confidence": float(np.clip(best_score, 0.0, 1.0)),
            "details": {
                "dominant_hue": _dominant_hue_deg(hsv, best),
                "n_blobs_considered": int(len(survivors)),
                "top_line_y": top_line_y,
                "bot_line_y": bot_line_y,
            },
        }

    except Exception as exc:
        log.warning("find_mineral_name_row: %r", exc)
        return None
