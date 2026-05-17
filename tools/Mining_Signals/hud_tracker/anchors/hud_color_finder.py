"""Pixel-only HUD-panel finder for the SC mining HUD.

Stage 1 of the multi-anchor HUD tracker (see
``hud_tracker/detector_inventory.md`` section 1). This module locates
the SCAN RESULTS panel as a *structural object* — purely from RGB
pixel evidence, without depending on any of the row-level detectors
in ``ocr/`` (which would create circular logic: the row detectors
need a HUD bbox to be reliable, and a "HUD finder" that calls them
just papers over the same false-positive problem).

Why this exists
---------------
The SC mining HUD chrome — title text, label text, and the two
horizontal bracket lines — uses two distinctive saturated colors:

  * **Cyan/teal text** for "SCAN RESULTS", "MASS:", "RESISTANCE:",
    "INSTABILITY:" and the value digits. In PIL HSV (H scale 0-255)
    this peaks around H ≈ 120-170 (≈ 170°–240° on the standard
    0-360° wheel).
  * **Yellow-green** for the chrome accents and difficulty bar fill.
    In PIL HSV this peaks around H ≈ 30-55 (≈ 42°–77° on 0-360°).

Both are unusual against the typical mining backdrop (asteroids,
space, dust) which is desaturated grey-brown. So a per-pixel mask
``(hue ∈ {cyan_band ∪ green_band}) ∧ (sat ≥ S_min) ∧ (val ≥ V_min)``
isolates the HUD chrome very cleanly. After a small morphological
close to fuse text characters into a connected blob, the largest
plausible component is the panel.

Algorithm
---------
1. RGB → HSV (PIL's HSV conversion; uint8, H scale 0-255).
2. Build chrome-mask: pixel passes if its hue lies in either of the
   two calibrated bands AND saturation ≥ S_min AND value ≥ V_min.
3. Binary close (3×3 cross, 2 iterations) to fuse text characters and
   bridge tiny anti-aliasing gaps in the chrome lines.
4. Connected components via ``scipy.ndimage.label``.
5. Per component, compute area, bbox, extent (= component_area /
   bbox_area). Reject components that are too small, that have an
   implausible aspect ratio for a SCAN RESULTS panel, or whose extent
   is so low the component is just noise.
6. Score the survivors and pick the best.

Calibration source
------------------
Hue/sat/value bands are read from
``hud_tracker/hud_color_calibration.json`` (alongside this module),
which is produced by ``calibrate_hud_colors.py`` from labeled GT
captures. If the calibration file is missing, the module falls back
to conservative defaults that should work but won't be tuned.

Public API
----------
``find_hud_panel(image) -> dict | None``

Returns the discovered panel bbox and confidence, or None when no
plausible panel was found.

Constraints
-----------
* No OpenCV dependency (not guaranteed installed). Uses
  ``scipy.ndimage`` for morphology + connected components.
* Read-only on ``ocr/`` — this finder runs upstream of the OCR
  pipeline.
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

__all__ = ["find_hud_panel", "load_calibration", "DEFAULT_CALIBRATION"]


# ─────────────────────────────────────────────────────────────────────
# Defaults — used when no calibration JSON is present.
#
# These were eyeballed from the 2 calibration captures and the bright
# pixel histograms in user_20260418_154408 / user_20260418_081525.
# Hue values are in PIL's 0-255 scale (H * 360/255 = degrees).
# ─────────────────────────────────────────────────────────────────────

DEFAULT_CALIBRATION: dict = {
    "version": 1,
    "source": "fallback-defaults",
    "n_captures": 0,
    # Cyan/teal band — text labels and SCAN RESULTS title.
    "cyan_band": {"h_min": 110, "h_max": 175},
    # Yellow-green band — chrome accents, difficulty bar fill.
    "green_band": {"h_min": 25, "h_max": 60},
    "sat_min": 130,
    "val_min": 100,
    # Geometry — the SCAN RESULTS panel on a region1 crop.
    "min_area_px": 1500,
    "min_bbox_aspect": 0.4,        # w/h
    "max_bbox_aspect": 1.5,        # w/h
    "min_extent": 0.05,            # component_pixels / bbox_area
    # Morphology — see _morph_close_panel for what these do.
    "morph_seed_iterations": 2,    # initial 3x3 close to fuse glyphs
    "morph_vert_close_px": 30,     # vertical span to bridge between rows
    "morph_horiz_close_px": 8,     # horizontal span to bridge within rows
    "bbox_aspect_peak": 1.0,       # canonical SCAN RESULTS panel aspect
}


# ─────────────────────────────────────────────────────────────────────
# Calibration I/O
# ─────────────────────────────────────────────────────────────────────


def _calibration_path() -> Path:
    """Where the calibration JSON lives (alongside this module's
    package, one directory up from ``anchors/``)."""
    return Path(__file__).resolve().parent.parent / "hud_color_calibration.json"


def load_calibration(path: Optional[Path] = None) -> dict:
    """Load calibration JSON or fall back to baked-in defaults.

    Never raises — bad/missing files just return the defaults with a
    log warning, so the finder can always run.
    """
    target = path if path is not None else _calibration_path()
    if not target.is_file():
        log.info(
            "hud_color_finder: no calibration at %s, using defaults",
            target,
        )
        return dict(DEFAULT_CALIBRATION)
    try:
        data = json.loads(Path(target).read_text())
    except Exception as exc:
        log.warning(
            "hud_color_finder: failed to read %s (%s); using defaults",
            target, exc,
        )
        return dict(DEFAULT_CALIBRATION)
    # Merge over defaults so missing keys are safely defaulted.
    merged = dict(DEFAULT_CALIBRATION)
    merged.update({k: v for k, v in data.items() if v is not None})
    return merged


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────


def _to_rgb_array(image: ImageLike) -> Optional[np.ndarray]:
    """Coerce input to a (H, W, 3) uint8 RGB array.

    Returns None on bad input; callers should check.
    """
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
    """RGB → HSV via PIL (uint8 H/S/V with H on the 0-255 scale).

    PIL is fast enough at this size and avoids the colorsys per-pixel
    Python loop. The returned array shape is (H, W, 3).
    """
    pil = Image.fromarray(arr, mode="RGB").convert("HSV")
    return np.asarray(pil)


def _build_chrome_mask(hsv: np.ndarray, calib: dict) -> np.ndarray:
    """Boolean (H, W) mask of plausible HUD chrome pixels.

    True where:
      hue ∈ [cyan_band] ∪ [green_band]
      AND  sat ≥ sat_min
      AND  val ≥ val_min
    """
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    cy = calib["cyan_band"]
    gn = calib["green_band"]
    s_min = int(calib["sat_min"])
    v_min = int(calib["val_min"])

    cyan = (H >= int(cy["h_min"])) & (H <= int(cy["h_max"]))
    green = (H >= int(gn["h_min"])) & (H <= int(gn["h_max"]))
    bright = (S >= s_min) & (V >= v_min)
    return (cyan | green) & bright


def _morph_close_panel(
    mask: np.ndarray,
    *,
    vert_close_px: int = 40,
    horiz_close_px: int = 8,
    seed_iterations: int = 2,
) -> np.ndarray:
    """Multi-stage close that fuses the SCAN RESULTS panel rows into
    one connected component.

    The HUD has 4-5 stacked rows separated by ~30 px of dark
    background (the chrome lines, the title, the three label rows,
    and the difficulty bar). A small 3×3 closing only merges
    individual character glyphs, leaving the rows as separate blobs.

    Strategy:
      1. Small 3×3 cross close × ``seed_iterations`` to fuse glyphs
         within a single row.
      2. Vertical close (``vert_close_px`` × 1) to bridge the
         background between rows.
      3. Horizontal close (1 × ``horiz_close_px``) to bridge the
         characters within a row that the vertical close missed.

    The vertical close bridges only ``vert_close_px`` of background
    so unrelated bright pixels far above/below the panel don't get
    sucked in unless they're already very close vertically.
    """
    structure = ndimage.generate_binary_structure(2, 1)
    out = ndimage.binary_closing(mask, structure=structure,
                                 iterations=int(seed_iterations))
    if vert_close_px > 1:
        vert = np.ones((int(vert_close_px), 1), dtype=bool)
        out = ndimage.binary_closing(out, structure=vert)
    if horiz_close_px > 1:
        hor = np.ones((1, int(horiz_close_px)), dtype=bool)
        out = ndimage.binary_closing(out, structure=hor)
    return out


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected labeling. Returns (labels, n)."""
    structure = ndimage.generate_binary_structure(2, 2)  # 3x3 full = 8-connected
    labels, n = ndimage.label(mask, structure=structure)
    return labels, int(n)


def _component_stats(labels: np.ndarray, n: int) -> list[dict]:
    """For each component, compute area + bbox + extent.

    Returns a list of dicts, one per component (skipping the
    background label 0).
    """
    if n == 0:
        return []
    # ndimage.find_objects returns slices [y_slice, x_slice] per label
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
        })
    return out


def _score_component(c: dict, calib: dict) -> float:
    """Heuristic combined score in [0, 1] for ranking surviving
    components. Higher = more panel-like.

    Combines:
      * area_score: log-scale area, capped at a reasonable max.
      * extent_score: prefer extent in a moderate range; very low =
        scattered noise, very high (>0.5) = solid-fill rectangle (also
        suspect for a HUD which is mostly background between text).
      * aspect_score: triangle window centered on the canonical panel
        aspect (~0.7 wide:tall on region1 crops).
    """
    area = c["area"]
    extent = c["extent"]
    bw, bh = c["w"], c["h"]
    aspect = bw / float(bh)

    # Saturating area score — anything ≥ 8000 px is "definitely big enough".
    area_score = float(np.clip(area / 8000.0, 0.0, 1.0))

    # Extent: best around 0.10 - 0.30, falls off above and below.
    if extent < 0.05:
        extent_score = 0.0
    elif extent <= 0.20:
        extent_score = extent / 0.20
    elif extent <= 0.50:
        extent_score = 1.0 - 0.6 * (extent - 0.20) / 0.30
    else:
        extent_score = 0.4 - 0.4 * min(1.0, (extent - 0.50) / 0.50)

    # Aspect: triangle window peaked at the calibrated peak (default
    # 1.0 — the SCAN RESULTS panel on region1 captures is roughly
    # square), falls to 0 at the calibrated bounds.
    a_lo = float(calib["min_bbox_aspect"])
    a_hi = float(calib["max_bbox_aspect"])
    a_peak = float(calib.get("bbox_aspect_peak", 1.0))
    if aspect <= a_lo or aspect >= a_hi:
        aspect_score = 0.0
    elif aspect <= a_peak:
        aspect_score = (aspect - a_lo) / max(1e-6, a_peak - a_lo)
    else:
        aspect_score = (a_hi - aspect) / max(1e-6, a_hi - a_peak)

    return float(0.45 * area_score + 0.30 * extent_score + 0.25 * aspect_score)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def find_hud_panel(
    image: ImageLike,
    *,
    calibration: Optional[dict] = None,
    return_details: bool = True,
) -> Optional[dict]:
    """Locate the SCAN RESULTS panel from RGB pixels.

    Parameters
    ----------
    image
        PIL ``Image.Image`` or numpy array (RGB, BGR-ignored, or gray
        will be replicated to 3 channels).
    calibration
        Optional calibration dict (same shape as
        ``hud_color_calibration.json``). If None, loads the JSON next
        to this module, falling back to baked-in defaults.
    return_details
        If True (default), include diagnostic fields in the result.

    Returns
    -------
    dict | None
        ``None`` if no plausible panel was found. Otherwise::

            {
              "bbox": (x, y, w, h),
              "confidence": float,
              "n_chrome_pixels": int,
              "n_candidates_considered": int,
              "details": {                  # only when return_details
                "color_mask_summary": {...},
                "rejected": [{...}, ...],
                "calibration": {...},
              }
            }
    """
    arr = _to_rgb_array(image)
    if arr is None:
        return None

    calib = calibration if calibration is not None else load_calibration()

    try:
        hsv = _rgb_to_hsv(arr)
        raw_mask = _build_chrome_mask(hsv, calib)
        n_chrome_raw = int(raw_mask.sum())
        if n_chrome_raw == 0:
            return None

        mask = _morph_close_panel(
            raw_mask,
            seed_iterations=int(calib.get("morph_seed_iterations", 2)),
            vert_close_px=int(calib.get("morph_vert_close_px", 40)),
            horiz_close_px=int(calib.get("morph_horiz_close_px", 8)),
        )
        labels, n_components = _label_components(mask)
        if n_components == 0:
            return None

        stats = _component_stats(labels, n_components)
        if not stats:
            return None

        min_area = int(calib.get("min_area_px", 1500))
        min_aspect = float(calib.get("min_bbox_aspect", 0.4))
        max_aspect = float(calib.get("max_bbox_aspect", 1.5))
        min_extent = float(calib.get("min_extent", 0.05))

        survivors: list[dict] = []
        rejected: list[dict] = []
        for c in stats:
            aspect = c["w"] / float(c["h"])
            reason = None
            if c["area"] < min_area:
                reason = f"area {c['area']} < {min_area}"
            elif aspect < min_aspect or aspect > max_aspect:
                reason = f"aspect {aspect:.2f} outside [{min_aspect}, {max_aspect}]"
            elif c["extent"] < min_extent:
                reason = f"extent {c['extent']:.3f} < {min_extent}"
            if reason is None:
                survivors.append(c)
            else:
                rejected.append({**c, "reason": reason})

        if not survivors:
            if return_details:
                return None
            return None

        # Score and pick the best.
        for s in survivors:
            s["score"] = _score_component(s, calib)
        survivors.sort(key=lambda c: c["score"], reverse=True)
        best = survivors[0]

        bbox = (int(best["x"]), int(best["y"]), int(best["w"]), int(best["h"]))
        confidence = float(np.clip(best["score"], 0.0, 1.0))

        result: dict = {
            "bbox": bbox,
            "confidence": confidence,
            "n_chrome_pixels": int(best["area"]),
            "n_candidates_considered": int(n_components),
        }
        if return_details:
            result["details"] = {
                "color_mask_summary": {
                    "n_chrome_pixels_raw": n_chrome_raw,
                    "n_chrome_pixels_after_close": int(mask.sum()),
                    "image_h": int(arr.shape[0]),
                    "image_w": int(arr.shape[1]),
                },
                "rejected": rejected[:8],          # cap noise
                "n_survivors": len(survivors),
                "best_extent": float(best["extent"]),
                "best_aspect": float(best["w"] / float(best["h"])),
                "calibration": {
                    "source": calib.get("source", "?"),
                    "n_captures": calib.get("n_captures"),
                    "cyan_band": calib.get("cyan_band"),
                    "green_band": calib.get("green_band"),
                    "sat_min": calib.get("sat_min"),
                    "val_min": calib.get("val_min"),
                },
            }
        return result

    except Exception as exc:
        log.warning("find_hud_panel: %r", exc)
        return None
