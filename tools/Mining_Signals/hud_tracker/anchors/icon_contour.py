"""Contour-based detector for the SC mining HUD location-pin icon.

This is the SECOND of two PRIMARY voters in the icon detection system,
designed to have **decorrelated failure modes** from
``icon_geometry.find_icon_by_geometry`` (which uses HSV warm-color
masking + structural decomposition).

This detector uses LUMA + EDGES instead of color, and matches the
icon's CONTOUR / SILHOUETTE rather than its color blob:

  1. Convert input to grayscale (luma).
  2. Gaussian-smooth + Sobel-magnitude + adaptive threshold -> edge mask.
  3. Pre-built canonical edge mask from ``training_data_blacklist/bad
     crop.png`` is rescaled across a fixed scale set.
  4. Per-scale: NCC of (dilated) edge masks finds candidate positions,
     followed by non-maximum suppression to extract peaks.
  5. Per-candidate: symmetric chamfer distance refines the score, and
     edge-density filters reject sparse-noise / busy-text regions.
  6. Filter on aspect ratio, chamfer threshold, and edge-density ratio;
     pick the LEFTMOST surviving candidate (icon convention - the
     icon is the leftmost UI element in the HUD row).

Failure modes (intentionally decorrelated from icon_geometry):
 * Fails on uniformly low-contrast captures (no clear edges).
 * Fails on extreme defocus / motion blur (edges smeared).
 * Fails when capture noise creates many spurious icon-shaped contours.
 * Can be fooled by a digit cluster's vertical-bar+circle silhouette,
   particularly a leading "1" or "0" -- but the leftmost-filter
   handles this in the typical HUD layout.

Public API: ``find_icon_by_contour(image, hud_bbox=None)``.

Constraints honored:
 * PIL + numpy + scipy (ndimage + signal) only. No opencv, no torch.
 * Defensive: bad input returns ``None`` instead of raising.
 * Designed to run in <30 ms on a 200x100 px input.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage  # type: ignore[import-untyped]
from scipy import signal as sp_signal  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_TEMPLATE_PATH = os.path.join(
    _REPO_ROOT, "training_data_blacklist", "bad crop.png"
)

# Edge-detection params
EDGE_GAUSS_SIGMA = 0.8
EDGE_QUANTILE = 0.85         # keep top 15% of gradient magnitude
EDGE_FLOOR = 30.0            # min absolute gradient threshold

# Template scales: target widths in pixels. Canonical template's
# bbox-cropped edge mask is ~62x62 px; these widths span typical icon
# sizes.
TEMPLATE_TARGET_WIDTHS = (16, 20, 24, 28, 32, 36, 44)

# Below this template width we don't accept matches on full-panel
# inputs (small templates fit any edge cluster). Small inputs (where
# the whole image is the icon) get the full scale set.
MIN_PANEL_TEMPLATE_W = 24

# Chamfer threshold for accepting a match.
CHAMFER_MIN = 0.85

# NCC threshold for the cheap pre-filter (NCC of binary edge masks
# can go negative even on the icon, so this is a permissive floor).
NCC_MIN = 0.05
# Stronger NCC threshold for accepting a candidate (filters out
# borderline matches at the edges of edge-rich regions).
NCC_ACCEPT = 0.10

# Edge-density ratio band: panel-edges-in-window / template-edges.
# Two regimes:
#  - On a panel, the icon's window contains the icon outline + some
#    overlap with neighboring digits/text => ratio ~ 1.5-3.0.
#    Empty/noise windows have ratio < 1.0; busy text ratio > 4.0.
#  - On an isolated icon-only crop (whole image is the icon), the
#    panel-edge count ~= template-edge count => ratio ~ 0.5-1.2.
#
# The detector uses input size to pick the appropriate band.
EDGE_RATIO_PANEL_MIN = 1.4
EDGE_RATIO_PANEL_MAX = 3.5
EDGE_RATIO_SMALL_MIN = 0.4   # below this is noise
EDGE_RATIO_SMALL_MAX = 3.5

# Minimum panel-edge perimeter inside the candidate window.
MIN_PERIMETER = 14

# Aspect ratio sanity (h/w of matched template -- always 1:1 since
# template is square, but keep the check for the post-expansion bbox).
MIN_ASPECT_HW = 0.7
MAX_ASPECT_HW = 2.0

# Y-position prior: for tall inputs (panels), the icon is in the
# upper portion. Below this fraction of the input height, candidates
# are rejected. For small inputs (full image is the icon), this
# fraction is effectively 1.0.
Y_FRACTION_MAX = 0.6
Y_FRACTION_DISABLE_HEIGHT = 80   # below this height the Y filter is off

# NMS window size (fraction of template width).
NMS_FRACTION = 0.7

# Combined-confidence weighting (final scalar in [0, 1]).
W_NCC = 0.20
W_CHAMFER = 0.80


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------


def _to_luma_array(image: Any) -> np.ndarray | None:
    """Coerce input to an HxW uint8 luma ndarray; return None on bad input."""
    if image is None:
        return None
    try:
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                pass
            elif arr.ndim == 3 and arr.shape[2] in (3, 4):
                rgb = arr[..., :3].astype(np.float32)
                arr = (
                    0.299 * rgb[..., 0]
                    + 0.587 * rgb[..., 1]
                    + 0.114 * rgb[..., 2]
                )
            else:
                return None
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("L"))
    except Exception:  # pragma: no cover - defensive
        return None
    return None


def _crop_to_hud(luma: np.ndarray, hud_bbox: tuple[int, int, int, int] | None):
    if hud_bbox is None:
        return luma, (0, 0)
    x, y, w, h = hud_bbox
    H, W = luma.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(W, int(x) + int(w))
    y1 = min(H, int(y) + int(h))
    if x1 <= x0 or y1 <= y0:
        return luma, (0, 0)
    return luma[y0:y1, x0:x1], (x0, y0)


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------


def _edges(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Sobel gradient magnitude and a binary edge mask."""
    if gray.size == 0:
        z = np.zeros_like(gray, dtype=np.float32)
        return z, np.zeros_like(gray, dtype=bool)
    smoothed = ndimage.gaussian_filter(gray.astype(np.float32), sigma=EDGE_GAUSS_SIGMA)
    gx = ndimage.sobel(smoothed, axis=1)
    gy = ndimage.sobel(smoothed, axis=0)
    mag = np.hypot(gx, gy).astype(np.float32)
    q = float(np.quantile(mag, EDGE_QUANTILE))
    thr = max(q, EDGE_FLOOR)
    mask = mag >= thr
    return mag, mask


_DILATE_STRUCT = np.ones((3, 3), dtype=bool)


def _dilate(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask
    return ndimage.binary_dilation(mask, structure=_DILATE_STRUCT, iterations=1)


# ---------------------------------------------------------------------------
# Template cache
# ---------------------------------------------------------------------------


_TEMPLATE_CACHE: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] | None = None
# Each entry: (z_edge_bool, z_dil_bool, z_dt_f32, n_edge_pixels, target_width)


def _load_template_edge() -> np.ndarray | None:
    if not os.path.exists(_TEMPLATE_PATH):
        return None
    try:
        gray = np.asarray(Image.open(_TEMPLATE_PATH).convert("L"))
    except Exception:  # pragma: no cover - defensive
        return None
    _, edge = _edges(gray)
    if not edge.any():
        return None
    ys, xs = np.where(edge)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return edge[y0:y1, x0:x1]


def _build_template_cache() -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
    base = _load_template_edge()
    if base is None or base.size == 0:
        return []
    bh, bw = base.shape
    if bw < 4 or bh < 4:
        return []
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    for tw in TEMPLATE_TARGET_WIDTHS:
        s = float(tw) / float(bw)
        if s <= 0 or s > 2.5:
            continue
        try:
            zoomed = ndimage.zoom(base.astype(np.float32), (s, s), order=1) > 0.4
        except Exception:  # pragma: no cover - defensive
            continue
        if zoomed.shape[0] < 8 or zoomed.shape[1] < 8:
            continue
        if not zoomed.any():
            continue
        z_dil = _dilate(zoomed)
        z_dt = ndimage.distance_transform_edt(~zoomed).astype(np.float32)
        out.append((zoomed, z_dil, z_dt, int(zoomed.sum()), int(zoomed.shape[1])))
    return out


def _ensure_cache() -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = _build_template_cache()
    return _TEMPLATE_CACHE


def reset_cache() -> None:
    """Force the next call to rebuild templates from disk."""
    global _TEMPLATE_CACHE
    _TEMPLATE_CACHE = None


# ---------------------------------------------------------------------------
# Matching primitives
# ---------------------------------------------------------------------------


def _ncc_valid(panel: np.ndarray, tpl: np.ndarray) -> np.ndarray | None:
    """FFT-based normalized cross-correlation, 'valid' mode."""
    if panel.shape[0] < tpl.shape[0] or panel.shape[1] < tpl.shape[1]:
        return None
    p = panel.astype(np.float32)
    p = p - p.mean()
    t = tpl.astype(np.float32)
    t = t - t.mean()
    try:
        corr = sp_signal.fftconvolve(p, t[::-1, ::-1], mode="valid")
    except Exception:  # pragma: no cover - defensive
        return None
    t_norm = float(np.sqrt((t ** 2).sum()))
    if t_norm < 1e-6:
        return None
    ones = np.ones_like(t, dtype=np.float32)
    p2_local = sp_signal.fftconvolve(p ** 2, ones, mode="valid")
    p_local_norm = np.sqrt(np.maximum(p2_local, 1e-6))
    return (corr / (p_local_norm * t_norm + 1e-6)).astype(np.float32)


def _sym_chamfer(
    panel_edge: np.ndarray,
    panel_dt: np.ndarray,
    tpl_edge: np.ndarray,
    tpl_dt: np.ndarray,
    x: int,
    y: int,
) -> tuple[float, int]:
    """Symmetric chamfer score in [0, 1] and panel-edge pixel count in window."""
    h, w = tpl_edge.shape
    if y < 0 or x < 0 or y + h > panel_edge.shape[0] or x + w > panel_edge.shape[1]:
        return 0.0, 0
    p_sub = panel_edge[y : y + h, x : x + w]
    if p_sub.shape != (h, w):
        return 0.0, 0
    ey, ex = np.where(tpl_edge)
    if ey.size == 0:
        return 0.0, 0
    fwd = float(panel_dt[y + ey, x + ex].mean())
    py, px = np.where(p_sub)
    if py.size == 0:
        return 0.0, 0
    bwd = float(tpl_dt[py, px].mean())
    diag = float(np.hypot(h, w))
    avg = 0.5 * (fwd + bwd)
    score = 1.0 - avg / max(1.0, diag)
    return float(max(0.0, min(1.0, score))), int(py.size)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_icon_by_contour(
    image: Any,
    hud_bbox: tuple[int, int, int, int] | None = None,
) -> dict | None:
    """Find the location-pin icon via contour/silhouette matching.

    Returns a dict::

        {
            "bbox": (x, y, w, h),
            "confidence": float,
            "details": {
                "contour_match_score": float,
                "edge_density": float,
                "n_contours_examined": int,
                "best_contour_perimeter": int,
            },
        }

    or ``None`` if no candidate passes the thresholds.
    """
    luma_full = _to_luma_array(image)
    if luma_full is None or luma_full.size == 0:
        return None
    if luma_full.ndim != 2:
        return None
    if luma_full.shape[0] < 8 or luma_full.shape[1] < 8:
        return None

    luma, (ox, oy) = _crop_to_hud(luma_full, hud_bbox)
    if luma.shape[0] < 8 or luma.shape[1] < 8:
        return None

    H, W = luma.shape
    # Three input regimes:
    #  - "icon-only": tiny square crop, the whole image is the icon
    #    => relaxed ratio (icon contributes most edges)
    #  - "hud-cropped": short and wide, contains pill+icon+digits
    #    => medium ratio
    #  - "panel": tall full panel with terrain background
    #    => stricter ratio + Y prior to focus on the upper HUD band
    is_panel = H >= Y_FRACTION_DISABLE_HEIGHT
    aspect_input = W / max(1, H)
    is_icon_only = (not is_panel) and aspect_input <= 1.5
    y_max_for_panel = int(H * Y_FRACTION_MAX) if is_panel else H
    if is_panel:
        ratio_min, ratio_max = EDGE_RATIO_PANEL_MIN, EDGE_RATIO_PANEL_MAX
    else:
        # Both icon-only and HUD-cropped use the relaxed band -- a
        # HUD crop has the icon adjacent to digits, so a window
        # tightly around the icon may have ratio just over 1.0.
        ratio_min, ratio_max = EDGE_RATIO_SMALL_MIN, EDGE_RATIO_SMALL_MAX

    templates = _ensure_cache()
    if not templates:
        return None

    try:
        _, p_edge = _edges(luma)
        if not p_edge.any():
            return None
        p_dil = _dilate(p_edge)
        p_dt = ndimage.distance_transform_edt(~p_edge).astype(np.float32)
    except Exception:  # pragma: no cover - defensive
        return None

    edge_density_global = float(p_edge.mean())

    n_examined = 0
    candidates: list[dict] = []

    for z_edge, z_dil, z_dt, t_pix, tw in templates:
        if z_edge.shape[0] > p_edge.shape[0] or z_edge.shape[1] > p_edge.shape[1]:
            continue
        # On panel-mode inputs (panels or wide HUD crops), drop tiny
        # templates -- they fit anywhere and produce false positives.
        # Icon-only inputs use the full scale set.
        if not is_icon_only and z_edge.shape[1] < MIN_PANEL_TEMPLATE_W:
            continue

        try:
            cmap = _ncc_valid(p_dil, z_dil)
        except Exception:  # pragma: no cover - defensive
            continue
        if cmap is None or cmap.size == 0:
            continue

        # Non-maximum suppression at this scale.
        nms_size = max(3, int(z_edge.shape[1] * NMS_FRACTION))
        try:
            nbr = ndimage.maximum_filter(cmap, size=nms_size)
        except Exception:  # pragma: no cover - defensive
            continue
        peak_mask = (cmap == nbr) & (cmap > NCC_MIN)
        ys, xs = np.where(peak_mask)
        if ys.size == 0:
            continue

        for yy, xx in zip(ys.tolist(), xs.tolist()):
            n_examined += 1

            # Y prior: on tall panel inputs, the icon is in the upper
            # portion (HUDs sit above the lower 40% of the image).
            if is_panel and yy > y_max_for_panel:
                continue
            _ = y_max_for_panel  # silence unused-warning when icon-only

            cs, n_panel = _sym_chamfer(p_edge, p_dt, z_edge, z_dt, xx, yy)
            if cs < CHAMFER_MIN:
                continue
            if n_panel < MIN_PERIMETER:
                continue
            ratio = n_panel / max(1, t_pix)
            if ratio < ratio_min or ratio > ratio_max:
                continue

            ncc_score = float(cmap[yy, xx])
            # On panel-mode inputs, require a stronger NCC peak: this
            # filters out matches at the edges of edge-rich regions
            # where chamfer is high but the silhouette correlation is
            # weak.
            if (not is_icon_only) and ncc_score < NCC_ACCEPT:
                continue
            combo = W_NCC * max(0.0, ncc_score) + W_CHAMFER * cs

            h_box, w_box = z_edge.shape
            aspect = h_box / max(1, w_box)
            if aspect < MIN_ASPECT_HW or aspect > MAX_ASPECT_HW:
                continue

            candidates.append(
                {
                    "x": int(xx),
                    "y": int(yy),
                    "w": int(w_box),
                    "h": int(h_box),
                    "chamfer": float(cs),
                    "ncc": float(ncc_score),
                    "combo": float(combo),
                    "ratio": float(ratio),
                    "n_panel": int(n_panel),
                    "tw": int(tw),
                    "t_pix": int(t_pix),
                }
            )

    if not candidates:
        return None

    # Selection strategy depends on input regime:
    #  - icon-only (square small input): pick highest combo score
    #    (the icon dominates the input).
    #  - HUD-cropped (wide short input): pick LEFTMOST candidate
    #    above NCC threshold; the icon is the leftmost UI element.
    #  - full panel: pick highest combo score, with the Y prior
    #    already filtering candidates outside the upper HUD band.
    if is_icon_only or is_panel:
        candidates.sort(key=lambda c: -c["combo"])
    else:
        # HUD-cropped: leftmost with strong NCC, tiebreak by combo.
        strong = [c for c in candidates if c["ncc"] >= NCC_ACCEPT]
        if not strong:
            strong = candidates
        strong.sort(key=lambda c: (c["x"], -c["combo"]))
        candidates = strong
    pick = candidates[0]

    # Use the matched template bbox as-is. Empirically, refining to
    # panel-edge extent or expanding bleeds the bbox into adjacent
    # UI lines (pill border on the left) more often than it
    # tightens to the icon.
    gx, gy = pick["x"], pick["y"]
    gw, gh = pick["w"], pick["h"]

    return {
        "bbox": (gx + ox, gy + oy, gw, gh),
        "confidence": float(pick["combo"]),
        "details": {
            "contour_match_score": float(pick["chamfer"]),
            "edge_density": float(edge_density_global),
            "n_contours_examined": int(n_examined),
            "best_contour_perimeter": int(pick["n_panel"]),
            "ncc_score": float(pick["ncc"]),
            "template_width": int(pick["tw"]),
            "edge_ratio": float(pick["ratio"]),
            "n_candidates": int(len(candidates)),
        },
    }


__all__ = ["find_icon_by_contour", "reset_cache"]
