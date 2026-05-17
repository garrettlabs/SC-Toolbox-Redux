"""Auto-annotation tool for the SC mining HUD training data.

Inverts the manual ``template_annotator.py`` flow: existing anchor
detectors propose bounding boxes for each fixed HUD element. The user
only corrects what's wrong instead of drawing every box by hand.

The annotator supports two HUD regions, switched live via the region
selector in the left panel (or via ``--region`` on the CLI):

REGION 1 — SCAN RESULTS panel (8 features):
    1.  scan_results       — panel title text
    2.  top_line           — HUD chrome line under SCAN RESULTS
    3.  resource           — mineral name row
    4.  mass_row           — entire MASS: <value> row
    5.  resistance_row     — entire RESISTANCE: <value> row
    6.  instability_row    — entire INSTABILITY: <value> row
    7.  outcome            — difficulty bar (always manual)
    8.  bot_line           — HUD chrome line above COMPOSITION

REGION 2 — Signature scanner pill (3 features):
    1.  pill               — rounded-rectangle cyan/teal stroke
    2.  icon               — yellow location-pin
    3.  value              — 4–5 digit numeric value

Each region has its own ELEMENTS list, default source folder, and
detector chain. Per-region config lives in ``REGION_CONFIG`` — adding a
hypothetical region3 is one tuple + one list addition.

Workflow:
    1. Pick a region (left panel) and an image from the file list.
    2. Tool runs the region's detector chain and overlays candidate
       boxes (each labeled with its source detector + score).
    3. Review:
         - Click "Accept all" to take detector output as-is.
         - Click a box to select; drag a corner to resize, drag the
           middle to translate. Press Del to delete.
         - Click "Draw <element>" or press 1-N to draw a missing one
           (N = number of elements in current region).
         - Shift+Drag in empty space defines a HUD-region constraint
           and re-runs detectors INSIDE that region only.
    4. Save (Ctrl+S, also auto-saves on advance) writes
       ``<image>.boxes.json`` with detector_meta side-channel that
       tracks how far each detector was off.

Keyboard shortcuts:
    1-N       arm draw mode for element 1..N (region-dependent)
    Del       delete the selected box
    Left      previous image
    Right     next image
    Ctrl+S    save now
    Ctrl+R    re-run detectors on current image
    Ctrl+A    accept-all (no-op — boxes already shown)

Detector accuracy log:
    Each save records, per-feature:
       detector name, raw score, user_corrected (bool), delta_px
    Reading those back across the labeled set gives a continuous
    accuracy measurement of every anchor without a separate eval
    harness.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import (
    QPointF, QRectF, Qt, Signal,
)
from PySide6.QtGui import (
    QBrush, QColor, QFont, QImage, QKeySequence, QMouseEvent,
    QPainter, QPen, QPixmap, QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QComboBox, QGraphicsItem, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QStatusBar, QVBoxLayout,
    QWidget,
)

# ────────────────────────────────────────────────────────────────────
# Path setup — resolve the Mining_Signals tool root regardless of
# whether this script is run from source or via the .bat launcher.
# ────────────────────────────────────────────────────────────────────
THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
if str(TOOL) not in sys.path:
    sys.path.insert(0, str(TOOL))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("auto_template_annotator")

DEFAULT_SOURCE = TOOL / "training_data_panels" / "user_20260418_154408" / "region1"
DEFAULT_SOURCE_REGION2 = TOOL / "training_data_panels" / "user_20260418_154408" / "region2"

# ────────────────────────────────────────────────────────────────────
# Per-region element lists. Each list is ``[(feature_name, rgb), ...]``
# in the order the user-facing buttons / shortcut keys 1..N follow.
# ────────────────────────────────────────────────────────────────────

# Region 1 — SCAN RESULTS panel (matches template_annotator).
REGION1_ELEMENTS: list[tuple[str, tuple[int, int, int]]] = [
    ("scan_results",    (255, 220, 0)),     # yellow
    ("top_line",        (255, 140, 0)),     # orange
    ("resource",        (0, 230, 100)),     # green
    ("mass_row",        (0, 200, 255)),     # cyan
    ("resistance_row",  (200, 100, 255)),   # purple
    ("instability_row", (255, 100, 200)),   # pink
    ("outcome",         (180, 180, 50)),    # olive
    ("bot_line",        (255, 140, 0)),     # orange
]

# Region 2 — Signature scanner pill (3 features). The ``pill`` plays
# the role ``scan_results`` plays in region1 — it's the anchor for the
# whole widget.
REGION2_ELEMENTS: list[tuple[str, tuple[int, int, int]]] = [
    ("pill",  (0, 230, 255)),        # cyan — the rounded rect of the whole widget
    ("icon",  (255, 200, 0)),        # yellow — the location-pin
    ("value", (200, 100, 255)),      # purple — the numeric value text
]

# Backwards-compatible aliases. ``ELEMENTS`` / ``ELEMENT_NAMES`` are
# kept at module level pointing at region1 so external callers + any
# tests still import a working default. The window swaps these via
# ``self._elements`` / ``self._element_names`` when the region changes.
ELEMENTS: list[tuple[str, tuple[int, int, int]]] = REGION1_ELEMENTS
ELEMENT_NAMES = [n for n, _ in ELEMENTS]


def _color_for(name: str) -> tuple[int, int, int]:
    """Resolve display color for ``name`` across both regions.

    Searches REGION1 + REGION2 element lists. White fallback so an
    unknown name doesn't crash rendering — it just paints white.
    """
    for region_elems in (REGION1_ELEMENTS, REGION2_ELEMENTS):
        for n, c in region_elems:
            if n == name:
                return c
    return (255, 255, 255)


# ────────────────────────────────────────────────────────────────────
# Detector adapters
#
# Each adapter takes (pil_image, optional hud_bbox=None) and returns a
# dict mapping feature_name → {"x", "y", "w", "h", "score", "detector"}.
# Score may be None if the underlying detector doesn't expose one.
#
# Adapters MUST swallow all detector errors and return {} on failure —
# never let one detector crash the whole proposal step.
#
# Adding a new detector: write an adapter, append it to DETECTORS.
# ────────────────────────────────────────────────────────────────────

# Proportional ratios for label-row widening, derived from
# hud_tracker/world_model.json. These are scan_results-relative
# (scan_results.w == 1.0 by definition there). Used to inflate the
# label-text bbox returned by find_label_positions to the full row
# width that template_annotator users would draw.
#
# Per the world model:
#   mass_row.x_frac        = -0.014  (just left of scan_results.x)
#   mass_row.w_frac        =  1.198  (~ 1.20× scan_results width)
#   resistance_row.x_frac  = -0.019
#   resistance_row.w_frac  =  1.227
#   instability_row.x_frac = -0.034
#   instability_row.w_frac =  1.246
#   *_row.h_frac           ≈  1.28 × scan_results.h
_ROW_RATIOS = {
    "mass_row":        {"x_frac": -0.014, "w_frac": 1.198, "h_frac": 1.281},
    "resistance_row":  {"x_frac": -0.019, "w_frac": 1.227, "h_frac": 1.281},
    "instability_row": {"x_frac": -0.034, "w_frac": 1.246, "h_frac": 1.281},
}


def _safe_call(fn, *args, **kwargs) -> Any:
    """Run a detector, swallowing all exceptions so one bad detector
    never breaks the proposal step."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log.warning("detector %s failed: %s", fn.__name__, exc)
        log.debug("traceback: %s", traceback.format_exc())
        return None


def _crop_for_hud(img: Image.Image, hud_bbox: Optional[tuple[int, int, int, int]]):
    """If a HUD constraint is given, crop the image to that region for
    detector input, and return (cropped_pil, dx, dy) where dx, dy is
    the offset to add to detector coordinates to map back to the
    full-image frame.

    If no HUD bbox, returns (img, 0, 0).
    """
    if hud_bbox is None:
        return img, 0, 0
    x, y, w, h = hud_bbox
    x = max(0, int(x))
    y = max(0, int(y))
    w = max(1, int(w))
    h = max(1, int(h))
    return img.crop((x, y, x + w, y + h)), x, y


def detect_scan_results(
    img: Image.Image,
    hud_bbox: Optional[tuple[int, int, int, int]] = None,
) -> dict[str, dict]:
    """find_scan_results_anchor → scan_results bbox."""
    try:
        from ocr.sc_ocr.scan_results_match import find_scan_results_anchor
    except Exception as exc:
        log.warning("scan_results detector unavailable: %s", exc)
        return {}
    crop, dx, dy = _crop_for_hud(img, hud_bbox)
    res = _safe_call(find_scan_results_anchor, crop)
    if not res:
        return {}
    return {
        "scan_results": {
            "x": int(res["title_x"]) + dx,
            "y": int(res["title_y"]) + dy,
            "w": int(res["title_w"]),
            "h": int(res["title_h"]),
            "score": float(res.get("score", 0.0)),
            "detector": "find_scan_results_anchor",
        }
    }


def detect_label_rows(
    img: Image.Image,
    hud_bbox: Optional[tuple[int, int, int, int]] = None,
) -> dict[str, dict]:
    """find_label_positions → mass_row / resistance_row / instability_row.

    The detector returns the LABEL-TEXT bbox only (e.g. "MASS:"). We
    widen it to the full-row bbox using proportional ratios from
    hud_tracker/world_model.json. The "anchor" for the row is the
    scan_results bbox if available; otherwise we fall back to the
    image width (which is roughly the panel width on these crops).
    """
    try:
        from ocr.sc_ocr.label_match import find_label_positions
    except Exception as exc:
        log.warning("label_rows detector unavailable: %s", exc)
        return {}

    crop, dx, dy = _crop_for_hud(img, hud_bbox)
    matches = _safe_call(find_label_positions, crop) or {}
    if not matches:
        return {}

    # Anchor for widening. Try scan_results first; fall back to the
    # image (or HUD-cropped) width. The world-model widths are quoted
    # as fractions of scan_results.w, where scan_results.w_frac=1.0,
    # so we only need a width estimate to inflate.
    sr_anchor = _safe_call(_get_scan_results_for_anchor, crop)
    if sr_anchor is not None:
        sr_x = int(sr_anchor["title_x"])
        sr_y = int(sr_anchor["title_y"])
        sr_w = int(sr_anchor["title_w"])
        sr_h = int(sr_anchor["title_h"])
    else:
        # No anchor — assume the panel fills the search image roughly.
        cw, ch = crop.size
        sr_x = 0
        sr_y = 0
        sr_w = cw
        sr_h = max(20, int(ch * 0.06))

    out: dict[str, dict] = {}
    name_map = {
        "mass":        "mass_row",
        "resistance":  "resistance_row",
        "instability": "instability_row",
    }
    for label_key, feature_name in name_map.items():
        m = matches.get(label_key)
        if not m:
            continue
        ratios = _ROW_RATIOS[feature_name]
        # Row width inflated from scan_results.w. Row height inflated
        # from the label-text height (which is the row's text band) —
        # h_frac in world_model.json is ~1.28 of scan_results.h, so
        # use ratios["h_frac"] * sr_h, but never less than the label's
        # own height.
        row_w = max(int(m["w"]), int(round(sr_w * ratios["w_frac"])))
        row_h = max(int(m["h"]), int(round(sr_h * ratios["h_frac"])))
        # Row x: world model says it's slightly left of scan_results.
        # But the LABEL-TEXT match also tells us where the "M" of MASS
        # actually is. Use min(label_x, sr_x + frac_offset) so we
        # never accidentally trim the label off the left.
        proposed_x = sr_x + int(round(sr_w * ratios["x_frac"]))
        row_x = max(0, min(proposed_x, int(m["x"])))
        # Row y: center the row band on the label-text vertical center.
        label_cy = int(m["y"]) + int(m["h"]) // 2
        row_y = max(0, label_cy - row_h // 2)
        out[feature_name] = {
            "x": row_x + dx,
            "y": row_y + dy,
            "w": row_w,
            "h": row_h,
            "score": float(m.get("score", 0.0)),
            "detector": "find_label_positions",
        }
    return out


def _get_scan_results_for_anchor(crop: Image.Image) -> Optional[dict]:
    """Same as the scan_results detector but returns the raw dict for
    use as the anchoring reference inside detect_label_rows."""
    from ocr.sc_ocr.scan_results_match import find_scan_results_anchor
    return find_scan_results_anchor(crop)


def find_hud_panel_color(
    img: Image.Image,
) -> Optional[dict]:
    """Run ``hud_tracker.anchors.hud_color_finder.find_hud_panel`` on
    the full image. Returns the finder's result dict (with bbox +
    confidence + chrome pixel count) or None.

    Same import-path dance as ``detect_panel_lines`` so this runs both
    in the WingmanAI dev tree and the production tree.
    """
    try:
        from hud_tracker.anchors.hud_color_finder import find_hud_panel
    except ImportError:
        import os as _os
        prod_tree = _os.path.expandvars(
            r"%LOCALAPPDATA%\SC_Toolbox\current\tools\Mining_Signals"
        )
        if _os.path.isdir(prod_tree) and prod_tree not in sys.path:
            sys.path.insert(0, prod_tree)
        try:
            from hud_tracker.anchors.hud_color_finder import find_hud_panel
        except Exception as exc:
            log.warning("hud_color_finder unavailable: %s", exc)
            return None
    except Exception as exc:
        log.warning("hud_color_finder unavailable: %s", exc)
        return None
    return _safe_call(find_hud_panel, img)


def detect_panel_lines(
    img: Image.Image,
    hud_bbox: Optional[tuple[int, int, int, int]] = None,
) -> dict[str, dict]:
    """find_chrome_lines → top_line, bot_line.

    Uses ``hud_tracker.anchors.chrome_lines.find_chrome_lines`` —
    the rev2 isolation-based detector that:
      * Reliably finds the bot_line on dim panels (where the
        legacy 80% column-fill rule sometimes culled it).
      * Rejects the COMPOSITION underline pair below the panel
        (where the legacy "lines[-1]" heuristic mis-fired).
      * Returns bboxes that include the end-bracket margins so
        they line up with manually-drawn GT boxes.
    """
    try:
        from hud_tracker.anchors.chrome_lines import find_chrome_lines
    except ImportError:
        # hud_tracker lives in the production tree, not the WingmanAI
        # dev tree where this annotator runs. Add it to sys.path and retry.
        import os as _os
        prod_tree = _os.path.expandvars(
            r"%LOCALAPPDATA%\SC_Toolbox\current\tools\Mining_Signals"
        )
        if _os.path.isdir(prod_tree) and prod_tree not in sys.path:
            sys.path.insert(0, prod_tree)
        try:
            from hud_tracker.anchors.chrome_lines import find_chrome_lines
        except Exception as exc:
            log.warning("chrome_lines detector unavailable: %s", exc)
            return {}
    except Exception as exc:
        log.warning("chrome_lines detector unavailable: %s", exc)
        return {}

    crop, dx, dy = _crop_for_hud(img, hud_bbox)
    result = _safe_call(find_chrome_lines, crop)
    if not result:
        return {}

    out: dict[str, dict] = {}
    for key in ("top_line", "bot_line"):
        bbox = result.get(key)
        if bbox is None:
            continue
        score = bbox.get("score")
        out[key] = {
            "x": int(bbox["x"]) + dx,
            "y": int(bbox["y"]) + dy,
            "w": int(bbox["w"]),
            "h": int(bbox["h"]),
            "score": float(score) if score is not None else None,
            "detector": "find_chrome_lines",
        }
    return out


def detect_resource(
    img: Image.Image,
    hud_bbox: Optional[tuple[int, int, int, int]] = None,
) -> dict[str, dict]:
    """_find_mineral_row_universal → resource bbox.

    Returns a y-band only; we widen to the cropped image width since
    the mineral-row spans the full panel width on these crops.
    """
    fn = None
    fn_name = "_find_mineral_row_universal"
    try:
        from ocr.sc_ocr.api import _find_mineral_row_universal as fn
    except Exception as exc:
        log.debug("universal mineral row not available: %s", exc)
    if fn is None:
        try:
            from ocr.onnx_hud_reader import _find_mineral_row as fn  # type: ignore
            fn_name = "_find_mineral_row"
        except Exception as exc:
            log.warning("mineral_row detector unavailable: %s", exc)
            return {}

    crop, dx, dy = _crop_for_hud(img, hud_bbox)
    band = _safe_call(fn, crop)
    if not band:
        return {}
    y1, y2 = band
    cw, _ = crop.size
    return {
        "resource": {
            "x": dx,
            "y": int(y1) + dy,
            "w": int(cw),
            "h": int(y2) - int(y1),
            "score": None,
            "detector": fn_name,
        }
    }


# ────────────────────────────────────────────────────────────────────
# Region 2 detectors — signature scanner pill widget.
#
# Three features:
#   * pill  — rounded-rectangle outline (the whole widget). Detected
#             via ``hud_color_finder.find_hud_panel`` reused with
#             region2-tuned HSV bands + tighter aspect/extent rules.
#   * icon  — location-pin. Detected via
#             ``ocr.sc_ocr.signal_anchor.find_icon`` (multi-scale NCC
#             + CNN re-rank).
#   * value — 4–7 digit numeric cluster. Detected via
#             ``ocr.sc_ocr.signal_anchor.find_digit_cluster``.
#
# All adapters honor the ``hud_bbox`` constraint (crop → run → unshift)
# the same way region1 detectors do.
# ────────────────────────────────────────────────────────────────────

# Calibration for the pill's color-segmentation pass. The pill is a
# saturated cyan/teal stroke; values inside are yellow location-pin +
# cyan digits. Tuned on region2 captures from
# user_20260418_154408 (~18/20 hit rate). Designed to be replaced by
# a per-region calibration JSON later when we have labeled GT.
PILL_CALIBRATION: dict = {
    "version": 1,
    "source": "region2-fallback-defaults",
    "n_captures": 0,
    # Cyan/teal band — the pill stroke + interior digit text.
    "cyan_band": {"h_min": 100, "h_max": 180},
    # Yellow band — the location-pin icon.
    "green_band": {"h_min": 15, "h_max": 60},
    "sat_min": 60,
    "val_min": 80,
    # Geometry — pill is a ~3.5:1 wide rectangle.
    "min_area_px": 600,
    "min_bbox_aspect": 1.5,
    "max_bbox_aspect": 5.5,
    "min_extent": 0.4,
    # Morphology — single row, glyphs don't span multiple lines so we
    # don't want to bridge vertically.
    "morph_seed_iterations": 2,
    "morph_vert_close_px": 3,
    "morph_horiz_close_px": 30,
    "bbox_aspect_peak": 3.5,
}


def detect_pill(
    img: Image.Image,
    hud_bbox: Optional[tuple[int, int, int, int]] = None,
) -> dict[str, dict]:
    """Reuse ``hud_color_finder.find_hud_panel`` with region2-tuned
    calibration to locate the signature scanner pill.

    Returns ``{"pill": {...}}`` or ``{}``.
    """
    try:
        from hud_tracker.anchors.hud_color_finder import find_hud_panel
    except ImportError:
        import os as _os
        prod_tree = _os.path.expandvars(
            r"%LOCALAPPDATA%\SC_Toolbox\current\tools\Mining_Signals"
        )
        if _os.path.isdir(prod_tree) and prod_tree not in sys.path:
            sys.path.insert(0, prod_tree)
        try:
            from hud_tracker.anchors.hud_color_finder import find_hud_panel
        except Exception as exc:
            log.warning("pill detector unavailable: %s", exc)
            return {}
    except Exception as exc:
        log.warning("pill detector unavailable: %s", exc)
        return {}

    crop, dx, dy = _crop_for_hud(img, hud_bbox)
    res = _safe_call(find_hud_panel, crop, calibration=PILL_CALIBRATION)
    if not res:
        return {}
    bbox = res.get("bbox")
    if not bbox or len(bbox) != 4:
        return {}
    return {
        "pill": {
            "x": int(bbox[0]) + dx,
            "y": int(bbox[1]) + dy,
            "w": int(bbox[2]),
            "h": int(bbox[3]),
            "score": float(res.get("confidence", 0.0)),
            "detector": "find_hud_panel",
        }
    }


def detect_icon(
    img: Image.Image,
    hud_bbox: Optional[tuple[int, int, int, int]] = None,
) -> dict[str, dict]:
    """Wrap ``signal_anchor.find_icon`` for the region2 element ``icon``.

    The detector wants a grayscale numpy array, so we convert here.
    Returns ``{"icon": {...}}`` or ``{}``.
    """
    try:
        from ocr.sc_ocr.signal_anchor import find_icon, reset_anchor_cache
    except Exception as exc:
        log.warning("icon detector unavailable: %s", exc)
        return {}

    # Reset find_icon's temporal-smoothing cache. The cache is meant
    # for live video where the icon stays in roughly the same spot
    # frame-to-frame; in the labeler the user jumps between unrelated
    # captures, and a stale cache entry causes the next image's call
    # to fail with the low-confidence-disagrees-with-cache → None
    # path. Each labeled image must be evaluated independently.
    try:
        reset_anchor_cache()
    except Exception:
        pass

    crop, dx, dy = _crop_for_hud(img, hud_bbox)
    try:
        gray = np.asarray(crop.convert("L"), dtype=np.uint8)
        rgb = np.asarray(crop.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        log.warning("icon detector: failed to convert: %s", exc)
        return {}

    # ── NEW PRIMARY PATH: RGB-aware structural localizer ─────────────
    # Run the geometry + RGB-NCC consensus localizer FIRST. Both
    # detectors leverage the icon's warm-color signature (R-channel
    # heavy NCC + HSV mask + structural primitives), which the legacy
    # grayscale NCC discards. On the canonical failure case
    # (cap_20260418_155446_555.png) the legacy NCC clustered all 8
    # candidates on the cyan digit area; this path lands them on the
    # actual icon. We only fall back to legacy when the primaries
    # disagree — see hud_tracker.anchors.icon_voter.localize_icon.
    try:
        from hud_tracker.anchors.icon_voter import localize_icon as _localize_icon
    except ImportError:
        # hud_tracker lives in the production tree, not the WingmanAI
        # dev tree where this annotator can also run. Same dance as
        # detect_panel_lines / find_hud_panel_color.
        import os as _os
        _prod_tree = _os.path.expandvars(
            r"%LOCALAPPDATA%\SC_Toolbox\current\tools\Mining_Signals"
        )
        if _os.path.isdir(_prod_tree) and _prod_tree not in sys.path:
            sys.path.insert(0, _prod_tree)
        try:
            from hud_tracker.anchors.icon_voter import localize_icon as _localize_icon
        except Exception as _exc:
            log.debug("icon detector: localize_icon unavailable: %s", _exc)
            _localize_icon = None  # type: ignore[assignment]
    except Exception as _exc:
        log.debug("icon detector: localize_icon unavailable: %s", _exc)
        _localize_icon = None  # type: ignore[assignment]

    if _localize_icon is not None:
        try:
            loc = _localize_icon(rgb, hud_bbox=None)
        except Exception as _exc:
            log.warning("icon detector: localize_icon raised: %s", _exc)
            loc = None
        if loc is not None:
            lx, ly, lw, lh = loc["bbox"]
            return {
                "icon": {
                    "x": int(lx) + dx,
                    "y": int(ly) + dy,
                    "w": int(lw),
                    "h": int(lh),
                    "score": float(loc.get("score", 0.0)),
                    "detector": str(loc.get("detector", "localize_icon")),
                }
            }

    # ── FALLBACK: legacy find_icon + voter pipeline ──────────────────
    # Pass rgb_image so find_icon's per-candidate voter can run the
    # full 4-tier hierarchy (geometry + contour + rgb_cnn_v2 + gray_cnn).
    # Without rgb_image the voter degrades to gray-only mode and falls
    # back to the legacy NCC+CNN path we built the new architecture
    # to replace.
    res = _safe_call(find_icon, gray, rgb_image=rgb)
    if not res:
        return {}
    x1, y1, x2, y2, score = res

    # Positional prior: the location-pin icon is ALWAYS at the LEFT
    # EDGE of the signal pill (structurally x_frac ~0.05-0.15 of pill
    # width — never past ~0.30). Any "icon" match further right is a
    # false positive — usually find_icon's NCC + CNN re-rank locking
    # onto a digit shape that loosely resembles the @ glyph.
    #
    # Threshold 0.30: the real icon's LEFT EDGE is well under this;
    # this rejects matches that land on or past the digit cluster
    # without losing any real icons. (Earlier 0.5 was too lenient —
    # a digit at 40% of pill width would still pass.)
    crop_w = gray.shape[1] if gray.ndim >= 2 else 0
    if crop_w > 0 and x1 > crop_w * 0.30:
        log.info(
            "icon detector: rejecting match at x1=%d "
            "(>30%% of crop_w=%d) as digit-area false positive",
            x1, crop_w,
        )
        return {}

    # Color prior: the location-pin icon is warm yellow/orange.
    # Digit values are cool cyan/teal. find_icon does shape NCC + CNN
    # re-rank, but neither uses color — so a shape that correlates
    # with the icon template can pass even when the underlying pixels
    # are cyan digits. Sample the candidate region's HSV; reject if
    # cyan-dominated.
    try:
        cand_crop = crop.crop((int(x1), int(y1), int(x2), int(y2)))
        hsv = np.asarray(cand_crop.convert("HSV"), dtype=np.uint8)
        # Mask out low-saturation / low-value pixels (panel background,
        # white antialias). PIL HSV scale: 0-255 each.
        sat = hsv[..., 1]
        val = hsv[..., 2]
        bright = (sat >= 80) & (val >= 80)
        if bright.sum() >= 8:
            hue = hsv[..., 0][bright]
            # Warm hues (yellow/orange): PIL hue 0-50 (≈ 0-70°).
            # Cool hues (cyan/teal): PIL hue 100-150 (≈ 140-210°).
            n_warm = int(((hue <= 50) | (hue >= 230)).sum())
            n_cool = int(((hue >= 100) & (hue <= 150)).sum())
            n_total = max(1, int(bright.sum()))
            warm_frac = n_warm / n_total
            cool_frac = n_cool / n_total
            if cool_frac > 0.30 and cool_frac > warm_frac:
                log.info(
                    "icon detector: rejecting match — region is "
                    "cyan-dominated (warm=%.2f cool=%.2f), almost "
                    "certainly a digit false positive, not the icon",
                    warm_frac, cool_frac,
                )
                return {}
    except Exception as exc:
        log.warning("icon color check failed (non-fatal): %s", exc)

    return {
        "icon": {
            "x": int(x1) + dx,
            "y": int(y1) + dy,
            "w": int(x2) - int(x1),
            "h": int(y2) - int(y1),
            "score": float(score),
            "detector": "find_icon",
        }
    }


# Cached world-model-region2 calibration. None until the first lookup;
# False if the file is missing or unreadable so we don't keep retrying.
_WORLD_MODEL_REGION2: Any = None  # None = unloaded, False = missing


def _load_region2_world_model() -> Optional[dict]:
    """Load + memoise hud_tracker/world_model_region2.json.

    Returns the parsed dict on success, or None if the calibration file
    is missing (calibration not yet run). Result is cached after the
    first call so subsequent detector invocations are free.
    """
    global _WORLD_MODEL_REGION2
    if _WORLD_MODEL_REGION2 is not None:
        return _WORLD_MODEL_REGION2 or None  # False -> None

    # Search both the production tree (preferred) and the dev tree
    # in WingmanAI custom_skills (only present when running this
    # annotator standalone from the source tree).
    candidates: list[Path] = []
    prod_tree = Path(os.path.expandvars(
        r"%LOCALAPPDATA%\SC_Toolbox\current\tools\Mining_Signals"
    ))
    candidates.append(prod_tree / "hud_tracker" / "world_model_region2.json")
    candidates.append(TOOL / "hud_tracker" / "world_model_region2.json")

    for p in candidates:
        try:
            if p.is_file():
                with p.open("r", encoding="utf-8") as fh:
                    _WORLD_MODEL_REGION2 = json.load(fh)
                log.info("detect_value: loaded world model from %s", p)
                return _WORLD_MODEL_REGION2
        except Exception as exc:
            log.debug("detect_value: failed to load %s: %s", p, exc)
            continue

    log.info(
        "detect_value: no world_model_region2.json found "
        "(calibration not run yet); will fall back to NCC"
    )
    _WORLD_MODEL_REGION2 = False  # type: ignore[assignment]
    return None


def _get_pill_for_value_anchor(
    img: Image.Image,
    hud_bbox: Optional[tuple[int, int, int, int]],
) -> Optional[tuple[int, int, int, int]]:
    """Run detect_pill and return its bbox tuple, or None on miss.

    Used by detect_value's proportional path to anchor the digit area
    inside the pill rectangle. Re-runs detect_pill rather than threading
    cached state through; pill detection is fast (single pass over a
    small crop) and the auto-annotator already calls detect_pill in
    run_all_detectors so each capture pays the cost twice — acceptable
    for the labeller's interactive cadence.
    """
    res = _safe_call(detect_pill, img, hud_bbox)
    if not res:
        return None
    pill = res.get("pill")
    if not pill:
        return None
    try:
        return (int(pill["x"]), int(pill["y"]), int(pill["w"]), int(pill["h"]))
    except (KeyError, TypeError, ValueError):
        return None


def _get_icon_for_value_anchor(
    img: Image.Image,
    hud_bbox: Optional[tuple[int, int, int, int]],
) -> Optional[tuple[int, int, int, int]]:
    """Run detect_icon and return its bbox tuple, or None on miss.

    Same rationale as _get_pill_for_value_anchor — re-runs are cheap and
    keep detect_value's signature unchanged.
    """
    res = _safe_call(detect_icon, img, hud_bbox)
    if not res:
        return None
    icon = res.get("icon")
    if not icon:
        return None
    try:
        return (int(icon["x"]), int(icon["y"]), int(icon["w"]), int(icon["h"]))
    except (KeyError, TypeError, ValueError):
        return None


def _value_bbox_from_proportions(
    pill_bbox: tuple[int, int, int, int],
    value_frac: dict,
) -> tuple[int, int, int, int]:
    """Apply mean fractional coords to a pill bbox -> (x, y, w, h)."""
    px, py, pw, ph = pill_bbox
    vx = int(round(px + float(value_frac["x_frac"]["mean"]) * pw))
    vy = int(round(py + float(value_frac["y_frac"]["mean"]) * ph))
    vw = int(round(float(value_frac["w_frac"]["mean"]) * pw))
    vh = int(round(float(value_frac["h_frac"]["mean"]) * ph))
    return vx, vy, vw, vh


def detect_value(
    img: Image.Image,
    hud_bbox: Optional[tuple[int, int, int, int]] = None,
) -> dict[str, dict]:
    """Region2 value (digit string) bbox detector.

    PRIMARY: derive from icon + pill via region2 world model proportions.
        The signal pill has a fixed structural layout — once the pill
        bbox is known, the digit cluster's position inside it is
        determined by stable mean fractions. With both pill AND icon
        present we additionally clamp the LHS to icon.right + gap
        (avoids overlapping the icon when the icon is unusually wide)
        and the RHS to pill.right - margin (avoids extending past the
        pill stroke into other UI).

    PILL-ONLY FALLBACK: when icon detection misses but pill is found,
        still derive the value bbox from pill proportions (lower
        confidence — we can't refine the LHS).

    LEGACY FALLBACK: when neither the world model is available NOR pill
        detection succeeds, fall back to ``signal_anchor.find_digit_cluster``
        (NCC-based 4-7 glyph spatial matcher). find_digit_cluster
        doesn't return a confidence score, so we synthesize one from
        the cluster glyph count.

    Returns ``{"value": {...}}`` or ``{}``.
    """
    crop, dx, dy = _crop_for_hud(img, hud_bbox)

    # ── 1. PRIMARY: proportional derivation from pill + icon ────────
    wmr = _load_region2_world_model()
    if wmr is not None:
        value_frac = (wmr.get("features") or {}).get("value")
        if value_frac is not None:
            pill_bbox = _get_pill_for_value_anchor(img, hud_bbox)
            if pill_bbox is not None:
                icon_bbox = _get_icon_for_value_anchor(img, hud_bbox)

                vx, vy, vw, vh = _value_bbox_from_proportions(
                    pill_bbox, value_frac,
                )
                px, py, pw, ph = pill_bbox

                if icon_bbox is not None:
                    # Refinement: clamp LHS to icon.right + small gap.
                    # The icon never overlaps the digit cluster in a
                    # well-formed pill; this rejects pure-proportional
                    # drift when the icon happens to be wider than usual.
                    ix, _iy, iw, _ih = icon_bbox
                    lhs_floor = ix + iw + max(2, int(pw * 0.03))
                    # And RHS to pill.right - small margin so the bbox
                    # doesn't bleed past the pill stroke.
                    rhs_ceiling = px + pw - max(2, int(pw * 0.05))
                    if vx < lhs_floor:
                        vw -= (lhs_floor - vx)
                        vx = lhs_floor
                    if vx + vw > rhs_ceiling:
                        vw = rhs_ceiling - vx
                    detector_name = "world_model_region2"
                    score = 1.0
                else:
                    # Pill-only path — still better than nothing.
                    detector_name = "world_model_region2_pill_only"
                    score = 0.7

                if vw > 0 and vh > 0:
                    return {
                        "value": {
                            "x": int(vx),
                            "y": int(vy),
                            "w": int(vw),
                            "h": int(vh),
                            "score": float(score),
                            "detector": detector_name,
                        }
                    }

    # ── 2. LEGACY FALLBACK: NCC find_digit_cluster ──────────────────
    try:
        from ocr.sc_ocr.signal_anchor import find_digit_cluster
    except Exception as exc:
        log.warning("value detector unavailable: %s", exc)
        return {}

    try:
        gray = np.asarray(crop.convert("L"), dtype=np.uint8)
    except Exception as exc:
        log.warning("value detector: failed to convert to gray: %s", exc)
        return {}
    res = _safe_call(find_digit_cluster, gray)
    if not res:
        return {}
    x1, y1, x2, y2 = res
    w = int(x2) - int(x1)
    h = int(y2) - int(y1)
    if w <= 0 or h <= 0:
        return {}
    # Synthetic confidence: glyph_count_estimate / 7. We approximate
    # the count from the bbox width (digits are ~6-10 px wide each in
    # region2 captures).
    approx_glyphs = max(4, min(7, w // 8))
    return {
        "value": {
            "x": int(x1) + dx,
            "y": int(y1) + dy,
            "w": w,
            "h": h,
            "score": float(approx_glyphs) / 7.0,
            "detector": "find_digit_cluster",
        }
    }


# Pluggable detector registries. Each entry is a ``(name, callable)``
# tuple where the callable takes (img, hud_bbox) and returns
# ``dict[feature_name -> bbox dict]``.
REGION1_DETECTORS: list[tuple[str, Callable[..., dict[str, dict]]]] = [
    ("scan_results",     detect_scan_results),
    ("label_rows",       detect_label_rows),
    ("chrome_lines",     detect_panel_lines),
    ("resource",         detect_resource),
]
REGION2_DETECTORS: list[tuple[str, Callable[..., dict[str, dict]]]] = [
    ("pill",             detect_pill),
    ("icon",             detect_icon),
    ("value",            detect_value),
]

# Backwards-compatible alias — module-level ``DETECTORS`` defaults to
# region1. Window state swaps this via ``self._detectors``.
DETECTORS: list[tuple[str, Callable[..., dict[str, dict]]]] = REGION1_DETECTORS


# ────────────────────────────────────────────────────────────────────
# REGION_CONFIG — single dict the window inspects to switch modes.
#
# Adding a hypothetical region3 is one entry: define
# ``REGION3_ELEMENTS``, ``REGION3_DETECTORS``, point at a default
# folder, and add a key here. The UI handles the rest.
# ────────────────────────────────────────────────────────────────────
REGION_CONFIG: dict[str, dict] = {
    "region1": {
        "label":          "Region 1 (SCAN RESULTS)",
        "elements":       REGION1_ELEMENTS,
        "detectors":      REGION1_DETECTORS,
        "default_source": DEFAULT_SOURCE,
        # Used by detect_label_rows to widen find_label_positions
        # output to full-row bboxes. Region2 has no analog.
        "row_ratios":     _ROW_RATIOS,
    },
    "region2": {
        "label":          "Region 2 (Signature scanner)",
        "elements":       REGION2_ELEMENTS,
        "detectors":      REGION2_DETECTORS,
        "default_source": DEFAULT_SOURCE_REGION2,
        "row_ratios":     None,
    },
}


def run_all_detectors(
    img: Image.Image,
    hud_bbox: Optional[tuple[int, int, int, int]] = None,
    detectors: Optional[list[tuple[str, Callable[..., dict[str, dict]]]]] = None,
) -> dict[str, dict]:
    """Run every detector in ``detectors`` (or the module-level default
    ``DETECTORS``) and merge their outputs.

    On feature collision (two detectors propose the same feature),
    keep the proposal with the higher score; if either has score=None
    we keep the first-registered one.
    """
    detector_chain = detectors if detectors is not None else DETECTORS
    merged: dict[str, dict] = {}
    for name, detector in detector_chain:
        try:
            result = detector(img, hud_bbox)
        except Exception as exc:
            log.warning(
                "detector %s threw outside try/except: %s",
                name, exc,
            )
            continue
        if not result:
            continue
        for feature, bbox in result.items():
            if feature not in merged:
                merged[feature] = bbox
                continue
            cur_score = merged[feature].get("score")
            new_score = bbox.get("score")
            if (
                cur_score is None
                or (new_score is not None and new_score > cur_score)
            ):
                merged[feature] = bbox
    return merged


# ────────────────────────────────────────────────────────────────────
# QGraphicsRectItem subclass with corner-handle resize.
# ────────────────────────────────────────────────────────────────────

_HANDLE_PX = 8  # corner-handle hit radius


class EditableRectItem(QGraphicsRectItem):
    """A rectangle that lets the user grab any corner to resize, or
    grab the middle to translate. Stores the feature name + a label
    item so the auto-annotator can find them by name later.
    """
    def __init__(self, feature_name: str, color: tuple[int, int, int],
                 x: int, y: int, w: int, h: int):
        super().__init__(QRectF(x, y, w, h))
        self.feature_name = feature_name
        self.label_item = None  # set by scene
        self._color = color
        self.setPen(QPen(QColor(*color), 2))
        self.setBrush(QBrush(Qt.NoBrush))
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)  # we move manually
        self.setAcceptHoverEvents(True)
        self._drag_mode: Optional[str] = None  # "move" or "tl"/"tr"/"bl"/"br"
        self._drag_start_scene: Optional[QPointF] = None
        self._drag_start_rect: Optional[QRectF] = None

    def _hit_handle(self, scene_pos: QPointF) -> Optional[str]:
        r = self.rect()
        x, y, w, h = r.x(), r.y(), r.width(), r.height()
        local = scene_pos - self.pos()
        lx, ly = local.x(), local.y()
        # Corners
        for handle, hx, hy in (
            ("tl", x, y),
            ("tr", x + w, y),
            ("bl", x, y + h),
            ("br", x + w, y + h),
        ):
            if abs(lx - hx) <= _HANDLE_PX and abs(ly - hy) <= _HANDLE_PX:
                return handle
        # Inside → move
        if x <= lx <= x + w and y <= ly <= y + h:
            return "move"
        return None

    def hoverMoveEvent(self, event):
        h = self._hit_handle(event.scenePos())
        if h in ("tl", "br"):
            self.setCursor(Qt.SizeFDiagCursor)
        elif h in ("tr", "bl"):
            self.setCursor(Qt.SizeBDiagCursor)
        elif h == "move":
            self.setCursor(Qt.SizeAllCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            mode = self._hit_handle(event.scenePos())
            if mode is not None:
                self._drag_mode = mode
                self._drag_start_scene = event.scenePos()
                self._drag_start_rect = QRectF(self.rect())
                self.setSelected(True)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_mode is None or self._drag_start_rect is None:
            super().mouseMoveEvent(event)
            return
        d = event.scenePos() - self._drag_start_scene
        r = QRectF(self._drag_start_rect)
        if self._drag_mode == "move":
            r.translate(d)
        elif self._drag_mode == "tl":
            r.setTopLeft(r.topLeft() + d)
        elif self._drag_mode == "tr":
            r.setTopRight(r.topRight() + d)
        elif self._drag_mode == "bl":
            r.setBottomLeft(r.bottomLeft() + d)
        elif self._drag_mode == "br":
            r.setBottomRight(r.bottomRight() + d)
        # Normalize so the rect always has positive width/height.
        nr = r.normalized()
        if nr.width() < 4 or nr.height() < 2:
            return
        self.setRect(nr)
        # Keep the label glued to the top-left.
        if self.label_item is not None:
            self.label_item.setPos(nr.x(), max(0, nr.y() - 16))
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._drag_mode is not None:
            self._drag_mode = None
            self._drag_start_scene = None
            self._drag_start_rect = None
            scene = self.scene()
            if isinstance(scene, AutoAnnotationScene):
                scene.box_edited.emit(
                    self.feature_name, QRectF(self.rect()),
                )
            event.accept()
            return
        super().mouseReleaseEvent(event)


# ────────────────────────────────────────────────────────────────────
# Scene
# ────────────────────────────────────────────────────────────────────


class AutoAnnotationScene(QGraphicsScene):
    box_drawn = Signal(str, QRectF)        # element_name, scene_rect
    box_edited = Signal(str, QRectF)       # name, new_rect
    box_deleted = Signal(str)              # name
    hud_bbox_drawn = Signal(QRectF)        # shift+drag → new HUD region

    def __init__(self):
        super().__init__()
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._active_element: Optional[str] = None
        self._dragging = False
        self._drag_start: Optional[QPointF] = None
        self._drag_rect_item: Optional[QGraphicsRectItem] = None
        self._drag_is_hud = False
        self._element_rects: dict[str, EditableRectItem] = {}
        self._element_labels: dict[str, Any] = {}
        self._hud_rect_item: Optional[QGraphicsRectItem] = None
        # MAGENTA: real color+geometry HUD finder output. Rendered
        # only when the upstream find_hud_panel succeeded; distinct
        # from the synthesis-cyan box that's a fallback "union of row
        # anchors" estimate.
        self._hud_color_rect_item: Optional[QGraphicsRectItem] = None
        self._hud_color_label_item: Optional[Any] = None

    def set_image(self, pil_img: Image.Image) -> None:
        self.clear()
        self._element_rects.clear()
        self._element_labels.clear()
        self._hud_rect_item = None
        # The HUD finder's rect + label were destroyed by clear() too;
        # null out the references so the next set_hud_finder_bbox
        # doesn't try to removeItem() on dead Qt objects.
        self._hud_finder_rect_item = None
        self._hud_finder_label_item = None
        self._hud_color_rect_item = None
        self._hud_color_label_item = None
        rgb = pil_img.convert("RGB")
        qimg = QImage(
            rgb.tobytes(), rgb.width, rgb.height,
            rgb.width * 3, QImage.Format_RGB888,
        )
        pix = QPixmap.fromImage(qimg)
        self._pixmap_item = self.addPixmap(pix)
        self.setSceneRect(0, 0, pix.width(), pix.height())

    def set_active_element(self, name: Optional[str]) -> None:
        self._active_element = name

    def add_box(
        self,
        name: str,
        x: int, y: int, w: int, h: int,
        label_text: Optional[str] = None,
    ) -> EditableRectItem:
        if name in self._element_rects:
            self.remove_box(name, emit=False)
        color = _color_for(name)
        item = EditableRectItem(name, color, x, y, w, h)
        self.addItem(item)
        text = label_text if label_text is not None else name
        label = self.addText(text, QFont("Arial", 8))
        label.setDefaultTextColor(QColor(*color))
        label.setPos(x, max(0, y - 16))
        item.label_item = label
        self._element_rects[name] = item
        self._element_labels[name] = label
        return item

    def remove_box(self, name: str, emit: bool = True) -> None:
        if name not in self._element_rects:
            return
        item = self._element_rects.pop(name)
        label = self._element_labels.pop(name, None)
        if label is not None:
            self.removeItem(label)
        self.removeItem(item)
        if emit:
            self.box_deleted.emit(name)

    def set_hud_bbox(self, x: int, y: int, w: int, h: int) -> None:
        if self._hud_rect_item is not None:
            self.removeItem(self._hud_rect_item)
        # User-drawn HUD constraint: thick bright yellow dash so it's
        # unmistakable in front of bright HUD imagery.
        pen = QPen(QColor(255, 220, 0, 255), 4, Qt.DashLine)
        self._hud_rect_item = self.addRect(QRectF(x, y, w, h), pen)
        # Z-value 7 = above pixmap + feature boxes, just below HUD finder (8).
        self._hud_rect_item.setZValue(7)

    def clear_hud_bbox(self) -> None:
        if self._hud_rect_item is not None:
            self.removeItem(self._hud_rect_item)
            self._hud_rect_item = None

    def set_hud_finder_bbox(
        self, x: int, y: int, w: int, h: int, n_anchors: int,
    ) -> None:
        """Draw the FALLBACK HUD-finder synthesis box (cyan), derived
        from the union of detected feature boxes.

        This is the *fallback* output we render when the real
        color+geometry HUD finder hasn't returned a bbox yet (no calib
        data, dim panel, etc.). Distinct from the user-drawn yellow
        constraint and from the magenta real-finder bbox.
        """
        # Lazy attr init for backward compatibility.
        if not hasattr(self, "_hud_finder_rect_item"):
            self._hud_finder_rect_item = None
            self._hud_finder_label_item = None
        # Defensive: if a previous slide left zombie references after
        # scene.clear(), removeItem on a dead item raises silently.
        # Catch and reset so the new addRect always succeeds.
        try:
            if self._hud_finder_rect_item is not None:
                self.removeItem(self._hud_finder_rect_item)
        except Exception:
            pass
        try:
            if self._hud_finder_label_item is not None:
                self.removeItem(self._hud_finder_label_item)
        except Exception:
            pass
        self._hud_finder_rect_item = None
        self._hud_finder_label_item = None
        # Thick bright cyan solid border = "HUD FINDER thinks the HUD is here"
        # Z-value 8 puts it ABOVE the pixmap (0) and above feature boxes
        # (also 0 by default) — outline-only doesn't obscure inner content.
        pen = QPen(QColor(0, 255, 240, 255), 6, Qt.SolidLine)
        self._hud_finder_rect_item = self.addRect(
            QRectF(x, y, w, h), pen,
        )
        self._hud_finder_rect_item.setZValue(8)
        # Label floating above
        from PySide6.QtWidgets import QGraphicsSimpleTextItem
        label = QGraphicsSimpleTextItem(
            f"◉ HUD FINDER (anchor union, fallback)  ({n_anchors} anchors)"
        )
        label.setBrush(QBrush(QColor(0, 255, 240)))
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        label.setFont(font)
        label.setPos(x + 4, max(0, y - 22))
        label.setZValue(10)
        self.addItem(label)
        self._hud_finder_label_item = label

    def clear_hud_finder_bbox(self) -> None:
        if hasattr(self, "_hud_finder_rect_item") and self._hud_finder_rect_item is not None:
            self.removeItem(self._hud_finder_rect_item)
            self._hud_finder_rect_item = None
        if hasattr(self, "_hud_finder_label_item") and self._hud_finder_label_item is not None:
            self.removeItem(self._hud_finder_label_item)
            self._hud_finder_label_item = None

    def set_hud_color_finder_bbox(
        self,
        x: int, y: int, w: int, h: int,
        confidence: float,
        chrome_pixels: int,
    ) -> None:
        """Draw the REAL color+geometry HUD finder output (magenta).

        This is the upstream Stage-1 output: the panel located purely
        from RGB pixels, before any row detector ran. Distinct from
        the cyan synthesis fallback so the developer can compare
        them side-by-side.
        """
        if not hasattr(self, "_hud_color_rect_item"):
            self._hud_color_rect_item = None
            self._hud_color_label_item = None
        try:
            if self._hud_color_rect_item is not None:
                self.removeItem(self._hud_color_rect_item)
        except Exception:
            pass
        try:
            if self._hud_color_label_item is not None:
                self.removeItem(self._hud_color_label_item)
        except Exception:
            pass
        self._hud_color_rect_item = None
        self._hud_color_label_item = None
        # Bright magenta solid border, slightly thicker than feature
        # boxes but thinner than the cyan synthesis (so when both are
        # drawn they don't overlap visually).
        pen = QPen(QColor(255, 60, 220, 255), 5, Qt.SolidLine)
        self._hud_color_rect_item = self.addRect(QRectF(x, y, w, h), pen)
        # Z-value 9 — above the cyan synthesis (8) and feature boxes.
        self._hud_color_rect_item.setZValue(9)
        from PySide6.QtWidgets import QGraphicsSimpleTextItem
        label = QGraphicsSimpleTextItem(
            f"◉ HUD FINDER (color+geom)  conf={confidence:.2f}  "
            f"px={chrome_pixels}"
        )
        label.setBrush(QBrush(QColor(255, 60, 220)))
        font = QFont()
        font.setBold(True)
        font.setPointSize(13)
        label.setFont(font)
        # Place it just below the top edge so it doesn't collide with
        # the cyan synthesis label which sits above.
        label.setPos(x + 4, y + 2)
        label.setZValue(11)
        self.addItem(label)
        self._hud_color_label_item = label

    def clear_hud_color_finder_bbox(self) -> None:
        if hasattr(self, "_hud_color_rect_item") and self._hud_color_rect_item is not None:
            self.removeItem(self._hud_color_rect_item)
            self._hud_color_rect_item = None
        if hasattr(self, "_hud_color_label_item") and self._hud_color_label_item is not None:
            self.removeItem(self._hud_color_label_item)
            self._hud_color_label_item = None

    def selected_feature(self) -> Optional[str]:
        for name, item in self._element_rects.items():
            if item.isSelected():
                return name
        return None

    # Mouse handling: shift+drag = HUD bbox; plain drag with active
    # element = draw new box; otherwise pass through (rect items
    # handle their own selection / resize).
    def mousePressEvent(self, event: QMouseEvent):
        # If the click is on an existing rect item, defer to it.
        item = self.itemAt(event.scenePos(), self.views()[0].transform()
                           if self.views() else None)  # type: ignore
        # Defer if the item under the cursor handles drags itself.
        if isinstance(item, EditableRectItem):
            super().mousePressEvent(event)
            return
        if event.button() == Qt.LeftButton:
            shift = bool(event.modifiers() & Qt.ShiftModifier)
            # Plain drag with armed element = draw that element.
            # Anything else (no element, OR shift held) = draw HUD bbox.
            # HUD-first is the primary workflow: drag → set HUD →
            # detectors auto-populate inside it.
            if self._active_element and not shift:
                self._dragging = True
                self._drag_is_hud = False
                self._drag_start = event.scenePos()
                color = _color_for(self._active_element)
                pen = QPen(QColor(*color), 2, Qt.DashLine)
                self._drag_rect_item = self.addRect(
                    QRectF(self._drag_start, self._drag_start),
                    pen, QBrush(Qt.NoBrush),
                )
                event.accept()
                return
            # HUD region drag (default when no element is selected,
            # or always when shift is held)
            self._dragging = True
            self._drag_is_hud = True
            self._drag_start = event.scenePos()
            pen = QPen(QColor(255, 255, 255, 200), 2, Qt.DashDotLine)
            self._drag_rect_item = self.addRect(
                QRectF(self._drag_start, self._drag_start),
                pen, QBrush(Qt.NoBrush),
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging and self._drag_rect_item is not None:
            r = QRectF(self._drag_start, event.scenePos()).normalized()
            self._drag_rect_item.setRect(r)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
            r = self._drag_rect_item.rect() if self._drag_rect_item else QRectF()
            if self._drag_rect_item is not None:
                self.removeItem(self._drag_rect_item)
                self._drag_rect_item = None
            if self._drag_is_hud:
                if r.width() >= 8 and r.height() >= 8:
                    self.hud_bbox_drawn.emit(r)
            else:
                if r.width() >= 4 and r.height() >= 4 and self._active_element:
                    self.box_drawn.emit(self._active_element, r)
            event.accept()
            return
        super().mouseReleaseEvent(event)


# ────────────────────────────────────────────────────────────────────
# Main window
# ────────────────────────────────────────────────────────────────────


class AutoAnnotatorWindow(QMainWindow):
    def __init__(
        self,
        source_dir: Optional[Path] = None,
        region: str = "region1",
    ):
        super().__init__()
        # Resolve initial region. If unknown, fall back to region1.
        self._region: str = region if region in REGION_CONFIG else "region1"
        cfg = REGION_CONFIG[self._region]
        # Per-region runtime state — swapped wholesale on region change.
        self._elements: list[tuple[str, tuple[int, int, int]]] = cfg["elements"]
        self._element_names: list[str] = [n for n, _ in self._elements]
        self._detectors: list[tuple[str, Callable[..., dict[str, dict]]]] = cfg["detectors"]
        # Source dir: explicit > region default.
        self._source_dir: Path = (
            source_dir if source_dir is not None else cfg["default_source"]
        )
        self.setWindowTitle(self._make_title())
        self.resize(1500, 900)
        self._current_path: Optional[Path] = None
        self._current_image: Optional[Image.Image] = None
        # User-visible boxes for the current image: feature -> [x,y,w,h]
        self._current_boxes: dict[str, list[int]] = {}
        # Detector proposals BEFORE any user edit. Used to compute
        # delta_px at save time. feature -> {x,y,w,h, detector, score}
        self._detector_proposals: dict[str, dict] = {}
        # Track which features were touched by the user (vs accepted
        # straight from the detector).
        self._user_corrected: set[str] = set()
        self._active_element: Optional[str] = None
        self._hud_bbox: Optional[tuple[int, int, int, int]] = None
        # Most recent color+geometry HUD finder result (or None if it
        # hasn't run / failed). Stored so we can render the magenta
        # bbox even when the user later overrides with their own
        # yellow constraint, and so we can compare the two boxes
        # in development.
        self._hud_color_finder_result: Optional[dict] = None
        # Was the current ``_hud_bbox`` derived from the color finder
        # (True) or set explicitly by the user (False)?  We track this
        # so that the user's manual draw always overrides the auto
        # output even if a re-run would re-detect the panel.
        self._hud_bbox_from_finder: bool = False
        # Visualization state
        self._show_predictions = False
        self._prediction_items: list = []  # QGraphicsRectItem ghosts
        self._world_model: Optional[dict] = None
        self._load_world_model()
        # Element-key shortcuts; rebuilt when region changes.
        self._element_shortcuts: list = []
        # Container the element buttons live in — rebuilt on region
        # change. Stored so we can clear + repopulate it.
        self._element_buttons_container: Optional[QVBoxLayout] = None
        # Suppress region-change auto-save during initial construction.
        self._region_change_in_progress: bool = False

        # ── Layout ──
        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)

        # Left: region selector + file list + nav
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("<b>Region</b>"))
        self._region_combo = QComboBox()
        for key, region_cfg in REGION_CONFIG.items():
            self._region_combo.addItem(region_cfg["label"], key)
        # Set the current region as the active item.
        idx = self._region_combo.findData(self._region)
        if idx >= 0:
            self._region_combo.setCurrentIndex(idx)
        # Connect AFTER initial selection so the slot doesn't fire
        # during construction.
        self._region_combo.currentIndexChanged.connect(
            self._on_region_combo_changed,
        )
        left_panel.addWidget(self._region_combo)
        left_panel.addSpacing(6)
        left_panel.addWidget(QLabel("Captures (annotated → ✓):"))
        self._file_list = QListWidget()
        self._file_list.setMinimumWidth(280)
        self._file_list.currentRowChanged.connect(self._on_current_row_changed)
        left_panel.addWidget(self._file_list, 1)
        nav_row = QHBoxLayout()
        prev_btn = QPushButton("◀ Prev")
        prev_btn.clicked.connect(self._prev_image)
        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(self._next_image)
        nav_row.addWidget(prev_btn)
        nav_row.addWidget(next_btn)
        left_panel.addLayout(nav_row)
        h.addLayout(left_panel)

        # Center: image canvas
        self._scene = AutoAnnotationScene()
        self._scene.box_drawn.connect(self._on_box_drawn)
        self._scene.box_edited.connect(self._on_box_edited)
        self._scene.box_deleted.connect(self._on_box_deleted)
        self._scene.hud_bbox_drawn.connect(self._on_hud_bbox_drawn)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.SmoothPixmapTransform)
        self._view.setMinimumWidth(700)
        h.addWidget(self._view, 1)

        # Right: actions
        right_panel = QVBoxLayout()

        # Auto-annotation buttons
        right_panel.addWidget(QLabel(
            "<b>1. Drag a rectangle around the HUD region.</b><br>"
            "<i>Anchors fire inside it automatically — you should "
            "not need to draw individual rows.</i>"
        ))
        rerun_btn = QPushButton("⟳ Re-run detectors (Ctrl+R)")
        rerun_btn.setStyleSheet(
            "QPushButton { background-color: #46a; color: white; "
            "padding: 6px; }"
        )
        rerun_btn.clicked.connect(self._rerun_detectors)
        right_panel.addWidget(rerun_btn)

        clear_hud_btn = QPushButton("Clear HUD constraint")
        clear_hud_btn.setToolTip(
            "Reset the shift+drag HUD region. Detectors will then "
            "search the whole image again."
        )
        clear_hud_btn.clicked.connect(self._clear_hud_bbox)
        right_panel.addWidget(clear_hud_btn)

        accept_btn = QPushButton("✓ Accept all detector proposals")
        accept_btn.setStyleSheet(
            "QPushButton { background-color: #2a8; color: white; "
            "font-weight: bold; padding: 6px; }"
        )
        accept_btn.clicked.connect(self._accept_all)
        right_panel.addWidget(accept_btn)

        # Visualization: world-model predictions overlay
        self._predictions_btn = QPushButton(
            "👁 Show world-model predictions (P)"
        )
        self._predictions_btn.setCheckable(True)
        self._predictions_btn.setToolTip(
            "Overlay dashed ghost rectangles showing where each "
            "feature SHOULD be per the proportional model. "
            "Lets you visually verify the proportions are consistent "
            "with what the detectors actually find."
        )
        self._predictions_btn.setStyleSheet(
            "QPushButton { background-color: #555; color: white; "
            "padding: 6px; }"
            "QPushButton:checked { background-color: #b80; }"
        )
        self._predictions_btn.clicked.connect(self._toggle_predictions)
        right_panel.addWidget(self._predictions_btn)

        right_panel.addSpacing(10)
        right_panel.addWidget(QLabel(
            "<b>2. Manual fallback</b> (only if a detector misses):"
        ))
        # The element buttons live in their own VBox layout so we can
        # tear them down + rebuild them when the region changes
        # (region1 has 8, region2 has 3, etc.).
        self._element_buttons: dict[str, QPushButton] = {}
        self._element_buttons_container = QVBoxLayout()
        right_panel.addLayout(self._element_buttons_container)
        self._build_element_buttons()
        right_panel.addSpacing(8)

        del_btn = QPushButton("Delete selected box (Del)")
        del_btn.clicked.connect(self._delete_selected_box)
        right_panel.addWidget(del_btn)

        clear_btn = QPushButton("Clear ALL boxes on this image")
        clear_btn.clicked.connect(self._clear_all_boxes)
        right_panel.addWidget(clear_btn)

        save_btn = QPushButton("💾 Save (Ctrl+S)")
        save_btn.setStyleSheet(
            "QPushButton { background-color: #888; color: white; "
            "padding: 6px; }"
        )
        save_btn.clicked.connect(self._save_current_boxes)
        right_panel.addWidget(save_btn)

        skip_btn = QPushButton("⤼ Skip slide (Ctrl+K)")
        skip_btn.setToolTip(
            "Mark this capture as not worth training on. Writes a "
            ".skip sidecar and advances. Removes any existing "
            ".boxes.json. Reversible: just save boxes again to "
            "un-skip."
        )
        skip_btn.setStyleSheet(
            "QPushButton { background-color: #b46; color: white; "
            "padding: 6px; }"
        )
        skip_btn.clicked.connect(self._skip_current_image)
        right_panel.addWidget(skip_btn)
        right_panel.addStretch(1)

        # Detector status panel
        right_panel.addWidget(QLabel("<b>Detector status</b> (current image):"))
        self._detector_status_label = QLabel("(no image loaded)")
        self._detector_status_label.setWordWrap(True)
        self._detector_status_label.setStyleSheet(
            "QLabel { background: #222; color: #ccc; "
            "padding: 6px; font-family: monospace; font-size: 10px; }"
        )
        right_panel.addWidget(self._detector_status_label)
        # Help blurb
        tip = QLabel(
            "<i><b>Workflow:</b> drag → mark HUD → 7 boxes appear → "
            "drag corners to fix → press 7 + drag for outcome row → "
            "Ctrl+S to save & advance.<br><br>"
            "Drag in empty space = HUD region (re-draw to redo).<br>"
            "Click any box to select; drag a corner to resize, "
            "drag inside to move.</i>"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #888; padding: 2px;")
        right_panel.addWidget(tip)

        h.addLayout(right_panel)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)

        # ── Shortcuts ──
        # Element-key shortcuts (1..N) are built per-region inside
        # _build_element_shortcuts so they can be rebuilt when the
        # region switches.
        self._build_element_shortcuts()
        QShortcut(QKeySequence(Qt.Key_Delete), self,
                  activated=self._delete_selected_box)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self._prev_image)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self._next_image)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_current_boxes)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._rerun_detectors)
        QShortcut(QKeySequence("Ctrl+A"), self, activated=self._accept_all)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self._skip_current_image)
        QShortcut(QKeySequence("P"), self, activated=self._toggle_predictions)

        # Populate file list
        self._scan_folder()
        if self._file_list.count() > 0:
            self._file_list.setCurrentRow(0)

    # ──────────────────────────────────────────
    # Region helpers
    # ──────────────────────────────────────────

    def _make_title(self) -> str:
        cfg = REGION_CONFIG.get(self._region, {})
        label = cfg.get("label", self._region)
        return f"SC HUD Auto-Annotator — {label}"

    def _build_element_buttons(self) -> None:
        """Tear down + repopulate the element-button list to match the
        current region's elements. Called from __init__ and from
        _set_region.
        """
        # Drop existing buttons + their layout entries.
        if self._element_buttons_container is None:
            return
        for btn in list(self._element_buttons.values()):
            self._element_buttons_container.removeWidget(btn)
            btn.deleteLater()
        self._element_buttons.clear()
        # Rebuild from self._elements (region-specific).
        for i, (name, color) in enumerate(self._elements, start=1):
            btn = QPushButton(f"{i}. {name}")
            btn.setCheckable(True)
            btn.setStyleSheet(
                "QPushButton { text-align: left; padding: 4px; }"
                f"QPushButton:checked {{ background-color: rgb{color}; "
                "color: black; font-weight: bold; }}"
            )
            btn.clicked.connect(
                lambda _checked, n=name: self._select_element(n)
            )
            self._element_buttons[name] = btn
            self._element_buttons_container.addWidget(btn)

    def _build_element_shortcuts(self) -> None:
        """Build (or rebuild) numeric shortcut keys 1..N for the
        current region's elements. Old shortcuts are explicitly
        deleted so two region's worth of shortcuts don't pile up.
        """
        for sc in self._element_shortcuts:
            try:
                sc.setEnabled(False)
                sc.deleteLater()
            except Exception:
                pass
        self._element_shortcuts.clear()
        for i, name in enumerate(self._element_names, start=1):
            sc = QShortcut(QKeySequence(str(i)), self)
            sc.activated.connect(lambda n=name: self._select_element(n))
            self._element_shortcuts.append(sc)

    def _on_region_combo_changed(self, idx: int) -> None:
        """User picked a different region from the combobox."""
        if idx < 0:
            return
        new_region = self._region_combo.itemData(idx)
        if not new_region or new_region == self._region:
            return
        self._set_region(new_region)

    def _set_region(self, region: str) -> None:
        """Switch the annotator to a new region.

        Saves the current image's boxes first, then swaps elements +
        detectors + source dir, rebuilds the element buttons +
        shortcuts, and reloads the file list.
        """
        if region not in REGION_CONFIG:
            log.warning("unknown region %r; ignoring", region)
            return
        if region == self._region:
            return
        # Persist whatever's on screen before tearing down state.
        if (
            self._current_path is not None
            and self._current_boxes
            and not self._region_change_in_progress
        ):
            try:
                self._save_current_boxes(silent=True)
            except Exception as exc:
                log.warning("save during region change failed: %s", exc)
        self._region_change_in_progress = True
        try:
            cfg = REGION_CONFIG[region]
            self._region = region
            self._elements = cfg["elements"]
            self._element_names = [n for n, _ in self._elements]
            self._detectors = cfg["detectors"]
            self._source_dir = cfg["default_source"]
            # Rebuild UI bits that depend on the element list.
            self._build_element_buttons()
            self._build_element_shortcuts()
            self.setWindowTitle(self._make_title())
            # Drop in-flight per-image state — features may differ.
            self._current_path = None
            self._current_image = None
            self._current_boxes = {}
            self._detector_proposals = {}
            self._user_corrected = set()
            self._active_element = None
            self._hud_bbox = None
            self._hud_color_finder_result = None
            self._hud_bbox_from_finder = False
            self._scene.clear()
            self._scene._element_rects.clear()
            self._scene._element_labels.clear()
            self._prediction_items = []
            # Sync combobox if this was triggered programmatically.
            idx = self._region_combo.findData(region)
            if idx >= 0 and self._region_combo.currentIndex() != idx:
                blocker = self._region_combo.blockSignals(True)
                try:
                    self._region_combo.setCurrentIndex(idx)
                finally:
                    self._region_combo.blockSignals(blocker)
            # Reload file list for the new source dir.
            self._scan_folder()
            if self._file_list.count() > 0:
                self._file_list.setCurrentRow(0)
            self._update_detector_status()
            self._update_status()
        finally:
            self._region_change_in_progress = False

    # ──────────────────────────────────────────
    # File handling
    # ──────────────────────────────────────────

    def _scan_folder(self) -> None:
        self._file_list.clear()
        if not self._source_dir.is_dir():
            QMessageBox.critical(
                self, "Folder not found",
                f"Source folder does not exist:\n{self._source_dir}",
            )
            return
        pngs = sorted(self._source_dir.glob("*.png"))
        n_annotated = 0
        n_skipped = 0
        for p in pngs:
            annotated = self._boxes_path(p).is_file()
            skipped = self._skip_path(p).is_file()
            if skipped:
                prefix = "🚫 "
                n_skipped += 1
            elif annotated:
                prefix = "✓ "
                n_annotated += 1
            else:
                prefix = "   "
            item = QListWidgetItem(prefix + p.name)
            item.setData(Qt.UserRole, str(p))
            self._file_list.addItem(item)
        self._status.showMessage(
            f"Loaded {len(pngs)} captures from {self._source_dir.name} "
            f"({n_annotated} labeled, {n_skipped} skipped, "
            f"{len(pngs) - n_annotated - n_skipped} pending)",
        )

    def _boxes_path(self, png_path: Path) -> Path:
        return png_path.with_suffix(".boxes.json")

    def _skip_path(self, png_path: Path) -> Path:
        """Sidecar marking the capture as unworthy of training data."""
        return png_path.with_suffix(".skip")

    # ──────────────────────────────────────────
    # World-model overlay (visualize what the tracker hypothesis
    # predicts, given scan_results as the anchor).
    # ──────────────────────────────────────────

    def _load_world_model(self) -> None:
        """Load proportional constants from hud_tracker/world_model.json."""
        import os as _os
        candidates = [
            _os.path.expandvars(
                r"%LOCALAPPDATA%\SC_Toolbox\current\tools\Mining_Signals"
                r"\hud_tracker\world_model.json"
            ),
            str(Path(__file__).resolve().parent.parent
                / "hud_tracker" / "world_model.json"),
        ]
        for p in candidates:
            if _os.path.isfile(p):
                try:
                    self._world_model = json.loads(Path(p).read_text())
                    log.info("loaded world_model from %s", p)
                    return
                except Exception as exc:
                    log.warning("failed to parse %s: %s", p, exc)
        self._world_model = None
        log.info("no world_model.json found; predictions overlay disabled")

    def _predicted_positions(self) -> dict[str, tuple[int, int, int, int]]:
        """Compute where each feature SHOULD be per world_model + the
        current scan_results bbox. Returns {feature: (x, y, w, h)}."""
        if self._world_model is None:
            return {}
        sr = self._current_boxes.get("scan_results")
        if sr is None:
            return {}
        sx, sy, sw, sh = sr
        feats = self._world_model.get("features", {})
        out: dict[str, tuple[int, int, int, int]] = {}
        for name, data in feats.items():
            if name == "scan_results":
                continue  # the anchor itself
            try:
                xf = data["x_frac"]["mean"]
                yf = data["y_frac"]["mean"]
                wf = data["w_frac"]["mean"]
                hf = data["h_frac"]["mean"]
            except (KeyError, TypeError):
                continue
            x = int(round(sx + xf * sw))
            y = int(round(sy + yf * sh))
            w = int(round(wf * sw))
            h = int(round(hf * sh))
            out[name] = (x, y, w, h)
        return out

    def _render_predictions(self) -> None:
        """Draw dashed ghost rectangles where world_model predicts each
        feature should be. Updates whenever scan_results changes."""
        self._clear_predictions()
        if not self._show_predictions:
            return
        predictions = self._predicted_positions()
        if not predictions:
            return
        for name, (x, y, w, h) in predictions.items():
            color = _color_for(name)
            pen = QPen(QColor(color[0], color[1], color[2], 140), 1, Qt.DashLine)
            rect = self._scene.addRect(
                QRectF(x, y, w, h), pen, QBrush(Qt.NoBrush),
            )
            rect.setZValue(-1)  # behind detected/user boxes
            self._prediction_items.append(rect)

    def _clear_predictions(self) -> None:
        for item in self._prediction_items:
            try:
                self._scene.removeItem(item)
            except Exception:
                pass
        self._prediction_items.clear()

    def _render_hud_finder(self) -> None:
        """Synthesize a single HUD bbox from detected feature boxes
        and draw it as the HUD-finder output. This is what a real
        HUD tracker would emit: 'I think the HUD is here, derived
        from these N anchors.'"""
        # Synthesis: union bbox of all current feature boxes, padded
        # slightly so the border doesn't clip the inner features.
        if not self._current_boxes:
            self._scene.clear_hud_finder_bbox()
            return
        xs = [b[0] for b in self._current_boxes.values()]
        ys = [b[1] for b in self._current_boxes.values()]
        xe = [b[0] + b[2] for b in self._current_boxes.values()]
        ye = [b[1] + b[3] for b in self._current_boxes.values()]
        pad = 6
        x0 = max(0, min(xs) - pad)
        y0 = max(0, min(ys) - pad)
        x1 = max(xe) + pad
        y1 = max(ye) + pad
        # Clamp to image bounds if we have an image.
        if self._current_image is not None:
            iw, ih = self._current_image.size
            x1 = min(x1, iw - 1)
            y1 = min(y1, ih - 1)
        n_anchors = len(self._current_boxes)
        self._scene.set_hud_finder_bbox(
            x0, y0, x1 - x0, y1 - y0, n_anchors=n_anchors,
        )

    def _toggle_predictions(self) -> None:
        self._show_predictions = not self._show_predictions
        self._render_predictions()
        if self._show_predictions:
            n = len(self._prediction_items)
            if n == 0:
                self._status.showMessage(
                    "World-model overlay ON — but scan_results not "
                    "detected yet, so nothing to predict from.",
                    4000,
                )
            else:
                self._status.showMessage(
                    f"World-model overlay ON — {n} ghost rectangles "
                    "show where each feature SHOULD be per the "
                    "proportional model.",
                    4000,
                )
        else:
            self._status.showMessage("World-model overlay OFF.", 2000)

    def _on_current_row_changed(self, row: int) -> None:
        if row < 0:
            return
        # Save the previous image's boxes before switching.
        if self._current_path is not None and self._current_boxes:
            self._save_current_boxes(silent=True)
        item = self._file_list.item(row)
        if item is None:
            return
        path = Path(item.data(Qt.UserRole))
        self._load_image(path)

    def _load_image(self, path: Path) -> None:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            QMessageBox.warning(self, "Open failed", str(exc))
            return
        self._current_path = path
        self._current_image = img
        self._current_boxes = {}
        self._detector_proposals = {}
        self._user_corrected = set()
        self._hud_bbox = None
        self._hud_color_finder_result = None
        self._hud_bbox_from_finder = False
        # The scene's clear() inside set_image() drops all rect items,
        # so our prediction-item references are now stale — drop them.
        self._prediction_items = []
        self._scene.set_image(img)
        self._scene.clear_hud_bbox()
        self._scene.clear_hud_color_finder_bbox()

        # If a sidecar already exists, load it (boxes + meta + previous
        # HUD bbox if present) so re-edit picks up where the previous
        # session stopped.
        existing_meta: dict[str, dict] = {}
        existing_hud_bbox: Optional[tuple[int, int, int, int]] = None
        bp = self._boxes_path(path)
        had_sidecar = bp.is_file()
        if had_sidecar:
            try:
                data = json.loads(bp.read_text())
                for name, box in data.get("boxes", {}).items():
                    # Filter incoming boxes to the CURRENT region's
                    # element set so e.g. a region2 sidecar opened in
                    # region1 mode doesn't try to render unknown
                    # features.
                    if name not in self._element_names:
                        continue
                    self._current_boxes[name] = [
                        int(box["x"]), int(box["y"]),
                        int(box["w"]), int(box["h"]),
                    ]
                existing_meta = data.get("detector_meta", {}) or {}
                hud = data.get("hud_bbox")
                if hud:
                    existing_hud_bbox = (
                        int(hud["x"]), int(hud["y"]),
                        int(hud["w"]), int(hud["h"]),
                    )
            except Exception as exc:
                log.warning("failed to load existing %s: %s", bp, exc)

        # Stage 1 — pixel-only HUD finder. Always runs on image load,
        # *before* row detectors. The magenta bbox is the system's
        # actual HUD-finder output (color+geometry, no row deps).
        finder_result = find_hud_panel_color(img)
        self._hud_color_finder_result = finder_result
        if finder_result is not None:
            fb = finder_result["bbox"]
            self._scene.set_hud_color_finder_bbox(
                int(fb[0]), int(fb[1]), int(fb[2]), int(fb[3]),
                confidence=float(finder_result.get("confidence", 0.0)),
                chrome_pixels=int(finder_result.get("n_chrome_pixels", 0)),
            )

        # HUD-first workflow: only run detectors when we have a HUD
        # bbox. Three sources, in priority order:
        #   1. existing user bbox in the .boxes.json sidecar
        #   2. fresh color+geometry HUD finder result
        #   3. nothing — wait for user drag
        # Detectors should operate "exclusively inside the parameters
        # of the HUD" — that's both the user's preferred workflow and
        # a useful test of anchor accuracy when constrained.
        if existing_hud_bbox is not None:
            self._hud_bbox = existing_hud_bbox
            self._hud_bbox_from_finder = False
            self._scene.set_hud_bbox(*existing_hud_bbox)
            proposals = run_all_detectors(
                img, hud_bbox=existing_hud_bbox,
                detectors=self._detectors,
            )
        elif finder_result is not None:
            fb = finder_result["bbox"]
            auto_bbox = (int(fb[0]), int(fb[1]), int(fb[2]), int(fb[3]))
            self._hud_bbox = auto_bbox
            self._hud_bbox_from_finder = True
            self._scene.set_hud_bbox(*auto_bbox)
            proposals = run_all_detectors(
                img, hud_bbox=auto_bbox,
                detectors=self._detectors,
            )
        else:
            # No HUD bbox yet — defer detection. The user drags a
            # rectangle to mark the HUD; _on_hud_bbox_drawn fires
            # detectors then.
            proposals = {}
        self._detector_proposals = proposals

        # Replay existing meta — if a feature was previously stored and
        # the detector also fired this run, prefer the saved bbox in
        # the UI (user already adjusted it). user_corrected stays True
        # because we're loading a previously edited file.
        for feature, meta in existing_meta.items():
            if feature not in self._element_names:
                continue
            if meta.get("user_corrected"):
                self._user_corrected.add(feature)
            # If the saved meta has a det name + score and we don't
            # have a fresh detector proposal, synthesize one from the
            # saved data so the delta_px math still works.
            if feature not in self._detector_proposals and meta.get("detector"):
                # Reconstruct the original proposal: saved_bbox - delta
                saved = self._current_boxes.get(feature)
                delta = meta.get("delta_px", [0, 0, 0, 0])
                if saved is not None and len(delta) == 4:
                    self._detector_proposals[feature] = {
                        "x": int(saved[0]) - int(delta[0]),
                        "y": int(saved[1]) - int(delta[1]),
                        "w": int(saved[2]) - int(delta[2]),
                        "h": int(saved[3]) - int(delta[3]),
                        "score": meta.get("score"),
                        "detector": meta.get("detector"),
                    }

        # Decide which boxes to display: saved > detector.
        for feature, prop in proposals.items():
            if feature not in self._current_boxes:
                self._current_boxes[feature] = [
                    int(prop["x"]), int(prop["y"]),
                    int(prop["w"]), int(prop["h"]),
                ]

        # Render every box.
        for feature, box in self._current_boxes.items():
            self._add_box_to_scene(feature, box)

        self._update_detector_status()
        self._update_status()
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def _add_box_to_scene(self, feature: str, box: list[int]) -> None:
        prop = self._detector_proposals.get(feature)
        if prop is not None:
            score_str = (
                f" — score {prop['score']:.2f}"
                if prop.get("score") is not None else ""
            )
            label_text = f"{feature} — {prop.get('detector','?')}{score_str}"
        else:
            label_text = f"{feature} — manual"
        self._scene.add_box(
            feature, box[0], box[1], box[2], box[3],
            label_text=label_text,
        )

    def _next_image(self) -> None:
        row = self._file_list.currentRow()
        if row < self._file_list.count() - 1:
            self._file_list.setCurrentRow(row + 1)

    def _prev_image(self) -> None:
        row = self._file_list.currentRow()
        if row > 0:
            self._file_list.setCurrentRow(row - 1)

    # ──────────────────────────────────────────
    # Box manipulation
    # ──────────────────────────────────────────

    def _select_element(self, name: str) -> None:
        # Toggle: clicking the active button again disarms.
        if self._active_element == name:
            self._active_element = None
            self._element_buttons[name].setChecked(False)
            self._scene.set_active_element(None)
        else:
            self._active_element = name
            for n, btn in self._element_buttons.items():
                btn.setChecked(n == name)
            self._scene.set_active_element(name)
        self._update_status()

    def _on_box_drawn(self, name: str, rect: QRectF) -> None:
        box = [int(rect.x()), int(rect.y()),
               int(rect.width()), int(rect.height())]
        self._current_boxes[name] = box
        self._user_corrected.add(name)
        self._add_box_to_scene(name, box)
        self._save_current_boxes(silent=True)
        self._update_status()

    def _on_box_edited(self, name: str, rect: QRectF) -> None:
        box = [int(rect.x()), int(rect.y()),
               int(rect.width()), int(rect.height())]
        self._current_boxes[name] = box
        # Only flag as corrected if it actually moved from the
        # detector proposal (or there was no proposal).
        prop = self._detector_proposals.get(name)
        if prop is None or [
            int(prop["x"]), int(prop["y"]),
            int(prop["w"]), int(prop["h"]),
        ] != box:
            self._user_corrected.add(name)
        self._save_current_boxes(silent=True)
        self._update_status()

    def _on_box_deleted(self, name: str) -> None:
        self._current_boxes.pop(name, None)
        self._user_corrected.add(name)
        self._save_current_boxes(silent=True)
        self._update_status()

    def _delete_selected_box(self) -> None:
        sel = self._scene.selected_feature()
        if sel is not None:
            self._scene.remove_box(sel)
            return
        # Fallback: delete the active-element box.
        if self._active_element and self._active_element in self._current_boxes:
            self._scene.remove_box(self._active_element)

    def _clear_all_boxes(self) -> None:
        if self._current_path is None:
            return
        if QMessageBox.question(
            self, "Clear all", "Delete all boxes on this image?",
        ) != QMessageBox.Yes:
            return
        for name in list(self._current_boxes.keys()):
            self._scene.remove_box(name, emit=False)
            self._user_corrected.add(name)
        self._current_boxes.clear()
        self._save_current_boxes(silent=True)
        self._update_status()

    def _accept_all(self) -> None:
        """Take every detector proposal as-is (no user_corrected flag)
        and save."""
        if self._current_image is None:
            return
        # Reset to detector proposals (drop any in-flight edits).
        for name in list(self._current_boxes.keys()):
            self._scene.remove_box(name, emit=False)
        self._current_boxes.clear()
        for feature, prop in self._detector_proposals.items():
            box = [int(prop["x"]), int(prop["y"]),
                   int(prop["w"]), int(prop["h"])]
            self._current_boxes[feature] = box
            self._add_box_to_scene(feature, box)
        # Accept-all means "the detectors got it right" — clear the
        # corrected flag for every detector-supplied feature.
        for feature in self._detector_proposals:
            self._user_corrected.discard(feature)
        self._save_current_boxes()
        self._update_status()

    # ──────────────────────────────────────────
    # HUD bbox + re-run
    # ──────────────────────────────────────────

    def _on_hud_bbox_drawn(self, rect: QRectF) -> None:
        self._hud_bbox = (
            int(rect.x()), int(rect.y()),
            int(rect.width()), int(rect.height()),
        )
        # User explicitly drew this — take precedence over any
        # color-finder auto-detect for this image.
        self._hud_bbox_from_finder = False
        self._scene.set_hud_bbox(*self._hud_bbox)
        # HUD-first workflow: as soon as the HUD is drawn, run all
        # detectors INSIDE it. The user shouldn't have to press another
        # button — the HUD bbox IS the trigger.
        self._rerun_detectors()
        n = len(self._detector_proposals)
        self._status.showMessage(
            f"HUD bbox set; {n} of {len(self._element_names)} features "
            "auto-populated. Drag corners to correct any wrong ones, "
            "then Ctrl+S to save.",
            6000,
        )

    def _clear_hud_bbox(self) -> None:
        self._hud_bbox = None
        self._hud_bbox_from_finder = False
        self._scene.clear_hud_bbox()
        self._status.showMessage("HUD bbox cleared. Detectors search the whole image again.", 3000)

    def _skip_current_image(self) -> None:
        """Mark current capture as not-worth-training and advance.

        Writes a ``.skip`` sidecar so training-data scrapers know to
        exclude this PNG. Also removes any existing ``.boxes.json`` so
        we don't keep partial labels on a discarded capture. Reversible
        — saving boxes again later clears the skip marker.
        """
        if self._current_path is None:
            return
        path = self._current_path
        skip_path = self._skip_path(path)
        try:
            skip_path.write_text(json.dumps({
                "image": path.name,
                "skipped": True,
            }, indent=2))
        except Exception as exc:
            QMessageBox.warning(self, "Skip failed", str(exc))
            return
        # Drop any existing boxes — we're saying "don't train on this"
        bp = self._boxes_path(path)
        if bp.is_file():
            try:
                bp.unlink()
            except Exception as exc:
                log.warning("could not remove %s: %s", bp, exc)
        # Clear in-memory state so the auto-save on row-change doesn't
        # re-write a boxes.json behind us.
        for name in list(self._current_boxes.keys()):
            self._scene.remove_box(name, emit=False)
        self._current_boxes = {}
        self._user_corrected = set()
        self._detector_proposals = {}
        # Update the file-list label for this row.
        for i in range(self._file_list.count()):
            it = self._file_list.item(i)
            if it.data(Qt.UserRole) == str(path):
                it.setText("🚫 " + path.name)
                break
        self._status.showMessage(
            f"Skipped {path.name} (won't be used for training)",
            3000,
        )
        self._next_image()

    def _rerun_detectors(self) -> None:
        if self._current_image is None:
            return
        proposals = run_all_detectors(
            self._current_image, hud_bbox=self._hud_bbox,
            detectors=self._detectors,
        )
        # Clear all boxes and replace with new proposals (preserve
        # any user-drawn boxes that weren't matched by a detector).
        user_only = {
            n: box for n, box in self._current_boxes.items()
            if n in self._user_corrected
            and n not in proposals
        }
        for name in list(self._current_boxes.keys()):
            self._scene.remove_box(name, emit=False)
        self._current_boxes.clear()
        self._user_corrected.clear()
        self._detector_proposals = proposals
        for feature, prop in proposals.items():
            box = [int(prop["x"]), int(prop["y"]),
                   int(prop["w"]), int(prop["h"])]
            self._current_boxes[feature] = box
            self._add_box_to_scene(feature, box)
        # Restore user-only boxes (no detector for these).
        for name, box in user_only.items():
            self._current_boxes[name] = box
            self._user_corrected.add(name)
            self._add_box_to_scene(name, box)
        self._update_detector_status()
        self._update_status()

    # ──────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────

    def _save_current_boxes(self, silent: bool = False) -> None:
        if self._current_path is None:
            return
        bp = self._boxes_path(self._current_path)
        boxes_payload = {
            name: {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}
            for name, b in self._current_boxes.items()
        }
        # Build detector_meta block. For every feature the user has
        # SAVED a box for (whether detector-proposed or user-drawn),
        # record:
        #   - which detector originally proposed (or null)
        #   - that detector's confidence score (or null)
        #   - whether the user changed it from the detector output
        #   - delta_px between detector output and user-saved bbox
        detector_meta: dict[str, dict] = {}
        for name, b in self._current_boxes.items():
            prop = self._detector_proposals.get(name)
            if prop is None:
                detector_meta[name] = {
                    "detector": None,
                    "score": None,
                    "user_corrected": True,
                    "delta_px": [0, 0, 0, 0],
                }
                continue
            delta = [
                int(b[0]) - int(prop["x"]),
                int(b[1]) - int(prop["y"]),
                int(b[2]) - int(prop["w"]),
                int(b[3]) - int(prop["h"]),
            ]
            corrected = (
                name in self._user_corrected
                or any(d != 0 for d in delta)
            )
            score = prop.get("score")
            detector_meta[name] = {
                "detector": prop.get("detector"),
                "score": float(score) if score is not None else None,
                "user_corrected": bool(corrected),
                "delta_px": delta,
            }

        try:
            payload = {
                "image": self._current_path.name,
                "boxes": boxes_payload,
                "detector_meta": detector_meta,
            }
            if self._hud_bbox is not None:
                payload["hud_bbox"] = {
                    "x": self._hud_bbox[0],
                    "y": self._hud_bbox[1],
                    "w": self._hud_bbox[2],
                    "h": self._hud_bbox[3],
                }
            bp.write_text(json.dumps(payload, indent=2))
            # Saving boxes implicitly un-skips the slide.
            sp = self._skip_path(self._current_path)
            if sp.is_file():
                try:
                    sp.unlink()
                except Exception as exc:
                    log.warning("could not clear skip marker %s: %s", sp, exc)
            if not silent:
                self._status.showMessage(
                    f"Saved {len(boxes_payload)} boxes → {bp.name}", 3000,
                )
        except Exception as exc:
            log.warning("save boxes failed: %s", exc)
            if not silent:
                QMessageBox.warning(self, "Save failed", str(exc))
            return

        # Update file-list checkmark.
        for i in range(self._file_list.count()):
            it = self._file_list.item(i)
            if it.data(Qt.UserRole) == str(self._current_path):
                it.setText(
                    ("✓ " if len(boxes_payload) > 0 else "   ")
                    + self._current_path.name
                )
                break

    # ──────────────────────────────────────────
    # Status display
    # ──────────────────────────────────────────

    def _update_detector_status(self) -> None:
        n_elem = len(self._element_names)
        if not self._detector_proposals:
            self._detector_status_label.setText(
                f"No detector candidates (draw all {n_elem} manually)."
            )
            return
        lines = []
        # Order by current region's element list for stability.
        for feature in self._element_names:
            prop = self._detector_proposals.get(feature)
            if prop is None:
                lines.append(f"  {feature}: (no detector)")
                continue
            score = prop.get("score")
            score_str = (
                f"score {score:.2f}" if score is not None else "score n/a"
            )
            lines.append(
                f"✓ {feature}: {prop.get('detector','?')} ({score_str})"
            )
        self._detector_status_label.setText("\n".join(lines))

    def _update_status(self) -> None:
        # Refresh visualization overlays whenever boxes change.
        # The HUD finder synthesis box always renders (it's THE
        # primary output we want visible). Predictions render only
        # when the toggle is on.
        self._render_hud_finder()
        self._render_predictions()
        n_boxes = len(self._current_boxes)
        active = (
            f" | drawing: {self._active_element}"
            if self._active_element else ""
        )
        annotated_count = sum(
            1 for i in range(self._file_list.count())
            if self._file_list.item(i).text().startswith("✓")
        )
        total = self._file_list.count()
        n_corrected = len(
            self._user_corrected & set(self._current_boxes.keys())
        )
        self._status.showMessage(
            f"{n_boxes}/{len(self._elements)} boxes "
            f"({n_corrected} user-corrected) | "
            f"{annotated_count}/{total} images annotated{active}",
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Auto-annotation tool for SC mining HUD training data. "
            "Existing detectors propose bounding boxes; you only "
            "correct what's wrong."
        )
    )
    ap.add_argument(
        "--region",
        choices=sorted(REGION_CONFIG.keys()),
        default="region1",
        help=(
            "Which HUD region to annotate. region1 = SCAN RESULTS "
            "panel (8 features). region2 = signature scanner pill "
            "(3 features). Sets the default source folder + element "
            "schema + detector chain. The region selector in the "
            "left panel can change this live."
        ),
    )
    ap.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Folder of PNGs to annotate. Defaults to the per-region "
            "default folder when omitted (region1 → "
            f"{DEFAULT_SOURCE}, region2 → {DEFAULT_SOURCE_REGION2})."
        ),
    )
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = AutoAnnotatorWindow(source_dir=args.source, region=args.region)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
