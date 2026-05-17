"""Signature Annotator Debugger.

A dedicated diagnostic GUI that loads a single region2 capture and
shows EVERY detector's internal state visually so the user can
diagnose why the icon, pill, or value boxes land where they do.

The auto-annotator (``scripts/auto_template_annotator.py``) shows the
final boxes; this tool shows the *intermediate state* leading up to
those final boxes:

  * HSV warm-color mask + cyan-color mask overlays.
  * Every NCC candidate ``find_icon`` produces, BEFORE leftmost-only,
    BEFORE the voter, BEFORE the position prior — all colour-coded by
    template scale and labelled with score.
  * Per-candidate voter decisions: green border = ACCEPTED, red =
    REJECTED, with a one-line reason floating above.
  * The position-prior threshold (vertical dashed line at
    ``x = crop_w * 0.30`` of the detected/manual HUD bbox).
  * The final pill / icon / value bboxes.
  * Right-panel readouts with full per-candidate voter tables.
  * A scrolling log of every detector's stdout (the ``[VOTE]``,
    ``[ANCHOR-DIAG]`` lines).

Usage
-----
::

    # Default: open file picker for a region2 PNG.
    python scripts/signature_annotator_debugger.py

    # Open a specific PNG.
    python scripts/signature_annotator_debugger.py --png "<path>.png"

    # Or a folder (browse with prev/next).
    python scripts/signature_annotator_debugger.py --folder "<dir>"

Architecture
------------
* **NCC candidate exposure** — ``find_icon`` doesn't expose its raw
  per-scale candidate list. Rather than reimplement the NCC search,
  we monkey-patch ``ocr.sc_ocr.signal_anchor._cnn_filter_icon_candidates``
  for the duration of one ``find_icon`` call. The patch records the
  ``candidates`` argument (= the post-shape-filter, pre-leftmost-only,
  pre-voter list) into a local capture buffer, then delegates to the
  real implementation so behavior is unchanged. We ALSO capture the
  return value (= post-voter list).
* **Voter introspection** — for each captured candidate we then call
  ``hud_tracker.anchors.icon_voter.vote_on_icon_candidate`` directly
  with the same crop the runtime would have used, recording the full
  ``votes`` dict + decision_path + per-tier probabilities.
* **Position-prior status** — the prior fires inside
  ``detect_icon`` (auto-template-annotator), AFTER ``find_icon``
  returned. We replicate the same check here on the final
  ``find_icon`` result so the right panel can say "rejected by
  position prior" or "passed".
* **Color-prior status** — same: replicated from
  ``detect_icon``.
* **Log capture** — we install a logging handler over the relevant
  loggers (``ocr.sc_ocr.signal_anchor``,
  ``hud_tracker.anchors.icon_voter``) so the bottom-panel log
  shows every ``[VOTE]`` / ``[ANCHOR-DIAG]`` line in real time.

This is a viewer; it does NOT modify any detector logic.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import textwrap
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

# ────────────────────────────────────────────────────────────────────
# Path setup — resolve the Mining_Signals tool root regardless of how
# this script was launched. The runtime detectors live in this tree.
# ────────────────────────────────────────────────────────────────────
THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
if str(TOOL) not in sys.path:
    sys.path.insert(0, str(TOOL))


# ────────────────────────────────────────────────────────────────────
# Default capture folder. Mirrors the path used by the auto-template-
# annotator REGION2 launcher.
# ────────────────────────────────────────────────────────────────────
DEFAULT_REGION2_FOLDER = Path(os.path.expandvars(
    r"%APPDATA%\ShipBit\WingmanAI\custom_skills\SC_Toolbox_Beta_V1.2"
    r"\tools\Mining_Signals\training_data_panels\user_20260418_154408"
    r"\region2"
))


# ────────────────────────────────────────────────────────────────────
# Theme — matches the rest of the toolbox tools.
# ────────────────────────────────────────────────────────────────────
ACCENT = "#33dd88"
RED = "#ff4444"
DIM = "#888888"
BG = "#1e1e1e"
FG = "#e0e0e0"
PANEL = "#2a2a2a"


# ────────────────────────────────────────────────────────────────────
# Layer color palette — RGBA tuples for QColor.
# ────────────────────────────────────────────────────────────────────
# NCC candidate scale color ramp (small → large).
_SCALE_COLOR_RAMP = [
    (255, 80, 80),    # red — smallest
    (255, 140, 60),   # orange
    (255, 200, 60),   # yellow
    (180, 220, 60),   # yellow-green
    (60, 220, 120),   # green
    (60, 220, 220),   # cyan
    (60, 140, 255),   # blue
    (160, 80, 220),   # purple — largest
]

WARM_OVERLAY_COLOR = (255, 200, 0, 90)
CYAN_OVERLAY_COLOR = (0, 200, 220, 90)
ACCEPT_COLOR = (60, 220, 100)
REJECT_COLOR = (220, 60, 60)
PILL_COLOR = (0, 200, 220)
ICON_COLOR = (255, 200, 0)
VALUE_COLOR = (200, 100, 255)
PRIOR_COLOR = (120, 200, 255)

# New-architecture detector colors. Distinct from the legacy palette so
# the new RGB-NCC + localize_icon consensus stand out visually:
#   * RGB NCC peak       — bright magenta (different from cyan/yellow).
#   * Geometry primary   — bright green   (whole-image scope, not the
#                                          per-candidate validator vote).
#   * localize_icon      — bright orange  (consensus combines the two).
RGB_NCC_COLOR = (255, 60, 220)
GEOM_PRIMARY_COLOR = (60, 255, 80)
LOCALIZE_ICON_COLOR = (255, 140, 0)

# Per-span palette for the glyph-spans layer. Six visually-distinct
# colors so adjacent spans don't blend (the segmenter typically emits
# 4-6 spans on a signature value). Distinct from the icon/pill/value
# palette so spans never get confused with the outer bboxes.
GLYPH_SPAN_COLORS = [
    (60, 220, 255),   # bright cyan
    (255, 220, 60),   # bright yellow
    (255, 100, 200),  # pink
    (180, 255, 100),  # lime
    (255, 160, 60),   # amber
    (140, 200, 255),  # sky blue
]

# Proportional segmenter overlay color. Distinct from the column-
# projection palette so the eye can compare both segmenters' bboxes
# on the same crop. Bright magenta — different hue from the cyan/
# yellow column-projection spans.
PROPORTIONAL_SPAN_COLOR = (255, 80, 255)
# Magenta variant for the proportional comma bbox so the comma slot
# is distinguishable from the digit slots even within the
# proportional layer.
PROPORTIONAL_COMMA_COLOR = (200, 60, 220)

# Comma anchor (find_comma_voted) overlay color. Bright magenta-pink
# DASHED rectangle to distinguish from the proportional layer's
# magenta and from any other detector. Drawn on the value crop's
# region in the original image so the user can see the precise
# x-axis anchor the proportional segmenter uses for digit layout.
COMMA_ANCHOR_COLOR = (255, 100, 180)


# ────────────────────────────────────────────────────────────────────
# Detector-introspection helpers.
# ────────────────────────────────────────────────────────────────────


class _LogCapture(logging.Handler):
    """Buffered logging handler. Each ``emit`` appends a single line
    string to ``self.lines``. The UI drains this buffer once per
    detector run.
    """

    def __init__(self, level: int = logging.INFO):
        super().__init__(level=level)
        self.lines: list[str] = []
        fmt = logging.Formatter("%(name)s %(levelname)s %(message)s")
        self.setFormatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            self.lines.append(self.format(record))
        except Exception:  # pragma: no cover — defensive
            pass

    def drain(self) -> list[str]:
        out, self.lines = self.lines, []
        return out


@contextmanager
def _capture_logs(*logger_names: str, level: int = logging.INFO):
    """Temporarily attach a :class:`_LogCapture` to each logger.

    Yields the capture; on exit, removes the handler.
    """
    cap = _LogCapture(level=level)
    loggers: list[logging.Logger] = []
    saved_levels: list[int] = []
    for name in logger_names:
        lg = logging.getLogger(name)
        loggers.append(lg)
        saved_levels.append(lg.level)
        lg.setLevel(min(level, lg.level if lg.level > 0 else level))
        lg.addHandler(cap)
    try:
        yield cap
    finally:
        for lg, lvl in zip(loggers, saved_levels):
            lg.removeHandler(cap)
            try:
                lg.setLevel(lvl)
            except Exception:
                pass


# Replicate the position-prior threshold the auto-annotator uses. We
# pin this to a constant rather than reading from detect_icon to keep
# the debugger truly read-only on the runtime code.
_POSITION_PRIOR_FRAC = 0.30


# Replicate the color-prior thresholds the auto-annotator uses. Same
# rationale — keep the debugger free of side effects.
_COLOR_PRIOR_COOL_FRAC_THR = 0.30
_COLOR_PRIOR_BRIGHT_S = 80
_COLOR_PRIOR_BRIGHT_V = 80


def _hsv_warm_mask(rgb: np.ndarray) -> np.ndarray:
    """Same calibration as ``icon_geometry._warm_mask``. Returns bool."""
    img = Image.fromarray(rgb).convert("HSV")
    hsv = np.asarray(img)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 5) & (h <= 55) & (s >= 60) & (v >= 100)


def _hsv_cyan_mask(rgb: np.ndarray) -> np.ndarray:
    """Cyan/teal pixels — the digit territory."""
    img = Image.fromarray(rgb).convert("HSV")
    hsv = np.asarray(img)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 100) & (h <= 150) & (s >= 60) & (v >= 80)


def _candidate_color_for_scale(tw: int) -> tuple[int, int, int]:
    """Pick a stable colour for an NCC scale's candidates."""
    # Map tw in [12, 72] to ramp index.
    tws = (12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 72)
    try:
        idx = tws.index(int(tw))
    except ValueError:
        idx = max(0, min(len(tws) - 1, (int(tw) - 12) // 6))
    ramp = _SCALE_COLOR_RAMP
    return ramp[idx % len(ramp)]


def _color_for_span(idx: int) -> tuple[int, int, int]:
    """Cycle through the per-span palette so adjacent spans don't blend."""
    return GLYPH_SPAN_COLORS[idx % len(GLYPH_SPAN_COLORS)]


def _read_gt_value(png_path: Optional[Path]) -> Optional[str]:
    """Read the ground-truth signature value from a sibling
    ``cap_*.json`` sidecar. The labeller stores ``{"value": "11,520"}``
    for region2 captures it has reviewed. Returns the trimmed string or
    None when the sidecar is absent / missing the field.
    """
    if png_path is None:
        return None
    try:
        jpath = png_path.with_suffix(".json")
        if not jpath.is_file():
            return None
        with jpath.open("r", encoding="utf-8") as fh:
            j = json.load(fh)
        v = j.get("value")
        if isinstance(v, str) and v.strip():
            return v.strip()
    except Exception:
        return None
    return None


def _capture_glyph_segmentation(
    pil_image: Image.Image,
    crop_box: Optional[tuple[int, int, int, int]],
) -> dict[str, Any]:
    """Run the runtime segmenter on the value crop and return the
    spans + per-span CNN classifications.

    ``crop_box`` is ``(x1, y1, x2, y2)`` in original-image coordinates
    (the OUTER value bbox). We slice the gray panel by it, apply the
    same preprocessing chain ``_signal_recognize_pil`` does up to the
    segmenter call (canonicalize → adaptive-binarize → strip-bridges →
    mask-commas), call ``_segment_glyphs(disable_gap_cut=True)``, then
    classify each crop via ``_classify_crops_signal`` (with HUD CNN
    fallback).

    Coordinates: spans come back from ``_segment_glyphs`` in
    *crop-local* ``(x, y, w, h)`` form. We translate them into
    original-image coords by adding ``crop_box[0:2]`` and store both
    forms in the result so the UI overlays land on the right pixels
    AND the readout shows local box coords for diagnostic comparison
    with runtime ``[DIAG]`` log lines.

    Returns a dict with keys ``available`` (bool — was the segmenter
    importable), ``spans`` (list of per-span dicts with ``bbox_local``
    and ``bbox_image`` and ``classification``/``confidence``), and
    optionally ``error`` (str, when the call raised) or ``reason``
    (str, when crop_box was None).
    """
    out: dict[str, Any] = {
        "available": False,
        "spans": [],
        "error": None,
        "reason": None,
        "n_spans": 0,
        "string_composed": "",
    }

    if crop_box is None:
        out["reason"] = "crop_box is None"
        return out

    x1, y1, x2, y2 = (
        int(crop_box[0]), int(crop_box[1]),
        int(crop_box[2]), int(crop_box[3]),
    )
    if x2 - x1 < 8 or y2 - y1 < 6:
        out["reason"] = (
            f"crop_box too small: {(x1, y1, x2, y2)}"
        )
        return out

    try:
        from ocr.sc_ocr.api import (  # type: ignore
            _canonicalize_polarity,
            _adaptive_binarize_multi,
            _strip_pill_outline_bridges,
            _mask_commas_in_signature_band,
            _segment_glyphs,
        )
        try:
            from ocr.sc_ocr.api import (  # type: ignore
                _classify_crops_signal,
            )
        except Exception:
            _classify_crops_signal = None  # type: ignore[assignment]
        try:
            from ocr.sc_ocr.api import (  # type: ignore
                _classify_crops,
            )
        except Exception:
            _classify_crops = None  # type: ignore[assignment]
        out["available"] = True
    except Exception as exc:
        out["error"] = f"segmenter import failed: {exc}"
        return out

    try:
        rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
        # Match runtime: max-of-channels grayscale (sidesteps the SC
        # HUD's chromatic aberration that destroys luma contrast).
        gray = rgb.max(axis=2).astype(np.uint8)
        # Slice to the value bbox.
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            out["reason"] = "value crop is empty after slicing"
            return out
        # Same chain `_signal_recognize_pil` runs up to segmentation.
        # Skipping the row-isolate + Lanczos upscale on purpose: those
        # would reshape the crop and force us to back-project span
        # coords through extra transforms. The user wants a visualization
        # in the ORIGINAL image's coord system, and the segmenter's
        # span placement is fundamentally driven by the projection
        # against this binary mask — Lanczos doesn't change WHICH cols
        # have ink, only how crisp the ink is.
        work_canon = _canonicalize_polarity(crop)
        binary = _adaptive_binarize_multi(work_canon, expected_count=5)
        binary = _strip_pill_outline_bridges(binary)
        binary = _mask_commas_in_signature_band(binary)
        crops, boxes = _segment_glyphs(
            work_canon, binary, disable_gap_cut=True,
        )
    except Exception as exc:
        out["error"] = f"segmenter call raised: {exc}"
        return out

    if not boxes:
        out["n_spans"] = 0
        return out

    # Classify per-span. Prefer the signal-specific CNN; fall back to
    # the HUD CNN. Either may be unavailable on a fresh install with
    # no signal training, so both are guarded.
    classifications: list[tuple[str, float]] = []
    try:
        if _classify_crops_signal is not None and crops:
            classifications = list(_classify_crops_signal(crops))
        if not classifications and _classify_crops is not None and crops:
            classifications = list(_classify_crops(crops))
    except Exception as exc:  # pragma: no cover — defensive
        out["error"] = f"classify failed: {exc}"
        classifications = []

    # Pad classifications to len(boxes) if the CNN returned fewer rows.
    while len(classifications) < len(boxes):
        classifications.append(("?", 0.0))

    spans: list[dict[str, Any]] = []
    for i, (bx, by, bw, bh) in enumerate(boxes):
        cls, conf = classifications[i]
        ix1 = x1 + int(bx)
        iy1 = y1 + int(by)
        ix2 = ix1 + int(bw)
        iy2 = iy1 + int(bh)
        spans.append({
            "idx": i,
            "bbox_local": (int(bx), int(by), int(bw), int(bh)),
            "bbox_image": (ix1, iy1, ix2, iy2),
            "classification": str(cls),
            "confidence": float(conf),
            "color": _color_for_span(i),
        })

    out["spans"] = spans
    out["n_spans"] = len(spans)
    out["string_composed"] = "".join(s["classification"] for s in spans)
    return out


def _capture_comma_anchor(
    pil_image: Image.Image,
    crop_box: Optional[tuple[int, int, int, int]],
) -> dict[str, Any]:
    """Run the dedicated RGB comma detector on the value crop.

    Returns a dict with the comma bbox + per-check breakdown for both
    polarities, plus the voted combiner's agreement state. Mirrors
    the structure ``_capture_proportional_segmentation`` produces so
    the UI can render the result alongside the proportional segmenter
    layer.

    On any error / unavailable module / no detection, fields default
    to ``None`` so the renderer can show a graceful placeholder.

    Returns:
        ``available`` (bool), ``found`` (bool), ``bbox_local`` (x,y,w,h
        in crop-local coords), ``bbox_image`` (x1,y1,x2,y2 in source-
        image coords), ``x_center_local`` (int), ``confidence`` (float),
        ``agreed`` (bool), ``polarity_used`` (str), ``checks``
        (per-check booleans), ``numbers`` (per-check supporting metrics),
        ``primary_summary`` / ``inverted_summary`` strings, and
        ``error`` / ``reason`` when something went wrong.
    """
    out: dict[str, Any] = {
        "available": False,
        "found": False,
        "bbox_local": None,
        "bbox_image": None,
        "x_center_local": None,
        "confidence": None,
        "agreed": None,
        "polarity_used": None,
        "checks": None,
        "numbers": None,
        "primary_summary": None,
        "inverted_summary": None,
        "error": None,
        "reason": None,
    }

    if crop_box is None:
        out["reason"] = "crop_box is None"
        return out

    x1, y1, x2, y2 = (
        int(crop_box[0]), int(crop_box[1]),
        int(crop_box[2]), int(crop_box[3]),
    )
    if x2 - x1 < 8 or y2 - y1 < 6:
        out["reason"] = f"crop_box too small: {(x1, y1, x2, y2)}"
        return out

    try:
        from hud_tracker.anchors.comma_finder import (  # type: ignore
            find_comma,
            find_comma_inv,
            find_comma_voted,
        )
        out["available"] = True
    except Exception as exc:
        out["error"] = f"comma_finder import failed: {exc}"
        return out

    try:
        rgb_crop = pil_image.convert("RGB").crop((x1, y1, x2, y2))
        primary = find_comma(rgb_crop)
        inverted = find_comma_inv(rgb_crop)
        voted = find_comma_voted(rgb_crop)
    except Exception as exc:
        out["error"] = f"comma_finder call raised: {exc}"
        return out

    def _summary(r: Optional[dict[str, Any]]) -> str:
        if r is None:
            return "None"
        return (
            f"x={int(r['x_center'])} "
            f"conf={float(r['confidence']):.2f} "
            f"({int(r['details']['n_checks_passed'])}/6) "
            f"polarity={r.get('polarity_used', '?')}"
        )

    out["primary_summary"] = _summary(primary)
    out["inverted_summary"] = _summary(inverted)

    if voted is None:
        out["reason"] = (
            "find_comma_voted returned None — both polarities failed "
            "to find a candidate that passed >= 4 of 6 checks"
        )
        return out

    out["found"] = True
    bx, by, bw, bh = voted["bbox"]
    out["bbox_local"] = (int(bx), int(by), int(bw), int(bh))
    out["bbox_image"] = (
        x1 + int(bx),
        y1 + int(by),
        x1 + int(bx) + int(bw),
        y1 + int(by) + int(bh),
    )
    out["x_center_local"] = int(voted["x_center"])
    out["confidence"] = float(voted["confidence"])
    out["agreed"] = bool(voted.get("agreed", False))
    out["polarity_used"] = str(voted.get("polarity_used", "?"))
    details = voted.get("details", {}) or {}
    out["checks"] = dict(details.get("checks", {}))
    out["numbers"] = dict(details.get("numbers", {}))
    return out


def _capture_proportional_segmentation(
    pil_image: Image.Image,
    crop_box: Optional[tuple[int, int, int, int]],
) -> dict[str, Any]:
    """Run the proportional segmenter on the value crop.

    Mirrors :func:`_capture_glyph_segmentation` but invokes the new
    structural-prior-aware ``segment_signal_proportional`` instead of
    the legacy column-projection ``_segment_glyphs``. Returns a dict
    with the same general shape so the UI can render BOTH segmenter
    outputs side by side on the same crop.

    The proportional segmenter exploits the fact that SC mining
    signatures are deterministically formatted (``D,DDD`` or
    ``DD,DDD``) — it detects the comma's bottom-only-ink signature,
    anchors digit slots to the comma's structural position, and runs
    the CNN on each digit bbox. Unlike column projection, it can't
    fragment a digit into multiple spans or merge adjacent digits.
    Both 4-digit and 5-digit hypotheses are evaluated and the higher-
    scoring one is reported.

    Returns:
        ``available`` (bool), ``digits`` (per-slot dicts: bbox_local,
        bbox_image, classification, confidence, is_comma), ``n_digits``
        (4 or 5), ``comma_position`` (1 or 2), ``confidence`` (float),
        ``hypotheses`` (per-hypothesis scoring detail), ``error`` /
        ``reason`` when something went wrong.
    """
    out: dict[str, Any] = {
        "available": False,
        "digits": [],
        "n_digits": None,
        "comma_position": None,
        "confidence": None,
        "hypotheses": [],
        "error": None,
        "reason": None,
        "string_composed": "",
    }

    if crop_box is None:
        out["reason"] = "crop_box is None"
        return out

    x1, y1, x2, y2 = (
        int(crop_box[0]), int(crop_box[1]),
        int(crop_box[2]), int(crop_box[3]),
    )
    if x2 - x1 < 8 or y2 - y1 < 6:
        out["reason"] = (
            f"crop_box too small: {(x1, y1, x2, y2)}"
        )
        return out

    try:
        from hud_tracker.anchors.signal_proportional_segmenter import (  # type: ignore
            segment_signal_proportional,
        )
        try:
            from ocr.sc_ocr.api import (  # type: ignore
                _classify_crops_signal,
                _KNOWN_SIGNAL_VALUES,
            )
        except Exception:
            _classify_crops_signal = None  # type: ignore[assignment]
            _KNOWN_SIGNAL_VALUES = set()  # type: ignore[assignment]
        out["available"] = True
    except Exception as exc:
        out["error"] = f"proportional segmenter import failed: {exc}"
        return out

    try:
        # Slice the value bbox out of the source RGB. The proportional
        # segmenter does its own grayscale conversion + upscale + canon
        # internally (it needs the RGB so its polarity-detection helper
        # can sample border colour). No row-isolate trim — the segmenter
        # operates on the bbox content directly, mirroring how the
        # runtime feeds it.
        rgb_crop = pil_image.convert("RGB").crop((x1, y1, x2, y2))
        result = segment_signal_proportional(
            rgb_crop,
            classifier=_classify_crops_signal,
            lexicon=(
                _KNOWN_SIGNAL_VALUES
                if _KNOWN_SIGNAL_VALUES else None
            ),
        )
    except Exception as exc:
        out["error"] = f"proportional segmenter call raised: {exc}"
        return out

    if result is None:
        out["reason"] = (
            "crop too small to plausibly hold a 4-digit signature"
        )
        return out

    digits_with_image_coords: list[dict[str, Any]] = []
    crop_w_post = int(result["details"].get("crop_w", x2 - x1))
    crop_h_post = int(result["details"].get("crop_h", y2 - y1))
    src_w = max(1, x2 - x1)
    src_h = max(1, y2 - y1)
    # The segmenter may have internally Lanczos-upscaled the crop, so
    # its bboxes are in the upscaled coordinate system. Project them
    # back to source image coords by scaling against the original
    # value-bbox dimensions.
    sx = src_w / float(crop_w_post)
    sy = src_h / float(crop_h_post)

    for i, d in enumerate(result.get("digits", [])):
        bx, by, bw, bh = d["bbox"]
        # Project bbox from segmenter-internal coords into source-image
        # coords. Floor for left/top, ceil for right/bottom, so the
        # overlay rectangle fully contains the underlying digit pixels
        # even after the integer rounding from the upscale.
        ix1 = x1 + int(round(bx * sx))
        iy1 = y1 + int(round(by * sy))
        ix2 = x1 + int(round((bx + bw) * sx))
        iy2 = y1 + int(round((by + bh) * sy))
        digits_with_image_coords.append({
            "idx": i,
            "bbox_local_post": (int(bx), int(by), int(bw), int(bh)),
            "bbox_image": (ix1, iy1, ix2, iy2),
            "is_comma": bool(d.get("is_comma")),
            "classification": (
                str(d.get("classification", ""))
                if not d.get("is_comma") else ""
            ),
            "confidence": (
                float(d.get("confidence", 0.0))
                if not d.get("is_comma") else 0.0
            ),
        })

    out["digits"] = digits_with_image_coords
    out["n_digits"] = int(result["n_digits"])
    out["comma_position"] = int(result["comma_position"])
    out["confidence"] = float(result["confidence"])
    out["hypotheses"] = list(result["details"].get("hypotheses", []))
    out["string_composed"] = str(
        result["details"].get("string_composed", "")
    )
    out["details"] = dict(result["details"])
    return out


# ────────────────────────────────────────────────────────────────────
# Run-detectors function — does ALL the heavy lifting and returns a
# dict the UI can render. Called both from the GUI and the headless
# verification path.
# ────────────────────────────────────────────────────────────────────


def run_full_diagnostics(
    pil_image: Image.Image,
    hud_bbox: Optional[tuple[int, int, int, int]] = None,
) -> dict[str, Any]:
    """Run pill / icon / value detectors with full instrumentation.

    Returns a dict with:
      * ``image``: the input PIL image (RGB)
      * ``hud_bbox``: ``(x, y, w, h)`` or None
      * ``warm_mask``, ``cyan_mask``: HxW bool arrays
      * ``pill``: dict with bbox + score + details, or None
      * ``ncc_raw_candidates``: list of dicts (everything ``find_icon``
        passed to ``_cnn_filter_icon_candidates``)
      * ``ncc_post_voter``: list of dicts (the survivors)
      * ``voter_table``: list of dicts, one per candidate, with the
        full ``votes`` / ``decision_path`` / per-tier probs.
      * ``icon_final``: final ``find_icon`` result, or None
      * ``icon_position_prior_passed``: bool
      * ``icon_color_prior_passed``: bool
      * ``icon_after_priors``: dict (the box detect_icon would emit),
        or None
      * ``digit_cluster``: dict from find_digit_cluster, or None
      * ``rgb_ncc_result``: dict from ``find_icon_rgb_ncc``, or None
      * ``rgb_ncc_available``: bool — whether the module imported
      * ``rgb_ncc_template_info``: dict with template count + source
      * ``geometry_primary_result``: dict from
        ``find_icon_by_geometry`` called whole-image (NOT per-
        candidate; the per-candidate voter decisions are a separate
        signal), or None
      * ``geometry_primary_available``: bool
      * ``localize_icon_result``: dict from ``localize_icon``, or None
      * ``localize_icon_available``: bool
      * ``localize_icon_iou``: float (IoU between geometry & rgb_ncc
        primary bboxes), or None when one is missing
      * ``localize_icon_no_consensus_reason``: str describing why
        consensus didn't fire, or None when it did
      * ``glyph_segmentation``: dict from
        ``_capture_glyph_segmentation`` (per-span bboxes +
        classifications run on the value crop), or None when no value
        crop was available.
      * ``glyph_segmentation_crop_source``: str — which crop source
        was used (``world_model_region2``, ``digit_cluster``, or
        ``unavailable``).
      * ``logs``: list of captured log lines.
      * ``errors``: list of (stage, str(exc)) from any failures.
    """
    from ocr.sc_ocr import signal_anchor as _sa
    from hud_tracker.anchors.icon_voter import vote_on_icon_candidate

    # Defensive imports for the new architecture. If these fail (e.g.
    # the module was renamed, dependencies missing) the debugger
    # still loads with the legacy layers; the new layers simply
    # report unavailable.
    try:
        from hud_tracker.anchors.icon_rgb_ncc import (  # type: ignore
            find_icon_rgb_ncc as _find_icon_rgb_ncc,
        )
        try:
            from hud_tracker.anchors.icon_rgb_ncc import availability as _rgb_ncc_avail  # type: ignore
        except Exception:
            _rgb_ncc_avail = None  # type: ignore[assignment]
        _rgb_ncc_available = True
    except Exception as _exc_rgb_ncc:
        _find_icon_rgb_ncc = None  # type: ignore[assignment]
        _rgb_ncc_avail = None  # type: ignore[assignment]
        _rgb_ncc_available = False

    try:
        from hud_tracker.anchors.icon_voter import (  # type: ignore
            localize_icon as _localize_icon,
        )
        _localize_icon_available = True
    except Exception:
        _localize_icon = None  # type: ignore[assignment]
        _localize_icon_available = False

    try:
        from hud_tracker.anchors.icon_geometry import (  # type: ignore
            find_icon_by_geometry as _find_icon_by_geometry,
        )
        _geom_primary_available = True
    except Exception:
        _find_icon_by_geometry = None  # type: ignore[assignment]
        _geom_primary_available = False

    out: dict[str, Any] = {
        "image": pil_image,
        "hud_bbox": hud_bbox,
        "warm_mask": None,
        "cyan_mask": None,
        "pill": None,
        "ncc_raw_candidates": [],
        "ncc_post_voter": [],
        "voter_table": [],
        "icon_final": None,
        "icon_position_prior_passed": True,
        "icon_color_prior_passed": True,
        "icon_after_priors": None,
        "digit_cluster": None,
        "rgb_ncc_result": None,
        "rgb_ncc_available": bool(_rgb_ncc_available),
        "rgb_ncc_template_info": None,
        "geometry_primary_result": None,
        "geometry_primary_available": bool(_geom_primary_available),
        "localize_icon_result": None,
        "localize_icon_available": bool(_localize_icon_available),
        "localize_icon_iou": None,
        "localize_icon_no_consensus_reason": None,
        "glyph_segmentation": None,
        "glyph_segmentation_crop_source": "unavailable",
        "logs": [],
        "errors": [],
    }

    rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
    gray = np.asarray(pil_image.convert("L"), dtype=np.uint8)

    # 1. HSV masks — diagnostic only, doesn't touch detectors.
    try:
        out["warm_mask"] = _hsv_warm_mask(rgb)
        out["cyan_mask"] = _hsv_cyan_mask(rgb)
    except Exception as exc:
        out["errors"].append(("hsv_masks", str(exc)))

    # 2. Pill — find_hud_panel with region2 calibration. We mirror the
    # PILL_CALIBRATION dict from auto_template_annotator.py so this
    # tool stays standalone if that module is ever moved.
    pill_calibration = {
        "version": 2,
        "source": "region2-pill-tuned",
        "cyan_band": {"h_min": 100, "h_max": 175},
        "green_band": {"h_min": 25, "h_max": 60},
        "sat_min": 60,
        "val_min": 80,
        "min_area_px": 600,
        "min_bbox_aspect": 1.5,
        "max_bbox_aspect": 5.5,
        "min_extent": 0.4,
        "morph_seed_iterations": 2,
        "morph_vert_close_px": 3,
        "morph_horiz_close_px": 30,
        "bbox_aspect_peak": 3.5,
    }

    with _capture_logs(
        "ocr.sc_ocr.signal_anchor",
        "ocr.sc_ocr.api",  # segmenter-related diagnostics
        "hud_tracker.anchors.icon_voter",
        "hud_tracker.anchors.hud_color_finder",
        "hud_tracker.anchors.icon_geometry",
        "hud_tracker.anchors.icon_contour",
        "hud_tracker.anchors.icon_rgb_ncc",  # new: RGB NCC primary
        level=logging.INFO,
    ) as cap:
        try:
            from hud_tracker.anchors.hud_color_finder import find_hud_panel
            res = find_hud_panel(pil_image, calibration=pill_calibration)
            if res and "bbox" in res:
                bx, by, bw, bh = res["bbox"]
                out["pill"] = {
                    "bbox": (int(bx), int(by), int(bx) + int(bw), int(by) + int(bh)),
                    "xywh": (int(bx), int(by), int(bw), int(bh)),
                    "score": float(res.get("confidence", 0.0)),
                    "details": res.get("details"),
                    "n_chrome_pixels": int(res.get("n_chrome_pixels", 0)),
                    "n_candidates_considered": int(res.get("n_candidates_considered", 0)),
                }
        except Exception as exc:
            out["errors"].append(("pill", str(exc)))
            out["errors"].append(("pill_tb", traceback.format_exc()))

        # 3. Icon — instrument find_icon by monkey-patching
        # _cnn_filter_icon_candidates so we can capture the candidate
        # list AND the post-voter list. Then run the voter ourselves
        # per-candidate to collect a full table.
        try:
            _sa.reset_anchor_cache()
        except Exception:
            pass

        captured_pre_voter: dict[str, list[tuple]] = {"cands": []}
        captured_post_voter: dict[str, list[tuple]] = {"cands": []}
        original_filter = _sa._cnn_filter_icon_candidates

        def _patched_filter(g, candidates, rgb_image=None):
            captured_pre_voter["cands"] = list(candidates)
            try:
                kept = original_filter(g, candidates, rgb_image=rgb_image)
            except Exception:
                kept = candidates
                raise
            finally:
                # If the filter raised, kept may be partial — capture
                # whatever we got. Outer try/except in the caller
                # propagates the error.
                pass
            captured_post_voter["cands"] = list(kept)
            return kept

        # Patch only for the duration of the find_icon call.
        try:
            _sa._cnn_filter_icon_candidates = _patched_filter
            try:
                result = _sa.find_icon(gray, min_score=0.40, rgb_image=rgb)
            except Exception as exc:
                out["errors"].append(("find_icon", str(exc)))
                out["errors"].append(("find_icon_tb", traceback.format_exc()))
                result = None
        finally:
            _sa._cnn_filter_icon_candidates = original_filter

        if result is not None:
            ix1, iy1, ix2, iy2, score = result
            out["icon_final"] = {
                "bbox": (int(ix1), int(iy1), int(ix2), int(iy2)),
                "score": float(score),
            }

        out["ncc_raw_candidates"] = [
            {
                "score": float(c[0]),
                "x1": int(c[1]),
                "y1": int(c[2]),
                "x2": int(c[3]),
                "y2": int(c[4]),
                "tw": int(c[5]),
                "color": _candidate_color_for_scale(int(c[5])),
            }
            for c in captured_pre_voter["cands"]
        ]
        out["ncc_post_voter"] = [
            {
                "score": float(c[0]),
                "x1": int(c[1]),
                "y1": int(c[2]),
                "x2": int(c[3]),
                "y2": int(c[4]),
                "tw": int(c[5]),
            }
            for c in captured_post_voter["cands"]
        ]

        # 4. Run the voter manually on each candidate (top-K) to fill
        # the per-candidate decision table. This re-runs the same
        # voter calls find_icon would have made — so the recorded
        # votes match what the runtime saw.
        cands = captured_pre_voter["cands"]
        # Sort by score desc and take all (typically <= 13 — one per
        # template scale), so the right panel can show the full
        # picture.
        cands_sorted = sorted(cands, key=lambda c: -float(c[0]))

        for cand in cands_sorted:
            x1, y1, x2, y2 = (
                int(cand[1]), int(cand[2]), int(cand[3]), int(cand[4]),
            )
            tw = int(cand[5])
            # Build the same 28×28 gray crop find_icon's _gray_crop_28
            # closure builds — this matches the gray CNN's training
            # distribution.
            pad_x = max(2, (x2 - x1) // 4)
            pad_y = max(2, (y2 - y1) // 4)
            rx1 = max(0, x1 - pad_x)
            ry1 = max(0, y1 - pad_y)
            rx2 = min(gray.shape[1], x2 + pad_x)
            ry2 = min(gray.shape[0], y2 + pad_y)
            region = gray[ry1:ry2, rx1:rx2]
            if region.size == 0:
                gray_crop = np.zeros((28, 28), dtype=np.float32)
            else:
                try:
                    pil_l = Image.fromarray(region.astype(np.uint8)).resize(
                        (28, 28), Image.BILINEAR,
                    )
                    gray_crop = np.asarray(pil_l, dtype=np.float32) / 255.0
                except Exception:
                    gray_crop = np.zeros((28, 28), dtype=np.float32)

            try:
                pil_for_voter = Image.fromarray(rgb).convert("RGB")
                vote = vote_on_icon_candidate(
                    rgb_image=pil_for_voter,
                    candidate_bbox=(x1, y1, x2, y2),
                    gray_cnn=None,
                    rgb_cnn=None,
                    gray_crop=gray_crop,
                )
            except Exception as exc:
                vote = {
                    "accepted": False,
                    "confidence": 0.0,
                    "votes": {
                        "geometry": "error",
                        "contour": "error",
                        "rgb_cnn": "error",
                        "gray_cnn": "error",
                    },
                    "decision_path": f"voter_raised: {exc}",
                    "details": {"rgb_at_prob": None, "gray_at_prob": None},
                }

            row = {
                "score": float(cand[0]),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w": x2 - x1, "h": y2 - y1, "tw": tw,
                "geom_vote": vote["votes"].get("geometry", ""),
                "contour_vote": vote["votes"].get("contour", ""),
                "rgb_cnn_vote": vote["votes"].get("rgb_cnn", ""),
                "gray_cnn_vote": vote["votes"].get("gray_cnn", ""),
                "rgb_at_prob": vote["details"].get("rgb_at_prob"),
                "gray_at_prob": vote["details"].get("gray_at_prob"),
                "decision_path": vote.get("decision_path", ""),
                "accepted": bool(vote.get("accepted", False)),
                "confidence": float(vote.get("confidence", 0.0)),
                "color": _candidate_color_for_scale(tw),
            }
            out["voter_table"].append(row)

        # 5. Position-prior + color-prior simulation (mirrors
        # detect_icon in auto_template_annotator).
        if result is not None:
            ix1, iy1, ix2, iy2, _score = result
            crop_w = gray.shape[1] if gray.ndim >= 2 else 0
            pos_passed = not (crop_w > 0 and ix1 > crop_w * _POSITION_PRIOR_FRAC)
            out["icon_position_prior_passed"] = bool(pos_passed)
            out["icon_position_prior_threshold_x"] = (
                int(crop_w * _POSITION_PRIOR_FRAC) if crop_w > 0 else None
            )

            # Color prior.
            color_passed = True
            try:
                cand_crop = pil_image.crop((int(ix1), int(iy1), int(ix2), int(iy2)))
                hsv = np.asarray(cand_crop.convert("HSV"), dtype=np.uint8)
                sat = hsv[..., 1]
                val = hsv[..., 2]
                bright = (
                    (sat >= _COLOR_PRIOR_BRIGHT_S)
                    & (val >= _COLOR_PRIOR_BRIGHT_V)
                )
                if bright.sum() >= 8:
                    hue = hsv[..., 0][bright]
                    n_warm = int(((hue <= 50) | (hue >= 230)).sum())
                    n_cool = int(((hue >= 100) & (hue <= 150)).sum())
                    n_total = max(1, int(bright.sum()))
                    warm_frac = n_warm / n_total
                    cool_frac = n_cool / n_total
                    if cool_frac > _COLOR_PRIOR_COOL_FRAC_THR and cool_frac > warm_frac:
                        color_passed = False
                    out["icon_color_warm_frac"] = float(warm_frac)
                    out["icon_color_cool_frac"] = float(cool_frac)
            except Exception:
                pass
            out["icon_color_prior_passed"] = bool(color_passed)

            if pos_passed and color_passed:
                out["icon_after_priors"] = {
                    "bbox": (int(ix1), int(iy1), int(ix2), int(iy2)),
                }

        # 6. Digit cluster (value).
        try:
            dc = _sa.find_digit_cluster(gray)
            if dc is not None:
                dx1, dy1, dx2, dy2 = dc
                out["digit_cluster"] = {
                    "bbox": (int(dx1), int(dy1), int(dx2), int(dy2)),
                }
        except Exception as exc:
            out["errors"].append(("find_digit_cluster", str(exc)))

        # 7. New-architecture detectors — run independently of the
        # legacy NCC pipeline. Each is wrapped in its own try/except
        # so a failure in one doesn't stop the others (and doesn't
        # break the legacy layers above).
        rgb_ncc_xywh: Optional[tuple[int, int, int, int]] = None
        geom_xywh: Optional[tuple[int, int, int, int]] = None

        # 7a. RGB NCC primary.
        if _rgb_ncc_available and _find_icon_rgb_ncc is not None:
            try:
                # Surface template provenance for the right panel.
                if _rgb_ncc_avail is not None:
                    try:
                        out["rgb_ncc_template_info"] = _rgb_ncc_avail()
                    except Exception:
                        out["rgb_ncc_template_info"] = None
                rres = _find_icon_rgb_ncc(rgb, hud_bbox=hud_bbox)
                if rres is not None and "bbox" in rres:
                    bx, by, bw, bh = rres["bbox"]
                    rgb_ncc_xywh = (int(bx), int(by), int(bw), int(bh))
                    details = rres.get("details") or {}
                    per_ch = details.get("per_channel_scores") or (0.0, 0.0, 0.0)
                    weights = details.get("weights") or (0.5, 0.2, 0.3)
                    crop_w = rgb.shape[1] if rgb.ndim >= 2 else 0
                    pos_pass = bool(crop_w > 0 and bx <= crop_w * 0.40)
                    out["rgb_ncc_result"] = {
                        "bbox": rgb_ncc_xywh,
                        "bbox_xyxy": (
                            int(bx), int(by),
                            int(bx) + int(bw), int(by) + int(bh),
                        ),
                        "score": float(rres.get("score", 0.0)),
                        "scale": float(details.get("scale", 1.0)),
                        "per_channel_scores": (
                            float(per_ch[0]), float(per_ch[1]), float(per_ch[2]),
                        ),
                        "weights": (
                            float(weights[0]), float(weights[1]), float(weights[2]),
                        ),
                        "n_candidates_above_thresh": int(
                            details.get("n_candidates_above_thresh", 0)
                        ),
                        "template_used": str(details.get("template_used", "")),
                        "position_prior_pass": pos_pass,
                    }
            except Exception as exc:
                out["errors"].append(("rgb_ncc", str(exc)))

        # 7b. Geometry primary — call find_icon_by_geometry directly
        # on the whole image. This is DIFFERENT from the per-
        # candidate geometry vote captured in voter_table above:
        # there geometry validates a fixed crop the NCC proposed;
        # here it gets to scan the whole image and propose its own.
        if _geom_primary_available and _find_icon_by_geometry is not None:
            try:
                gres = _find_icon_by_geometry(rgb, hud_bbox=hud_bbox)
                if gres is not None and "bbox" in gres:
                    gx, gy, gw, gh = gres["bbox"]
                    geom_xywh = (int(gx), int(gy), int(gw), int(gh))
                    gdetails = gres.get("details") or {}
                    out["geometry_primary_result"] = {
                        "bbox": geom_xywh,
                        "bbox_xyxy": (
                            int(gx), int(gy),
                            int(gx) + int(gw), int(gy) + int(gh),
                        ),
                        "confidence": float(gres.get("confidence", 0.0)),
                        "score": int(gdetails.get("score", 0)),
                        "checks": dict(gdetails.get("checks", {})),
                        "tiny_mode": bool(gdetails.get("tiny_mode", False)),
                    }
            except Exception as exc:
                out["errors"].append(("geometry_primary", str(exc)))

        # 7c. localize_icon consensus.
        if _localize_icon_available and _localize_icon is not None:
            try:
                lres = _localize_icon(rgb, hud_bbox=hud_bbox)
                if lres is not None and "bbox" in lres:
                    lx, ly, lw, lh = lres["bbox"]
                    ldetails = lres.get("details") or {}
                    out["localize_icon_result"] = {
                        "bbox": (int(lx), int(ly), int(lw), int(lh)),
                        "bbox_xyxy": (
                            int(lx), int(ly),
                            int(lx) + int(lw), int(ly) + int(lh),
                        ),
                        "score": float(lres.get("score", 0.0)),
                        "detector": str(lres.get("detector", "")),
                        "iou": float(ldetails.get("iou", 0.0)),
                    }
                    out["localize_icon_iou"] = float(ldetails.get("iou", 0.0))
                else:
                    # No consensus — diagnose why.
                    if rgb_ncc_xywh is None and geom_xywh is None:
                        reason = "neither geometry nor rgb_ncc returned a bbox"
                    elif rgb_ncc_xywh is None:
                        reason = (
                            f"geometry returned {geom_xywh}, rgb_ncc returned None"
                        )
                    elif geom_xywh is None:
                        reason = (
                            f"geometry returned None, rgb_ncc returned "
                            f"{rgb_ncc_xywh}"
                        )
                    else:
                        # Both returned but consensus IoU was below 0.4.
                        # Compute IoU here for the readout.
                        try:
                            from hud_tracker.anchors.icon_voter import (  # type: ignore
                                _iou_xywh as _local_iou,
                            )
                            iou_v = float(_local_iou(geom_xywh, rgb_ncc_xywh))
                        except Exception:
                            iou_v = 0.0
                        out["localize_icon_iou"] = iou_v
                        reason = (
                            f"IoU {iou_v:.2f} below 0.4 threshold "
                            f"(geometry={geom_xywh}, rgb_ncc={rgb_ncc_xywh})"
                        )
                    out["localize_icon_no_consensus_reason"] = reason
            except Exception as exc:
                out["errors"].append(("localize_icon", str(exc)))

        # If localize_icon was unavailable but both primaries fired,
        # we can still report the IoU for the readout panel (graceful
        # degradation).
        if (
            out["localize_icon_iou"] is None
            and rgb_ncc_xywh is not None
            and geom_xywh is not None
        ):
            try:
                ax1, ay1, ax2, ay2 = (
                    geom_xywh[0], geom_xywh[1],
                    geom_xywh[0] + geom_xywh[2],
                    geom_xywh[1] + geom_xywh[3],
                )
                bx1, by1, bx2, by2 = (
                    rgb_ncc_xywh[0], rgb_ncc_xywh[1],
                    rgb_ncc_xywh[0] + rgb_ncc_xywh[2],
                    rgb_ncc_xywh[1] + rgb_ncc_xywh[3],
                )
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
                bb = max(0, bx2 - bx1) * max(0, by2 - by1)
                ua = aa + bb - inter
                out["localize_icon_iou"] = (
                    float(inter) / float(ua) if ua > 0 else 0.0
                )
            except Exception:
                pass

        # 8. Glyph segmentation — call the runtime ``_segment_glyphs``
        # on the value crop and capture per-span bboxes + per-span CNN
        # classifications. This is the most direct way to see WHERE
        # the segmenter placed each digit and what each one was
        # classified as, which is exactly the diagnostic the user
        # needs after recent segmenter fixes still mis-align ~50% of
        # captures.
        #
        # Crop source priority (mirrors the runtime in
        # ``_signal_recognize_pil``):
        #   1. ``world_model_region2`` proportional crop, when both
        #      the JSON calibration AND a pill bbox are available.
        #      This is what the runtime uses on the steady-state path
        #      so debugger output matches what live OCR sees.
        #   2. ``find_digit_cluster`` legacy bbox — fallback when no
        #      world model / pill is available (older installs, dim
        #      captures).
        #   3. None — no crop available; segmenter not run.
        try:
            seg_crop_box: Optional[tuple[int, int, int, int]] = None
            seg_crop_source = "unavailable"
            # Try world-model-region2 first (matches runtime).
            try:
                from ocr.sc_ocr.api import (  # type: ignore
                    _load_region2_world_model_for_api,
                    _find_pill_for_signal,
                )
                _wmr = _load_region2_world_model_for_api()
                if _wmr is not None:
                    _vfrac = (_wmr.get("features") or {}).get("value")
                    _pill_xywh = (
                        _find_pill_for_signal(rgb) if _vfrac else None
                    )
                    if _vfrac and _pill_xywh is not None:
                        _px, _py, _pw, _ph = _pill_xywh
                        _vx = int(round(
                            _px + float(_vfrac["x_frac"]["mean"]) * _pw
                        ))
                        _vy = int(round(
                            _py + float(_vfrac["y_frac"]["mean"]) * _ph
                        ))
                        _vw = int(round(
                            float(_vfrac["w_frac"]["mean"]) * _pw
                        ))
                        _vh = int(round(
                            float(_vfrac["h_frac"]["mean"]) * _ph
                        ))
                        # Apply the same icon-anchored LHS refinement
                        # the runtime does so debugger spans land in
                        # the same coords production reads from.
                        try:
                            from hud_tracker.anchors.icon_voter import (
                                localize_icon as _li_for_seg,
                            )
                            _icon_loc_seg = _li_for_seg(rgb)
                            if _icon_loc_seg is not None:
                                _ix, _iy, _iw, _ih = _icon_loc_seg["bbox"]
                                _icon_anchor = (
                                    _ix + _iw + max(2, int(_pw * 0.03))
                                )
                                _delta = _vx - _icon_anchor
                                _vx = _icon_anchor
                                _vw = _vw + _delta
                        except Exception:
                            pass
                        _rhs_ceil = (
                            _px + _pw - max(2, int(_pw * 0.05))
                        )
                        _x1s = max(0, _vx)
                        _y1s = max(0, _vy)
                        _x2s = min(
                            _vx + _vw, _rhs_ceil, gray.shape[1],
                        )
                        _y2s = min(_vy + _vh, gray.shape[0])
                        if (_x2s - _x1s >= 20) and (_y2s - _y1s >= 8):
                            seg_crop_box = (_x1s, _y1s, _x2s, _y2s)
                            seg_crop_source = "world_model_region2"
            except Exception as _wm_exc:
                out["errors"].append(("segmenter_world_model", str(_wm_exc)))
            # Fallback: digit_cluster bbox from the legacy detector.
            if seg_crop_box is None and out.get("digit_cluster") is not None:
                seg_crop_box = out["digit_cluster"]["bbox"]
                seg_crop_source = "digit_cluster"
            seg_result = _capture_glyph_segmentation(
                pil_image, seg_crop_box,
            )
            seg_result["crop_source"] = seg_crop_source
            seg_result["crop_box"] = (
                tuple(int(v) for v in seg_crop_box)
                if seg_crop_box is not None else None
            )
            out["glyph_segmentation"] = seg_result
            out["glyph_segmentation_crop_source"] = seg_crop_source

            # ── Proportional segmenter (parallel readout) ──
            # Run the new constrained-format-aware segmenter on the
            # SAME value bbox the column-projection segmenter used.
            # Letting both read off the same crop gives the user a
            # direct, like-for-like comparison: which segmenter's
            # bboxes land on individual digits, which one fragments,
            # which one fuses adjacent digits. The runtime promoted
            # the proportional segmenter to PRIMARY and fell back to
            # column-projection on low-confidence reads, so this
            # diagnostic mirrors the production decision surface.
            try:
                prop_result = _capture_proportional_segmentation(
                    pil_image, seg_crop_box,
                )
                prop_result["crop_source"] = seg_crop_source
                prop_result["crop_box"] = (
                    tuple(int(v) for v in seg_crop_box)
                    if seg_crop_box is not None else None
                )
                out["proportional_segmentation"] = prop_result
            except Exception as _prop_exc:
                out["errors"].append(("proportional_segmentation", str(_prop_exc)))

            # ── Comma anchor (find_comma_voted) ──
            # Run the dedicated RGB comma detector on the same value
            # bbox. This is the structural X-axis anchor the
            # proportional segmenter now uses internally as its
            # primary; surfacing it here lets the user verify the
            # comma column directly and see which checks the
            # detector passed/failed.
            try:
                comma_result = _capture_comma_anchor(
                    pil_image, seg_crop_box,
                )
                comma_result["crop_source"] = seg_crop_source
                comma_result["crop_box"] = (
                    tuple(int(v) for v in seg_crop_box)
                    if seg_crop_box is not None else None
                )
                out["comma_anchor"] = comma_result
            except Exception as _ca_exc:
                out["errors"].append(("comma_anchor", str(_ca_exc)))
        except Exception as _seg_exc:
            out["errors"].append(("glyph_segmentation", str(_seg_exc)))

        out["logs"] = list(cap.lines)

    return out


# ────────────────────────────────────────────────────────────────────
# GUI imports — kept lazy so the headless verification path doesn't
# pay the Qt cost.
# ────────────────────────────────────────────────────────────────────


def _import_qt():
    from PySide6.QtCore import Qt, QPointF, QRectF, Signal as QSignal
    from PySide6.QtGui import (
        QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap,
    )
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QFileDialog, QGraphicsItem,
        QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
        QGraphicsTextItem, QGraphicsView, QGroupBox, QHBoxLayout,
        QLabel, QListWidget, QListWidgetItem, QMainWindow, QPushButton,
        QPlainTextEdit, QScrollArea, QSplitter, QStatusBar, QTableWidget,
        QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView,
    )
    return locals()


def _pil_to_qpixmap(pil: Image.Image, qt: dict) -> Any:
    """Convert a PIL RGB image to a QPixmap."""
    QImage = qt["QImage"]
    QPixmap = qt["QPixmap"]
    if pil.mode != "RGBA":
        pil = pil.convert("RGBA")
    data = pil.tobytes("raw", "RGBA")
    qimg = QImage(data, pil.width, pil.height, QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimg)


def _mask_to_pixmap(mask: np.ndarray, color_rgba: tuple[int, int, int, int], qt: dict) -> Any:
    """Render a bool mask as a coloured RGBA pixmap (transparent where mask is False)."""
    QImage = qt["QImage"]
    QPixmap = qt["QPixmap"]
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[mask, 0] = color_rgba[0]
    rgba[mask, 1] = color_rgba[1]
    rgba[mask, 2] = color_rgba[2]
    rgba[mask, 3] = color_rgba[3]
    img = QImage(rgba.tobytes(), W, H, 4 * W, QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(img)


def build_gui(initial_diag: Optional[dict] = None,
              folder: Optional[Path] = None,
              png: Optional[Path] = None) -> None:
    qt = _import_qt()
    Qt = qt["Qt"]
    QColor = qt["QColor"]
    QPen = qt["QPen"]
    QBrush = qt["QBrush"]
    QFont = qt["QFont"]
    QPainter = qt["QPainter"]
    QApplication = qt["QApplication"]
    QMainWindow = qt["QMainWindow"]
    QSplitter = qt["QSplitter"]
    QWidget = qt["QWidget"]
    QVBoxLayout = qt["QVBoxLayout"]
    QHBoxLayout = qt["QHBoxLayout"]
    QLabel = qt["QLabel"]
    QPushButton = qt["QPushButton"]
    QCheckBox = qt["QCheckBox"]
    QGroupBox = qt["QGroupBox"]
    QListWidget = qt["QListWidget"]
    QListWidgetItem = qt["QListWidgetItem"]
    QFileDialog = qt["QFileDialog"]
    QGraphicsScene = qt["QGraphicsScene"]
    QGraphicsView = qt["QGraphicsView"]
    QGraphicsRectItem = qt["QGraphicsRectItem"]
    QGraphicsPixmapItem = qt["QGraphicsPixmapItem"]
    QGraphicsTextItem = qt["QGraphicsTextItem"]
    QPlainTextEdit = qt["QPlainTextEdit"]
    QTableWidget = qt["QTableWidget"]
    QTableWidgetItem = qt["QTableWidgetItem"]
    QHeaderView = qt["QHeaderView"]
    QScrollArea = qt["QScrollArea"]

    SCALE = 4  # zoom factor for the captured image (small captures → readable)

    class DebuggerWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(
                "Signature Annotator Debugger — region2 detector internals"
            )
            self.setMinimumSize(1500, 900)
            self.setStyleSheet(f"background: {BG}; color: {FG};")

            self._folder: Optional[Path] = folder
            self._files: list[Path] = []
            self._current_index: int = -1
            self._diag: Optional[dict] = None
            # Path of the currently-loaded PNG (used to read the
            # cap_*.json sidecar's ground-truth ``value`` field for
            # the segmenter readout's GT match column).
            self._png_path: Optional[Path] = None

            # Layer items (re-built each time we render a diagnostic).
            self._scene = QGraphicsScene(self)
            self._scene.setBackgroundBrush(QColor("#0a0a0a"))
            self._image_item: Optional[QGraphicsPixmapItem] = None
            self._layer_items: dict[str, list[Any]] = {}
            self._layer_visible: dict[str, bool] = {
                "image": True,
                "warm_mask": False,
                "cyan_mask": False,
                "ncc_candidates": True,
                "voter_decisions": True,
                "position_prior": True,
                "pill_final": True,
                "icon_final": True,
                "value_final": True,
                # New-architecture layers — default OFF for the
                # individual primaries (keeps screen uncluttered),
                # default ON for the consensus (the user mostly cares
                # about whether the new primary path agreed).
                "rgb_ncc_peak": False,
                "geometry_primary": False,
                "localize_icon_consensus": True,
                # Per-glyph segmenter spans — default ON because
                # diagnosing segmenter mis-alignment is the primary
                # use case post the recent region2 segmenter fix.
                "glyph_spans": True,
                # Comma anchor — default ON. The comma column is
                # the X-axis anchor every digit's position is
                # measured from, so visualizing it directly is the
                # most useful diagnostic for diagnosing leading-
                # digit clipping.
                "comma_anchor": True,
            }

            self._build_ui()

            # Hydrate.
            if folder is not None and folder.is_dir():
                self._set_folder(folder)
            if png is not None and png.is_file():
                self._open_file(png)
            elif initial_diag is not None:
                self._show_diag(initial_diag)
            else:
                # Fall through — user can pick from a file dialog.
                pass

        # ── UI ──────────────────────────────────────────────────────
        def _build_ui(self) -> None:
            central = QWidget(self)
            self.setCentralWidget(central)
            outer = QVBoxLayout(central)
            outer.setContentsMargins(8, 8, 8, 8)

            # Top toolbar
            top = QHBoxLayout()
            self._open_file_btn = QPushButton("Open PNG…")
            self._open_file_btn.clicked.connect(self._open_file_dialog)
            self._open_folder_btn = QPushButton("Open folder…")
            self._open_folder_btn.clicked.connect(self._open_folder_dialog)
            self._prev_btn = QPushButton("◀ Prev")
            self._prev_btn.clicked.connect(lambda: self._step(-1))
            self._next_btn = QPushButton("Next ▶")
            self._next_btn.clicked.connect(lambda: self._step(+1))
            self._reload_btn = QPushButton("Re-run detectors")
            self._reload_btn.clicked.connect(self._rerun)
            self._file_label = QLabel("No file loaded")
            self._file_label.setStyleSheet(f"color: {DIM};")
            for w in (
                self._open_file_btn, self._open_folder_btn,
                self._prev_btn, self._next_btn, self._reload_btn,
            ):
                w.setStyleSheet(
                    f"background: {PANEL}; color: {FG}; padding: 4px 12px;"
                )
                top.addWidget(w)
            top.addWidget(self._file_label, 1)
            outer.addLayout(top)

            # Splitter — left = scene + log; right = readouts.
            mid_split = QSplitter(Qt.Orientation.Horizontal, central)
            outer.addWidget(mid_split, 1)

            # Left side — scene on top, log on bottom.
            left = QSplitter(Qt.Orientation.Vertical, mid_split)
            self._view = QGraphicsView(self._scene)
            self._view.setBackgroundBrush(QColor("#0a0a0a"))
            self._view.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            self._view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            left.addWidget(self._view)
            self._log = QPlainTextEdit()
            self._log.setReadOnly(True)
            self._log.setMaximumBlockCount(1000)
            self._log.setStyleSheet(
                f"background: {PANEL}; color: {FG}; "
                f"font-family: Consolas; font-size: 10pt;"
            )
            left.addWidget(self._log)
            left.setSizes([700, 220])
            mid_split.addWidget(left)

            # Right side — readouts.
            right = QWidget()
            right_lay = QVBoxLayout(right)
            right_lay.setContentsMargins(4, 4, 4, 4)
            mid_split.addWidget(right)
            mid_split.setSizes([1000, 500])

            # Layer toggles
            tog_box = QGroupBox("Layers")
            tog_box.setStyleSheet(f"QGroupBox {{ color: {FG}; }}")
            tog_lay = QVBoxLayout(tog_box)
            self._toggle_widgets: dict[str, Any] = {}
            for key, label in [
                ("warm_mask", "HSV warm mask (yellow tint)"),
                ("cyan_mask", "HSV cyan mask (cyan tint)"),
                ("ncc_candidates", "NCC candidates (color-coded by scale)"),
                ("voter_decisions", "Voter decisions (green=accept, red=reject)"),
                ("position_prior", "Position prior threshold (dashed line)"),
                ("pill_final", "Final pill bbox (cyan)"),
                ("icon_final", "Final icon bbox (yellow)"),
                ("value_final", "Final value bbox (purple)"),
                # New-architecture layers (RGB-NCC + localize_icon).
                ("rgb_ncc_peak", "RGB NCC peak (magenta)"),
                ("geometry_primary", "Geometry primary [whole-image] (green)"),
                ("localize_icon_consensus", "localize_icon consensus (orange)"),
                # Per-glyph segmenter output — default ON so the most
                # useful view for diagnosing segmenter mis-alignment is
                # visible without an extra click. User can untoggle if
                # they want a cleaner view.
                ("glyph_spans", "Glyph spans (column projection)"),
                # Proportional segmenter overlay — drawn in magenta
                # to contrast with the column-projection cyan/yellow.
                # Default ON so the side-by-side comparison is
                # immediately visible.
                ("proportional_spans", "Glyph spans (proportional)"),
                # Comma anchor — drawn as a bright magenta-pink dashed
                # rectangle on the value crop so the user can verify
                # the X-axis anchor that drives the proportional
                # layout. Default ON.
                ("comma_anchor", "Comma anchor (find_comma_voted)"),
            ]:
                cb = QCheckBox(label)
                cb.setChecked(self._layer_visible.get(key, True))
                cb.setStyleSheet(f"color: {FG};")
                cb.toggled.connect(lambda v, k=key: self._set_layer_visible(k, v))
                tog_lay.addWidget(cb)
                self._toggle_widgets[key] = cb
            right_lay.addWidget(tog_box)

            # Pill summary
            self._pill_summary = QLabel("Pill: —")
            self._pill_summary.setWordWrap(True)
            self._pill_summary.setStyleSheet(
                f"background: {PANEL}; color: {FG}; padding: 6px;"
            )
            right_lay.addWidget(self._pill_summary)

            # NCC summary
            self._ncc_summary = QLabel("NCC: —")
            self._ncc_summary.setWordWrap(True)
            self._ncc_summary.setStyleSheet(
                f"background: {PANEL}; color: {FG}; padding: 6px;"
            )
            right_lay.addWidget(self._ncc_summary)

            # Voter table
            voter_box = QGroupBox("Voter decisions per candidate")
            voter_box.setStyleSheet(f"QGroupBox {{ color: {FG}; }}")
            voter_lay = QVBoxLayout(voter_box)
            self._voter_table = QTableWidget()
            self._voter_table.setColumnCount(11)
            self._voter_table.setHorizontalHeaderLabels([
                "x1", "y1", "w", "h", "tw", "ncc",
                "geom", "contour", "rgb@", "gray@", "decision",
            ])
            hdr = self._voter_table.horizontalHeader()
            hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self._voter_table.setStyleSheet(
                f"QTableWidget {{ background: {PANEL}; color: {FG}; "
                f"gridline-color: #444; }} "
                f"QHeaderView::section {{ background: #222; color: {ACCENT}; "
                f"padding: 4px; }}"
            )
            voter_lay.addWidget(self._voter_table)
            right_lay.addWidget(voter_box, 1)

            # ── New-architecture readouts ─────────────────────────
            # RGB NCC summary (magenta)
            self._rgb_ncc_summary = QLabel("RGB NCC: —")
            self._rgb_ncc_summary.setWordWrap(True)
            self._rgb_ncc_summary.setStyleSheet(
                f"background: {PANEL}; color: {FG}; padding: 6px; "
                f"border-left: 3px solid rgb"
                f"({RGB_NCC_COLOR[0]},{RGB_NCC_COLOR[1]},{RGB_NCC_COLOR[2]});"
            )
            right_lay.addWidget(self._rgb_ncc_summary)

            # Geometry primary (whole-image) summary (green)
            self._geom_primary_summary = QLabel("Geometry (whole-image): —")
            self._geom_primary_summary.setWordWrap(True)
            self._geom_primary_summary.setStyleSheet(
                f"background: {PANEL}; color: {FG}; padding: 6px; "
                f"border-left: 3px solid rgb"
                f"({GEOM_PRIMARY_COLOR[0]},{GEOM_PRIMARY_COLOR[1]},"
                f"{GEOM_PRIMARY_COLOR[2]});"
            )
            right_lay.addWidget(self._geom_primary_summary)

            # localize_icon consensus summary (orange)
            self._localize_icon_summary = QLabel("localize_icon: —")
            self._localize_icon_summary.setWordWrap(True)
            self._localize_icon_summary.setStyleSheet(
                f"background: {PANEL}; color: {FG}; padding: 6px; "
                f"border-left: 3px solid rgb"
                f"({LOCALIZE_ICON_COLOR[0]},{LOCALIZE_ICON_COLOR[1]},"
                f"{LOCALIZE_ICON_COLOR[2]});"
            )
            right_lay.addWidget(self._localize_icon_summary)

            # Segmenter output summary (per-span readout). Border
            # color matches the first span's palette entry so the eye
            # naturally connects this readout to the on-image overlay.
            self._segmenter_summary = QLabel("Segmenter output: —")
            self._segmenter_summary.setWordWrap(True)
            _seg_border = GLYPH_SPAN_COLORS[0]
            self._segmenter_summary.setStyleSheet(
                f"background: {PANEL}; color: {FG}; padding: 6px; "
                f"font-family: Consolas; font-size: 9pt; "
                f"border-left: 3px solid rgb"
                f"({_seg_border[0]},{_seg_border[1]},{_seg_border[2]});"
            )
            right_lay.addWidget(self._segmenter_summary)

            # Proportional segmenter readout. Magenta border to mirror
            # the on-image overlay color so the user can visually map
            # the panel back to the bboxes. Shows both 4-digit and
            # 5-digit hypothesis scores side by side, which one was
            # picked, and the per-digit reads.
            self._proportional_summary = QLabel("Proportional segmenter: —")
            self._proportional_summary.setWordWrap(True)
            self._proportional_summary.setStyleSheet(
                f"background: {PANEL}; color: {FG}; padding: 6px; "
                f"font-family: Consolas; font-size: 9pt; "
                f"border-left: 3px solid rgb"
                f"({PROPORTIONAL_SPAN_COLOR[0]},"
                f"{PROPORTIONAL_SPAN_COLOR[1]},"
                f"{PROPORTIONAL_SPAN_COLOR[2]});"
            )
            right_lay.addWidget(self._proportional_summary)

            # Comma-anchor readout. Magenta-pink border to mirror the
            # on-image dashed rectangle. Shows the voted result, both
            # individual polarities, agreement state, and the per-
            # check breakdown (which structural test passed/failed).
            self._comma_anchor_summary = QLabel("Comma anchor: —")
            self._comma_anchor_summary.setWordWrap(True)
            self._comma_anchor_summary.setStyleSheet(
                f"background: {PANEL}; color: {FG}; padding: 6px; "
                f"font-family: Consolas; font-size: 9pt; "
                f"border-left: 3px solid rgb"
                f"({COMMA_ANCHOR_COLOR[0]},"
                f"{COMMA_ANCHOR_COLOR[1]},"
                f"{COMMA_ANCHOR_COLOR[2]});"
            )
            right_lay.addWidget(self._comma_anchor_summary)

            # Final-decisions summary
            self._final_summary = QLabel("Final: —")
            self._final_summary.setWordWrap(True)
            self._final_summary.setStyleSheet(
                f"background: {PANEL}; color: {FG}; padding: 6px;"
            )
            right_lay.addWidget(self._final_summary)

        # ── State helpers ───────────────────────────────────────────
        def _set_folder(self, folder: Path) -> None:
            self._folder = folder
            self._files = sorted(folder.glob("*.png"))
            self._current_index = -1
            if self._files:
                self._current_index = 0
                self._open_file(self._files[0])

        def _step(self, delta: int) -> None:
            if not self._files:
                return
            self._current_index = max(
                0, min(len(self._files) - 1, self._current_index + delta),
            )
            self._open_file(self._files[self._current_index])

        def _open_file_dialog(self) -> None:
            start_dir = str(self._folder) if self._folder else str(DEFAULT_REGION2_FOLDER)
            picked, _ = QFileDialog.getOpenFileName(
                self, "Open region2 capture", start_dir, "PNG (*.png)",
            )
            if picked:
                self._open_file(Path(picked))

        def _open_folder_dialog(self) -> None:
            start_dir = str(self._folder) if self._folder else str(DEFAULT_REGION2_FOLDER)
            picked = QFileDialog.getExistingDirectory(
                self, "Open folder of region2 captures", start_dir,
            )
            if picked:
                self._set_folder(Path(picked))

        def _open_file(self, path: Path) -> None:
            try:
                pil = Image.open(path).convert("RGB")
            except Exception as exc:
                self._log.appendPlainText(f"[ERROR] Could not open {path}: {exc}")
                return
            # Try to read sibling .boxes.json for hud_bbox.
            hud_bbox = None
            try:
                jpath = path.with_suffix("").with_suffix(".boxes.json")
                if jpath.is_file():
                    j = json.loads(jpath.read_text(encoding="utf-8"))
                    h = j.get("hud_bbox")
                    if h:
                        hud_bbox = (
                            int(h["x"]), int(h["y"]), int(h["w"]), int(h["h"]),
                        )
            except Exception:
                pass
            t0 = time.monotonic()
            diag = run_full_diagnostics(pil, hud_bbox=hud_bbox)
            dt = (time.monotonic() - t0) * 1000.0
            self._log.appendPlainText(
                f"---- Loaded {path.name} (detectors {dt:.0f} ms) ----"
            )
            self._diag = diag
            self._png_path = path
            self._file_label.setText(str(path))
            if path in self._files:
                self._current_index = self._files.index(path)
            self._show_diag(diag)

        def _rerun(self) -> None:
            if self._diag is None:
                return
            pil = self._diag.get("image")
            hud_bbox = self._diag.get("hud_bbox")
            if pil is None:
                return
            t0 = time.monotonic()
            diag = run_full_diagnostics(pil, hud_bbox=hud_bbox)
            self._log.appendPlainText(
                f"---- Re-ran ({(time.monotonic() - t0)*1000.0:.0f} ms) ----"
            )
            self._diag = diag
            self._show_diag(diag)

        def _set_layer_visible(self, key: str, visible: bool) -> None:
            self._layer_visible[key] = visible
            for it in self._layer_items.get(key, []):
                it.setVisible(visible)

        # ── Rendering ───────────────────────────────────────────────
        def _show_diag(self, diag: dict) -> None:
            self._scene.clear()
            self._layer_items = {k: [] for k in self._layer_visible.keys()}

            pil = diag.get("image")
            if pil is None:
                return

            # Base image
            base_pix = _pil_to_qpixmap(pil, qt)
            self._image_item = self._scene.addPixmap(base_pix)
            self._image_item.setScale(SCALE)
            self._scene.setSceneRect(0, 0, pil.width * SCALE, pil.height * SCALE)
            self._layer_items["image"] = [self._image_item]

            def _add_rect(x1, y1, x2, y2, color_rgb,
                          width_px: int = 1, dashed: bool = False,
                          fill_alpha: int = 0, layer: str = "ncc_candidates"):
                pen = QPen(QColor(*color_rgb))
                pen.setWidth(width_px)
                pen.setCosmetic(True)
                if dashed:
                    pen.setStyle(Qt.PenStyle.DashLine)
                rect = QGraphicsRectItem(
                    x1 * SCALE, y1 * SCALE,
                    (x2 - x1) * SCALE, (y2 - y1) * SCALE,
                )
                rect.setPen(pen)
                if fill_alpha > 0:
                    fill_color = QColor(*color_rgb)
                    fill_color.setAlpha(fill_alpha)
                    rect.setBrush(QBrush(fill_color))
                else:
                    rect.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                rect.setVisible(self._layer_visible.get(layer, True))
                self._scene.addItem(rect)
                self._layer_items.setdefault(layer, []).append(rect)
                return rect

            def _add_text(x, y, text, color_rgb, size: int = 8, layer: str = "ncc_candidates"):
                tx = QGraphicsTextItem(text)
                tx.setDefaultTextColor(QColor(*color_rgb))
                f = QFont("Consolas", size)
                f.setBold(True)
                tx.setFont(f)
                tx.setPos(x * SCALE, y * SCALE)
                tx.setVisible(self._layer_visible.get(layer, True))
                self._scene.addItem(tx)
                self._layer_items.setdefault(layer, []).append(tx)
                return tx

            # HSV overlays
            warm = diag.get("warm_mask")
            if warm is not None:
                px = _mask_to_pixmap(warm, WARM_OVERLAY_COLOR, qt)
                it = self._scene.addPixmap(px)
                it.setScale(SCALE)
                it.setZValue(0.5)
                it.setVisible(self._layer_visible.get("warm_mask", False))
                self._layer_items["warm_mask"].append(it)

            cyan = diag.get("cyan_mask")
            if cyan is not None:
                px = _mask_to_pixmap(cyan, CYAN_OVERLAY_COLOR, qt)
                it = self._scene.addPixmap(px)
                it.setScale(SCALE)
                it.setZValue(0.6)
                it.setVisible(self._layer_visible.get("cyan_mask", False))
                self._layer_items["cyan_mask"].append(it)

            # NCC raw candidates — color-coded by template scale.
            for cand in diag.get("ncc_raw_candidates", []):
                col = cand["color"]
                _add_rect(
                    cand["x1"], cand["y1"], cand["x2"], cand["y2"],
                    col, width_px=1, layer="ncc_candidates",
                )
                _add_text(
                    cand["x1"], max(0, cand["y1"] - 9),
                    f"tw={cand['tw']} s={cand['score']:.2f}",
                    col, size=6, layer="ncc_candidates",
                )

            # Voter decisions per candidate (overlays on top of NCC).
            for row in diag.get("voter_table", []):
                col = ACCEPT_COLOR if row["accepted"] else REJECT_COLOR
                _add_rect(
                    row["x1"], row["y1"], row["x2"], row["y2"],
                    col, width_px=2, layer="voter_decisions",
                )
                short_path = (row["decision_path"] or "").replace(
                    "primaries_", "p_",
                ).replace("gray_cnn=", "g=").replace("rgb_cnn=", "r=")
                # Compose a one-line reason.
                if row["accepted"]:
                    reason = f"OK {short_path}"
                else:
                    reason = f"NO {short_path}"
                _add_text(
                    row["x1"], max(0, row["y2"] + 1),
                    reason[:46], col, size=6, layer="voter_decisions",
                )

            # Position-prior threshold line.
            crop_w = pil.width
            x_thresh = int(crop_w * _POSITION_PRIOR_FRAC)
            if "icon_position_prior_threshold_x" in diag:
                x_thresh = diag["icon_position_prior_threshold_x"] or x_thresh
            line = QGraphicsRectItem(
                x_thresh * SCALE, 0,
                1 * SCALE, pil.height * SCALE,
            )
            pen = QPen(QColor(*PRIOR_COLOR))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            pen.setWidth(2)
            line.setPen(pen)
            line.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            line.setVisible(self._layer_visible.get("position_prior", True))
            self._scene.addItem(line)
            self._layer_items["position_prior"].append(line)
            ttx = _add_text(
                x_thresh + 1, 1,
                f"pos_prior x={x_thresh} (30%)",
                PRIOR_COLOR, size=7, layer="position_prior",
            )

            # Final pill
            pill = diag.get("pill")
            if pill is not None:
                px1, py1, px2, py2 = pill["bbox"]
                _add_rect(
                    px1, py1, px2, py2, PILL_COLOR,
                    width_px=3, layer="pill_final",
                )
                _add_text(
                    px1, max(0, py1 - 11),
                    f"pill conf={pill['score']:.2f}",
                    PILL_COLOR, size=7, layer="pill_final",
                )

            # Final icon (after find_icon — pre position prior)
            icon = diag.get("icon_final")
            if icon is not None:
                ix1, iy1, ix2, iy2 = icon["bbox"]
                # If the position prior would reject this, draw it red
                # dashed; otherwise yellow solid.
                pos_ok = diag.get("icon_position_prior_passed", True)
                col_ok = diag.get("icon_color_prior_passed", True)
                final_color = ICON_COLOR if (pos_ok and col_ok) else REJECT_COLOR
                _add_rect(
                    ix1, iy1, ix2, iy2, final_color,
                    width_px=3, dashed=not (pos_ok and col_ok),
                    layer="icon_final",
                )
                tag = f"icon ncc={icon['score']:.2f}"
                if not pos_ok:
                    tag += " (POS_PRIOR_REJECT)"
                if not col_ok:
                    tag += " (COLOR_PRIOR_REJECT)"
                _add_text(
                    ix1, max(0, iy1 - 11),
                    tag, final_color, size=7, layer="icon_final",
                )

            # Final value (digit cluster)
            dc = diag.get("digit_cluster")
            if dc is not None:
                vx1, vy1, vx2, vy2 = dc["bbox"]
                _add_rect(
                    vx1, vy1, vx2, vy2, VALUE_COLOR,
                    width_px=3, layer="value_final",
                )
                _add_text(
                    vx1, max(0, vy1 - 11),
                    "value (digit cluster)",
                    VALUE_COLOR, size=7, layer="value_final",
                )

            # ── New-architecture layers (RGB-NCC + localize_icon) ──
            # Z-ordering: these go ON TOP of the legacy layers
            # because they represent the new primary path. The
            # consensus rectangle goes last (highest z) with the
            # thickest border so it's the visually dominant signal
            # when both individual primaries agreed.
            #
            # We use setZValue on the QGraphicsRectItem so that
            # toggling layers above doesn't push them under the
            # image.

            # 1. RGB NCC peak — magenta solid rectangle.
            rgb_ncc = diag.get("rgb_ncc_result")
            if rgb_ncc is not None:
                rx1, ry1, rx2, ry2 = rgb_ncc["bbox_xyxy"]
                rect = _add_rect(
                    rx1, ry1, rx2, ry2, RGB_NCC_COLOR,
                    width_px=3, layer="rgb_ncc_peak",
                )
                rect.setZValue(10.0)
                tlabel = _add_text(
                    rx1, max(0, ry1 - 11),
                    f"RGB NCC: score={rgb_ncc['score']:.2f}, "
                    f"scale={rgb_ncc['scale']:.2g}",
                    RGB_NCC_COLOR, size=7, layer="rgb_ncc_peak",
                )
                tlabel.setZValue(10.1)

            # 2. Geometry primary (whole-image) — bright green.
            geom_primary = diag.get("geometry_primary_result")
            if geom_primary is not None:
                gx1, gy1, gx2, gy2 = geom_primary["bbox_xyxy"]
                rect = _add_rect(
                    gx1, gy1, gx2, gy2, GEOM_PRIMARY_COLOR,
                    width_px=3, layer="geometry_primary",
                )
                rect.setZValue(10.5)
                tlabel = _add_text(
                    gx1, max(0, gy1 - 11),
                    f"geometry: score={geom_primary['score']}/6, "
                    f"conf={geom_primary['confidence']:.2f}",
                    GEOM_PRIMARY_COLOR, size=7, layer="geometry_primary",
                )
                tlabel.setZValue(10.6)

            # 3. localize_icon consensus — bright orange, thickest
            # border, on top.
            localize = diag.get("localize_icon_result")
            no_consensus_reason = diag.get("localize_icon_no_consensus_reason")
            if localize is not None:
                lx1, ly1, lx2, ly2 = localize["bbox_xyxy"]
                rect = _add_rect(
                    lx1, ly1, lx2, ly2, LOCALIZE_ICON_COLOR,
                    width_px=4, layer="localize_icon_consensus",
                )
                rect.setZValue(11.0)
                tlabel = _add_text(
                    lx1, max(0, ly1 - 11),
                    f"localize_icon: {localize['detector']}, "
                    f"score={localize['score']:.2f}",
                    LOCALIZE_ICON_COLOR, size=7,
                    layer="localize_icon_consensus",
                )
                tlabel.setZValue(11.1)
            elif no_consensus_reason is not None:
                # Render a corner badge so the user can see at a
                # glance that consensus didn't fire. Anchored to top-
                # left of the image; small so it doesn't occlude.
                tlabel = _add_text(
                    1, 1,
                    "localize_icon: NO CONSENSUS",
                    LOCALIZE_ICON_COLOR, size=8,
                    layer="localize_icon_consensus",
                )
                tlabel.setZValue(11.1)

            # ── Glyph spans (segmenter output) ─────────────────────
            # Per-span thin rectangles in unique cycling colors with
            # ``<idx>: '<class>' <conf>`` labels above each. Placed at
            # z=6 — below ``localize_icon_consensus`` (z=11) and the
            # other new-architecture layers (z=10, z=10.5), but above
            # NCC candidates / voter decisions / mask overlays.
            seg = diag.get("glyph_segmentation")
            if seg is not None and seg.get("spans"):
                for span in seg["spans"]:
                    sx1, sy1, sx2, sy2 = span["bbox_image"]
                    col = span["color"]
                    rect = _add_rect(
                        sx1, sy1, sx2, sy2, col,
                        width_px=1, layer="glyph_spans",
                    )
                    rect.setZValue(6.0)
                    label_text = (
                        f"{span['idx']}: '{span['classification']}' "
                        f"{span['confidence']:.2f}"
                    )
                    tlabel = _add_text(
                        sx1, max(0, sy1 - 9),
                        label_text, col, size=6,
                        layer="glyph_spans",
                    )
                    tlabel.setZValue(6.1)
            elif seg is not None and (
                seg.get("error") or seg.get("reason")
            ):
                # Show a corner badge so the user knows the segmenter
                # didn't run / failed. Doesn't occlude any other layer.
                msg = seg.get("error") or seg.get("reason") or "unknown"
                tlabel = _add_text(
                    1, 11,
                    f"segmenter: {str(msg)[:60]}",
                    GLYPH_SPAN_COLORS[0], size=7,
                    layer="glyph_spans",
                )
                tlabel.setZValue(6.1)

            # ── Proportional segmenter spans (parallel layer) ──────
            # Drawn in magenta on top of the column-projection layer,
            # at z=6.5 so the proportional bboxes sit above the
            # column-projection ones (the user usually wants to see
            # "where the proportional segmenter says the digits are"
            # against the underlying image, not against the projection
            # spans). Comma slot uses a slightly darker magenta to
            # distinguish it from the digit slots.
            prop = diag.get("proportional_segmentation")
            if prop is not None and prop.get("digits"):
                # Render the structurally-known label below each
                # bbox (e.g. "[L0]" for left-of-comma slot 0). This
                # makes it explicit which slot each bbox represents
                # — important because proportional layout assigns
                # slots by position, not by detected blob.
                for d in prop["digits"]:
                    px1, py1, px2, py2 = d["bbox_image"]
                    if d.get("is_comma"):
                        col = PROPORTIONAL_COMMA_COLOR
                        layer_lbl = "comma"
                    else:
                        col = PROPORTIONAL_SPAN_COLOR
                        cls = d.get("classification", "?")
                        conf = float(d.get("confidence", 0.0))
                        layer_lbl = f"{d['idx']}:'{cls}' {conf:.2f}"
                    rect = _add_rect(
                        px1, py1, px2, py2, col,
                        width_px=2, layer="proportional_spans",
                    )
                    rect.setZValue(6.5)
                    tlabel = _add_text(
                        px1, max(0, py1 - 19),
                        layer_lbl, col, size=6,
                        layer="proportional_spans",
                    )
                    tlabel.setZValue(6.6)
            elif prop is not None and (
                prop.get("error") or prop.get("reason")
            ):
                # Corner badge on segmenter failure — anchored below
                # the column-projection error message badge so they
                # don't overlap.
                msg = prop.get("error") or prop.get("reason") or "unknown"
                tlabel = _add_text(
                    1, 21,
                    f"proportional: {str(msg)[:60]}",
                    PROPORTIONAL_SPAN_COLOR, size=7,
                    layer="proportional_spans",
                )
                tlabel.setZValue(6.6)

            # ── Comma anchor (find_comma_voted) ─────────────────────
            # Bright magenta-pink DASHED rectangle on the comma's
            # bbox, plus a small label above with confidence and
            # check count. Z=7 (above the proportional spans at 6.5
            # so it stands out as the structural anchor).
            comma_anchor = diag.get("comma_anchor")
            if comma_anchor is not None and comma_anchor.get("found"):
                cx1, cy1, cx2, cy2 = comma_anchor["bbox_image"]
                rect = _add_rect(
                    cx1, cy1, cx2, cy2, COMMA_ANCHOR_COLOR,
                    width_px=2, dashed=True, layer="comma_anchor",
                )
                rect.setZValue(7.0)
                conf = float(comma_anchor.get("confidence", 0.0))
                ck = comma_anchor.get("checks") or {}
                n_passed = sum(1 for v in ck.values() if v)
                tlabel = _add_text(
                    cx1, max(0, cy1 - 11),
                    f"Comma: conf={conf:.2f} ({n_passed}/6)",
                    COMMA_ANCHOR_COLOR, size=7, layer="comma_anchor",
                )
                tlabel.setZValue(7.1)
            elif comma_anchor is not None and (
                comma_anchor.get("error") or comma_anchor.get("reason")
            ):
                # Corner badge on detection miss — placed below the
                # other diagnostic badges so they don't overlap.
                msg = (
                    comma_anchor.get("error")
                    or comma_anchor.get("reason")
                    or "unknown"
                )
                tlabel = _add_text(
                    1, 31,
                    f"comma: {str(msg)[:60]}",
                    COMMA_ANCHOR_COLOR, size=7, layer="comma_anchor",
                )
                tlabel.setZValue(7.1)

            # ── Right panel readouts ───────────────────────────────
            self._render_readouts(diag)

            # ── Bottom log ─────────────────────────────────────────
            for line in diag.get("logs", []):
                self._log.appendPlainText(line)

            self._view.fitInView(
                self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio,
            )

        def _render_readouts(self, diag: dict) -> None:
            # Pill summary
            pill = diag.get("pill")
            if pill is None:
                self._pill_summary.setText("Pill: NOT FOUND (find_hud_panel returned None)")
            else:
                self._pill_summary.setText(textwrap.dedent(f"""\
                    Pill: bbox={pill['bbox']} (xywh={pill['xywh']})
                    confidence={pill['score']:.3f}
                    n_chrome_pixels={pill['n_chrome_pixels']}
                    n_candidates_considered={pill['n_candidates_considered']}
                    """).strip())

            # NCC summary
            cands = diag.get("ncc_raw_candidates", [])
            post = diag.get("ncc_post_voter", [])
            if not cands:
                self._ncc_summary.setText("NCC: 0 candidates")
            else:
                top10 = sorted(cands, key=lambda c: -c["score"])[:10]
                lines = [
                    f"NCC raw candidates: {len(cands)} (post-voter: {len(post)})",
                    "Top by score (x1,y1,x2,y2 score tw):",
                ]
                for c in top10:
                    lines.append(
                        f"  ({c['x1']:3d},{c['y1']:3d},{c['x2']:3d},"
                        f"{c['y2']:3d}) s={c['score']:.2f} tw={c['tw']:2d}"
                    )
                self._ncc_summary.setText("\n".join(lines))

            # Voter table
            rows = diag.get("voter_table", [])
            self._voter_table.setRowCount(len(rows))
            for r_idx, row in enumerate(rows):
                cells = [
                    str(row["x1"]), str(row["y1"]), str(row["w"]),
                    str(row["h"]), str(row["tw"]),
                    f"{row['score']:.2f}",
                    row["geom_vote"], row["contour_vote"],
                    row["rgb_cnn_vote"]
                    + (
                        f" ({row['rgb_at_prob']:.2f})"
                        if row["rgb_at_prob"] is not None else ""
                    ),
                    row["gray_cnn_vote"]
                    + (
                        f" ({row['gray_at_prob']:.2f})"
                        if row["gray_at_prob"] is not None else ""
                    ),
                    ("OK " if row["accepted"] else "NO ")
                    + (row["decision_path"] or ""),
                ]
                for c_idx, txt in enumerate(cells):
                    item = QTableWidgetItem(txt)
                    if c_idx == 10:
                        item.setForeground(
                            QBrush(QColor(*ACCEPT_COLOR))
                            if row["accepted"]
                            else QBrush(QColor(*REJECT_COLOR))
                        )
                    self._voter_table.setItem(r_idx, c_idx, item)

            # ── New-architecture readouts ──────────────────────────
            self._render_rgb_ncc_readout(diag)
            self._render_geometry_primary_readout(diag)
            self._render_localize_icon_readout(diag)
            self._render_segmenter_readout(diag)
            self._render_proportional_readout(diag)
            self._render_comma_anchor_readout(diag)

            # Final-decisions summary
            icon = diag.get("icon_final")
            dc = diag.get("digit_cluster")
            pos_ok = diag.get("icon_position_prior_passed", True)
            col_ok = diag.get("icon_color_prior_passed", True)
            warm_f = diag.get("icon_color_warm_frac")
            cool_f = diag.get("icon_color_cool_frac")
            x_thr = diag.get("icon_position_prior_threshold_x")
            lines = []
            if icon is not None:
                lines.append(
                    f"find_icon picked: {icon['bbox']} score={icon['score']:.2f}"
                )
            else:
                lines.append("find_icon: returned None")
            if x_thr is not None:
                lines.append(
                    f"position_prior x_threshold={x_thr} "
                    f"(passed={pos_ok})"
                )
            if warm_f is not None:
                lines.append(
                    f"color_prior warm_frac={warm_f:.2f} "
                    f"cool_frac={cool_f:.2f} (passed={col_ok})"
                )
            if dc is not None:
                lines.append(f"digit_cluster bbox={dc['bbox']}")
            else:
                lines.append("digit_cluster: None")
            errs = diag.get("errors", [])
            if errs:
                lines.append("Errors:")
                for stage, msg in errs:
                    if stage.endswith("_tb"):
                        continue
                    lines.append(f"  [{stage}] {msg}")
            self._final_summary.setText("\n".join(lines))

        # ── New-architecture readout helpers ────────────────────────
        def _render_rgb_ncc_readout(self, diag: dict) -> None:
            if not diag.get("rgb_ncc_available", False):
                self._rgb_ncc_summary.setText(
                    "─── RGB NCC ─────────────────────────────────────\n"
                    "MODULE NOT AVAILABLE — find_icon_rgb_ncc could not "
                    "be imported. Falling back to legacy NCC."
                )
                return

            tinfo = diag.get("rgb_ncc_template_info") or {}
            n_t = int(tinfo.get("n_templates", 0))
            real_dir = tinfo.get("real_dir", "")
            real_dir_short = real_dir.split(os.sep)[-1] if real_dir else ""

            res = diag.get("rgb_ncc_result")
            if res is None:
                self._rgb_ncc_summary.setText(
                    "─── RGB NCC ─────────────────────────────────────\n"
                    f"Templates loaded: {n_t} real, "
                    f"from {real_dir_short or '<unknown>'}\n"
                    "Peak found: NO\n"
                    "Bbox: —\n"
                    "Score: —"
                )
                return

            per = res["per_channel_scores"]
            wts = res["weights"]
            combined = (
                wts[0] * per[0] + wts[1] * per[1] + wts[2] * per[2]
            )
            x, y, w, h = res["bbox"]
            self._rgb_ncc_summary.setText(textwrap.dedent(f"""\
                ─── RGB NCC ─────────────────────────────────────
                Templates loaded: {n_t} real, from {real_dir_short or '<unknown>'}
                Peak found: yes
                Bbox: ({x}, {y}, {w}, {h})
                Score: {res['score']:.3f}
                Scale: {res['scale']:.2g}
                Per-channel scores: R={per[0]:.2f}, G={per[1]:.2f}, B={per[2]:.2f}
                Combined (R-heavy): {combined:.3f}
                Position prior (≤40% width): {'PASS' if res['position_prior_pass'] else 'FAIL'}
                Template used: {res['template_used']}
                Candidates above thresh: {res['n_candidates_above_thresh']}
                """).rstrip())

        def _render_geometry_primary_readout(self, diag: dict) -> None:
            if not diag.get("geometry_primary_available", False):
                self._geom_primary_summary.setText(
                    "─── Geometry (whole-image primary) ──────────\n"
                    "MODULE NOT AVAILABLE — find_icon_by_geometry "
                    "could not be imported."
                )
                return

            res = diag.get("geometry_primary_result")
            if res is None:
                self._geom_primary_summary.setText(
                    "─── Geometry (whole-image primary mode) ─────────\n"
                    "Bbox: —\n"
                    "Returned None (no warm-mask component passed the "
                    "structural checks)."
                )
                return

            checks = res["checks"]
            check_lines = []
            short_names = {
                "color_warm": "color_warm",
                "two_components": "two_components",
                "teardrop_has_hole": "teardrop_has_hole",
                "oval_below_teardrop": "oval_below",
                "oval_has_notch": "oval_has_notch",
                "aspect_ratio_global": "aspect_ratio",
            }
            cur_line = []
            for k, label in short_names.items():
                glyph = "PASS" if checks.get(k, False) else "FAIL"
                cur_line.append(f"{glyph} {label}")
                if len(cur_line) == 3:
                    check_lines.append(" ".join(cur_line))
                    cur_line = []
            if cur_line:
                check_lines.append(" ".join(cur_line))

            x, y, w, h = res["bbox"]
            tiny = " (tiny_mode)" if res.get("tiny_mode") else ""
            text = (
                "─── Geometry (whole-image primary mode) ─────────\n"
                f"Bbox: ({x}, {y}, {w}, {h})\n"
                f"Confidence: {res['confidence']:.2f}\n"
                f"Score: {res['score']}/6{tiny}\n"
                "Checks: " + ("\n        ".join(check_lines))
            )
            self._geom_primary_summary.setText(text)

        def _render_localize_icon_readout(self, diag: dict) -> None:
            if not diag.get("localize_icon_available", False):
                self._localize_icon_summary.setText(
                    "─── localize_icon (consensus) ───────────────────\n"
                    "MODULE NOT AVAILABLE — localize_icon could not "
                    "be imported."
                )
                self._localize_icon_summary.setStyleSheet(
                    f"background: {PANEL}; color: {FG}; padding: 6px; "
                    f"border-left: 3px solid {DIM};"
                )
                return

            res = diag.get("localize_icon_result")
            geom_res = diag.get("geometry_primary_result")
            ncc_res = diag.get("rgb_ncc_result")
            iou = diag.get("localize_icon_iou")
            geom_bb = geom_res["bbox"] if geom_res else None
            ncc_bb = ncc_res["bbox"] if ncc_res else None

            if res is not None:
                # Consensus fired.
                lx, ly, lw, lh = res["bbox"]
                iou_str = (
                    f"{iou:.2f}" if iou is not None else f"{res.get('iou', 0.0):.2f}"
                )
                text = textwrap.dedent(f"""\
                    ─── localize_icon (consensus) ───────────────────
                    Geometry bbox:  {geom_bb}  IoU vs RGB NCC = {iou_str}
                    RGB NCC bbox:   {ncc_bb}
                    Consensus result: ACCEPT — IoU {iou_str} > 0.4
                    Returned bbox: ({lx}, {ly}, {lw}, {lh})
                    Returned detector tag: {res['detector']}
                    Score: {res['score']:.2f}
                    """).rstrip()
                self._localize_icon_summary.setText(text)
                self._localize_icon_summary.setStyleSheet(
                    f"background: {PANEL}; color: {FG}; padding: 6px; "
                    f"border-left: 3px solid rgb"
                    f"({LOCALIZE_ICON_COLOR[0]},{LOCALIZE_ICON_COLOR[1]},"
                    f"{LOCALIZE_ICON_COLOR[2]});"
                )
            else:
                # No consensus — show prominently in red.
                reason = (
                    diag.get("localize_icon_no_consensus_reason")
                    or "unknown"
                )
                text = textwrap.dedent(f"""\
                    ─── localize_icon (consensus) ───────────────────
                    NO CONSENSUS — falling back to legacy NCC + voter
                    Reason: {reason}
                    """).rstrip()
                self._localize_icon_summary.setText(text)
                self._localize_icon_summary.setStyleSheet(
                    f"background: {PANEL}; color: {RED}; padding: 6px; "
                    f"border-left: 3px solid {RED};"
                )

        def _render_segmenter_readout(self, diag: dict) -> None:
            """Render the per-span segmenter table + GT comparison."""
            seg = diag.get("glyph_segmentation")
            crop_source = diag.get(
                "glyph_segmentation_crop_source", "unavailable",
            )
            gt_value = _read_gt_value(self._png_path)

            header = "─── Segmenter output ───────────────────────────"
            if seg is None:
                self._segmenter_summary.setText(
                    f"{header}\nValue crop unavailable — segmenter not run"
                )
                return
            if seg.get("error"):
                self._segmenter_summary.setText(
                    f"{header}\n"
                    f"Crop source: {crop_source}\n"
                    f"Segmenter failed: {seg['error']}"
                )
                return
            if seg.get("reason"):
                self._segmenter_summary.setText(
                    f"{header}\n"
                    f"Crop source: {crop_source}\n"
                    f"Value crop unavailable — segmenter not run "
                    f"({seg['reason']})"
                )
                return
            spans = seg.get("spans", [])
            if not spans:
                self._segmenter_summary.setText(
                    f"{header}\n"
                    f"Crop source: {crop_source}\n"
                    "No spans detected"
                )
                return

            # Compose readout. Match format from the spec.
            crop_box_str = (
                str(seg.get("crop_box")) if seg.get("crop_box") else "—"
            )
            n = len(spans)
            digits_only_gt: Optional[str] = None
            if gt_value is not None:
                digits_only_gt = "".join(
                    ch for ch in gt_value if ch.isdigit()
                )

            lines: list[str] = [
                header,
                f"Spans found:    {n}",
                f"Crop source:    {crop_source}  bbox={crop_box_str}",
            ]
            if gt_value is not None:
                lines.append(
                    f"GT (if known):  '{gt_value}' "
                    f"({len(digits_only_gt or '')} digits)"
                )
            lines.append("")

            # Build the per-span table. Whether we have a GT to compare
            # against drives whether the OK column appears.
            if digits_only_gt is not None:
                lines.append(
                    "  Idx | Bbox             | Class | Conf  | OK"
                )
                lines.append(
                    "  ----+------------------+-------+-------+-----"
                )
                for i, span in enumerate(spans):
                    bx, by, bw, bh = span["bbox_local"]
                    cls = span["classification"]
                    conf = span["confidence"]
                    expected = (
                        digits_only_gt[i]
                        if i < len(digits_only_gt) else None
                    )
                    ok_mark = (
                        "yes" if expected is not None and cls == expected
                        else ("no " if expected is not None else "?  ")
                    )
                    lines.append(
                        f"  {i:>3d} | ({bx:>3d},{by:>3d},{bw:>3d},"
                        f"{bh:>3d}) | '{cls:>1s}'   | "
                        f"{conf:.2f} | {ok_mark}"
                    )
            else:
                lines.append("  Idx | Bbox             | Class | Conf")
                lines.append("  ----+------------------+-------+------")
                for i, span in enumerate(spans):
                    bx, by, bw, bh = span["bbox_local"]
                    cls = span["classification"]
                    conf = span["confidence"]
                    lines.append(
                        f"  {i:>3d} | ({bx:>3d},{by:>3d},{bw:>3d},"
                        f"{bh:>3d}) | '{cls:>1s}'   | {conf:.2f}"
                    )

            composed = seg.get("string_composed", "")
            lines.append("")
            if digits_only_gt is not None:
                gt_match = (composed == digits_only_gt)
                match_mark = "yes" if gt_match else "no"
                lines.append(
                    f"String composed: '{composed}'    "
                    f"GT match: {match_mark}"
                )
            else:
                lines.append(f"String composed: '{composed}'")

            self._segmenter_summary.setText("\n".join(lines))

        def _render_proportional_readout(self, diag: dict) -> None:
            """Render the proportional segmenter's per-hypothesis table
            and the picked-winner readout. Mirrors the column-projection
            readout layout so the user can compare both at a glance.
            """
            prop = diag.get("proportional_segmentation")
            gt_value = _read_gt_value(self._png_path)
            digits_only_gt: Optional[str] = None
            if gt_value is not None:
                digits_only_gt = "".join(
                    ch for ch in gt_value if ch.isdigit()
                )

            header = "─── Proportional segmenter ──────────────────────"
            if prop is None:
                self._proportional_summary.setText(
                    f"{header}\nNot run — value bbox unavailable"
                )
                return
            if prop.get("error"):
                self._proportional_summary.setText(
                    f"{header}\nFailed: {prop['error']}"
                )
                return
            if prop.get("reason"):
                self._proportional_summary.setText(
                    f"{header}\nNot run: {prop['reason']}"
                )
                return

            n_digits = prop.get("n_digits")
            comma_pos = prop.get("comma_position")
            score = prop.get("confidence")
            hyps = prop.get("hypotheses", [])
            composed = prop.get("string_composed", "")

            lines: list[str] = [
                header,
                f"Picked: N={n_digits}-digit  comma_pos={comma_pos}  "
                f"composed={composed!r}  score={score:.3f}",
            ]
            if gt_value is not None:
                gt_match = (composed == digits_only_gt)
                lines.append(
                    f"GT:     '{gt_value}' "
                    f"({'MATCH' if gt_match else 'MISS'})"
                )
            lines.append("")
            lines.append("Hypothesis comparison:")
            lines.append(
                "  N=  composed   mean_conf  in_lex  used_anchor  score"
            )
            lines.append(
                "  --  ---------  ---------  ------  -----------  -----"
            )
            for h in hyps:
                lines.append(
                    f"  {h.get('n_digits'):>1}  "
                    f"'{str(h.get('composed','')):>5s}'    "
                    f"{float(h.get('mean_conf', 0.0)):>7.3f}    "
                    f"{'yes' if h.get('in_lexicon') else 'no ':>5s}    "
                    f"{'yes' if h.get('used_blob_centers') else 'no ':>9s}    "
                    f"{float(h.get('score', 0.0)):>5.3f}"
                )

            # Per-digit table
            digits = prop.get("digits", [])
            if digits:
                lines.append("")
                if digits_only_gt is not None:
                    lines.append(
                        "  Idx | Bbox(local)         | Class | Conf  | OK"
                    )
                    lines.append(
                        "  ----+---------------------+-------+-------+----"
                    )
                else:
                    lines.append(
                        "  Idx | Bbox(local)         | Class | Conf"
                    )
                    lines.append(
                        "  ----+---------------------+-------+------"
                    )
                # Track only digit-slot index for GT comparison
                digit_iter_idx = 0
                for d in digits:
                    bx, by, bw, bh = d["bbox_local_post"]
                    if d.get("is_comma"):
                        lines.append(
                            f"  {d['idx']:>3d} | "
                            f"({bx:>4d},{by:>3d},{bw:>4d},{bh:>3d}) | "
                            f"  ,    | comma"
                        )
                    else:
                        cls = d.get("classification", "?")
                        conf = float(d.get("confidence", 0.0))
                        if digits_only_gt is not None:
                            expected = (
                                digits_only_gt[digit_iter_idx]
                                if digit_iter_idx < len(digits_only_gt)
                                else None
                            )
                            ok_mark = (
                                "yes" if expected is not None and cls == expected
                                else ("no " if expected is not None else "?  ")
                            )
                            lines.append(
                                f"  {d['idx']:>3d} | "
                                f"({bx:>4d},{by:>3d},{bw:>4d},{bh:>3d}) | "
                                f"'{cls:>1s}'   | {conf:.2f} | {ok_mark}"
                            )
                        else:
                            lines.append(
                                f"  {d['idx']:>3d} | "
                                f"({bx:>4d},{by:>3d},{bw:>4d},{bh:>3d}) | "
                                f"'{cls:>1s}'   | {conf:.2f}"
                            )
                        digit_iter_idx += 1

            self._proportional_summary.setText("\n".join(lines))

        def _render_comma_anchor_readout(self, diag: dict) -> None:
            """Render the comma_anchor readout. Shows the voted result,
            both individual polarities, agreement state, and the per-
            check breakdown (which structural test passed/failed).
            """
            header = "---Comma anchor---------------------------------"
            ca = diag.get("comma_anchor")
            if ca is None:
                self._comma_anchor_summary.setText(
                    f"{header}\nNot run -- value bbox unavailable"
                )
                return
            if not ca.get("available"):
                self._comma_anchor_summary.setText(
                    f"{header}\nMODULE NOT AVAILABLE: "
                    f"{ca.get('error', 'comma_finder import failed')}"
                )
                return
            if ca.get("error"):
                self._comma_anchor_summary.setText(
                    f"{header}\nFailed: {ca['error']}"
                )
                return
            if not ca.get("found"):
                msg = ca.get("reason", "find_comma_voted returned None")
                self._comma_anchor_summary.setText(
                    f"{header}\n"
                    f"Comma not found - fallback to legacy inline "
                    f"detection\n"
                    f"  primary  : {ca.get('primary_summary', 'None')}\n"
                    f"  inverted : {ca.get('inverted_summary', 'None')}\n"
                    f"  reason   : {msg}"
                )
                return

            bx, by, bw, bh = ca["bbox_local"]
            x_center = ca.get("x_center_local")
            conf = float(ca.get("confidence") or 0.0)
            agreed = bool(ca.get("agreed"))
            polarity = ca.get("polarity_used", "?")
            checks = ca.get("checks") or {}
            numbers = ca.get("numbers") or {}
            n_passed = sum(1 for v in checks.values() if v)

            agreement_str = "agreed by both polarities" if agreed else (
                "polarities disagreed -- using higher-confidence result"
            )

            check_order = [
                ("bottom_heavy_ink", "bottom_heavy_ink", "bottom_frac"),
                ("top_empty", "top_empty", "top_frac"),
                ("narrow_width", "narrow_width", "width_px"),
                ("small_mass", "small_mass", "mass_px"),
                ("aspect_compact", "aspect_compact", None),
                ("isolated_horizontally", "isolated_horizontally",
                 "top_ink_frac_in_comma_cols"),
            ]
            lines: list[str] = [
                header,
                f"Voted result:        x={x_center} (conf {conf:.2f}, "
                f"{agreement_str})",
                f"Polarity used:       {polarity}",
                f"Bbox (crop-local):   "
                f"({bx},{by},{bw},{bh})",
                f"Checks passed ({n_passed}/6):",
            ]
            for key, label, num_key in check_order:
                ok = bool(checks.get(key, False))
                mark = "+" if ok else "-"
                detail = ""
                if num_key and num_key in numbers:
                    val = numbers[num_key]
                    if isinstance(val, float):
                        detail = f"  {val:.2f}"
                    else:
                        detail = f"  {val}"
                lines.append(f"  {mark} {label:<22s}{detail}")

            lines.append("")
            lines.append(f"  primary  result: {ca.get('primary_summary', 'None')}")
            lines.append(f"  inverted result: {ca.get('inverted_summary', 'None')}")
            self._comma_anchor_summary.setText("\n".join(lines))

    app = QApplication.instance() or QApplication(sys.argv)
    win = DebuggerWindow()
    win.show()
    if not initial_diag and not png and not folder and DEFAULT_REGION2_FOLDER.is_dir():
        win._set_folder(DEFAULT_REGION2_FOLDER)
    sys.exit(app.exec())


# ────────────────────────────────────────────────────────────────────
# Headless verification — exposed so a CI / smoke test can dump a
# per-candidate voter table to text without Qt.
# ────────────────────────────────────────────────────────────────────


def headless_dump(png_path: Path) -> str:
    """Run all detectors on ``png_path`` and return a multi-line text
    dump of the per-candidate voter table.
    """
    pil = Image.open(png_path).convert("RGB")
    diag = run_full_diagnostics(pil)
    gt_value = _read_gt_value(png_path)
    buf = io.StringIO()
    buf.write(f"=== {png_path.name} ===\n")
    buf.write(f"image size: {pil.size}\n")

    pill = diag.get("pill")
    if pill is None:
        buf.write("pill: NOT FOUND\n")
    else:
        buf.write(
            f"pill: bbox={pill['bbox']} conf={pill['score']:.3f} "
            f"chrome_px={pill['n_chrome_pixels']}\n"
        )

    cands = diag.get("ncc_raw_candidates", [])
    post = diag.get("ncc_post_voter", [])
    buf.write(f"ncc_raw_candidates: {len(cands)} (post-voter: {len(post)})\n")
    crop_w = pil.size[0]
    x_thr = int(crop_w * _POSITION_PRIOR_FRAC)
    buf.write(f"position_prior x_threshold = {x_thr} (30% of {crop_w})\n")

    n_left = sum(1 for c in cands if c["x1"] <= x_thr)
    n_right = sum(1 for c in cands if c["x1"] > x_thr)
    buf.write(
        f"  candidates with x1 <= {x_thr} (left/icon zone): {n_left}\n"
    )
    buf.write(
        f"  candidates with x1 >  {x_thr} (right/digit zone): {n_right}\n"
    )

    buf.write("\nPer-candidate voter table (sorted by NCC score desc):\n")
    buf.write(
        f"  {'x1':>3} {'y1':>3} {'x2':>3} {'y2':>3} "
        f"{'tw':>2} {'ncc':>5}  "
        f"{'geom':>10} {'contour':>11} {'rgb@':>14} {'gray@':>14}  "
        f"acc?  decision_path\n"
    )
    for row in diag.get("voter_table", []):
        rgb_str = (
            f"{row['rgb_cnn_vote']}({row['rgb_at_prob']:.2f})"
            if row["rgb_at_prob"] is not None else row["rgb_cnn_vote"]
        )
        gray_str = (
            f"{row['gray_cnn_vote']}({row['gray_at_prob']:.2f})"
            if row["gray_at_prob"] is not None else row["gray_cnn_vote"]
        )
        buf.write(
            f"  {row['x1']:>3d} {row['y1']:>3d} {row['x2']:>3d} {row['y2']:>3d} "
            f"{row['tw']:>2d} {row['score']:>5.2f}  "
            f"{row['geom_vote']:>10s} {row['contour_vote']:>11s} "
            f"{rgb_str:>14s} {gray_str:>14s}  "
            f"{'OK' if row['accepted'] else 'NO'}    {row['decision_path']}\n"
        )

    icon = diag.get("icon_final")
    if icon is None:
        buf.write("\nfind_icon final: None\n")
    else:
        buf.write(
            f"\nfind_icon final: bbox={icon['bbox']} score={icon['score']:.2f}\n"
        )
        pos_ok = diag.get("icon_position_prior_passed", True)
        col_ok = diag.get("icon_color_prior_passed", True)
        buf.write(
            f"  position_prior_passed={pos_ok}  "
            f"color_prior_passed={col_ok}\n"
        )
        if "icon_color_warm_frac" in diag:
            buf.write(
                f"  warm_frac={diag['icon_color_warm_frac']:.2f}  "
                f"cool_frac={diag['icon_color_cool_frac']:.2f}\n"
            )
        if not (pos_ok and col_ok):
            buf.write("  -> auto-annotator's detect_icon would emit {} (rejected)\n")

    dc = diag.get("digit_cluster")
    if dc is None:
        buf.write("digit_cluster: None\n")
    else:
        buf.write(f"digit_cluster: {dc['bbox']}\n")

    # ── New-architecture detector dump ────────────────────────────
    buf.write("\n─── RGB NCC ─────────────────────────────────────\n")
    if not diag.get("rgb_ncc_available", False):
        buf.write("MODULE NOT AVAILABLE\n")
    else:
        tinfo = diag.get("rgb_ncc_template_info") or {}
        n_t = int(tinfo.get("n_templates", 0))
        real_dir = tinfo.get("real_dir", "")
        real_dir_short = real_dir.split(os.sep)[-1] if real_dir else ""
        buf.write(
            f"Templates loaded: {n_t} real, "
            f"from {real_dir_short or '<unknown>'}\n"
        )
        rres = diag.get("rgb_ncc_result")
        if rres is None:
            buf.write("Peak found: NO\n")
        else:
            per = rres["per_channel_scores"]
            wts = rres["weights"]
            combined = wts[0] * per[0] + wts[1] * per[1] + wts[2] * per[2]
            buf.write("Peak found: yes\n")
            buf.write(f"Bbox: {rres['bbox']}\n")
            buf.write(f"Score: {rres['score']:.3f}\n")
            buf.write(f"Scale: {rres['scale']:.2g}\n")
            buf.write(
                f"Per-channel scores: R={per[0]:.2f}, G={per[1]:.2f}, "
                f"B={per[2]:.2f}\n"
            )
            buf.write(f"Combined (R-heavy): {combined:.3f}\n")
            buf.write(
                "Position prior (<=40% width): "
                f"{'PASS' if rres['position_prior_pass'] else 'FAIL'}\n"
            )
            buf.write(f"Template used: {rres['template_used']}\n")

    buf.write("\n─── Geometry (whole-image primary mode) ─────────\n")
    if not diag.get("geometry_primary_available", False):
        buf.write("MODULE NOT AVAILABLE\n")
    else:
        gres = diag.get("geometry_primary_result")
        if gres is None:
            buf.write(
                "Bbox: None (no warm-mask component passed structural "
                "checks)\n"
            )
        else:
            buf.write(f"Bbox: {gres['bbox']}\n")
            buf.write(f"Confidence: {gres['confidence']:.2f}\n")
            buf.write(f"Score: {gres['score']}/6")
            if gres.get("tiny_mode"):
                buf.write(" (tiny_mode)")
            buf.write("\n")
            checks = gres["checks"]
            check_strs = []
            for k in (
                "color_warm", "two_components", "teardrop_has_hole",
                "oval_below_teardrop", "oval_has_notch",
                "aspect_ratio_global",
            ):
                check_strs.append(
                    f"{'PASS' if checks.get(k, False) else 'FAIL'} {k}"
                )
            buf.write("Checks: " + ", ".join(check_strs) + "\n")

    buf.write("\n─── localize_icon (consensus) ───────────────────\n")
    if not diag.get("localize_icon_available", False):
        buf.write("MODULE NOT AVAILABLE\n")
    else:
        lres = diag.get("localize_icon_result")
        if lres is not None:
            geom_bb = (
                diag["geometry_primary_result"]["bbox"]
                if diag.get("geometry_primary_result") else None
            )
            ncc_bb = (
                diag["rgb_ncc_result"]["bbox"]
                if diag.get("rgb_ncc_result") else None
            )
            iou = diag.get("localize_icon_iou")
            iou_str = f"{iou:.2f}" if iou is not None else "n/a"
            buf.write(
                f"Geometry bbox:  {geom_bb}  IoU vs RGB NCC = {iou_str}\n"
            )
            buf.write(f"RGB NCC bbox:   {ncc_bb}\n")
            buf.write(
                f"Consensus result: ACCEPT — IoU {iou_str} > 0.4\n"
            )
            buf.write(f"Returned bbox: {lres['bbox']}\n")
            buf.write(f"Returned detector tag: {lres['detector']}\n")
            buf.write(f"Score: {lres['score']:.2f}\n")
        else:
            buf.write(
                "NO CONSENSUS — falling back to legacy NCC + voter\n"
            )
            buf.write(
                f"Reason: {diag.get('localize_icon_no_consensus_reason')}\n"
            )

    buf.write("\n─── Segmenter output ───────────────────────────\n")
    seg = diag.get("glyph_segmentation")
    crop_source = diag.get("glyph_segmentation_crop_source", "unavailable")
    if seg is None:
        buf.write("Value crop unavailable — segmenter not run\n")
    elif seg.get("error"):
        buf.write(f"Crop source: {crop_source}\n")
        buf.write(f"Segmenter failed: {seg['error']}\n")
    elif seg.get("reason"):
        buf.write(f"Crop source: {crop_source}\n")
        buf.write(
            f"Value crop unavailable — segmenter not run "
            f"({seg['reason']})\n"
        )
    else:
        spans = seg.get("spans", [])
        n = len(spans)
        crop_box_str = (
            str(seg.get("crop_box")) if seg.get("crop_box") else "—"
        )
        buf.write(f"Spans found:    {n}\n")
        buf.write(f"Crop source:    {crop_source}  bbox={crop_box_str}\n")
        digits_only_gt: Optional[str] = None
        if gt_value is not None:
            digits_only_gt = "".join(
                ch for ch in gt_value if ch.isdigit()
            )
            buf.write(
                f"GT (if known):  '{gt_value}' "
                f"({len(digits_only_gt)} digits)\n"
            )
        if not spans:
            buf.write("No spans detected\n")
        else:
            buf.write("\n")
            if digits_only_gt is not None:
                buf.write(
                    "  Idx | Bbox             | Class | Conf  | OK\n"
                )
                buf.write(
                    "  ----+------------------+-------+-------+-----\n"
                )
                for i, span in enumerate(spans):
                    bx, by, bw, bh = span["bbox_local"]
                    cls = span["classification"]
                    conf = span["confidence"]
                    expected = (
                        digits_only_gt[i]
                        if i < len(digits_only_gt) else None
                    )
                    ok_mark = (
                        "yes" if expected is not None and cls == expected
                        else ("no " if expected is not None else "?  ")
                    )
                    buf.write(
                        f"  {i:>3d} | ({bx:>3d},{by:>3d},{bw:>3d},"
                        f"{bh:>3d}) | '{cls:>1s}'   | "
                        f"{conf:.2f} | {ok_mark}\n"
                    )
            else:
                buf.write(
                    "  Idx | Bbox             | Class | Conf\n"
                )
                buf.write(
                    "  ----+------------------+-------+------\n"
                )
                for i, span in enumerate(spans):
                    bx, by, bw, bh = span["bbox_local"]
                    cls = span["classification"]
                    conf = span["confidence"]
                    buf.write(
                        f"  {i:>3d} | ({bx:>3d},{by:>3d},{bw:>3d},"
                        f"{bh:>3d}) | '{cls:>1s}'   | {conf:.2f}\n"
                    )
            composed = seg.get("string_composed", "")
            if digits_only_gt is not None:
                gt_match = (composed == digits_only_gt)
                buf.write(
                    f"\nString composed: '{composed}'    "
                    f"GT match: {'yes' if gt_match else 'no'}\n"
                )
            else:
                buf.write(f"\nString composed: '{composed}'\n")

    errs = diag.get("errors", [])
    if errs:
        buf.write("\nErrors during run:\n")
        for stage, msg in errs:
            if stage.endswith("_tb"):
                continue
            buf.write(f"  [{stage}] {msg}\n")

    if diag.get("logs"):
        buf.write("\nCaptured log lines (last 30):\n")
        for line in diag["logs"][-30:]:
            buf.write(f"  {line}\n")

    return buf.getvalue()


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--png", type=Path, default=None,
        help="Path to a single region2 PNG to inspect.",
    )
    parser.add_argument(
        "--folder", type=Path, default=None,
        help="Path to a folder of region2 PNGs (browse via prev/next).",
    )
    parser.add_argument(
        "--headless-dump", action="store_true",
        help="Skip the GUI; print the per-candidate voter table for --png "
             "(or every PNG in --folder) to stdout.",
    )
    args = parser.parse_args(argv)

    if args.headless_dump:
        if args.png is None and args.folder is None:
            print("--headless-dump requires --png or --folder")
            return 2
        targets: list[Path] = []
        if args.png is not None:
            targets.append(args.png)
        if args.folder is not None:
            targets.extend(sorted(args.folder.glob("*.png")))
        for p in targets:
            try:
                print(headless_dump(p))
            except Exception as exc:
                print(f"[ERR] {p}: {exc}")
                traceback.print_exc()
        return 0

    # Default: launch GUI.
    build_gui(folder=args.folder, png=args.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
