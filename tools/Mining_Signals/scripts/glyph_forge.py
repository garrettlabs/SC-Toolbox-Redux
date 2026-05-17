"""Glyph Forge — Tesseract-proposes / human-verifies glyph extractor.

Per capture in ``training_data_panels/user_*/region2/``:

  1. Run Tesseract with a digit whitelist over multiple PSM + scale
     variants. Pick the variant whose digit count matches the user's
     typed label.
  2. Use Tesseract's per-character bounding boxes to extract one
     28×28 tile per digit.
  3. Display the row of tiles with each tile's destination class
     dropdown pre-filled with the expected char from the typed label.
  4. The user eyeballs each tile and either:
       - hits Save & Next to accept all defaults
       - or changes any tile's class dropdown to "skip" (don't save)
         or to a different digit class (move to right folder)
  5. Saves the approved glyphs into
     ``training_data_user_sig/<class>/user_<src>_<i>.png``.
  6. Captures already processed (output PNGs already exist) are
     skipped on relaunch so you can resume.

This sidesteps every column-projection failure mode (icon
contamination, merged digits, comma artifacts) by leaning on
Tesseract for segmentation + your eyes for correctness.

Run with:
    python scripts/glyph_forge.py
    or double-click LAUNCH_GlyphForge.bat in training_data_panels/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QPalette, QPixmap, QColor, QFont, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSplitter, QVBoxLayout, QWidget, QProgressBar,
)


THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
sys.path.insert(0, str(TOOL))
sys.path.insert(0, str(TOOL / "scripts"))

# Glyph Forge resolves all of its dependencies — training_registry,
# the api blacklist+voter helpers, the extract_labeled_glyphs module,
# AND the ONNX model weights — out of the WingmanAI tool tree it
# lives in. The WingmanAI tree has its own copy of the ocr package
# and its own ocr/models/ ONNX files. We deliberately do NOT add the
# production-install tree to sys.path here: a previous version did,
# and that caused ``from ocr import training_registry`` to resolve to
# the production tree's registry which didn't know about the
# WingmanAI training_data_panels source dirs — so ``_all_caps`` came
# out empty and the user saw "No labeled captures found." in the UI.

from ocr import training_registry  # noqa: E402
import extract_labeled_glyphs as _xlg  # noqa: E402

# Production helpers — the icon blacklist + 4-CNN voter. We DEFER
# this import until first actual use (lazy via ``_get_prod_api``)
# because importing api.py triggers an eager onnxruntime load + many
# transitive imports that can take 30s+ on cold disk / Windows
# Defender. Doing it on the UI thread before show() makes the app
# look hung. Lazy load => GUI appears instantly, and the FIRST tile
# render pays the cold cost (visible to the user as "loading...").
#
# Disable entirely with ``GLYPH_FORGE_NO_PIPELINE=1`` env var — useful
# when you only want the Tesseract proposer behaviour without the
# blacklist + voter overlay. Lets the user keep working if the
# production helpers are broken or slow on their machine.
_prod_api = None
_prod_api_load_attempted = False
_PIPELINE_DISABLED = bool(os.environ.get("GLYPH_FORGE_NO_PIPELINE"))


def _get_prod_api():
    """Return the production sc_ocr.api module, loading it lazily on
    first call. Returns ``None`` (and logs the reason once) when the
    module isn't importable in this environment, or when the user
    has set ``GLYPH_FORGE_NO_PIPELINE=1``.
    """
    global _prod_api, _prod_api_load_attempted
    if _prod_api is not None:
        return _prod_api
    if _prod_api_load_attempted:
        return None
    _prod_api_load_attempted = True
    if _PIPELINE_DISABLED:
        print("glyph_forge: GLYPH_FORGE_NO_PIPELINE=1, skipping prod api import")
        return None
    try:
        print("glyph_forge: loading production helpers (cold start, ~5-30s)...",
              flush=True)
        import time as _t
        _t0 = _t.perf_counter()
        from ocr.sc_ocr import api as _api_mod  # type: ignore
        _prod_api = _api_mod
        print(f"glyph_forge: production helpers ready in "
              f"{_t.perf_counter() - _t0:.1f}s", flush=True)
        return _prod_api
    except Exception as _exc:
        print(f"glyph_forge: production helpers unavailable ({_exc}) - "
              f"falling back to Tesseract-only proposals", flush=True)
        return None

# Theme matching coverage HUD / blacklist manager
ACCENT = "#33dd88"
RED = "#ff4444"
DIM = "#888888"
BG = "#1e1e1e"
FG = "#e0e0e0"
WARN = "#ffc107"

# Active region — currently signal-only since this tool was built to
# solve the signal scanner's specific extraction problems. Adding HUD
# would require choosing which JSON field is the "label" (mass vs
# resistance vs instability) per capture; defer until needed.
KIND = "signal"
SKIP_OPTION = "—"


# ─────────────────────────────────────────────────────────────
# Tesseract proposer
# ─────────────────────────────────────────────────────────────

def _propose_glyphs(
    img_path: Path, label: dict,
) -> tuple[Optional[list[np.ndarray]], list[str], str,
           Optional[np.ndarray], Optional[list[tuple[int, int]]],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """Run Tesseract on `img_path` with multi-variant search. Pick the
    variant whose digit count matches the user's typed label. Return
    (glyphs_28x28, expected_chars, tag, post_mask_gray, spans, post_mask_rgb)
    where:
      glyphs_28x28: list of 28×28 uint8 arrays (one per glyph)
      expected_chars: digit chars from typed label (no comma)
      tag: which variant won (e.g. "2x/psm7"), or "" if none did
      post_mask_gray: the masked + row-isolated grayscale source array
                     used for cropping (so the UI can re-crop after the
                     user nudges a tile's bounds without re-running
                     Tesseract / re-masking)
      spans: list of (x1, x2) per tile, in post_mask_gray coords.
             The UI uses these as the editable starting bounds.
      post_mask_rgb: the same row-isolated row but as 3-channel RGB,
                     for the production RGB CNN's per-tile classifier.
                     ``None`` if RGB extraction failed (defensive — UI
                     falls back to gray-only voting in that case).

    Returns (None, chars, "", None, None, None) if Tesseract never agreed.
    """
    spec = training_registry.get(KIND)
    value = str(label.get(spec.label_field, "")).strip().replace(",", "")
    chars = [c for c in value if c.isdigit()]
    if not chars or not img_path.is_file():
        return None, chars, "", None, None, None, None

    try:
        img = Image.open(img_path).convert("L")
    except Exception:
        return None, chars, "", None, None, None, None
    gray = np.asarray(img, dtype=np.uint8)
    img_w = gray.shape[1]

    # Also load the same image as RGB so the RGB CNN can classify each
    # tile in colour. ``Image.open(img_path).convert("RGB")`` produces
    # the SAME pixel grid (same width/height) as the grayscale path
    # above, just with 3 channels — so the row-isolate y-bounds we
    # derive on gray transfer 1:1 to RGB by row index.
    try:
        img_rgb = Image.open(img_path).convert("RGB")
        rgb = np.asarray(img_rgb, dtype=np.uint8)
    except Exception:
        rgb = None

    # Apply same icon mask + row isolate as the offline extractor so
    # Tesseract sees the cleaned-up image.
    bg = int(np.median(gray))
    gray = gray.copy()
    icon_right = _xlg._locate_icon_via_blacklist_match(gray)
    floor_mask = int(img_w * 0.30)
    mask_w = max(floor_mask, icon_right + 4 if icon_right > 0 else 0)
    # Save the PRE-MASK gray for the leading-digit recovery pass.
    # When Tesseract's icon-mask hides a leading digit (because the
    # icon mask extends past the icon graphic into the digit's
    # columns), we need to look at the original UN-masked pixels to
    # find the lost digit. Without this, the leading "1" of values
    # like "17,020" disappears: the icon mask paints columns 0..114
    # with bg, but the "1" lives at ~107..115 and gets fully erased.
    gray_unmasked_full = gray.copy()
    if 0 < mask_w < img_w:
        gray[:, :mask_w] = bg
    # Row-isolate gray. ``_isolate_main_row`` slices along axis 0 to
    # a narrow band around the digit row. We want RGB sliced at
    # EXACTLY the same band, so we use the helper's sibling
    # ``_find_main_row_bounds`` which returns the (y1, y2) bounds
    # rather than the trimmed array — letting us apply the SAME
    # bounds to RGB for shared coordinates with the gray spans.
    band_bounds = _xlg._find_main_row_bounds(gray)
    if band_bounds is not None:
        y1, y2 = band_bounds
        # Mirror ``_isolate_main_row``'s ±2 px padding so the bounds
        # we apply to RGB + the unmasked-gray-recovery view match the
        # bounds the gray slice received.
        y1 = max(0, y1 - 2)
        y2 = min(gray.shape[0], y2 + 2)
        gray = gray[y1:y2, :]
        rgb_row: Optional[np.ndarray] = (
            rgb[y1:y2, :, :] if rgb is not None else None
        )
        gray_unmasked = gray_unmasked_full[y1:y2, :]
    else:
        # No band detected — keep gray as-is (matches
        # ``_isolate_main_row`` fallback). RGB + unmasked stay at
        # full height for the same coordinate space as gray.
        rgb_row = rgb
        gray_unmasked = gray_unmasked_full

    base = Image.fromarray(gray, mode="L")
    variants = [
        (base, "1x"),
        (base.resize((base.width * 2, base.height * 2), Image.LANCZOS), "2x"),
        (base.resize((base.width * 3, base.height * 3), Image.LANCZOS), "3x"),
        (Image.fromarray(255 - gray, mode="L"), "1x_inv"),
    ]
    label_clean = "".join(chars)

    for psm in ("7", "13", "8", "6"):
        for img_v, tag in variants:
            try:
                tess_boxes = _xlg._tesseract_char_boxes(
                    img_v, whitelist="0123456789.", psm=psm,
                )
            except Exception:
                continue
            if not tess_boxes:
                continue
            tess_clean = "".join(
                b[0] for b in tess_boxes
                if b[0].isdigit() or b[0] == "."
            ).replace(".", "")
            if tess_clean != label_clean:
                continue
            if len(tess_boxes) < len(chars):
                continue
            # Map box coords back to ORIGINAL gray coords.
            scale = 2 if tag.startswith("2x") else (
                3 if tag.startswith("3x") else 1)
            digit_boxes = [b for b in tess_boxes if b[0].isdigit()][:len(chars)]
            spans = []
            for i, b in enumerate(digit_boxes):
                x1 = b[1] // scale
                x2 = b[3] // scale
                spans.append((x1, x2))
            # Same overlap fix as offline extractor: midpoint between
            # consecutive overlapping centers.
            for i, (x1, x2) in enumerate(spans):
                if i + 1 < len(spans):
                    nx1, nx2 = spans[i+1]
                    if nx1 < x2:
                        cur_c = (x1 + x2) / 2.0
                        nxt_c = (nx1 + nx2) / 2.0
                        if nxt_c > cur_c:
                            boundary = int((cur_c + nxt_c) / 2.0)
                            spans[i] = (x1, boundary)
                            spans[i+1] = (boundary, nx2)
            # Render each span to 28×28
            glyphs: list[np.ndarray] = []
            ok = True
            for (x1, x2) in spans:
                if x2 <= x1:
                    ok = False
                    break
                g = _xlg._glyph_to_28x28(gray, x1, x2)
                if g is None:
                    ok = False
                    break
                glyphs.append(g)
            if not ok or len(glyphs) != len(chars):
                continue
            return (
                glyphs, chars, f"{tag}/psm{psm}",
                gray, spans, rgb_row, gray_unmasked,
            )
    return None, chars, "", None, None, None, None


# ─────────────────────────────────────────────────────────────
# Leading-digit recovery (Layer B)
# ─────────────────────────────────────────────────────────────
#
# When Tesseract's icon-mask is wider than the icon graphic itself,
# the mask paints over leading digit columns and Tesseract never
# sees them. The icon-prefilter then drops the icon's tile, leaving
# fewer tiles than the typed label has digits — even though the
# digit's ink IS present in the unmasked source.
#
# The recovery scan: look at the unmasked-but-row-isolated gray in
# the columns BEFORE the leftmost surviving tile, find ink-bearing
# blobs, classify each via the icon-blacklist + voter to confirm
# it's a digit (not the icon itself), and propose new leading tiles
# for any digit blobs found. Operates entirely on existing pixel
# data — no new image loads, no Tesseract re-runs.

# Minimum ink-density (per column, fraction of row height) to count
# a column as "has digit ink." Same threshold the production
# segmenter uses for ink-extent detection.
_RECOVERY_INK_COL_FRAC = 0.20
# Maximum gap between adjacent ink columns to still count as the
# same blob — bridges chromatic-aberration drops within a digit
# stem. 3 px is generous enough to catch SC HUD's worst-rendered
# "1" at 1080p, tight enough to keep two adjacent digits separate.
_RECOVERY_GAP_PX = 3
# Minimum / maximum blob width (in source-image pixels) that
# qualifies as a single digit. Smaller = noise speck or mid-stroke
# artifact; larger = the icon graphic itself or a comma + digit
# fusion. Calibrated to typical 1080p capture scale.
_RECOVERY_MIN_BLOB_PX = 4
_RECOVERY_MAX_BLOB_PX = 28


def _scan_for_missed_leading_digits(
    gray_unmasked: np.ndarray,
    leftmost_kept_x: int,
) -> list[tuple[int, int]]:
    """Find ink-bearing blobs left of ``leftmost_kept_x`` in the
    unmasked gray. Returns a list of ``(x1, x2)`` blob spans in
    left-to-right order, INCLUDING any icon-shaped blobs (the caller
    is responsible for filtering those out via the
    blacklist+voter).

    The threshold logic mirrors the production segmenter's blob
    detector: per-column ink density above ``_RECOVERY_INK_COL_FRAC``
    of row height, gap-bridging up to ``_RECOVERY_GAP_PX``, blob
    width filter ``_RECOVERY_MIN/MAX_BLOB_PX``.
    """
    if gray_unmasked is None or gray_unmasked.size == 0:
        return []
    H, W = gray_unmasked.shape[:2]
    if leftmost_kept_x <= 0:
        return []

    # Polarity-canonicalize: treat digits as the BRIGHT class (matches
    # what the production segmenter expects).
    if int(np.median(gray_unmasked)) > 140:
        work = 255 - gray_unmasked
    else:
        work = gray_unmasked
    try:
        thr = _xlg._otsu(work)
    except Exception:
        thr = int(work.mean())
    binary = (work > thr).astype(np.uint8)
    col_density = binary.sum(axis=0)
    col_thr = max(1, int(H * _RECOVERY_INK_COL_FRAC))

    # Walk left-to-right up to leftmost_kept_x, grouping ink columns
    # into blobs with gap-bridging.
    blobs: list[tuple[int, int]] = []
    blob_start: Optional[int] = None
    last_inky: Optional[int] = None
    scan_end = max(0, leftmost_kept_x)
    for x in range(scan_end):
        is_inky = col_density[x] >= col_thr
        if is_inky:
            if blob_start is None:
                blob_start = x
            last_inky = x
            continue
        if blob_start is not None and last_inky is not None:
            if x - last_inky > _RECOVERY_GAP_PX:
                # Close current blob.
                w = last_inky + 1 - blob_start
                if _RECOVERY_MIN_BLOB_PX <= w <= _RECOVERY_MAX_BLOB_PX:
                    blobs.append((blob_start, last_inky + 1))
                blob_start = None
                last_inky = None
    # Close any blob still open at scan_end.
    if blob_start is not None and last_inky is not None:
        w = last_inky + 1 - blob_start
        if _RECOVERY_MIN_BLOB_PX <= w <= _RECOVERY_MAX_BLOB_PX:
            blobs.append((blob_start, last_inky + 1))
    return blobs


# ─────────────────────────────────────────────────────────────
# Bbox tighten-to-ink (Layer H)
# ─────────────────────────────────────────────────────────────
#
# After Tesseract proposes tiles + Layer D refines boundaries,
# individual tile bboxes can still include trailing/leading
# background or partial-neighbour ink (comma-edge contamination,
# pre-icon halo, etc.). The CNN was trained on tightly-cropped
# digit glyphs; bboxes that include extra background pixels
# stretch the digit less during the resize-to-28×28 step, which
# moves the input out of the CNN's training distribution.
#
# Fix: per-tile, scan the column-density projection within the
# bbox and contract horizontally until ink-bearing columns are
# reached on both sides. Adds 1-px margin so digit edges aren't
# clipped. Conservative — only contracts (never expands), and
# never produces a sliver narrower than 4 px.
#
# This addresses the 11,520 capture's "5 -> 3" misclassification:
# Tesseract proposed tile[2] at x=117..134 but the "5" digit
# occupies x=120..134 and the comma's right edge contaminates
# x=117..120. Tightening contracts the bbox to x=120..134 so the
# CNN sees a clean "5".

_TIGHTEN_INK_COL_FRAC = 0.10   # column counts as inky if ≥ this fraction of row height
_TIGHTEN_MARGIN_PX = 1         # extra px on each side of detected ink range
_TIGHTEN_MIN_WIDTH = 4         # never produce a tile narrower than this


def _tighten_tiles_to_ink(
    source_gray: np.ndarray,
    kept_spans: list[tuple[int, int]],
    kept_glyphs: list[np.ndarray],
) -> tuple[list[tuple[int, int]], list[np.ndarray], int]:
    """Contract each tile bbox horizontally to its actual ink
    extent. Returns ``(new_spans, new_glyphs, n_tightened)`` where
    ``n_tightened`` is the count of bboxes that actually shrank
    (a tile whose original bbox was already tight contributes 0).

    Per-tile pipeline:
      1. Polarity-canonicalize the source gray once (cached
         outside the loop).
      2. Threshold via Otsu, project columns.
      3. For each tile bbox, find the leftmost+rightmost columns
         within ``[x1, x2)`` whose ink density crosses the row-
         height-fractional threshold.
      4. Apply ``_TIGHTEN_MARGIN_PX`` margin and re-extract the
         28×28 glyph at the new bounds.

    Skips tiles that:
      - have no ink columns within their bbox (probably a noise
        tile; leave alone),
      - would shrink to less than ``_TIGHTEN_MIN_WIDTH``,
      - already match their tight ink range (no change to make).
    """
    if source_gray is None or source_gray.size == 0:
        return kept_spans, kept_glyphs, 0
    if not kept_spans:
        return kept_spans, kept_glyphs, 0

    # Polarity canon + threshold + projection — once per call.
    if int(np.median(source_gray)) > 140:
        work = 255 - source_gray
    else:
        work = source_gray
    try:
        thr = _xlg._otsu(work)
    except Exception:
        thr = int(work.mean())
    binary = (work > thr).astype(np.uint8)
    col_density = binary.sum(axis=0)
    H = source_gray.shape[0]
    # Vertical-coverage threshold: a column counts as "real digit
    # ink" only when it has ink in at least 25% of the row height.
    # Rationale:
    #   - Real digit strokes span most of the row (40-80% coverage)
    #   - Comma's body + halo columns span 10-22% coverage
    #     (comma sits in the bottom band of the row, ~5-9 px tall
    #     out of 30-42 px row)
    #   - 25% threshold cleanly separates the two categories on
    #     captures we've checked. 20% was too lenient (comma's
    #     densest column hit exactly 20% on cap_085431_378 and
    #     prevented tightening of tile[2]'s left edge into the
    #     digit body). 30%+ risks clipping thin digit-stroke
    #     columns at the digit's vertical extremes.
    abs_thr = max(1, int(H * 0.25))

    new_spans = list(kept_spans)
    new_glyphs = list(kept_glyphs)
    n_tightened = 0

    for i, (x1, x2) in enumerate(new_spans):
        sub = col_density[x1:x2]
        if sub.size == 0:
            continue
        ink_mask = sub >= abs_thr
        if not ink_mask.any():
            continue
        ink_cols = np.where(ink_mask)[0]
        first_ink_local = int(ink_cols[0])
        last_ink_local = int(ink_cols[-1])

        new_x1 = x1 + max(0, first_ink_local - _TIGHTEN_MARGIN_PX)
        new_x2 = x1 + min(sub.size, last_ink_local + 1 + _TIGHTEN_MARGIN_PX)

        if new_x2 - new_x1 < _TIGHTEN_MIN_WIDTH:
            continue
        if new_x1 == x1 and new_x2 == x2:
            continue

        new_g = _xlg._glyph_to_28x28(source_gray, new_x1, new_x2)
        if new_g is None:
            continue

        new_spans[i] = (new_x1, new_x2)
        new_glyphs[i] = new_g
        n_tightened += 1

    return new_spans, new_glyphs, n_tightened


# ─────────────────────────────────────────────────────────────
# Wide-tile splitter (Layer F)
# ─────────────────────────────────────────────────────────────
#
# Tesseract sometimes merges two adjacent digits into a single
# bbox when their columns share enough ink (e.g. on
# cap_20260418_085435_190 the "2" and "6" of "26,000" came back
# as one tile at x=103..130 w=27). Symptom: kept tile count is
# short of expected, and one tile is dramatically wider than its
# peers.
#
# Fix: while kept_count < expected, find the widest tile. If it's
# >1.5x the median width, find the lowest-ink valley in its
# central region and split it into two tiles. Iterate until count
# matches expected OR no tile is wide enough to split.
#
# Adapts the production segmenter's ``_split_wide_signature_spans``
# logic (in ocr/sc_ocr/api.py) but trimmed for Glyph Forge's
# simpler needs (no expected_count guarantees, no overlap fixes).

_SPLIT_WIDE_FACTOR = 1.5      # tile must be ≥ this × median to split
_SPLIT_MIN_HALF_PX = 4        # each split half must be ≥ this wide
_SPLIT_MARGIN_FRAC = 0.20     # search window: middle (1 - 2*margin)
_VALLEY_DEPTH_FRAC = 0.25     # valley column must be < this × peak
_VALLEY_EDGE_MARGIN_PX = 4    # interior region for valley search


def _has_internal_valley(
    col_density_sub: np.ndarray,
    edge_margin_px: int = _VALLEY_EDGE_MARGIN_PX,
    valley_depth_frac: float = _VALLEY_DEPTH_FRAC,
) -> bool:
    """Detect a clear ink-density valley INSIDE a tile bbox.

    Returns True iff the tile's column-density profile has at least
    one interior column whose density is below
    ``valley_depth_frac * peak_density``. The interior excludes the
    edge ``edge_margin_px`` pixels on each side (avoids classifying
    the natural empty-ish edges of any digit as a valley).

    Used by Layer F (wide-tile splitter) to identify merged-narrow-
    digit-pair tiles that don't trip the 1.5x-median width threshold:
    e.g. a "17" merge at w=17 looks the same width as a clean "5"
    at w=17, but only the merge has a stem-gap-body signature where
    one interior column drops to <25% of the peak.

    Returns False on tiles that are too narrow to support a margin,
    on tiles whose peak density is too low to define a meaningful
    fraction (< 4 px), and on tiles whose interior never dips below
    the threshold (clean single digits).
    """
    if col_density_sub is None:
        return False
    n = col_density_sub.size
    if n < 2 * edge_margin_px + 1:
        return False
    interior = col_density_sub[edge_margin_px:n - edge_margin_px]
    if interior.size == 0:
        return False
    peak = int(col_density_sub.max())
    if peak < 4:
        return False
    valley = int(interior.min())
    return valley < peak * valley_depth_frac


def _split_wide_tiles(
    source_gray: np.ndarray,
    kept_spans: list[tuple[int, int]],
    kept_glyphs: list[np.ndarray],
    expected_count: int,
) -> tuple[list[tuple[int, int]], list[np.ndarray], int]:
    """Iteratively split outsized tiles until kept_count == expected
    OR no tile remains wide enough to plausibly contain two digits.

    Returns ``(new_spans, new_glyphs, n_split)`` where ``n_split`` is
    the number of split operations performed (== additional tiles
    added).

    Conservative — only fires when we have FEWER tiles than
    expected. A capture where Tesseract found exactly the right
    count never splits, even if one tile happens to be a wide
    glyph. Prevents over-segmentation of clean reads.
    """
    if source_gray is None or source_gray.size == 0:
        return kept_spans, kept_glyphs, 0
    if len(kept_spans) >= expected_count:
        return kept_spans, kept_glyphs, 0
    if len(kept_spans) < 1:
        return kept_spans, kept_glyphs, 0
    # NOTE: a previous version allowed splitting up to ``expected_count + 1``
    # tiles to handle cases like 21,200 where tile[0] was still a
    # merge even after the count matched expected. But on 11,520
    # the lower-half-median was dragged down by a narrow leading
    # "1", making a legitimate w=17 "5" look like a 2.83x outlier
    # and triggering an unwanted split that cost the read 2 tiles.
    # Reverted to the conservative rule: only split when count is
    # short. The 21,200 tile[0] case is then a manual-nudge job
    # rather than auto-fix; safer than corrupting clean reads.
    max_tiles = expected_count

    # Polarity-canonicalize → Otsu → column ink density. Same
    # convention as Layer D so the valley-finding behaves
    # consistently across both refinement passes.
    if int(np.median(source_gray)) > 140:
        work = 255 - source_gray
    else:
        work = source_gray
    try:
        thr = _xlg._otsu(work)
    except Exception:
        thr = int(work.mean())
    binary = (work > thr).astype(np.uint8)
    col_density = binary.sum(axis=0)

    out_spans = list(kept_spans)
    out_glyphs = list(kept_glyphs)
    n_split = 0
    safety = 8  # max splits per capture (avoids infinite loops)

    while len(out_spans) < max_tiles and safety > 0:
        safety -= 1
        # Refresh single-digit-width estimate on each iteration. We
        # use the LOWER-HALF median (i.e. the median of the
        # narrower half of tiles) instead of the full median so
        # that several merged-digit blobs in the same row don't
        # inflate the reference width above what a single digit
        # actually is. Example: widths=[6, 12, 20, 24] for one '1'
        # + one '0' + two merged-digit blobs has full-median 16
        # (between 12 and 20), but the real single-digit width is
        # 6-12 — we want the threshold derived from that, so the
        # 20/24 tiles get split. Lower-half median = 9 here, and
        # 1.5x = 13.5 catches both wide tiles.
        widths = sorted([e - s for s, e in out_spans])
        if not widths:
            break
        half = max(1, len(widths) // 2)
        lower_widths = widths[:half]
        median_w = lower_widths[len(lower_widths) // 2]
        if median_w < 4:
            break
        # Find the widest tile that's a plausible split candidate.
        # A tile is a candidate if EITHER:
        #   * width-based: width >= 1.5 * lower-half median, OR
        #   * valley-based: the tile has a clear internal gap in
        #     ink density (one column inside the tile has < 25% of
        #     the tile's peak column density, with at least 4 px
        #     margin from each edge).
        # The valley path catches narrow-digit-pair merges that
        # don't trip the width threshold — e.g. a merged "17" at
        # w=17 looks the same width as a clean "5" at w=17, but
        # only the merged version has a stem-gap-body density
        # signature inside.
        candidates: list[int] = []
        for idx, (cs, ce) in enumerate(out_spans):
            cw = ce - cs
            if cw < _SPLIT_MIN_HALF_PX * 2:
                continue
            is_wide = cw >= median_w * _SPLIT_WIDE_FACTOR
            sub_for_valley = col_density[cs:ce]
            has_valley = _has_internal_valley(sub_for_valley)
            if is_wide or has_valley:
                candidates.append(idx)
        if not candidates:
            break
        widest_idx = max(
            candidates,
            key=lambda i: out_spans[i][1] - out_spans[i][0],
        )
        s, e = out_spans[widest_idx]
        tile_w = e - s

        # Search the central region for the lowest-ink column.
        margin = max(2, int(tile_w * _SPLIT_MARGIN_FRAC))
        lo = s + margin
        hi = e - margin
        if hi <= lo:
            break
        sub = col_density[lo:hi]
        split_x = lo + int(np.argmin(sub))

        # Reject splits that produce a sliver on either side.
        if (split_x - s) < _SPLIT_MIN_HALF_PX:
            break
        if (e - split_x) < _SPLIT_MIN_HALF_PX:
            break

        # Re-extract glyphs for the two halves.
        left_g = _xlg._glyph_to_28x28(source_gray, s, split_x)
        right_g = _xlg._glyph_to_28x28(source_gray, split_x, e)
        if left_g is None or right_g is None:
            break

        # Apply the split: replace the widest tile with two halves.
        out_spans[widest_idx] = (split_x, e)
        out_glyphs[widest_idx] = right_g
        out_spans.insert(widest_idx, (s, split_x))
        out_glyphs.insert(widest_idx, left_g)
        n_split += 1

    return out_spans, out_glyphs, n_split


# ─────────────────────────────────────────────────────────────
# Ink-valley boundary refinement (Layer D)
# ─────────────────────────────────────────────────────────────
#
# Tesseract's per-character bboxes occasionally place a boundary
# inside a digit instead of in the gap between digits. Symptom:
# adjacent tiles show "split-digit" content (e.g. tile_n captures
# only the left half of a "2", tile_{n+1} captures the right half
# of "2" + the full "0"). The voter then misclassifies both.
#
# Fix: walk adjacent TOUCHING tile pairs. Compute column-ink density
# between the two tile centers. Find the lowest-ink column (the
# valley between the actual digit bodies). If the valley has notably
# less ink than the current boundary, shift the boundary to the
# valley. Re-extract glyphs from the refined spans.
#
# Conservative: only fires on TOUCHING pairs (e_left == s_right ±
# small tolerance). Doesn't fabricate boundaries when there's
# already a real gap. Min-half-width floor prevents the refiner
# from creating slivers.

_REFINE_MIN_HALF_PX = 4    # each half must be ≥ this wide
_REFINE_MIN_GAIN = 2       # valley ink must beat current by ≥ this many px


def _refine_adjacent_tile_boundaries(
    source_gray: np.ndarray,
    kept_spans: list[tuple[int, int]],
    kept_glyphs: list[np.ndarray],
) -> tuple[list[tuple[int, int]], list[np.ndarray], int]:
    """Snap touching tile boundaries to local ink valleys.

    Returns ``(new_spans, new_glyphs, n_shifted)`` where ``n_shifted``
    is the number of boundaries that moved. Glyphs are re-extracted
    from refined spans via ``_xlg._glyph_to_28x28``; if re-extract
    fails for any tile, the original glyph is preserved.
    """
    if source_gray is None or source_gray.size == 0:
        return kept_spans, kept_glyphs, 0
    if len(kept_spans) < 2:
        return kept_spans, kept_glyphs, 0

    # Polarity-canonicalize → Otsu → column ink density (matches the
    # production segmenter's projection convention).
    if int(np.median(source_gray)) > 140:
        work = 255 - source_gray
    else:
        work = source_gray
    try:
        thr = _xlg._otsu(work)
    except Exception:
        thr = int(work.mean())
    binary = (work > thr).astype(np.uint8)
    col_density = binary.sum(axis=0)
    W = source_gray.shape[1]

    new_spans = list(kept_spans)
    n_shifted = 0
    for i in range(len(new_spans) - 1):
        s_left, e_left = new_spans[i]
        s_right, e_right = new_spans[i + 1]
        # Only refine TOUCHING / overlapping pairs. Allow up to 2 px
        # gap as still-touching (chromatic aberration can split a
        # boundary into two thin strips with empty cols between).
        if e_left + 2 < s_right:
            continue
        c_left = (s_left + e_left) // 2
        c_right = (s_right + e_right) // 2
        lo = max(s_left + _REFINE_MIN_HALF_PX, c_left)
        hi = min(e_right - _REFINE_MIN_HALF_PX, c_right)
        if hi <= lo:
            continue
        sub = col_density[lo:hi]
        if sub.size == 0:
            continue
        # argmin is leftmost minimum on ties; that's fine — gives
        # consistent placement when several adjacent columns share
        # the lowest density.
        valley_x = lo + int(np.argmin(sub))
        valley_density = int(col_density[valley_x])
        current_boundary = max(s_left + 1, min(W - 1, e_left))
        current_density = int(col_density[current_boundary])
        if current_density - valley_density < _REFINE_MIN_GAIN:
            continue
        # Apply the shift.
        new_spans[i] = (s_left, valley_x)
        new_spans[i + 1] = (valley_x, e_right)
        n_shifted += 1

    if n_shifted == 0:
        return kept_spans, kept_glyphs, 0

    # Re-extract glyphs from refined spans. Fall back to original
    # glyph for any span where re-extract fails (defensive — keeps
    # the tile present in the UI even if the shift produced an
    # ink-empty crop).
    new_glyphs: list[np.ndarray] = []
    for i, (s, e) in enumerate(new_spans):
        g = _xlg._glyph_to_28x28(source_gray, s, e)
        if g is None and i < len(kept_glyphs):
            g = kept_glyphs[i]
        if g is None:
            g = np.full((28, 28), 255, dtype=np.uint8)
        new_glyphs.append(g)
    return new_spans, new_glyphs, n_shifted


# ─────────────────────────────────────────────────────────────
# Kerning-based tile proposer (Layer C)
# ─────────────────────────────────────────────────────────────
#
# Alternative to the Tesseract proposer. Workflow:
#
#   1. Run our find_comma_voted on the source RGB to locate the
#      comma's center column in source coordinates.
#   2. Read kerning_model.json for per-slot offsets (in row-height
#      units) measured from prior calibrated captures.
#   3. Project N=4-or-5 (from typed label) digit slot centers as
#      ``comma_center + slot_offset * row_height``.
#   4. Crop a 28×28 tile at each projected center with a fixed
#      digit-pitch-derived width.
#
# Returns the same tuple shape as ``_propose_glyphs`` so the UI
# layer can use either proposer interchangeably. Caller decides
# which to use based on whether Tesseract's count matched the
# typed label after the icon prefilter.
#
# Doesn't depend on Tesseract at all — sidesteps both bbox-quality
# and missed-leading-digit problems for captures where the comma
# is detectable.

_KERNING_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "hud_tracker" / "anchors" / "kerning_model.json"
)
_KERNING_MODEL: Optional[dict] = None


def _load_kerning_model() -> Optional[dict]:
    """Lazy-load kerning_model.json. Returns ``None`` if the file
    is missing, malformed, or the schema doesn't match — caller
    should then fall back to the Tesseract proposer."""
    global _KERNING_MODEL
    if _KERNING_MODEL is not None:
        return _KERNING_MODEL
    if not _KERNING_MODEL_PATH.is_file():
        return None
    try:
        doc = json.loads(_KERNING_MODEL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(doc, dict) or doc.get("schema") != "kerning_v1":
        return None
    slots = doc.get("slots") or {}
    for s in ("1", "2", "3", "4"):
        if s not in slots:
            return None
    _KERNING_MODEL = doc
    return doc


def _propose_glyphs_kerning(
    img_path: Path, label: dict,
) -> tuple[Optional[list[np.ndarray]], list[str], str,
           Optional[np.ndarray], Optional[list[tuple[int, int]]],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """Comma-anchored alternative to ``_propose_glyphs`` (Tesseract).

    Returns the same tuple shape so the UI doesn't care which
    proposer ran. Tag in third position is ``"kerning"`` so the
    user can see in the status bar which path produced the tiles.
    """
    spec = training_registry.get(KIND)
    value = str(label.get(spec.label_field, "")).strip().replace(",", "")
    chars = [c for c in value if c.isdigit()]
    if not chars or len(chars) not in (4, 5) or not img_path.is_file():
        return None, chars, "", None, None, None, None

    kerning = _load_kerning_model()
    if kerning is None:
        return None, chars, "", None, None, None, None

    # Comma detector lives in the WingmanAI hud_tracker package
    # (copied from production at integration time). Lazy import so
    # a missing/broken module doesn't break Tesseract path startup.
    try:
        from hud_tracker.anchors.comma_finder import find_comma_voted
    except Exception:
        return None, chars, "", None, None, None, None

    # Load source as both gray + RGB (mirrors the Tesseract path).
    try:
        img_l = Image.open(img_path).convert("L")
        img_rgb = Image.open(img_path).convert("RGB")
    except Exception:
        return None, chars, "", None, None, None, None
    gray_full = np.asarray(img_l, dtype=np.uint8)
    rgb_full = np.asarray(img_rgb, dtype=np.uint8)

    # Row-isolate so coordinates match what GlyphTile expects (the
    # post-isolate band is what gets shown + nudged in the UI).
    band = _xlg._find_main_row_bounds(gray_full)
    if band is not None:
        y1, y2 = band
        y1 = max(0, y1 - 2)
        y2 = min(gray_full.shape[0], y2 + 2)
        gray_iso = gray_full[y1:y2, :]
        rgb_iso = rgb_full[y1:y2, :, :]
    else:
        gray_iso = gray_full
        rgb_iso = rgb_full

    row_h = float(gray_iso.shape[0])
    if row_h < 8:
        return None, chars, "", None, None, None, None

    # Find comma — operates on the row-isolated RGB so the x
    # coordinates it returns are directly usable as tile centers.
    try:
        comma_result = find_comma_voted(rgb_iso)
    except Exception:
        comma_result = None
    if comma_result is None:
        return None, chars, "", None, None, None, None
    comma_center = float(comma_result.get("x_center", 0.0))
    if comma_center <= 0:
        return None, chars, "", None, None, None, None

    # Project slot centers from kerning offsets.
    n_digits = len(chars)
    slot_keys = ["1", "2", "3", "4"] if n_digits == 4 else ["0", "1", "2", "3", "4"]
    slots = kerning["slots"]
    slot_centers: list[float] = []
    for s in slot_keys:
        info = slots[s]
        offset = float(
            info.get("center_offset_median", info.get("center_offset_mean", 0.0))
        )
        slot_centers.append(comma_center + offset * row_h)

    # Compute a single tile width from the median digit pitch
    # between adjacent slot offsets (excluding the over-comma pitch
    # which is naturally inflated). Mirrors the same calculation
    # the production segmenter's kerning tier did before it was
    # reverted — keeps the tile bbox wide enough to absorb ±sigma
    # ink-position drift without bleeding into neighbour digits.
    pitches = []
    sorted_slot_keys = sorted(slots.keys())
    for i in range(len(sorted_slot_keys) - 1):
        ki, kj = sorted_slot_keys[i], sorted_slot_keys[i + 1]
        # Skip slot 1 -> slot 2 (over-comma pitch).
        if ki == "1" and kj == "2":
            continue
        oi = float(slots[ki].get("center_offset_median", 0.0))
        oj = float(slots[kj].get("center_offset_median", 0.0))
        delta = oj - oi
        if delta > 0.05:
            pitches.append(delta)
    if not pitches:
        return None, chars, "", None, None, None, None
    pitch_unit = sorted(pitches)[len(pitches) // 2]
    bbox_w_px = max(6.0, pitch_unit * row_h * 0.85)

    spans: list[tuple[int, int]] = []
    glyphs: list[np.ndarray] = []
    W_iso = gray_iso.shape[1]
    for cx in slot_centers:
        x1 = int(round(cx - bbox_w_px * 0.5))
        x2 = int(round(cx + bbox_w_px * 0.5))
        x1 = max(0, x1)
        x2 = min(W_iso, x2)
        if x2 - x1 < 4:
            return None, chars, "", None, None, None, None
        glyph = _xlg._glyph_to_28x28(gray_iso, x1, x2)
        if glyph is None:
            return None, chars, "", None, None, None, None
        spans.append((x1, x2))
        glyphs.append(glyph)

    # The unmasked gray is just gray_iso here — kerning proposer
    # never applies an icon mask, so masked == unmasked == gray_iso.
    return (
        glyphs, chars, f"kerning(comma_x={comma_center:.0f})",
        gray_iso, spans, rgb_iso, gray_iso,
    )


# ─────────────────────────────────────────────────────────────
# Production-pipeline integration: icon-blacklist + 4-CNN voter
# ─────────────────────────────────────────────────────────────
#
# The Tesseract proposer above produces tile candidates that may
# include the location-pin icon (when Tesseract reads the icon shape
# as "1") and may assign incorrect digit classes (Tesseract has a
# known SC-font bias on ``0`` → ``6``/``8``). Both problems show up
# as user labour in the UI: the user has to skip the icon tile
# manually and override Tesseract's wrong digits via the dropdown.
#
# We wire in two production helpers to remove that labour:
#
# * ``api._drop_blacklisted_signature_glyphs``: NCC against the
#   icon template at ≥0.55 → "this tile is the icon, auto-skip it"
# * 4-CNN voter (``_classify_crops_signal`` + ``_inv`` + ``_rgb`` +
#   ``_rgb_inv``): four decorrelated reads on the same 28×28 crop;
#   majority agreement at high confidence → "this tile is digit X"
#
# Both layers are advisory — the UI still shows the dropdown so the
# user can override anything that looks wrong. The blacklist lowers
# the dropdown default to ``—`` (skip); the voter lowers the dropdown
# default to its predicted class (which agrees with the typed label
# in the common case but differs when Tesseract's tile alignment
# put the icon in slot 0 and pushed the rest one-off).

_BLACKLIST_NCC_THR = 0.55   # mirrors api._SIG_BLACKLIST_NCC_THR
_VOTER_CONF_FLOOR = 0.50    # below this we don't surface a CNN suggestion
_VOTER_AGREE_BOOST = 0.05   # confidence boost when 2+ CNNs agree on top-1
# Icon-class CNN vote: when 2+ of the 4 CNNs predict the icon class
# ``@`` at this confidence or higher, the tile is considered the
# icon EVEN IF the NCC falls short of its threshold. Empirically the
# CNNs are more reliable than NCC at sub-32-px tile scales — NCC
# tops out around 0.6-0.7 even on real icon crops, while the gray
# CNN's @ class hits 0.95+ when the icon is in-distribution.
_ICON_CNN_CONF = 0.80
_ICON_CNN_VOTES = 2         # need at least this many CNNs voting @


def _gray_glyph_to_rgb_28x28(
    source_rgb: np.ndarray,
    x1: int,
    x2: int,
) -> Optional[np.ndarray]:
    """Crop RGB source at (x1, x2) and produce a 28×28 RGB tile in
    the SAME normalization the production RGB CNN was trained on:
    pad with 255 (white) by 2 px, BILINEAR resize to 28×28, return
    uint8 (28, 28, 3).

    Mirrors ``_xlg._glyph_to_28x28`` but for RGB. The y-tightening
    step uses the same Otsu-binary signal the gray helper does — we
    re-derive it from the RGB by taking max-of-channels (matches how
    the production pipeline derives gray from RGB elsewhere).
    """
    if source_rgb.ndim != 3 or source_rgb.shape[2] < 3:
        return None
    H, W = source_rgb.shape[:2]
    x1 = max(0, int(x1))
    x2 = min(W, int(x2))
    if x2 - x1 < 2:
        return None
    # Derive a gray equivalent for ink-row detection (max-of-channels,
    # matches production's ``rgb.max(axis=2)`` convention).
    gray = source_rgb.max(axis=2)
    if int(np.median(gray)) > 140:
        gray_for_thresh = 255 - gray
    else:
        gray_for_thresh = gray
    try:
        thr = _xlg._otsu(gray_for_thresh)
    except Exception:
        thr = int(gray_for_thresh.mean())
    binary_col = (gray_for_thresh[:, x1:x2] > thr).astype(np.uint8)
    ys = np.where(np.any(binary_col > 0, axis=1))[0]
    if len(ys) < 2:
        return None
    ya, yb = int(ys[0]), int(ys[-1]) + 1

    crop = source_rgb[ya:yb, x1:x2].astype(np.float32)
    pad = 2
    padded = np.full(
        (crop.shape[0] + pad * 2, crop.shape[1] + pad * 2, 3),
        255.0, dtype=np.float32,
    )
    padded[pad:pad + crop.shape[0], pad:pad + crop.shape[1]] = crop
    pil = Image.fromarray(padded.astype(np.uint8), mode="RGB").resize(
        (28, 28), Image.BILINEAR,
    )
    return np.asarray(pil, dtype=np.uint8)


def _classify_tile(
    gray_glyph_28x28: np.ndarray,
    rgb_glyph_28x28: Optional[np.ndarray],
) -> dict:
    """Run blacklist NCC + the 4-CNN voter on a single tile.

    Returns a dict::

        {
          "is_icon": bool,
          "icon_ncc": float,          # best NCC vs blacklist templates
          "predictions": {            # per-CNN predictions (top-1)
              "gray":     ("7", 0.93) or None,
              "gray_inv": ("7", 0.91) or None,
              "rgb":      ("7", 0.89) or None,
              "rgb_inv":  ("7", 0.85) or None,
          },
          "consensus": ("7", 0.91)    # voted top-1 + mean confidence,
                                      # or None if no agreement
        }

    All four CNN reads are gated on agreement: if 3 of 4 agree on a
    digit class with mean confidence ≥ ``_VOTER_CONF_FLOOR``, the
    consensus is that digit. If only 2 of 4 agree, consensus is set
    only when the two agreeing reads both clear the floor and the
    other two return non-digit / lower-confidence reads. Otherwise
    consensus is ``None`` and the UI falls back to the typed-label
    expected char as the dropdown default.

    Returns predictions={} + consensus=None when production helpers
    aren't available (Glyph Forge degrades to Tesseract-only mode).
    """
    out: dict = {
        "is_icon": False,
        "icon_ncc": 0.0,
        "predictions": {
            "gray": None, "gray_inv": None,
            "rgb": None, "rgb_inv": None,
        },
        "consensus": None,
    }
    api = _get_prod_api()  # lazy: pays cold-start cost on first call
    if api is None:
        return out
    if gray_glyph_28x28 is None or gray_glyph_28x28.shape != (28, 28):
        return out

    # ── Blacklist NCC (signal #1 for icon detection) ──
    # We replicate api's NCC inline (rather than calling
    # _drop_blacklisted_signature_glyphs which operates on lists +
    # mutates them) so we get the actual NCC score back, not just a
    # drop/keep decision.
    try:
        templates = api._ensure_signature_blacklist_templates()
    except Exception:
        templates = []
    if templates:
        try:
            cand = gray_glyph_28x28.astype(np.float32)
            if cand.max() > 1.5:
                cand = cand / 255.0
            cand_mean = float(cand.mean())
            cand_std = float(cand.std())
            if cand_std >= 1e-6:
                cand_norm = (cand - cand_mean) / cand_std
                best_ncc = -2.0
                for tmpl in templates:
                    ncc = float(np.mean(cand_norm * tmpl))
                    if ncc > best_ncc:
                        best_ncc = ncc
                out["icon_ncc"] = best_ncc
        except Exception:
            pass

    # ── 4-CNN voter (always runs) ──
    # We DON'T early-return when blacklist NCC ≥ threshold, because
    # at sub-32-px scales NCC is noisy enough to false-positive on
    # real digits (smoke test: real "7" hit NCC=0.65, real icon hit
    # NCC=0.50). The CNN's @ class is a more reliable signal. We
    # gather all four predictions FIRST, then combine CNN + NCC to
    # decide is_icon below.
    #
    # Production format expectations:
    #   _classify_crops_signal     : float32 [0,1] (28,28)  — list of crops
    #   _classify_crops_signal_inv : float32 [0,1] (28,28)  — list of crops
    #   _classify_crops_signal_rgb : uint8       (28,28,3)  — list of crops
    #   _classify_crops_signal_rgb_inv : uint8   (28,28,3)  — list of crops
    g_float = gray_glyph_28x28.astype(np.float32) / 255.0
    g_float_inv = np.clip(1.0 - g_float, 0.0, 1.0).astype(np.float32)

    pred_keys_grays = (
        ("gray",     "_classify_crops_signal",     [g_float]),
        ("gray_inv", "_classify_crops_signal_inv", [g_float_inv]),
    )
    for key, fn_name, crops in pred_keys_grays:
        try:
            fn = getattr(api, fn_name)
            results = fn(crops)
            if results:
                ch, conf = results[0]
                out["predictions"][key] = (str(ch), float(conf))
        except Exception:
            pass

    if rgb_glyph_28x28 is not None and rgb_glyph_28x28.shape == (28, 28, 3):
        for key, fn_name in (
            ("rgb",     "_classify_crops_signal_rgb"),
            ("rgb_inv", "_classify_crops_signal_rgb_inv"),
        ):
            try:
                fn = getattr(api, fn_name)
                results = fn([rgb_glyph_28x28])
                if results:
                    ch, conf = results[0]
                    out["predictions"][key] = (str(ch), float(conf))
            except Exception:
                pass

    # ── Icon-candidate decision (per-tile) ──
    # We compute a CANDIDATE flag here, not the final is_icon.
    # ``_show_current`` enforces a leftmost-only rule on the
    # collected candidates — the SC mining HUD always renders the
    # location-pin icon at the leftmost position of the value
    # strip, so any icon-shaped tile to the right of the leftmost
    # candidate is necessarily a false positive (typically a
    # merged-digit blob whose silhouette tripped the gray CNN's
    # ``@`` class). The leftmost rule is the cheap, robust
    # discriminator NCC alone couldn't provide because real-icon
    # NCC and merged-digit NCC overlap (~0.40-0.70 each).
    #
    # The candidate criteria stay permissive: catch any tile that
    # could plausibly be the icon. Filtering happens downstream.
    icon_votes_high = sum(
        1 for pred in out["predictions"].values()
        if pred is not None
        and not pred[0].isdigit()
        and pred[1] >= _ICON_CNN_CONF
    )
    if icon_votes_high >= _ICON_CNN_VOTES:
        out["is_icon"] = True

    # ── Digit consensus (tiered) ──
    # The 4 CNN reads are NOT all equally decorrelated:
    #
    #   gray  + gray_inv  ← same gray CNN, inverted pixels (tightly correlated)
    #   rgb   + rgb_inv   ← same RGB CNN, inverted pixels (tightly correlated)
    #
    # A simple count-of-agreements vote lets the gray family
    # dominate (2 votes for whatever gray sees) and overrules the
    # RGB family even when RGB is confident. On the SC font the
    # RGB CNN is empirically more reliable than gray (especially
    # for 0-vs-8 disambiguation: gray confuses 0 with 8 because the
    # binarization reads internal stroke variations as a middle
    # bar; RGB has colour-channel info that distinguishes "no bar"
    # from "bar"). So we tier the consensus rules:
    #
    #   Tier 1: 3+ CNNs agree on the same digit at conf >= floor
    #     -> super-majority, accept regardless of family
    #   Tier 2: RGB primary alone at conf >= 0.95
    #     -> strong RGB signal beats gray-family agreement
    #   Tier 3: 2 CNNs agree on a digit at conf >= floor
    #     -> minimum decorrelated agreement (current default)
    #   else no consensus
    if not out["is_icon"]:
        votes: dict[str, list[float]] = {}
        for pred in out["predictions"].values():
            if pred is None:
                continue
            ch, conf = pred
            if ch.isdigit() and conf >= _VOTER_CONF_FLOOR:
                votes.setdefault(ch, []).append(conf)

        # Tier 1: super-majority (3+ CNNs).
        super_maj = [c for c, cs in votes.items() if len(cs) >= 3]
        if super_maj:
            best_ch = max(
                super_maj,
                key=lambda c: float(np.mean(votes[c])),
            )
            mean_conf = float(np.mean(votes[best_ch]))
            n_agree = len(votes[best_ch])
            boosted = min(1.0, mean_conf + (n_agree - 1) * _VOTER_AGREE_BOOST)
            out["consensus"] = (best_ch, boosted)
            return out

        # Tier 2: confident RGB primary.
        rgb_pred = out["predictions"].get("rgb")
        if (
            rgb_pred is not None
            and rgb_pred[0].isdigit()
            and rgb_pred[1] >= 0.95
        ):
            out["consensus"] = (rgb_pred[0], float(rgb_pred[1]))
            return out

        # Tier 3: minimum 2-CNN agreement.
        if votes:
            best_ch = max(
                votes.keys(),
                key=lambda c: (len(votes[c]), float(np.mean(votes[c]))),
            )
            n_agree = len(votes[best_ch])
            if n_agree >= 2:
                mean_conf = float(np.mean(votes[best_ch]))
                boosted = min(
                    1.0, mean_conf + (n_agree - 1) * _VOTER_AGREE_BOOST,
                )
                out["consensus"] = (best_ch, boosted)

    return out


# ─────────────────────────────────────────────────────────────
# Tile widget
# ─────────────────────────────────────────────────────────────

class GlyphTile(QWidget):
    """One glyph row item: thumbnail + nudge controls + class dropdown.

    Holds a reference to the post-mask source array and the current
    crop bounds (x1, x2). Nudge buttons mutate the bounds and
    re-render the thumbnail in place; the FINAL glyph saved comes
    from the LAST bounds at save time. This lets the user fix
    Tesseract bbox misalignments without re-running OCR.
    """

    NUDGE_PX = 2  # pixels per nudge in source-image coords

    def __init__(
        self,
        source_gray: np.ndarray,
        x1: int,
        x2: int,
        expected: str,
        parent=None,
        *,
        source_rgb: Optional[np.ndarray] = None,
    ):
        super().__init__(parent)
        self._source = source_gray
        self._source_rgb = source_rgb
        self._x1 = int(x1)
        self._x2 = int(x2)
        self._expected = expected
        # Per-tile production-pipeline classification, recomputed each
        # time the user nudges the bounds. ``None`` until the first
        # render or when the production helpers aren't available.
        self._classification: Optional[dict] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignTop)

        # Thumbnail (upscaled to 96 px)
        self._thumb = QLabel()
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setFixedSize(96, 96)
        layout.addWidget(self._thumb)

        # Bounds readout (debug-ish, helps user know how far they nudged)
        self._bounds_lbl = QLabel("")
        self._bounds_lbl.setAlignment(Qt.AlignCenter)
        self._bounds_lbl.setStyleSheet(
            f"color: {DIM}; font-family: Consolas; font-size: 7pt;"
        )
        layout.addWidget(self._bounds_lbl)

        # Pipeline-status readout: shows the icon-blacklist NCC + the
        # 4-CNN consensus when the tile is non-icon. One row, ~9-12
        # chars total, e.g. "ncc=0.78 ICON" or "voter: 7@0.94".
        # Always present so the layout doesn't shift between tiles.
        self._pipeline_lbl = QLabel("")
        self._pipeline_lbl.setAlignment(Qt.AlignCenter)
        self._pipeline_lbl.setStyleSheet(
            f"color: {DIM}; font-family: Consolas; font-size: 7pt;"
        )
        layout.addWidget(self._pipeline_lbl)

        # Nudge controls — two rows of three small buttons each
        # Row 1: shift entire window left / right
        # Row 2: contract / expand window
        nudge_row1 = QHBoxLayout()
        nudge_row1.setSpacing(2)
        b_l = self._mini_btn("◀", "Shift left")
        b_l.clicked.connect(lambda: self._nudge(-self.NUDGE_PX, -self.NUDGE_PX))
        b_r = self._mini_btn("▶", "Shift right")
        b_r.clicked.connect(lambda: self._nudge(self.NUDGE_PX, self.NUDGE_PX))
        nudge_row1.addStretch(1)
        nudge_row1.addWidget(b_l)
        nudge_row1.addWidget(b_r)
        nudge_row1.addStretch(1)
        layout.addLayout(nudge_row1)

        nudge_row2 = QHBoxLayout()
        nudge_row2.setSpacing(2)
        b_narrow_l = self._mini_btn("L+", "Trim left edge")
        b_narrow_l.clicked.connect(lambda: self._nudge(self.NUDGE_PX, 0))
        b_widen_l = self._mini_btn("L-", "Expand left edge")
        b_widen_l.clicked.connect(lambda: self._nudge(-self.NUDGE_PX, 0))
        b_widen_r = self._mini_btn("R+", "Expand right edge")
        b_widen_r.clicked.connect(lambda: self._nudge(0, self.NUDGE_PX))
        b_narrow_r = self._mini_btn("R-", "Trim right edge")
        b_narrow_r.clicked.connect(lambda: self._nudge(0, -self.NUDGE_PX))
        nudge_row2.addWidget(b_widen_l)
        nudge_row2.addWidget(b_narrow_l)
        nudge_row2.addStretch(1)
        nudge_row2.addWidget(b_narrow_r)
        nudge_row2.addWidget(b_widen_r)
        layout.addLayout(nudge_row2)

        # Expected label
        exp_lbl = QLabel(f"expected: {expected!r}")
        exp_lbl.setAlignment(Qt.AlignCenter)
        exp_lbl.setStyleSheet(
            f"color: {DIM}; font-family: Consolas; font-size: 8pt;"
        )
        layout.addWidget(exp_lbl)

        # Class dropdown — initial value is set BELOW after we run the
        # production pipeline classification on the current crop, so
        # icons can default to SKIP and voter-confident tiles can
        # default to the voted class. Pre-built with all options so
        # ``setCurrentIndex`` calls below have indices to pick.
        self._combo = QComboBox()
        for ch in "0123456789":
            self._combo.addItem(ch)
        self._combo.addItem(SKIP_OPTION)
        self._combo.setStyleSheet(
            f"background: #2a2a2a; color: {FG}; padding: 2px 4px; "
            f"border: 1px solid #444; min-height: 20px;"
        )
        self._combo.currentTextChanged.connect(self._refresh_border)
        layout.addWidget(self._combo)

        # Initial render + classification + dropdown defaulting. Order
        # matters: thumb first so ``_classification`` is populated,
        # then dropdown init reads it, then border refresh paints
        # based on dropdown vs expected.
        self._refresh_thumb()
        self._apply_initial_dropdown_default()
        self._refresh_border()

    def _apply_initial_dropdown_default(self) -> None:
        """Set the dropdown's initial value.

        The typed label is the user's ground truth for what each
        position SHOULD contain — we always trust it as the dropdown
        default. The voter's prediction is shown in ``_pipeline_lbl``
        as an advisory signal: when it disagrees with ``expected``,
        the border turns RED so the user notices the tile may be
        bbox-misaligned or otherwise problematic.

        The ONLY case where we override the typed-label default is
        when blacklist NCC + the CNN @ vote unanimously identify the
        tile as the icon — that's a structural, not classification,
        error and the right action is "skip this tile" regardless of
        what the typed label says was supposed to be at this slot.

        Earlier versions defaulted the dropdown to the voter's read
        on non-icon tiles. That was actively dangerous: when the
        voter misclassified (e.g. read a real ``0`` as ``8`` because
        the bbox was misaligned and clipped the digit), Save & Next
        would happily file the ``0`` PNG into the ``8`` class folder,
        poisoning the training corpus. Defaulting to ``expected``
        keeps the user's typed value as the safe-by-default
        classification; voter disagreement just lights up the border.
        """
        target: str
        if self._classification and self._classification.get("is_icon"):
            target = SKIP_OPTION
        else:
            target = self._expected
        idx = self._combo.findText(target)
        if idx < 0:
            idx = self._combo.findText(self._expected)
        if idx >= 0:
            self._combo.blockSignals(True)
            try:
                self._combo.setCurrentIndex(idx)
            finally:
                self._combo.blockSignals(False)

    def _mini_btn(self, label: str, tooltip: str) -> QPushButton:
        b = QPushButton(label)
        b.setFixedSize(22, 18)
        b.setToolTip(tooltip)
        b.setStyleSheet(
            f"background: #333; color: {FG}; border: 1px solid #555; "
            f"font-family: Consolas; font-size: 8pt; padding: 0;"
        )
        return b

    def _nudge(self, dx1: int, dx2: int) -> None:
        """Shift / resize the crop window by (dx1, dx2) in source-image
        pixels. Clamps to source bounds and refuses to go below 4 px
        wide."""
        H, W = self._source.shape
        nx1 = max(0, min(W - 4, self._x1 + dx1))
        nx2 = max(nx1 + 4, min(W, self._x2 + dx2))
        if nx1 == self._x1 and nx2 == self._x2:
            return
        self._x1 = nx1
        self._x2 = nx2
        self._refresh_thumb()

    def _current_glyph(self) -> Optional[np.ndarray]:
        try:
            return _xlg._glyph_to_28x28(self._source, self._x1, self._x2)
        except Exception:
            return None

    def _refresh_thumb(self) -> None:
        g = self._current_glyph()
        if g is None:
            self._thumb.setText("(invalid)")
            self._bounds_lbl.setText(f"x={self._x1}..{self._x2} (BAD)")
            self._pipeline_lbl.setText("")
            self._classification = None
            return
        pil = Image.fromarray(g).resize((96, 96), Image.NEAREST)
        qim = ImageQt(pil)
        self._thumb.setPixmap(QPixmap.fromImage(qim))
        self._bounds_lbl.setText(
            f"x={self._x1}..{self._x2} (w={self._x2 - self._x1})"
        )
        # ── Production-pipeline classification ──
        # Recompute on every nudge so the user sees the voter's
        # updated read as they slide the crop window. Icon NCC is
        # cheap (one matmul over <10 templates); the 4-CNN voter is
        # ~10ms per tile on CPU. Cost is negligible vs the user's
        # nudge cadence.
        rgb_28 = None
        if self._source_rgb is not None:
            rgb_28 = _gray_glyph_to_rgb_28x28(
                self._source_rgb, self._x1, self._x2,
            )
        self._classification = _classify_tile(g, rgb_28)
        self._update_pipeline_label()

    def _update_pipeline_label(self) -> None:
        """Render the production-pipeline read into the small
        ``_pipeline_lbl`` text below the bounds readout. Three states:

          * Icon detected:  ``ncc=0.78 ICON`` (in WARN orange)
          * Voter consensus: ``voter: 7 @0.94`` (ACCENT green if it
                            agrees with ``expected``, RED if it
                            disagrees)
          * No consensus:   `(no voter consensus)`` in DIM gray
        """
        if self._classification is None:
            self._pipeline_lbl.setText("")
            return
        if self._classification.get("is_icon"):
            ncc = self._classification.get("icon_ncc", 0.0)
            self._pipeline_lbl.setText(f"ncc={ncc:.2f} ICON")
            self._pipeline_lbl.setStyleSheet(
                f"color: {WARN}; font-family: Consolas; font-size: 7pt;"
            )
            return
        cons = self._classification.get("consensus")
        if cons is not None:
            ch, conf = cons
            self._pipeline_lbl.setText(f"voter: {ch} @{conf:.2f}")
            agrees = (ch == self._expected)
            color = ACCENT if agrees else RED
            self._pipeline_lbl.setStyleSheet(
                f"color: {color}; font-family: Consolas; font-size: 7pt;"
            )
            return
        # No consensus and not an icon — surface the icon NCC + best
        # gray pred so the user has SOMETHING to look at when the
        # voter abstained (typically means the four CNNs disagreed).
        ncc = self._classification.get("icon_ncc", 0.0)
        gray_pred = self._classification.get("predictions", {}).get("gray")
        gray_str = (
            f"gray: {gray_pred[0]} @{gray_pred[1]:.2f}"
            if gray_pred is not None else "gray: ?"
        )
        self._pipeline_lbl.setText(f"ncc={ncc:.2f} {gray_str}")
        self._pipeline_lbl.setStyleSheet(
            f"color: {DIM}; font-family: Consolas; font-size: 7pt;"
        )

    def _refresh_border(self) -> None:
        val = self._combo.currentText()
        if val == self._expected:
            color = ACCENT
        elif val == SKIP_OPTION:
            color = WARN
        else:
            color = RED
        self._thumb.setStyleSheet(
            f"background: #2a2a2a; border: 2px solid {color}; "
            f"border-radius: 4px;"
        )

    @property
    def chosen_class(self) -> Optional[str]:
        v = self._combo.currentText()
        return None if v == SKIP_OPTION else v

    @property
    def glyph(self) -> Optional[np.ndarray]:
        # ALWAYS render at the current bounds — guarantees the saved
        # glyph reflects whatever the user last nudged to.
        return self._current_glyph()

    # ── Sidecar-JSON accessors ──
    # Exposed so the save loop can persist verified per-tile
    # positions + the production-pipeline classification snapshot
    # alongside the per-glyph PNGs. These are read-only views; the
    # widget owns mutation.
    @property
    def x1(self) -> int:
        return int(self._x1)

    @property
    def x2(self) -> int:
        return int(self._x2)

    @property
    def expected(self) -> str:
        return self._expected

    @property
    def classification_snapshot(self) -> Optional[dict]:
        """Last computed icon-NCC + voter result, or ``None`` if not
        yet rendered. Consumed by the sidecar JSON serializer."""
        return self._classification


# ─────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────

class GlyphForge(QWidget):
    def __init__(self, *, defer_first_render: bool = False):
        super().__init__()
        self.setWindowTitle("Glyph Forge — Tesseract proposes, you verify")
        self.resize(1100, 580)
        self.setStyleSheet(f"background: {BG}; color: {FG};")

        spec = training_registry.get(KIND)
        self._spec = spec
        self._out_root = spec.glyph_staging_dir
        self._out_root.mkdir(parents=True, exist_ok=True)

        # Build capture queue: every labeled JSON in registered sources,
        # MINUS any whose first glyph already exists in staging (so
        # this tool is resumable across runs).
        all_caps: list[Path] = []
        for src_dir in training_registry.get_training_sources(KIND):
            for j in src_dir.glob("cap_*.json"):
                if j.name.endswith(".boxes.json"):
                    continue
                try:
                    d = json.loads(j.read_text(encoding="utf-8"))
                except Exception:
                    continue
                v = (d.get(spec.label_field) or "").strip()
                if v:
                    all_caps.append(j)
        all_caps.sort()
        self._all_caps = all_caps

        self._idx = self._first_pending()

        self._build_ui()
        # Defer the first capture render until AFTER ``show()`` so the
        # GUI appears immediately. The first capture render triggers
        # cold ONNX session loads (4 models, can take 5-30s on first
        # use); without this defer the user sees a hung window.
        # ``main()`` calls ``_show_current`` via ``QTimer.singleShot``
        # right after ``show()``.
        if not defer_first_render:
            self._show_current()
        else:
            self._status_lbl.setText(
                "Loading first capture + production helpers (cold start, "
                "may take 5-30s)..."
            )

    def _first_pending(self) -> int:
        """Find first capture whose 'glyph 0' hasn't been saved yet.

        Single-pass scan: walks every class dir ONCE, builds a set of
        capture stems with at least one saved glyph, then linear-scans
        the capture queue for the first one missing from the set.

        Why this matters: the previous implementation re-globbed every
        class dir for every capture (O(captures × classes × files_per_class)).
        With ~232 captures × 12 class dirs × 2000+ aug_auto + user PNGs
        per class, that's millions of stat calls — Windows + antivirus
        was running this for several minutes before the GUI appeared.
        Single-pass is O(total_files), bounded by the size of the
        output folder — typically <1s even on slow disks.
        """
        import time as _t
        _t0 = _t.perf_counter()
        # Collect every saved <stem> across all class dirs in one scan.
        saved_stems: set[str] = set()
        try:
            for cls_dir in self._out_root.iterdir():
                if not cls_dir.is_dir() or cls_dir.name.startswith("_"):
                    continue
                # ``user_<stem>_<idx>.png`` — strip the ``user_`` prefix
                # and the trailing ``_<idx>.png`` to recover the stem.
                # We use direct directory iteration instead of glob to
                # skip the pattern-match cost on Windows (glob walks
                # the full directory then filters).
                for entry in cls_dir.iterdir():
                    name = entry.name
                    if not name.startswith("user_") or not name.endswith(".png"):
                        continue
                    base = name[len("user_"):-len(".png")]
                    # base now looks like "<stem>_<idx>". Strip
                    # rightmost _<idx>.
                    cut = base.rsplit("_", 1)
                    if len(cut) == 2 and cut[1].isdigit():
                        saved_stems.add(cut[0])
                    else:
                        saved_stems.add(base)
        except Exception as exc:
            print(f"glyph_forge: _first_pending scan partial ({exc}) - "
                  f"defaulting to index 0")
            return 0

        result = 0
        for i, j in enumerate(self._all_caps):
            png = j.with_suffix(".png")
            if not png.is_file():
                continue
            if png.stem not in saved_stems:
                result = i
                break
        else:
            # All captures already processed — start at the last index
            # so the user sees the most recently completed one.
            result = max(0, len(self._all_caps) - 1)

        elapsed = _t.perf_counter() - _t0
        print(f"glyph_forge: scanned {len(saved_stems)} processed captures "
              f"in {elapsed:.2f}s, resuming at index {result}")
        return result

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # Header
        title = QLabel("GLYPH FORGE", self)
        tf = QFont("Consolas")
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        title.setStyleSheet(f"color: {ACCENT}; background: transparent;")
        root.addWidget(title)

        sub = QLabel(
            "Tesseract proposes one tile per typed digit. Eyeball each "
            "tile, change the dropdown if the wrong digit is shown, then "
            "Save & Next. Skip captures Tesseract reads wrong.",
            self,
        )
        sub.setStyleSheet(
            f"color: {DIM}; font-size: 9pt; background: transparent;"
        )
        sub.setWordWrap(True)
        root.addWidget(sub)

        # Capture preview (full original image, upscaled 2x for visibility)
        self._capture_lbl = QLabel("(loading)", self)
        self._capture_lbl.setMinimumHeight(120)
        self._capture_lbl.setAlignment(Qt.AlignCenter)
        self._capture_lbl.setStyleSheet(
            f"background: #181818; border: 1px solid #333; padding: 8px;"
        )
        root.addWidget(self._capture_lbl)

        # Status line
        self._status_lbl = QLabel("", self)
        self._status_lbl.setStyleSheet(
            f"color: {FG}; font-family: Consolas; font-size: 10pt; "
            f"background: transparent;"
        )
        root.addWidget(self._status_lbl)

        # Tile row
        self._tile_holder = QWidget()
        self._tile_layout = QHBoxLayout(self._tile_holder)
        self._tile_layout.setContentsMargins(0, 4, 0, 4)
        self._tile_layout.setSpacing(8)
        self._tile_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        root.addWidget(self._tile_holder)

        # Buttons
        btn_row = QHBoxLayout()
        prev_btn = QPushButton("◀ Prev (PgUp)", self)
        prev_btn.clicked.connect(lambda: self._step(-1))
        btn_row.addWidget(prev_btn)

        skip_btn = QPushButton("Skip (PgDn)", self)
        skip_btn.clicked.connect(lambda: self._step(+1))
        btn_row.addWidget(skip_btn)

        save_btn = QPushButton("Save & Next  (Ctrl+Enter)", self)
        save_btn.setStyleSheet(
            f"background: {ACCENT}; color: black; padding: 6px 16px; "
            f"font-weight: bold; border: none; border-radius: 3px;"
        )
        save_btn.clicked.connect(self._save_and_next)
        btn_row.addWidget(save_btn)

        btn_row.addStretch(1)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, max(1, len(self._all_caps)))
        self._progress.setFixedWidth(260)
        btn_row.addWidget(self._progress)

        root.addLayout(btn_row)

        # Shortcuts
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._save_and_next)
        QShortcut(QKeySequence("Ctrl+S"),      self, activated=self._save_and_next)
        QShortcut(QKeySequence("PgDown"),      self, activated=lambda: self._step(+1))
        QShortcut(QKeySequence("PgUp"),        self, activated=lambda: self._step(-1))

    def _step(self, delta: int) -> None:
        new = self._idx + delta
        if 0 <= new < len(self._all_caps):
            self._idx = new
            self._show_current()

    def _clear_tiles(self) -> None:
        while self._tile_layout.count():
            it = self._tile_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()

    def _show_current(self) -> None:
        if not self._all_caps:
            self._status_lbl.setText("No labeled captures found.")
            return
        self._clear_tiles()

        json_path = self._all_caps[self._idx]
        png = json_path.with_suffix(".png")
        try:
            label = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            label = {}
        # Capture preview
        try:
            cap_img = Image.open(png).convert("RGB")
            scale = 2
            cap_img = cap_img.resize(
                (cap_img.width * scale, cap_img.height * scale), Image.NEAREST,
            )
            qim = ImageQt(cap_img)
            self._capture_lbl.setPixmap(QPixmap.fromImage(qim))
        except Exception as e:
            self._capture_lbl.setText(f"(failed: {e})")

        glyphs, expected_chars, tag, source_gray, spans, source_rgb, source_gray_unmasked = (
            _propose_glyphs(png, label)
        )
        value = label.get(self._spec.label_field, "")
        self._progress.setValue(self._idx + 1)
        self._progress.setFormat(
            f"{self._idx + 1} / {len(self._all_caps)} captures"
        )

        # If Tesseract couldn't agree on a tile set, try the kerning
        # proposer immediately as a fallback (Layer C). Kerning uses
        # find_comma_voted + measured per-slot offsets so it doesn't
        # depend on Tesseract bboxes at all. May succeed when
        # Tesseract's variant-search rejected every PSM/scale combo.
        if glyphs is None or source_gray is None or spans is None:
            kg, kchars, ktag, kgray, kspans, krgb, kgu = _propose_glyphs_kerning(
                png, label,
            )
            if (
                kg is not None and kgray is not None and kspans is not None
            ):
                glyphs, expected_chars, tag = kg, kchars, ktag
                source_gray, spans, source_rgb = kgray, kspans, krgb
                source_gray_unmasked = kgu
            else:
                self._status_lbl.setText(
                    f"[{self._idx + 1}/{len(self._all_caps)}]  "
                    f"{png.name}  label={value!r}  "
                    f"⚠ Tesseract + kerning both failed - Skip."
                )
                return

        # ── Icon prefilter ──────────────────────────────────────
        # Run blacklist NCC + 4-CNN @ vote on every proposed tile
        # BEFORE building widgets. Drop tiles flagged as icons so
        # they never appear in the row at all (was: shown with skip
        # dropdown, still cluttered the row + pushed expected-char
        # alignment off by one). Then re-zip the surviving tiles
        # with ``expected_chars`` taking the LAST K entries — the
        # icon almost always eats the leading position(s), so the
        # remaining tiles correspond to the trailing K typed digits.
        # The first ``_classify_tile`` invocation here triggers the
        # production-helper cold load (4 ONNX sessions); subsequent
        # tiles are warm.
        # Per-tile icon-candidate classification (sorted left-to-right
        # by Tesseract). We collect candidates here, then enforce a
        # leftmost-only rule below — SC HUD always puts the icon at
        # the leftmost position of the value strip, so any icon-
        # candidate tile to the right of the leftmost one is a false
        # positive (typically a merged-digit blob whose silhouette
        # tripped the gray CNN's ``@`` class).
        per_tile_classifications: list[dict] = []
        for (x1, x2), g in zip(spans, glyphs):
            rgb_28 = (
                _gray_glyph_to_rgb_28x28(source_rgb, x1, x2)
                if source_rgb is not None else None
            )
            per_tile_classifications.append(_classify_tile(g, rgb_28))

        # Find the leftmost candidate. Tiles after it that are still
        # marked is_icon get UN-marked here.
        first_candidate_idx: Optional[int] = None
        for i, cls in enumerate(per_tile_classifications):
            if cls.get("is_icon"):
                first_candidate_idx = i
                break

        kept_spans: list[tuple[int, int]] = []
        kept_glyphs: list[np.ndarray] = []
        n_icon_dropped = 0
        for i, ((x1, x2), g) in enumerate(zip(spans, glyphs)):
            cls = per_tile_classifications[i]
            # Drop ONLY the leftmost icon candidate (real SC icon is
            # always at the leftmost x). All later "icon" votes are
            # spurious — keep those tiles so their digit content
            # doesn't get lost.
            is_real_icon = (
                cls.get("is_icon") and i == first_candidate_idx
            )
            if is_real_icon:
                n_icon_dropped += 1
                continue
            kept_spans.append((x1, x2))
            kept_glyphs.append(g)

        # ── Layer B: leading-digit recovery ─────────────────────
        # When Tesseract's icon-mask was wider than the icon graphic,
        # leading digit columns get zeroed before Tesseract sees
        # them. Scan the unmasked-but-row-isolated gray for ink
        # blobs LEFT of the leftmost surviving tile. Each blob runs
        # through the icon-blacklist + voter; non-icon digit blobs
        # get prepended to ``kept_spans`` as recovered tiles.
        n_recovered = 0
        if (
            source_gray_unmasked is not None
            and len(kept_spans) < len(expected_chars)
            and kept_spans
        ):
            leftmost_x = kept_spans[0][0]
            recovery_blobs = _scan_for_missed_leading_digits(
                source_gray_unmasked, leftmost_x,
            )
            recovered_to_prepend: list[
                tuple[tuple[int, int], np.ndarray]
            ] = []
            for (bx1, bx2) in recovery_blobs:
                rg = _xlg._glyph_to_28x28(
                    source_gray_unmasked, bx1, bx2,
                )
                if rg is None:
                    continue
                rrgb_28 = (
                    _gray_glyph_to_rgb_28x28(source_rgb, bx1, bx2)
                    if source_rgb is not None else None
                )
                rcls = _classify_tile(rg, rrgb_28)
                # Skip blobs the pipeline classifies as the icon —
                # those are icon residue we already meant to drop.
                if rcls.get("is_icon"):
                    continue
                recovered_to_prepend.append(((bx1, bx2), rg))
            # Prepend recovered tiles in left-to-right order.
            recovered_to_prepend.sort(key=lambda kv: kv[0][0])
            for (bx1, bx2), rg in recovered_to_prepend:
                kept_spans.insert(0, (bx1, bx2))
                kept_glyphs.insert(0, rg)
                n_recovered += 1
                # Stop once we've recovered enough to match expected.
                if len(kept_spans) >= len(expected_chars):
                    break
        # Sort kept_spans left-to-right after recovery (any tiles
        # we prepended need to keep the left-to-right invariant the
        # downstream zip(spans, expected) relies on).
        if n_recovered > 0:
            paired = sorted(
                zip(kept_spans, kept_glyphs), key=lambda kv: kv[0][0],
            )
            kept_spans = [k for k, _ in paired]
            kept_glyphs = [g for _, g in paired]

        # ── Layer F: wide-tile splitter ────────────────────────
        # When kept count is short of expected AND a tile is much
        # wider than its peers, Tesseract probably merged two
        # digits. Split at the lowest-ink valley column in the
        # tile's central region. Iterates until count matches OR
        # no tile remains wide enough to split.
        ref_source = (
            source_gray_unmasked
            if source_gray_unmasked is not None else source_gray
        )
        n_split = 0
        if ref_source is not None and len(kept_spans) < len(expected_chars):
            kept_spans, kept_glyphs, n_split = _split_wide_tiles(
                ref_source, kept_spans, kept_glyphs, len(expected_chars),
            )

        # ── Layer D: ink-valley boundary refinement ────────────
        # When Tesseract places a boundary inside a digit instead
        # of in the gap between digits, adjacent tiles end up
        # showing split-digit content (e.g. tile_n=left-half-of-2,
        # tile_{n+1}=right-half-of-2 + full 0). Snap touching
        # boundaries to local ink minima so each tile contains
        # exactly one digit's ink. Operates on the unmasked gray
        # so the icon-mask doesn't deform the per-column projection
        # we're searching for valleys in.
        n_refined = 0
        ref_source = source_gray_unmasked if source_gray_unmasked is not None else source_gray
        if ref_source is not None and len(kept_spans) >= 2:
            kept_spans, kept_glyphs, n_refined = _refine_adjacent_tile_boundaries(
                ref_source, kept_spans, kept_glyphs,
            )

        # ── Layer H: bbox tighten-to-ink ───────────────────────
        # Per-tile, contract bbox to actual ink extent. Removes
        # leading/trailing background pixels that contaminate the
        # CNN's view of the digit (comma-edge residue, pre-icon
        # halo, etc.). Eliminates the 11,520 "5->3" pattern where
        # tile[2]'s left edge included the comma's right halo.
        n_tightened = 0
        if ref_source is not None and kept_spans:
            kept_spans, kept_glyphs, n_tightened = _tighten_tiles_to_ink(
                ref_source, kept_spans, kept_glyphs,
            )

        # ── Layer C: kerning fallback ──────────────────────────
        # Recovery couldn't bring kept count up to expected (or had
        # nothing to scan). Try the kerning-anchored proposer as a
        # whole-image alternative: find the comma, project N slot
        # centers, crop tiles. Doesn't depend on Tesseract's bboxes
        # at all. We only adopt the kerning result if it gets the
        # tile count exactly right — partial agreement isn't worth
        # discarding the recovery progress we already made.
        if len(kept_spans) != len(expected_chars):
            kg, kchars, ktag, kgray, kspans, krgb, kgu = _propose_glyphs_kerning(
                png, label,
            )
            if (
                kg is not None and kgray is not None and kspans is not None
                and len(kspans) == len(expected_chars)
            ):
                # Re-run icon prefilter on kerning's tiles too, in
                # case slot[0] happened to project onto the icon.
                k_kept_spans: list[tuple[int, int]] = []
                k_kept_glyphs: list[np.ndarray] = []
                for (kx1, kx2), kgl in zip(kspans, kg):
                    krgb_28 = (
                        _gray_glyph_to_rgb_28x28(krgb, kx1, kx2)
                        if krgb is not None else None
                    )
                    kcls = _classify_tile(kgl, krgb_28)
                    if kcls.get("is_icon"):
                        continue
                    k_kept_spans.append((kx1, kx2))
                    k_kept_glyphs.append(kgl)
                if len(k_kept_spans) == len(expected_chars):
                    kept_spans = k_kept_spans
                    kept_glyphs = k_kept_glyphs
                    # Adopt kerning's source arrays so GlyphTile's
                    # crop coordinates resolve into the same image
                    # the spans were measured against.
                    source_gray = kgray
                    source_rgb = krgb
                    source_gray_unmasked = kgu
                    tag = ktag
                    n_icon_dropped += (len(kspans) - len(k_kept_spans))

        # Re-align expected_chars to surviving tiles. Three cases:
        #   * len(kept) == len(expected): zip directly. Most common
        #     when no icons were detected, OR when Tesseract proposed
        #     N+icon tiles and we dropped the icon (so kept count
        #     matches expected).
        #   * len(kept) <  len(expected): Tesseract lost some digits
        #     to icon fusion. Assume the icon ate LEADING positions;
        #     align expected[-K:] with kept tiles. Status line warns
        #     so user can Skip the capture.
        #   * len(kept) >  len(expected): Tesseract over-segmented.
        #     Take the last len(expected) tiles (rightmost are most
        #     reliable on SC HUD captures, where left-edge icon /
        #     stroke residue tends to leak in).
        if len(kept_spans) <= len(expected_chars):
            aligned_expected = expected_chars[-len(kept_spans):]
        else:
            kept_spans = kept_spans[-len(expected_chars):]
            kept_glyphs = kept_glyphs[-len(expected_chars):]
            aligned_expected = list(expected_chars)

        # Status line surfaces pipeline state + alignment outcome.
        # Pipeline is "available" if the lazy loader could resolve it
        # (or hasn't been disabled via env var). The cold load may
        # have just happened during the prefilter loop above.
        helper_status = (
            "pipeline: ON"
            if (_prod_api is not None or not _PIPELINE_DISABLED)
            else "pipeline: OFF"
        )
        rgb_status = (
            "rgb: yes" if source_rgb is not None else "rgb: no"
        )
        align_warn = ""
        if len(kept_spans) != len(expected_chars):
            align_warn = (
                f"  !! kept={len(kept_spans)} vs "
                f"expected={len(expected_chars)} - SKIP RECOMMENDED"
            )

        recovered_str = (
            f" +{n_recovered} recovered" if n_recovered else ""
        )
        split_str = (
            f" +{n_split} split" if n_split else ""
        )
        refined_str = (
            f" +{n_refined} bboxes refined" if n_refined else ""
        )
        tightened_str = (
            f" +{n_tightened} tightened" if n_tightened else ""
        )
        self._status_lbl.setText(
            f"[{self._idx + 1}/{len(self._all_caps)}]  "
            f"{png.name}  label={value!r}  "
            f"variant={tag}  -> {len(kept_spans)} tiles  "
            f"(dropped {n_icon_dropped} icon{recovered_str}"
            f"{split_str}{refined_str}{tightened_str})  "
            f"{helper_status}  {rgb_status}{align_warn}"
        )
        for (x1, x2), ch in zip(kept_spans, aligned_expected):
            tile = GlyphTile(
                source_gray, x1, x2, ch, self,
                source_rgb=source_rgb,
            )
            self._tile_layout.addWidget(tile)
        self._tile_layout.addStretch(1)

    def _save_and_next(self) -> None:
        # Iterate the tiles, save each one whose dropdown != skip
        json_path = self._all_caps[self._idx]
        png = json_path.with_suffix(".png")
        src_name = png.stem
        saved = 0
        # Snapshot per-tile state for the sidecar JSON. We capture
        # FINAL bounds (post-nudge), final dropdown choice, and the
        # production-pipeline classification result so downstream
        # consumers (kerning calibration, decontamination passes,
        # whole-row training) have everything they need without
        # re-running Tesseract / the CNN voter on the source image.
        sidecar_tiles: list[dict] = []
        for i in range(self._tile_layout.count()):
            w = self._tile_layout.itemAt(i).widget()
            if not isinstance(w, GlyphTile):
                continue
            cls = w.chosen_class  # None when SKIP_OPTION selected
            g = w.glyph  # re-renders at CURRENT bounds (post-nudge)
            tile_record: dict = {
                "x1": w.x1,
                "x2": w.x2,
                "expected": w.expected,
                "saved_class": cls,
                "skipped": cls is None,
            }
            # Embed the production-pipeline read so we can later
            # cross-check the user's choice against what the icon-
            # blacklist + 4-CNN voter said. Useful for auditing
            # disagreements (false positives on the blacklist,
            # surprising voter consensus, etc.).
            snap = w.classification_snapshot
            if snap is not None:
                tile_record["pipeline"] = {
                    "is_icon": bool(snap.get("is_icon", False)),
                    "icon_ncc": float(snap.get("icon_ncc", 0.0)),
                    "predictions": {
                        k: list(v) if v is not None else None
                        for k, v in (
                            snap.get("predictions", {}) or {}
                        ).items()
                    },
                    "consensus": (
                        list(snap["consensus"])
                        if snap.get("consensus") is not None else None
                    ),
                }
            sidecar_tiles.append(tile_record)

            if cls is None or g is None:
                continue
            if _xlg._save_glyph(g, cls, src_name, self._out_root):
                saved += 1

        # Persist sidecar JSON next to the source PNG. Schema is
        # self-describing so future tools can detect the version +
        # gracefully degrade if fields they expect are missing.
        sidecar_path = png.with_suffix(".glyphs.json")
        try:
            label_dict: dict = {}
            try:
                label_dict = json.loads(
                    json_path.read_text(encoding="utf-8"),
                )
            except Exception:
                pass
            label_value = (
                label_dict.get(self._spec.label_field, "")
                if isinstance(label_dict, dict) else ""
            )
            sidecar_doc = {
                "schema": "glyph_forge_v1",
                "capture": src_name,
                "label": str(label_value),
                "kind": KIND,
                "tile_count": len(sidecar_tiles),
                "saved_count": saved,
                "skipped_count": sum(1 for t in sidecar_tiles if t["skipped"]),
                "tiles": sidecar_tiles,
            }
            sidecar_path.write_text(
                json.dumps(sidecar_doc, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            self._status_lbl.setText(
                f"Saved {saved} glyphs from {png.name}; sidecar FAILED: {exc}"
            )
            QTimer.singleShot(800, lambda: self._step(+1))
            return

        self._status_lbl.setText(
            f"Saved {saved} glyphs from {png.name} "
            f"+ sidecar -> advancing"
        )
        # Advance after a short delay so the user can see the message
        QTimer.singleShot(120, lambda: self._step(+1))


def main() -> None:
    import time as _t
    print("glyph_forge: starting", flush=True)
    _t0 = _t.perf_counter()

    app = QApplication.instance() or QApplication(sys.argv)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG))
    palette.setColor(QPalette.WindowText, QColor(FG))
    palette.setColor(QPalette.Base, QColor("#2a2a2a"))
    palette.setColor(QPalette.Text, QColor(FG))
    palette.setColor(QPalette.Button, QColor("#444"))
    palette.setColor(QPalette.ButtonText, QColor(FG))
    app.setPalette(palette)
    print(f"glyph_forge: Qt initialized in {_t.perf_counter() - _t0:.2f}s",
          flush=True)

    _t1 = _t.perf_counter()
    forge = GlyphForge(defer_first_render=True)
    print(f"glyph_forge: GlyphForge() built in {_t.perf_counter() - _t1:.2f}s "
          f"(GUI only - first capture render deferred to post-show)",
          flush=True)

    forge.show()
    forge.raise_()
    print(f"glyph_forge: window shown, cold-start "
          f"{_t.perf_counter() - _t0:.2f}s; rendering first capture next "
          f"(may take 5-30s for ONNX cold load)", flush=True)

    # Render first capture on the next event-loop tick so the window
    # paints first. ``QTimer.singleShot(0, ...)`` runs the callback as
    # soon as the event loop returns to its idle state — i.e. after
    # the initial paint events are flushed.
    def _render_first():
        _r0 = _t.perf_counter()
        forge._show_current()
        print(f"glyph_forge: first capture rendered in "
              f"{_t.perf_counter() - _r0:.2f}s", flush=True)

    QTimer.singleShot(0, _render_first)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
