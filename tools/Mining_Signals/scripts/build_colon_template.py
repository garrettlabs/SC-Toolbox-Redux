"""Build ``ocr/sc_templates/colon.npz`` — a canonical colon-only NCC template.

The HUD has three rows (MASS, RESISTANCE, INSTABILITY) and each label
ends with a colon (":") followed by the value. The full label-template
NCC anchors (``label_match.find_label_positions``) fail when the label
text is partially occluded / dim / dirty — but the colon, being a tiny
two-stacked-dot glyph that's identical across all three labels and at
all HUD scales, is much more recoverable. A standalone colon detector
gives ``_label_rows_from_anchor`` a second, independent anchor to fall
back on (see ``ocr/sc_ocr/colon_anchor.py``).

This script extracts the colon glyph from the existing label templates
in ``ocr/sc_templates/labels.npz``:

  1. Load each (mass / resistance / instability) template.
  2. Scan columns for the characteristic colon pattern: two short
     bright runs (each 2-5 px) separated by a short gap (~3-8 px),
     with NO other tall strokes in the same column.
  3. Find the RIGHTMOST cluster of such columns — that's the colon
     (preceding text characters never produce this clean two-dot
     pattern; trailing "%" or value digits might, but the rightmost
     cluster is biased toward the colon since the colon sits between
     the label and the value crop area).
  4. Slice the colon region (~4-7 cols wide) with 1 px padding either
     side.
  5. Right-align and average all extracted glyphs into one canonical
     colon template at the labels' native height (28 px).
  6. Save to ``ocr/sc_templates/colon.npz`` with keys ``colon``
     (uint8 H×W) and ``height`` (int32).

If a label template doesn't contain a detectable colon (e.g. the
current MASS template was over-trimmed by ``rebuild_label_templates_tight.py``
and lost its colon), we skip it and average what we have — typically
RESISTANCE and INSTABILITY both have intact colons at their right edges.

Run:
    python scripts/build_colon_template.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
)
log = logging.getLogger("build_colon_template")

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
LABELS_NPZ = TOOL / "ocr" / "sc_templates" / "labels.npz"
COLON_NPZ = TOOL / "ocr" / "sc_templates" / "colon.npz"
DEBUG_DIR = TOOL / "ocr" / "sc_templates" / "labels_debug"
CANONICAL_HEIGHT = 28
PAD_COLS = 1  # 1 px padding either side of detected colon


def _binarize(template: np.ndarray) -> tuple[np.ndarray, int]:
    """Midpoint-threshold binarization. Returns (binary uint8, thr)."""
    t = template.astype(np.int32)
    thr = (int(t.min()) + int(t.max())) // 2
    bn = (t > thr).astype(np.uint8)
    return bn, thr


def _column_runs(col: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, length) bright-run pairs in a single column."""
    H = col.shape[0]
    runs: list[tuple[int, int]] = []
    start = -1
    for y in range(H):
        if col[y] and start < 0:
            start = y
        elif not col[y] and start >= 0:
            runs.append((start, y - start))
            start = -1
    if start >= 0:
        runs.append((start, H - start))
    return runs


def _is_colon_column(col: np.ndarray, H: int) -> bool:
    """True iff a single column matches a colon-dot pattern.

    Criteria:
      - Exactly 2 bright runs.
      - Each run is 2..6 px tall (typical colon dot).
      - Total ink ≤ 12 px (excludes tall strokes like the bar of E or T).
      - First run starts at y ≥ 4 (excludes top-line decoration).
      - Second run ends at y ≤ H - 4 (excludes baseline strokes).
      - Gap between runs is 2..12 px.
    """
    runs = _column_runs(col)
    if len(runs) != 2:
        return False
    r1, r2 = runs
    if not (2 <= r1[1] <= 6 and 2 <= r2[1] <= 6):
        return False
    if (r1[1] + r2[1]) > 12:
        return False
    if r1[0] < 4 or (r2[0] + r2[1]) > H - 4:
        return False
    gap = r2[0] - (r1[0] + r1[1])
    if not (2 <= gap <= 12):
        return False
    return True


def _find_colon_columns(binary: np.ndarray) -> list[int]:
    """Return all column indices that look like a colon-dot pattern."""
    H, W = binary.shape
    out: list[int] = []
    for x in range(W):
        if _is_colon_column(binary[:, x], H):
            out.append(x)
    return out


def _rightmost_colon_cluster(columns: list[int]) -> Optional[tuple[int, int]]:
    """Group consecutive colon columns into clusters; return (left, right)
    of the rightmost cluster (inclusive). Returns None if no clusters.

    Two columns belong to the same cluster if they are within 2 px of
    each other (allows for anti-aliasing gaps at the edges of the dots).
    """
    if not columns:
        return None
    clusters: list[list[int]] = [[columns[0]]]
    for x in columns[1:]:
        if x - clusters[-1][-1] <= 2:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    last = clusters[-1]
    return (last[0], last[-1])


def _extract_colon_glyph(
    template: np.ndarray,
    label_name: str,
) -> Optional[np.ndarray]:
    """Find and slice the colon from one label template.

    Returns a uint8 (H, W) slice tight to the colon dots plus 1-px
    padding either side, or None if no clean colon cluster is found.
    """
    binary, thr = _binarize(template)
    cols = _find_colon_columns(binary)
    if not cols:
        log.info("  %s: no colon-pattern columns found", label_name)
        return None
    cluster = _rightmost_colon_cluster(cols)
    if cluster is None:
        log.info("  %s: no colon cluster", label_name)
        return None
    left, right = cluster
    if (right - left + 1) < 2:
        log.info(
            "  %s: rightmost cluster too narrow (cols %d..%d)",
            label_name, left, right,
        )
        return None
    W = template.shape[1]
    x_start = max(0, left - PAD_COLS)
    x_end = min(W, right + 1 + PAD_COLS)
    slc = template[:, x_start:x_end].astype(np.uint8)
    log.info(
        "  %s: colon cluster cols %d..%d (sliced %d..%d, w=%d)",
        label_name, left, right, x_start, x_end, slc.shape[1],
    )
    return slc


def _average_colons(glyphs: list[np.ndarray]) -> np.ndarray:
    """Right-align and average glyphs into one canonical template.

    All inputs share the same height (templates are pre-normalized to
    CANONICAL_HEIGHT). Width may differ slightly; we right-align (since
    the colon sits flush to the right of its containing box, regardless
    of trailing padding) and average overlapping regions.
    """
    if not glyphs:
        raise ValueError("no glyphs to average")
    h = glyphs[0].shape[0]
    for g in glyphs:
        if g.shape[0] != h:
            raise ValueError(f"height mismatch: {g.shape} vs {h}")
    max_w = max(g.shape[1] for g in glyphs)
    accum = np.zeros((h, max_w), dtype=np.float32)
    counts = np.zeros((h, max_w), dtype=np.float32)
    for g in glyphs:
        pad = max_w - g.shape[1]
        accum[:, pad:] += g.astype(np.float32)
        counts[:, pad:] += 1.0
    avg = np.where(counts > 0, accum / np.maximum(counts, 1.0), 0.0)
    return avg.astype(np.uint8)


def main() -> int:
    if not LABELS_NPZ.is_file():
        log.error("labels.npz not found at %s", LABELS_NPZ)
        return 1
    data = np.load(LABELS_NPZ)
    height = int(data["height"]) if "height" in data else CANONICAL_HEIGHT
    log.info("loaded labels.npz (height=%d)", height)

    glyphs: list[np.ndarray] = []
    for key in ("mass", "resistance", "instability"):
        if key not in data:
            log.warning("  %s: not in labels.npz", key)
            continue
        tpl = data[key]
        log.info("  %s: shape=%s", key, tpl.shape)
        g = _extract_colon_glyph(tpl, key)
        if g is not None:
            glyphs.append(g)

    if not glyphs:
        log.error("no colons extracted — labels.npz may need rebuilding")
        return 1

    avg = _average_colons(glyphs)
    log.info("averaged %d colon glyphs -> canonical template shape=%s",
             len(glyphs), avg.shape)

    payload = {"colon": avg, "height": np.int32(height)}
    COLON_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(COLON_NPZ), **payload)
    log.info("wrote %s", COLON_NPZ)

    # Save debug PNG (8x upscale for visual inspection).
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    big = np.repeat(np.repeat(avg, 8, axis=0), 8, axis=1)
    Image.fromarray(big).save(DEBUG_DIR / "colon_template_8x.png")
    Image.fromarray(avg).save(DEBUG_DIR / "colon_template.png")
    log.info("debug PNGs saved under %s", DEBUG_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
