"""Split captured SCAN RESULTS panel images into per-field training crops.

Usage:
    python split_panels.py [--input training_data_panels] [--output training_data_split]

Reuses ``_find_label_rows`` from ocr/onnx_hud_reader.py to locate the
MASS / RESISTANCE / INSTABILITY rows, then derives the rest of the
field positions geometrically (mineral name above MASS, difficulty
below INSTABILITY, COMPOSITION header + rows below that).

If a panel is malformed (no SCAN RESULTS header, missing labels, etc.)
it is skipped silently and reported in the failed-panels summary.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# Make ocr.onnx_hud_reader importable.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ocr.onnx_hud_reader import _find_label_rows  # noqa: E402


# ---------------------------------------------------------------- helpers


def _bright_mask(img: Image.Image, thr: int = 170) -> np.ndarray:
    """Boolean mask of any bright HUD text pixel (color-agnostic).

    Different SC video captures have very different hue tints — some
    panels render white labels as pink, orange values as peach, etc.
    Brightness (max channel) is the only robust signal across all of
    them: text glows against a dark asteroid background regardless of
    its colour.
    """
    rgb = np.array(img.convert("RGB"), dtype=np.int16)
    return rgb.max(axis=2) >= thr


# Backwards-compatible aliases — both refer to the same brightness mask.
def _orange_mask(img: Image.Image) -> np.ndarray:
    return _bright_mask(img, thr=160)


def _white_mask(img: Image.Image) -> np.ndarray:
    return _bright_mask(img, thr=180)


def _row_runs(mask_rows: np.ndarray, min_density: int, min_h: int = 3) -> list[tuple[int, int]]:
    """Find vertical runs of rows whose density exceeds ``min_density``."""
    hot = mask_rows >= min_density
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for y, v in enumerate(hot):
        if v and not in_run:
            in_run = True
            start = y
        elif not v and in_run:
            in_run = False
            if y - start >= min_h:
                runs.append((start, y))
    if in_run and len(hot) - start >= min_h:
        runs.append((start, len(hot)))
    return runs


def _find_scan_results_y(img: Image.Image, mass_y1: int) -> int | None:
    """Return the y-bottom of the SCAN RESULTS header.

    Searches the band above MASS for a dense bright row + a horizontal
    underline. The mineral name lives between SCAN RESULTS and MASS.
    """
    wm = _white_mask(img)
    counts = wm[:mass_y1, :].sum(axis=1)
    runs = _row_runs(counts, min_density=18, min_h=3)
    if not runs:
        return None
    # SCAN RESULTS is the densest run in the band above MASS.
    best = max(runs, key=lambda r: counts[r[0]:r[1]].sum())
    return best[1]


def _find_first_orange_row_between(
    img: Image.Image, y_top: int, y_bot: int
) -> tuple[int, int] | None:
    om = _orange_mask(img)
    counts = om[y_top:y_bot, :].sum(axis=1)
    runs = _row_runs(counts, min_density=8, min_h=3)
    if not runs:
        return None
    y1, y2 = runs[0]
    return (y_top + y1, y_top + y2)


def _find_composition_header(img: Image.Image, y_after: int) -> tuple[int, int] | None:
    """Find the COMPOSITION white header below ``y_after``."""
    wm = _white_mask(img)
    counts = wm[y_after:, :].sum(axis=1)
    runs = _row_runs(counts, min_density=15, min_h=4)
    if not runs:
        return None
    y1, y2 = runs[0]
    return (y_after + y1, y_after + y2)


def _find_orange_rows_below(
    img: Image.Image, y_after: int, max_rows: int = 12
) -> list[tuple[int, int]]:
    om = _orange_mask(img)
    counts = om[y_after:, :].sum(axis=1)
    runs = _row_runs(counts, min_density=8, min_h=4)
    return [(y_after + a, y_after + b) for a, b in runs[:max_rows]]


def _column_runs(mask_cols: np.ndarray, min_density: int, min_w: int = 3) -> list[tuple[int, int]]:
    hot = mask_cols >= min_density
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for x, v in enumerate(hot):
        if v and not in_run:
            in_run = True
            start = x
        elif not v and in_run:
            in_run = False
            if x - start >= min_w:
                runs.append((start, x))
    if in_run and len(hot) - start >= min_w:
        runs.append((start, len(hot)))
    return runs


def _split_comp_row(
    img: Image.Image, y1: int, y2: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    """Split a composition row into (pct, name, count) x-spans."""
    om = _orange_mask(img)
    band = om[y1:y2, :]
    col_counts = band.sum(axis=0)
    runs = _column_runs(col_counts, min_density=1, min_w=2)
    if len(runs) < 3:
        return None
    # Merge close runs (within ~6 px) — letters in a word produce small
    # gaps. We need 3 final groups: percent | name | count.
    merged: list[tuple[int, int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= 6:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    if len(merged) < 3:
        return None
    # Percent is leftmost; count is rightmost; name is everything between.
    pct = merged[0]
    cnt = merged[-1]
    name_a = merged[1][0]
    name_b = merged[-2][1]
    if name_b <= name_a:
        return None
    name = (name_a, name_b)
    return pct, name, cnt


# ---------------------------------------------------------------- core


FIELDS = (
    "mineral_name",
    "mass_value",
    "resistance_value",
    "instability_value",
    "difficulty_word",
    "scu_value",
    "comp_pct",
    "comp_name",
    "comp_count",
)


def _safe_crop(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image | None:
    x1, y1, x2, y2 = box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.width, x2)
    y2 = min(img.height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return img.crop((x1, y1, x2, y2))


def split_panel(
    img: Image.Image, out_root: Path, video_id: str, frame_stem: str
) -> tuple[Counter, str | None]:
    """Split one panel; return (counts, error). ``error`` None on success-ish.

    Even partial successes return None for error; ``error`` is only set
    when the panel is unusable (no SCAN RESULTS or no labels).
    """
    counts: Counter = Counter()

    label_rows = _find_label_rows(img)
    if not all(k in label_rows for k in ("mass", "resistance", "instability")):
        return counts, f"missing labels (got {sorted(label_rows.keys())})"

    mass_y1, mass_y2, mass_x = label_rows["mass"]
    scan_y = _find_scan_results_y(img, mass_y1)
    if scan_y is None:
        return counts, "no SCAN RESULTS header"
    resi_y1, resi_y2, resi_x = label_rows["resistance"]
    inst_y1, inst_y2, inst_x = label_rows["instability"]

    W = img.width
    right_margin = W - 6

    def _save(field: str, idx: int, crop: Image.Image | None) -> None:
        if crop is None:
            return
        d = out_root / field
        d.mkdir(parents=True, exist_ok=True)
        out = d / f"{video_id}_{frame_stem}_{idx}.png"
        crop.save(out, format="PNG", optimize=True)
        counts[field] += 1

    # --- Mineral name: between SCAN RESULTS bottom and MASS top.
    mineral_band = _find_first_orange_row_between(img, scan_y + 1, mass_y1 - 1)
    if mineral_band is not None:
        my1, my2 = mineral_band
        pad = 2
        _save(
            "mineral_name",
            0,
            _safe_crop(img, (4, my1 - pad, right_margin, my2 + pad)),
        )

    # --- Value crops to right of labels.
    _save(
        "mass_value",
        0,
        _safe_crop(img, (mass_x + 6, mass_y1, right_margin, mass_y2)),
    )
    _save(
        "resistance_value",
        0,
        _safe_crop(img, (resi_x + 6, resi_y1, right_margin, resi_y2)),
    )
    _save(
        "instability_value",
        0,
        _safe_crop(img, (inst_x + 6, inst_y1, right_margin, inst_y2)),
    )

    # --- Difficulty word: bracketed bar below INSTABILITY.
    # Look for the next non-empty row band below inst_y2.
    om = _orange_mask(img)
    diff_search_top = inst_y2 + 2
    diff_search_bot = min(img.height, inst_y2 + int((inst_y2 - inst_y1) * 4) + 30)
    if diff_search_top < diff_search_bot:
        sub = om[diff_search_top:diff_search_bot, :].sum(axis=1)
        runs = _row_runs(sub, min_density=10, min_h=3)
        if runs:
            d_y1, d_y2 = runs[0]
            d_y1 += diff_search_top
            d_y2 += diff_search_top
            # Crop the centre region between brackets — use the
            # central ~60% of the panel width.
            cx1 = int(W * 0.20)
            cx2 = int(W * 0.80)
            pad = 2
            _save(
                "difficulty_word",
                0,
                _safe_crop(img, (cx1, d_y1 - pad, cx2, d_y2 + pad)),
            )
            comp_search_top = d_y2 + 2
        else:
            comp_search_top = diff_search_top
    else:
        comp_search_top = diff_search_top

    # --- COMPOSITION header + SCU value on same row.
    comp_hdr = _find_composition_header(img, comp_search_top)
    if comp_hdr is None:
        return counts, None  # partial success
    ch_y1, ch_y2 = comp_hdr
    # SCU value is on the same y-band as COMPOSITION header but in the
    # right half of the panel and is ORANGE — search orange columns
    # within this row.
    scu_band = om[ch_y1:ch_y2, :]
    if scu_band.any():
        col_counts = scu_band.sum(axis=0)
        # The COMPOSITION header occupies the left half; SCU value sits
        # in the right half. Restrict search to right of the header.
        right_start = int(W * 0.45)
        right_counts = col_counts[right_start:]
        runs = _column_runs(right_counts, min_density=1, min_w=2)
        if runs:
            # Merge runs separated by small gaps so "5.56 SCU" comes
            # back as a single span (gaps for the dot, the space).
            merged: list[tuple[int, int]] = []
            for a, b in runs:
                if merged and a - merged[-1][1] <= 8:
                    merged[-1] = (merged[-1][0], b)
                else:
                    merged.append((a, b))
            sx1, sx2 = merged[-1]
            sx1 += right_start
            sx2 += right_start
            pad = 2
            _save(
                "scu_value",
                0,
                _safe_crop(img, (sx1 - pad, ch_y1 - pad, sx2 + pad, ch_y2 + pad)),
            )

    # --- Composition rows: orange rows below the comp bar.
    # Skip the composition bar itself (first dense row right after
    # header). Bar density is high and uniform; rows have the 3-segment
    # signature.
    bar_search_top = ch_y2 + 1
    # Heuristic: skip past anything within ~14 px of header (the bar).
    rows = _find_orange_rows_below(img, bar_search_top, max_rows=20)
    # Filter out the composition bar — it tends to be a single dense
    # block whose orange columns span almost the full width with no
    # 3-segment structure.
    comp_rows: list[tuple[int, int]] = []
    for y1, y2 in rows:
        # Reject the composition bar: it's a row of tight `|` marks
        # between brackets, which produces a long alternating pattern
        # (on/off/on/off...) across the row. Real text rows have only
        # a handful of segments (pct, name, count). Count segment
        # runs: bars yield 15+ runs; text rows yield well under 10.
        band = om[y1:y2, :]
        col_hot = band.sum(axis=0) > 0
        transitions = int(np.sum(col_hot[1:] & ~col_hot[:-1]))
        if transitions > 30:
            continue
        seg = _split_comp_row(img, y1, y2)
        if seg is None:
            continue
        # Sanity: pct segment should be on the left third.
        pct_x = seg[0][1]
        if pct_x > W * 0.45:
            continue
        comp_rows.append((y1, y2))

    for i, (y1, y2) in enumerate(comp_rows):
        seg = _split_comp_row(img, y1, y2)
        if seg is None:
            continue
        (px1, px2), (nx1, nx2), (cx1c, cx2c) = seg
        pad = 2
        _save(
            "comp_pct",
            i,
            _safe_crop(img, (px1 - pad, y1 - pad, px2 + pad, y2 + pad)),
        )
        _save(
            "comp_name",
            i,
            _safe_crop(img, (nx1 - pad, y1 - pad, nx2 + pad, y2 + pad)),
        )
        _save(
            "comp_count",
            i,
            _safe_crop(img, (cx1c - pad, y1 - pad, cx2c + pad, y2 + pad)),
        )

    return counts, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="training_data_panels")
    ap.add_argument("--output", default="training_data_split")
    args = ap.parse_args()

    in_root = Path(args.input)
    if not in_root.is_absolute():
        in_root = _ROOT / in_root
    out_root = Path(args.output)
    if not out_root.is_absolute():
        out_root = _ROOT / out_root

    if not in_root.is_dir():
        print(f"input dir not found: {in_root}", file=sys.stderr)
        return 1

    totals: Counter = Counter()
    failed: list[tuple[str, str]] = []
    n_panels = 0

    for vid_dir in sorted(p for p in in_root.iterdir() if p.is_dir()):
        video_id = vid_dir.name
        for png in sorted(vid_dir.glob("f_*.png")):
            n_panels += 1
            stem = png.stem
            try:
                img = Image.open(png).convert("RGB")
            except Exception as e:
                failed.append((f"{video_id}/{png.name}", f"open failed: {e}"))
                continue
            try:
                counts, err = split_panel(img, out_root, video_id, stem)
            except Exception as e:
                failed.append((f"{video_id}/{png.name}", f"crash: {e}"))
                continue
            totals.update(counts)
            if err is not None and not counts:
                failed.append((f"{video_id}/{png.name}", err))

    print(f"\nProcessed {n_panels} panel(s) from {in_root}")
    print(f"Output: {out_root}")
    print("\nPer-field crop counts:")
    for f in FIELDS:
        print(f"  {f:20s} {totals.get(f, 0)}")
    if failed:
        print(f"\nFailed panels ({len(failed)}):")
        for name, why in failed:
            print(f"  {name}: {why}")
    else:
        print("\nNo failed panels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
