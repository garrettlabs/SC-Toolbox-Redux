"""CRNN-as-oracle auto-labeler for RGB signal training data.

The strict Tesseract-verified extractor (``extract_rgb_signal_glyphs.py``)
only accepts ~32 of 233 signal panels because Tesseract struggles on
SC's chromatically-aberrated cyan-on-dark digits. This skews the per-
class distribution badly (class ``9`` had 1 real sample after
extraction).

The CRNN runs at significantly higher accuracy than Tesseract on this
font (it's been trained on SC's actual character set, not generic
documents), and its read can be checked against the user-typed label
in ``cap_*.json`` for ground-truth verification:

  CRNN read AGREES WITH typed label  →  trust the segmentation, save
                                        each span as a labeled RGB
                                        sample (label = CRNN's digit
                                        at that position).

  CRNN read DISAGREES                →  skip this panel; segmentation
                                        is suspect.

This is **semi-supervised auto-labeling** — using a more reliable peer
classifier (CRNN) to label samples for a less reliable student
(per-glyph RGB CNN). Should give us 4–5× more training data than the
Tesseract-gated extractor without any human labeling effort.

Output goes to ``training_data_user_sig_rgb/<class>/`` with the prefix
``auto_`` so we can distinguish from Tesseract-verified samples.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
sys.path.insert(0, str(TOOL))
sys.path.insert(0, str(TOOL / "scripts"))

import extract_labeled_glyphs as _xlg  # noqa: E402
from ocr import training_registry  # noqa: E402
from ocr.sc_ocr import api as _api  # noqa: E402
from ocr.sc_ocr import signal_anchor as _sa  # noqa: E402

OUT_ROOT = TOOL / "training_data_user_sig_rgb"


def _save_rgb_glyph_auto(
    glyph: np.ndarray, char: str, src_name: str, out_root: Path,
) -> bool:
    """Save (28, 28, 3) glyph under ``out_root/<class>/auto_<src>_<i>.png``."""
    if glyph.ndim != 3 or glyph.shape != (28, 28, 3):
        return False
    if not char.isdigit():
        return False
    cls = char
    d = out_root / cls
    d.mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        out = d / f"auto_{src_name}_{i}.png"
        if not out.exists():
            break
        i += 1
    Image.fromarray(glyph, mode="RGB").save(out)
    return True


def auto_label_panel(img_png: Path, label: dict) -> dict[str, int]:
    """Auto-label one panel using CRNN-as-oracle.

    Returns per-class counts of glyphs saved.
    """
    counts: dict[str, int] = {}
    value = label.get("value")
    if value is None:
        return counts
    typed_chars = [c for c in str(value) if c.isdigit()]
    if not typed_chars:
        return counts
    typed_string = "".join(typed_chars)

    # Load full panel
    try:
        pil_rgb = Image.open(img_png).convert("RGB")
    except Exception:
        return counts
    rgb_full = np.asarray(pil_rgb, dtype=np.uint8)
    luma_full = np.asarray(pil_rgb.convert("L"), dtype=np.uint8)
    # Use max-of-channels for the anchor pipeline (matches runtime)
    gray_full = rgb_full.max(axis=2).astype(np.uint8)

    # ── Anchor: find digit crop box ──
    try:
        crop_box = _sa.find_digit_crop_box(gray_full)
    except Exception:
        crop_box = None
    if crop_box is None:
        return counts
    cx1, cy1, cx2, cy2 = crop_box
    work_gray = gray_full[cy1:cy2, cx1:cx2]
    work_rgb = rgb_full[cy1:cy2, cx1:cx2]
    work_luma = luma_full[cy1:cy2, cx1:cx2]

    # ── Row isolate (matches runtime + extractor pipeline) ──
    band = (
        _xlg._find_main_row_bounds(work_gray)
        if hasattr(_xlg, "_find_main_row_bounds") else None
    )
    if band is not None:
        by1, by2 = band
        work_gray = work_gray[by1:by2]
        work_rgb = work_rgb[by1:by2]
        work_luma = work_luma[by1:by2]

    if work_gray.shape[0] < 6 or work_gray.shape[1] < 12:
        return counts

    # ── Verification approach: segmentation count == typed length ──
    # We initially tried CRNN agreement as the oracle, but the CRNN
    # underperforms on signal panels (only got partial reads on most
    # samples — '21350' read as '5', '10620' as '4', etc.). Trusting
    # the user-typed label directly is actually MORE reliable than
    # the CRNN at this resolution.
    #
    # New rule: if the segmenter finds exactly len(typed_label)
    # spans, we accept the panel and assign per-position labels
    # directly from the typed string. This is a standard
    # weakly-supervised labeling pattern: trust the human label as
    # ground truth, use segmentation only to localise glyphs.
    #
    # Risk: if segmentation finds the wrong count (off-by-one due to
    # comma fusion, dropped leading digit, etc.), we skip the panel
    # rather than save misaligned labels. The skip rate becomes our
    # quality filter.

    # ── Segment digits on the work crop (luma path) ──
    # Use the multi-recipe binarize (Fix B) so panels that the
    # legacy ``_adaptive_binarize`` collapsed into 1 huge span (and
    # were therefore unrecoverable) get a chance with one of the
    # alternate recipes. The recipe chosen is whichever produces a
    # span count closest to ``expected_count``.
    work_canon = _api._canonicalize_polarity(work_luma)
    bin_work = _api._adaptive_binarize_multi(
        work_canon, expected_count=len(typed_chars),
    )
    try:
        bin_work = _api._mask_commas_in_signature_band(bin_work)
    except Exception:
        pass
    pri_crops, pri_boxes = _api._segment_glyphs(work_canon, bin_work)
    if len(pri_boxes) != len(typed_chars):
        # Try _split_wide_signature_spans if count mismatches
        try:
            pri_crops, pri_boxes = _api._split_wide_signature_spans(
                work_canon, bin_work, pri_crops, pri_boxes,
                expected_count=len(typed_chars),
            )
        except Exception:
            pass
    if len(pri_boxes) != len(typed_chars):
        return counts

    # ── Per-glyph RGB extraction at the verified spans ──
    # Use _glyph_to_28x28_rgb from extract_rgb_signal_glyphs.
    from extract_rgb_signal_glyphs import _glyph_to_28x28_rgb
    src_name = img_png.stem
    h_w, w_w = work_rgb.shape[:2]
    for (bx, by, bw, bh), ch in zip(pri_boxes, typed_chars):
        x1, x2 = int(bx), int(bx) + int(bw)
        if x1 < 0 or x2 > w_w or x2 - x1 < 1:
            continue
        glyph_rgb = _glyph_to_28x28_rgb(work_rgb, work_luma, x1, x2)
        if glyph_rgb is None:
            continue
        if _save_rgb_glyph_auto(glyph_rgb, ch, src_name, OUT_ROOT):
            counts[ch] = counts.get(ch, 0) + 1
    return counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--clear-auto", action="store_true",
        help="Delete existing auto_*.png before generating.",
    )
    args = p.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.clear_auto:
        n_cleared = 0
        for cls_dir in OUT_ROOT.iterdir():
            if not cls_dir.is_dir():
                continue
            for f in cls_dir.glob("auto_*.png"):
                f.unlink()
                n_cleared += 1
        print(f"  Cleared {n_cleared} previous auto-labeled PNGs.")

    sources = training_registry.get_training_sources("signal")
    print(f"=== CRNN auto-labeler ===")
    print(f"  Output: {OUT_ROOT}")
    print(f"  Sources: {len(sources)}")

    panels_seen = 0
    panels_accepted = 0
    panels_crnn_disagreed = 0
    panels_segment_count_mismatch = 0
    panels_anchor_failed = 0
    counts_total: dict[str, int] = {}
    for src_dir in sources:
        for img_path in sorted(src_dir.glob("cap_*.png")):
            json_path = img_path.with_suffix(".json")
            if not json_path.is_file():
                continue
            try:
                label = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            panels_seen += 1
            counts = auto_label_panel(img_path, label)
            if counts:
                panels_accepted += 1
                for k, v in counts.items():
                    counts_total[k] = counts_total.get(k, 0) + v

    print(f"\n=== Auto-labeler summary ===")
    print(f"  Panels seen:     {panels_seen}")
    print(f"  Panels accepted: {panels_accepted}")
    print(f"  Per-class counts (auto-labeled, NEW):")
    for k in sorted(counts_total):
        print(f"    {k}: {counts_total[k]}")

    print(f"\n=== Combined pool (all *.png in {OUT_ROOT.name}/<class>/): ===")
    for cls in "0123456789":
        cls_dir = OUT_ROOT / cls
        if cls_dir.is_dir():
            n = len(list(cls_dir.glob("*.png")))
            print(f"    {cls}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
