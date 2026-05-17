"""Re-extract signal-panel digit glyphs preserving RGB color.

Mirrors ``extract_labeled_glyphs.extract_region2_glyphs`` for everything
except the final per-glyph crop step. Where the legacy extractor saves
a 28×28 single-channel uint8 PNG (luma-only), this script saves a
28×28×3 RGB PNG at the same ``(x1, x2)`` span location — preserving
the full color information that the original capture had.

Output goes to ``training_data_user_sig_rgb/<class>/`` (parallel to the
existing ``training_data_user_sig/`` directory). The icon class ``@``
is NOT generated here — it'll be augmented separately from the same
``training_data_blacklist/bad crop.png`` source via a parallel
``augment_icon_class_rgb.py`` once the digit extraction passes.

Pipeline per panel:
  1. Load source PNG as RGB (3-channel) AND as luma (1-channel).
  2. Use luma for icon-mask + row-isolate + Tesseract verify + segment
     (the existing pipeline that's been tuned over months — switching
     these stages to RGB would break Tesseract's verifier).
  3. Apply the same y-trim (row-isolate's bounds) to the RGB array so
     they share coordinates.
  4. For each verified digit span: slice RGB, find glyph y-extent via
     luma binary mask (same _glyph_to_28x28 logic), pad with white
     (255,255,255), bilinear-resize 28×28 — but in 3 channels.
  5. Save as RGB PNG.

Usage::

    python scripts/extract_rgb_signal_glyphs.py
    python scripts/extract_rgb_signal_glyphs.py --reset
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

OUT_ROOT = TOOL / "training_data_user_sig_rgb"


# ─────────────────────────────────────────────────────────────
# RGB-preserving glyph extractor — mirrors _glyph_to_28x28 logic
# ─────────────────────────────────────────────────────────────

def _glyph_to_28x28_rgb(
    rgb_crop: np.ndarray,  # shape (H, W, 3), uint8
    luma_crop: np.ndarray,  # shape (H, W), uint8 — for binary y-extent
    x1: int, x2: int,
) -> Optional[np.ndarray]:
    """RGB-channel-preserving variant of ``_glyph_to_28x28``.

    Identical y-extent logic (use luma binary to find ink rows) but
    saves the RGB region instead of the luma. Pads with WHITE
    (255, 255, 255) per-channel and bilinear-resizes to 28×28×3.

    Returns a (28, 28, 3) uint8 array, or None when the column window
    has insufficient ink to localise the glyph vertically.
    """
    # Polarity-canonicalize luma to find ink rows
    if np.median(luma_crop) > 140:
        work = 255 - luma_crop
    else:
        work = luma_crop
    thr = _xlg._otsu(work)
    binary = (work > thr).astype(np.uint8)
    glyph_col = binary[:, x1:x2]
    ys = np.where(np.any(glyph_col > 0, axis=1))[0]
    if len(ys) < 2:
        return None
    ya, yb = int(ys[0]), int(ys[-1]) + 1

    # Slice RGB at the same coordinates as the luma binary identified
    rgb_glyph = rgb_crop[ya:yb, x1:x2].astype(np.float32)
    pad = 2
    h, w, _ = rgb_glyph.shape
    padded = np.full((h + pad * 2, w + pad * 2, 3), 255.0, dtype=np.float32)
    padded[pad:pad + h, pad:pad + w] = rgb_glyph
    pil = Image.fromarray(padded.astype(np.uint8), mode="RGB").resize(
        (28, 28), Image.BILINEAR,
    )
    return np.asarray(pil, dtype=np.uint8)


def _save_rgb_glyph(
    glyph: np.ndarray, char: str, src_name: str, out_root: Path,
) -> bool:
    """Save (28, 28, 3) glyph under ``out_root/<class>/user_<src>_<i>.png``."""
    if glyph.ndim != 3 or glyph.shape != (28, 28, 3):
        return False
    if not (char.isdigit() or char in (".", "%")):
        return False
    cls = char  # digit-only for signal
    d = out_root / cls
    d.mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        out = d / f"user_{src_name}_{i}.png"
        if not out.exists():
            break
        i += 1
    Image.fromarray(glyph, mode="RGB").save(out)
    return True


# ─────────────────────────────────────────────────────────────
# Per-panel extraction (mirrors extract_region2_glyphs structure)
# ─────────────────────────────────────────────────────────────

def extract_panel_rgb(
    img_png: Path, label: dict, out_root: Path,
    left_mask_pct: float = 0.30,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    value = label.get("value")
    if value is None:
        return counts
    chars = [c for c in str(value) if c.isdigit()]
    if not chars:
        return counts

    try:
        pil_rgb = Image.open(img_png).convert("RGB")
    except Exception:
        return counts
    rgb = np.asarray(pil_rgb, dtype=np.uint8)
    luma = np.asarray(pil_rgb.convert("L"), dtype=np.uint8)
    img_w = luma.shape[1]

    # 1. Icon mask (uses luma + pHash blacklist match — same as original)
    bg_luma = int(np.median(luma))
    luma = luma.copy()
    rgb = rgb.copy()
    icon_right = _xlg._locate_icon_via_blacklist_match(luma)
    floor_mask = int(img_w * left_mask_pct) if left_mask_pct > 0 else 0
    mask_w = max(floor_mask, icon_right + 4 if icon_right > 0 else 0)
    if 0 < mask_w < img_w:
        luma[:, :mask_w] = bg_luma
        # Mask RGB with the per-channel median to avoid introducing
        # high-saturation artifacts.
        for ch in range(3):
            ch_med = int(np.median(rgb[:, :, ch]))
            rgb[:, :mask_w, ch] = ch_med

    # 2. Row isolation (find band on luma, apply same y-bounds to RGB)
    pre_h = luma.shape[0]
    band = _xlg._find_main_row_bounds(luma) if hasattr(
        _xlg, "_find_main_row_bounds",
    ) else None
    if band is not None:
        by1, by2 = band
        luma = luma[by1:by2, :]
        rgb = rgb[by1:by2, :]
    elif pre_h > 6:
        # Fallback: legacy _isolate_main_row returns trimmed array;
        # we re-derive bounds by finding which rows survived.
        trimmed = _xlg._isolate_main_row(luma)
        if trimmed.shape[0] != luma.shape[0]:
            # Brute-force: scan for matching row signature
            for y in range(luma.shape[0] - trimmed.shape[0] + 1):
                if np.array_equal(luma[y:y + trimmed.shape[0]], trimmed):
                    luma = trimmed
                    rgb = rgb[y:y + trimmed.shape[0], :]
                    break

    # 3. Tesseract char-box verification + span extraction (luma path)
    label_clean = "".join(c for c in chars if c.isdigit())
    spans: list[tuple[int, int]] = []
    used_tesseract = False
    try:
        variants: list[tuple["Image.Image", str]] = []
        try:
            base = Image.fromarray(luma, mode="L")
            variants.append((base, "1x"))
            variants.append((
                base.resize((base.width * 2, base.height * 2), Image.LANCZOS),
                "2x",
            ))
            variants.append((
                base.resize((base.width * 3, base.height * 3), Image.LANCZOS),
                "3x",
            ))
            inv = Image.fromarray(255 - luma, mode="L")
            variants.append((inv, "1x_inv"))
            variants.append((
                inv.resize((inv.width * 2, inv.height * 2), Image.LANCZOS),
                "2x_inv",
            ))
        except Exception:
            variants = []

        for psm in ("7", "13", "8", "6"):
            for img_v, _tag in variants:
                if used_tesseract:
                    break
                try:
                    tess_boxes = _xlg._tesseract_char_boxes(
                        img_v, whitelist="0123456789.", psm=psm,
                    )
                except Exception:
                    continue
                if not tess_boxes:
                    continue
                tess_clean = "".join(
                    b[0] for b in tess_boxes if b[0].isdigit() or b[0] == "."
                )
                if (
                    tess_clean.replace(".", "") == label_clean
                    and len(tess_boxes) >= len(chars)
                ):
                    spans = _xlg._segment_digits(luma, expected_count=len(chars))
                    if len(spans) == len(chars):
                        used_tesseract = True
                        break
    except Exception:
        pass

    if not used_tesseract or len(spans) != len(chars):
        return counts

    # 4. Per-glyph filter (drop spurious narrow / over-wide spans)
    MAX_SPAN_FRACTION = 0.45
    MIN_SPAN_WIDTH = 3
    filtered: list[tuple[int, int]] = []
    for (x1, x2) in spans:
        w = x2 - x1
        if w < MIN_SPAN_WIDTH or w > img_w * MAX_SPAN_FRACTION:
            continue
        filtered.append((x1, x2))
    if len(filtered) != len(chars):
        return counts

    if len(filtered) >= 3:
        widths = sorted(x2 - x1 for x1, x2 in filtered)
        median = widths[len(widths) // 2]
        if median > 0:
            outliers = [
                (x1, x2) for (x1, x2) in filtered
                if (x2 - x1) >= int(median * 1.7)
            ]
            if outliers:
                return counts

    # 5. Extract RGB glyphs at verified spans
    src_name = img_png.stem
    for (x1, x2), ch in zip(filtered, chars):
        glyph_rgb = _glyph_to_28x28_rgb(rgb, luma, x1, x2)
        if glyph_rgb is None:
            continue
        if _save_rgb_glyph(glyph_rgb, ch, src_name, out_root):
            counts[ch] = counts.get(ch, 0) + 1
    return counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true",
                   help="Wipe out_root before extracting.")
    args = p.parse_args()

    if args.reset and OUT_ROOT.is_dir():
        import shutil
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    sources = training_registry.get_training_sources("signal")
    print(f"=== RGB signal-glyph extraction ===")
    print(f"  Output: {OUT_ROOT}")
    print(f"  Sources: {len(sources)}")

    total_panels = 0
    total_glyphs = 0
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
            counts = extract_panel_rgb(img_path, label, OUT_ROOT)
            if counts:
                total_panels += 1
                for k, v in counts.items():
                    counts_total[k] = counts_total.get(k, 0) + v
                total_glyphs += sum(counts.values())

    print(f"\n=== Summary ===")
    print(f"  Panels accepted: {total_panels}")
    print(f"  Glyphs extracted: {total_glyphs}")
    print(f"  Per-class counts:")
    for k in sorted(counts_total):
        print(f"    {k}: {counts_total[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
