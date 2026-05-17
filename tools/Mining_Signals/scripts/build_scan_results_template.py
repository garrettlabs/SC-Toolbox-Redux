"""Bootstrap the SCAN RESULTS title template for NCC anchor matching.

Replaces the Tesseract-based ``_find_scan_results_anchor`` with a
template-matching approach that's faster, polarity-independent, and
tolerant of the slight tilt SC's 3D-billboard panel renders introduce.

Reads annotated panel PNGs that have a ``.boxes.json`` sidecar with a
``scan_results`` ground-truth bounding box. The annotations live next
to the PNGs in directories like::

    training_data_panels/<run_id>/<region>/cap_<timestamp>.png
    training_data_panels/<run_id>/<region>/cap_<timestamp>.boxes.json

Each ``.boxes.json`` looks like::

    {
        "image": "cap_20260418_155705_329.png",
        "boxes": {
            "scan_results": {"x": 72, "y": 61, "w": 207, "h": 32},
            ...
        }
    }

For each annotated panel:
  1. Crop the ``scan_results`` box plus a small margin
  2. Polarity-canonicalize so text is bright on dark background
  3. Resize to a canonical 28-px height (preserves aspect ratio)
  4. Add to the per-key crop pool

After processing all panels, average the pool to produce a noise-reduced
template that captures the consistent "SCAN RESULTS" pixel signature
across panels with different backgrounds, lighting, and minerals.

Output: ``tools/Mining_Signals/ocr/sc_templates/scan_results.npz`` with::

    {
        "scan_results": np.ndarray (28, W) uint8,  # canonicalized, averaged
        "height":       np.int32 28,
    }

Run once per significant SC font/HUD-scale change. The runtime
``scan_results_match.find_scan_results_anchor`` matches this template
via multi-scale NCC.

Usage::

    python scripts/build_scan_results_template.py
    # or with a custom source dir:
    python scripts/build_scan_results_template.py --source-dir <path>
"""
from __future__ import annotations

import argparse
import json
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
log = logging.getLogger("build_scan_results_template")

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
DEFAULT_SOURCE = TOOL / "training_data_panels"
DEFAULT_OUTPUT = TOOL / "ocr" / "sc_templates" / "scan_results.npz"
CANONICAL_HEIGHT = 28


def _otsu(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.sum(np.arange(256) * hist)
    sum_bg, w_bg = 0.0, 0
    max_var, threshold = 0.0, 127
    for t in range(256):
        w_bg += int(hist[t])
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * int(hist[t])
        m_bg = sum_bg / w_bg
        m_fg = (sum_total - sum_bg) / w_fg
        var = w_bg * w_fg * (m_bg - m_fg) ** 2
        if var > max_var:
            max_var = var
            threshold = t
    return int(threshold)


def _canonicalize_text_polarity(gray: np.ndarray) -> np.ndarray:
    """Force text to be BRIGHT pixels regardless of source polarity.

    Text is the minority class — Otsu-split, count both classes,
    invert if dark < bright (i.e. text was the dark minority).
    """
    thr = _otsu(gray)
    bright = int((gray > thr).sum())
    dark = gray.size - bright
    if dark < bright:
        return (255 - gray).astype(np.uint8)
    return gray.astype(np.uint8)


def _crop_box(
    img: Image.Image,
    bbox: dict,
    pad_x: int = 4,
    pad_y: int = 3,
) -> np.ndarray:
    """Crop a {x, y, w, h} bbox + small margin, return polarity-canonical uint8."""
    x = int(bbox["x"])
    y = int(bbox["y"])
    w = int(bbox["w"])
    h = int(bbox["h"])
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img.width, x + w + pad_x)
    y2 = min(img.height, y + h + pad_y)
    crop = img.crop((x1, y1, x2, y2)).convert("L")
    arr = np.array(crop, dtype=np.uint8)
    return _canonicalize_text_polarity(arr)


def _resize_to_height(arr: np.ndarray, target_h: int) -> np.ndarray:
    """Resize a uint8 2D array to target_h, preserving aspect ratio."""
    h, w = arr.shape
    if h == target_h:
        return arr
    scale = target_h / h
    new_w = max(8, int(round(w * scale)))
    pil = Image.fromarray(arr).resize((new_w, target_h), Image.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def _average_left_aligned(crops: list[np.ndarray]) -> Optional[np.ndarray]:
    """Average a list of (H, W) uint8 arrays, left-aligned.

    SCAN RESULTS templates start at the same letter ('S'), so left-
    align (pad on the right) so the leading 'S' edge stacks across all
    crops. Right-aligning would smear letters since trailing 'S' of
    'RESULTS' floats relative to the left.
    """
    if not crops:
        return None
    h = crops[0].shape[0]
    max_w = max(c.shape[1] for c in crops)
    accum = np.zeros((h, max_w), dtype=np.float32)
    counts = np.zeros((h, max_w), dtype=np.float32)
    for c in crops:
        # Left-align: pad on the RIGHT so leading 'S' edges line up.
        accum[:, :c.shape[1]] += c.astype(np.float32)
        counts[:, :c.shape[1]] += 1.0
    avg = np.where(counts > 0, accum / np.maximum(counts, 1.0), 0.0)
    return avg.astype(np.uint8)


def _find_annotated_panels(source_dir: Path) -> list[tuple[Path, Path]]:
    """Recursively find (image, boxes_json) pairs that have a scan_results box."""
    pairs: list[tuple[Path, Path]] = []
    for boxes_path in sorted(source_dir.rglob("*.boxes.json")):
        try:
            with open(boxes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        boxes = data.get("boxes", {})
        if "scan_results" not in boxes:
            continue
        # Image is alongside the json, with the .png extension.
        # Strip ".boxes.json" → ".png".
        stem = boxes_path.name.removesuffix(".boxes.json")
        img_path = boxes_path.parent / f"{stem}.png"
        if not img_path.is_file():
            continue
        pairs.append((img_path, boxes_path))
    return pairs


def _bootstrap_via_existing_matcher(
    source_dir: Path,
    canonical_height: int,
    min_score: float = 0.50,
) -> list[np.ndarray]:
    """Use the existing NCC matcher to find SCAN RESULTS in *all* panels.

    Returns a list of canonicalized, resized crops — same shape as
    those produced from .boxes.json annotations — extracted from
    every PNG in ``source_dir`` (recursive) where the existing
    template hits with score ≥ ``min_score``.

    This is bootstrapping: a small initial template (built from a
    handful of hand-annotated panels) is used to auto-locate SCAN
    RESULTS in the rest of the capture pool, then those auto-located
    crops join the original ones in the next averaging pass. The
    re-averaged template is dramatically more robust because it
    captures variation across lighting / mineral / occlusion that 2
    panels can't represent.

    Returns ``[]`` if the existing template isn't available yet
    (first-ever bootstrap call has nothing to bootstrap from).
    """
    # Import the matcher only here so the build script still works
    # before scan_results.npz exists (initial annotations-only build).
    sys_path_added = False
    try:
        if str(TOOL) not in sys.path:
            sys.path.insert(0, str(TOOL))
            sys_path_added = True
        try:
            from ocr.sc_ocr import scan_results_match as _srm
            from ocr.sc_ocr import label_match as _lm
        except Exception as exc:
            log.warning(
                "bootstrap unavailable (matcher import failed: %s) — "
                "skipping", exc,
            )
            return []
        # Force a fresh load so we pick up whatever template was just
        # written by the annotations-only first pass.
        _srm.reset_cache()
    finally:
        if sys_path_added:
            sys.path.remove(str(TOOL))

    crops: list[np.ndarray] = []
    pngs = sorted(source_dir.rglob("*.png"))
    for png in pngs:
        try:
            img = Image.open(png).convert("RGB")
        except Exception:
            continue
        anchor = _srm.find_scan_results_anchor(img)
        if anchor is None or anchor.get("score", 0.0) < min_score:
            continue
        # The matcher returns (x, y, w, h) at the matched scale —
        # this is the exact box of the title pixels. Crop with the
        # same padding the annotation path uses so the two crop pools
        # mix cleanly.
        try:
            box = {
                "x": int(anchor["title_x"]),
                "y": int(anchor["title_y"]),
                "w": int(anchor["title_w"]),
                "h": int(anchor["title_h"]),
            }
            crop = _crop_box(img, box)
            crop = _resize_to_height(crop, canonical_height)
            crops.append(crop)
        except Exception as exc:
            log.warning(
                "  bootstrap crop failed for %s: %s", png.name, exc,
            )
    log.info(
        "bootstrap: %d / %d panels matched at score ≥ %.2f",
        len(crops), len(pngs), min_score,
    )
    return crops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-dir", type=Path, default=DEFAULT_SOURCE,
        help="Directory to recursively scan for *.boxes.json annotations",
    )
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--canonical-height", type=int, default=CANONICAL_HEIGHT,
        help="Resize all crops to this height before averaging",
    )
    ap.add_argument(
        "--bootstrap", action="store_true",
        help="After building from .boxes.json annotations, run the "
             "resulting matcher across ALL PNGs in source-dir and "
             "merge auto-located crops into a second-pass template. "
             "Dramatically improves robustness when only a few panels "
             "are hand-annotated.",
    )
    ap.add_argument(
        "--bootstrap-min-score", type=float, default=0.50,
        help="Minimum NCC score for an auto-located crop to be merged "
             "(only used with --bootstrap)",
    )
    args = ap.parse_args()

    source_dir = args.source_dir
    if not source_dir.is_dir():
        log.error("source dir not found: %s", source_dir)
        return 1

    pairs = _find_annotated_panels(source_dir)
    if not pairs:
        log.error(
            "no annotated panels with scan_results boxes found under %s",
            source_dir,
        )
        return 1
    log.info("found %d annotated panel(s) with scan_results box", len(pairs))

    crops: list[np.ndarray] = []
    for img_path, boxes_path in pairs:
        try:
            with open(boxes_path, "r", encoding="utf-8") as f:
                boxes_data = json.load(f)
            box = boxes_data["boxes"]["scan_results"]
        except Exception as exc:
            log.warning("  %s: load failed: %s", boxes_path.name, exc)
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            log.warning("  %s: open failed: %s", img_path.name, exc)
            continue
        try:
            crop = _crop_box(img, box)
            crop = _resize_to_height(crop, args.canonical_height)
        except Exception as exc:
            log.warning("  %s: crop/resize failed: %s", img_path.name, exc)
            continue
        crops.append(crop)
        log.info(
            "  %s: bbox=(x=%d, y=%d, w=%d, h=%d) → crop %s",
            img_path.name, box["x"], box["y"], box["w"], box["h"],
            crop.shape,
        )

    if not crops:
        log.error("no usable crops extracted")
        return 1

    avg = _average_left_aligned(crops)
    if avg is None:
        log.error("averaging failed")
        return 1

    log.info(
        "averaged %d source crops → template shape %s",
        len(crops), avg.shape,
    )

    payload = {
        "scan_results": avg,
        "height": np.array(args.canonical_height, dtype=np.int32),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    log.info("wrote %s (annotations-only pass: %d crops)", args.output, len(crops))

    # Save the averaged template as a PNG for visual inspection
    debug_dir = args.output.parent / "labels_debug"
    debug_dir.mkdir(exist_ok=True)
    Image.fromarray(avg).save(debug_dir / "scan_results_template.png")

    # ── Optional bootstrap pass ──
    # Re-run the matcher (which now picks up the just-written template)
    # against every PNG in the source dir. Auto-located crops join the
    # annotated crops, and we re-average for a second-pass template
    # that's far more robust to lighting / mineral / occlusion variation.
    if args.bootstrap:
        log.info(
            "bootstrap: scanning %s for additional crops via NCC matcher...",
            source_dir,
        )
        boot_crops = _bootstrap_via_existing_matcher(
            source_dir,
            args.canonical_height,
            args.bootstrap_min_score,
        )
        if boot_crops:
            combined = crops + boot_crops
            log.info(
                "bootstrap: combining %d annotated + %d auto-located = %d",
                len(crops), len(boot_crops), len(combined),
            )
            avg2 = _average_left_aligned(combined)
            if avg2 is not None:
                payload["scan_results"] = avg2
                np.savez(args.output, **payload)
                Image.fromarray(avg2).save(
                    debug_dir / "scan_results_template.png"
                )
                log.info(
                    "bootstrap: re-wrote %s (template shape %s, "
                    "averaged from %d crops)",
                    args.output, avg2.shape, len(combined),
                )
        else:
            log.info("bootstrap: no auto-located crops added")

    log.info("wrote %s/scan_results_template.png", debug_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
