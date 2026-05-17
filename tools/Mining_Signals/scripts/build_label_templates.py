"""Bootstrap label templates for the SC mining HUD label matcher.

Reads PNG panels from ``tools/Mining_Signals/template_source_panels/``,
uses Tesseract one-shot to find each panel's MASS:, RESISTANCE:,
INSTABILITY: label bounding boxes, crops them, polarity-canonicalizes
to a uniform signature, then averages each label across all panels
(reduces noise, yields a robust template).

Output: ``tools/Mining_Signals/ocr/sc_templates/labels.npz`` containing:

    {
        "mass":         np.ndarray (28, W_mass) float32,
        "resistance":   np.ndarray (28, W_resistance) float32,
        "instability":  np.ndarray (28, W_instability) float32,
        "height":       28,
    }

Run once per significant SC font/HUD-scale change. The runtime
``label_match.find_label_positions`` matches these templates via
multi-scale NCC, so a single template per label works across all
panel capture sizes.

Usage:
    python scripts/build_label_templates.py
    # Optional: --source-dir, --output, --canonical-height
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
)
log = logging.getLogger("build_label_templates")

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
DEFAULT_SOURCE = TOOL / "template_source_panels"
DEFAULT_OUTPUT = TOOL / "ocr" / "sc_templates" / "labels.npz"
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
    """Force text to be BRIGHT pixels regardless of source polarity."""
    thr = _otsu(gray)
    bright = int((gray > thr).sum())
    dark = gray.size - bright
    if dark < bright:
        return (255 - gray).astype(np.uint8)
    return gray.astype(np.uint8)


def _find_labels_via_tesseract(
    img: Image.Image,
) -> dict[str, tuple[int, int, int, int]]:
    """Returns {"mass": (x,y,w,h), "resistance": ..., "instability": ...}."""
    try:
        import pytesseract
    except ImportError:
        log.error("pytesseract not installed — pip install pytesseract")
        sys.exit(1)
    # Point pytesseract at the installed Tesseract binary if not on PATH.
    _candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""),
                     "Programs", "Tesseract-OCR", "tesseract.exe"),
    ]
    for p in _candidate_paths:
        if p and os.path.isfile(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break

    # Restrict to left 60% of image
    W, H = img.size
    left = img.crop((0, 0, int(W * 0.60), H))
    gray = np.array(left.convert("L"), dtype=np.uint8)

    # Try both polarity variants
    thr = _otsu(gray)
    variants = [
        np.where(gray > thr, 0, 255).astype(np.uint8),
        np.where(gray < thr, 0, 255).astype(np.uint8),
    ]
    targets = {"mass": "mass", "resistance": "resi", "instability": "inst"}
    best: dict[str, tuple[int, int, int, int, int]] = {}

    for binary in variants:
        binary_pil = Image.fromarray(binary)
        try:
            data = pytesseract.image_to_data(
                binary_pil,
                config=(
                    "--psm 11 -c tessedit_char_whitelist="
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:"
                ),
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            log.warning("tesseract failed on variant: %s", exc)
            continue
        n = len(data.get("text", []))
        for i in range(n):
            text = (data["text"][i] or "").strip().lower()
            stripped = "".join(c for c in text if c.isalpha())
            if len(stripped) < 4:
                continue
            text = stripped
            x = int(data["left"][i])
            y = int(data["top"][i])
            ww = int(data["width"][i])
            hh = int(data["height"][i])
            for key, needle in targets.items():
                if needle in text:
                    score = len(text)
                    prev = best.get(key)
                    if prev is None or score > prev[4]:
                        best[key] = (x, y, ww, hh, score)
                    break
    return {k: (v[0], v[1], v[2], v[3]) for k, v in best.items()}


def _crop_label(
    img: Image.Image, bbox: tuple[int, int, int, int],
    pad_x: int = 4, pad_y: int = 3,
) -> np.ndarray:
    """Crop label bbox + small margin, return polarity-canonical uint8."""
    x, y, w, h = bbox
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


def _average_templates(templates: list[np.ndarray]) -> Optional[np.ndarray]:
    """Average a list of (H, W) uint8 arrays. Pad to max width before averaging.
    Templates are right-aligned (since labels end in the same colon)."""
    if not templates:
        return None
    h = templates[0].shape[0]
    max_w = max(t.shape[1] for t in templates)
    accum = np.zeros((h, max_w), dtype=np.float32)
    counts = np.zeros((h, max_w), dtype=np.float32)
    for t in templates:
        # Right-align: pad on the LEFT so colon edges line up.
        pad = max_w - t.shape[1]
        accum[:, pad:] += t.astype(np.float32)
        counts[:, pad:] += 1.0
    avg = np.where(counts > 0, accum / np.maximum(counts, 1.0), 0.0)
    return avg.astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--canonical-height", type=int, default=CANONICAL_HEIGHT,
        help="Resize all crops to this height before averaging",
    )
    args = ap.parse_args()

    source_dir = args.source_dir
    if not source_dir.is_dir():
        log.error("source dir not found: %s", source_dir)
        return 1

    pngs = sorted(source_dir.glob("*.png"))
    if not pngs:
        log.error("no PNG files in %s", source_dir)
        return 1
    log.info("found %d source panel(s)", len(pngs))

    per_label_crops: dict[str, list[np.ndarray]] = {
        "mass": [], "resistance": [], "instability": [],
    }

    for png in pngs:
        log.info("processing %s", png.name)
        try:
            img = Image.open(png).convert("RGB")
        except Exception as exc:
            log.warning("  open failed: %s", exc)
            continue
        bboxes = _find_labels_via_tesseract(img)
        if not bboxes:
            log.warning("  no labels found via Tesseract")
            continue
        for key in ("mass", "resistance", "instability"):
            bbox = bboxes.get(key)
            if bbox is None:
                log.info("  %s: not detected", key)
                continue
            log.info(
                "  %s: bbox=(x=%d, y=%d, w=%d, h=%d)",
                key, bbox[0], bbox[1], bbox[2], bbox[3],
            )
            crop = _crop_label(img, bbox)
            crop = _resize_to_height(crop, args.canonical_height)
            per_label_crops[key].append(crop)

    payload: dict[str, np.ndarray] = {}
    for key in ("mass", "resistance", "instability"):
        crops = per_label_crops[key]
        if not crops:
            log.warning("no crops for %r — template won't be saved", key)
            continue
        avg = _average_templates(crops)
        if avg is None:
            continue
        payload[key] = avg
        log.info(
            "  template %s: %d source crops averaged → shape %s",
            key, len(crops), avg.shape,
        )

    if not payload:
        log.error("no templates generated — check Tesseract output")
        return 1

    payload["height"] = np.array(args.canonical_height, dtype=np.int32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    log.info("wrote %s", args.output)

    # Also save individual crops as PNGs for visual inspection
    debug_dir = args.output.parent / "labels_debug"
    debug_dir.mkdir(exist_ok=True)
    for key, arr in payload.items():
        if key == "height":
            continue
        Image.fromarray(arr).save(debug_dir / f"{key}_template.png")
    log.info("wrote per-label debug PNGs to %s", debug_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
