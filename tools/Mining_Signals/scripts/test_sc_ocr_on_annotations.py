"""One-shot test: feed SC-OCR's digit CNN clean annotated value crops.

Loads each annotated panel (those with .boxes.json sidecars), uses the
``mass_row``, ``resistance_row``, ``instability_row`` boxes to extract
the value text, runs the SC-OCR mining-HUD digit CNN on each, and
prints what it reads.

If this prints the correct numeric values, we've proven:
  1. The digit CNN works (the network on the inside).
  2. The annotation captures the right region.
  3. Therefore the only remaining bug is the runtime ``_find_value_crop``
     pipeline that's NOT delivering crops as clean as the annotated ones.

Usage:
    python scripts/test_sc_ocr_on_annotations.py
    python scripts/test_sc_ocr_on_annotations.py --save-crops
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

# Make the toolbox importable
THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
sys.path.insert(0, str(TOOL))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DEFAULT_SOURCE = TOOL / "training_data_panels" / "user_20260418_154408" / "region1"


# ────────────────────────────────────────────────────────────────────
# SC-OCR helpers (subset duplicated here so this script is standalone)
# ────────────────────────────────────────────────────────────────────


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


def _canonicalize_polarity(gray: np.ndarray) -> np.ndarray:
    """Force text to be BRIGHT pixels (matches CNN training convention)."""
    thr = _otsu(gray)
    bright = int((gray > thr).sum())
    dark = gray.size - bright
    if dark < bright:
        return (255 - gray).astype(np.uint8)
    return gray.astype(np.uint8)


def _segment_glyphs(gray: np.ndarray, binary: np.ndarray) -> list[np.ndarray]:
    """Project to columns, find character spans, crop + pad to 28x28."""
    h, w = gray.shape
    proj = np.sum(binary > 0, axis=0)
    spans = []
    in_char = False
    start = 0
    for x in range(w + 1):
        val = proj[x] if x < w else 0
        if val > 0 and not in_char:
            in_char = True
            start = x
        elif val == 0 and in_char:
            in_char = False
            if x - start >= 2:
                spans.append((start, x))
    crops = []
    for x1, x2 in spans:
        ys = np.where(np.any(binary[:, x1:x2] > 0, axis=1))[0]
        if len(ys) < 3:
            continue
        y1, y2 = ys[0], ys[-1] + 1
        crop = gray[y1:y2, x1:x2].astype(np.float32)
        pad = 2
        padded = np.full(
            (crop.shape[0] + pad * 2, crop.shape[1] + pad * 2),
            255.0, dtype=np.float32,
        )
        padded[pad:pad + crop.shape[0], pad:pad + crop.shape[1]] = crop
        pil = Image.fromarray(padded.astype(np.uint8)).resize(
            (28, 28), Image.BILINEAR,
        )
        crops.append(np.array(pil, dtype=np.float32) / 255.0)
    return crops


_session = None
_char_classes = "0123456789.%"


def _load_cnn():
    global _session, _char_classes
    if _session is not None:
        return True
    try:
        import onnxruntime as ort
    except ImportError:
        log.error("onnxruntime not installed")
        return False
    model_path = TOOL / "ocr" / "models" / "model_cnn.onnx"
    meta_path = TOOL / "ocr" / "models" / "model_cnn.json"
    if not model_path.is_file():
        log.error("model not found: %s", model_path)
        return False
    _session = ort.InferenceSession(str(model_path))
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            if "char_classes" in meta:
                _char_classes = meta["char_classes"]
        except Exception:
            pass
    log.info("loaded CNN: %s (classes=%r)", model_path.name, _char_classes)
    return True


def _classify_crops(crops: list[np.ndarray]) -> list[tuple[str, float]]:
    if not crops or not _load_cnn():
        return []
    inp_name = _session.get_inputs()[0].name
    batch = np.array(crops, dtype=np.float32).reshape(-1, 1, 28, 28)
    logits = _session.run(None, {inp_name: batch})[0]
    results = []
    for i in range(len(crops)):
        probs = np.exp(logits[i] - np.max(logits[i]))
        probs /= probs.sum()
        idx = int(np.argmax(probs))
        results.append((_char_classes[idx], float(probs[idx])))
    return results


# ────────────────────────────────────────────────────────────────────
# Per-row value extraction from annotated row box
# ────────────────────────────────────────────────────────────────────


def _extract_value_from_row(
    img: Image.Image, row_box: dict, save_path: Optional[Path] = None,
) -> str:
    """Crop the row, isolate the VALUE portion (rightmost text cluster),
    run the digit CNN, return the decoded string."""
    x, y, w, h = row_box["x"], row_box["y"], row_box["w"], row_box["h"]
    row = img.crop((x, y, x + w, y + h)).convert("L")
    gray = np.array(row, dtype=np.uint8)
    gray_canon = _canonicalize_polarity(gray)
    thr = _otsu(gray_canon)
    binary = (gray_canon > thr).astype(np.uint8) * 255

    # Find the largest gap between text clusters → split label from value.
    proj = (binary > 0).sum(axis=0)
    hot = proj >= 2
    spans = []
    in_run = False
    s = 0
    for i in range(len(hot)):
        if hot[i] and not in_run:
            in_run = True
            s = i
        elif not hot[i] and in_run:
            in_run = False
            spans.append((s, i))
    if in_run:
        spans.append((s, len(hot)))

    if not spans:
        return ""

    # Filter narrow noise spans (< 2 px wide)
    spans = [(a, b) for a, b in spans if b - a >= 2]
    if not spans:
        return ""

    # Find the LARGEST gap between consecutive spans — that's the
    # label-to-value separator.
    gaps = [(spans[i + 1][0] - spans[i][1], i) for i in range(len(spans) - 1)]
    if gaps:
        max_gap, gap_idx = max(gaps, key=lambda g: g[0])
        if max_gap >= 8:
            # Spans to the RIGHT of the largest gap = value
            value_spans = spans[gap_idx + 1:]
        else:
            # No big gap — assume all spans are value
            value_spans = spans
    else:
        value_spans = spans

    if not value_spans:
        return ""

    # Crop from the start of the first value span to the end of the
    # last value span, with small margin.
    v_left = max(0, value_spans[0][0] - 4)
    v_right = min(binary.shape[1], value_spans[-1][1] + 4)
    value_gray = gray_canon[:, v_left:v_right]
    value_bin = binary[:, v_left:v_right]

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(value_gray).save(save_path)

    # Now segment + classify
    crops = _segment_glyphs(value_gray, value_bin)
    if not crops:
        return ""
    results = _classify_crops(crops)
    return "".join(ch for ch, _ in results), \
           [conf for _, conf in results]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--save-crops", action="store_true",
                    help="Save the value-crop PNG for each row to "
                         "scripts/_test_value_crops/ for inspection")
    ap.add_argument("--max", type=int, default=20,
                    help="Process at most N annotated images")
    args = ap.parse_args()

    if not _load_cnn():
        return 1

    annotated = sorted(args.source.glob("*.boxes.json"))
    if not annotated:
        log.error("no .boxes.json files found in %s", args.source)
        log.error("annotate at least one image with template_annotator.py first")
        return 1

    log.info("found %d annotated image(s)", len(annotated))
    annotated = annotated[:args.max]
    save_dir = TOOL / "scripts" / "_test_value_crops" if args.save_crops else None

    log.info("")
    log.info(
        "%-40s %-12s %-12s %-12s",
        "image", "MASS", "RESISTANCE", "INSTABILITY",
    )
    log.info("-" * 90)
    correct_count = 0
    total_count = 0

    for box_path in annotated:
        img_path = box_path.with_suffix("").with_suffix(".png")
        if not img_path.is_file():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            data = json.loads(box_path.read_text())
        except Exception:
            continue
        boxes = data.get("boxes", {})

        results = {}
        for field, key in (
            ("mass", "mass_row"),
            ("resistance", "resistance_row"),
            ("instability", "instability_row"),
        ):
            box = boxes.get(key)
            if box is None:
                results[field] = "NOBOX"
                continue
            sp = (
                save_dir / f"{img_path.stem}_{field}.png"
                if save_dir else None
            )
            try:
                text, confs = _extract_value_from_row(img, box, sp)
                mc = min(confs) if confs else 0.0
                results[field] = f"{text} ({mc:.2f})"
                if text:
                    total_count += 1
            except Exception as exc:
                results[field] = f"ERR: {exc}"

        log.info(
            "%-40s %-12s %-12s %-12s",
            img_path.name, results["mass"],
            results["resistance"], results["instability"],
        )

    if save_dir:
        log.info("")
        log.info("value crops saved to: %s", save_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
