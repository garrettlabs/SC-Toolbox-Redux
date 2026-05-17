"""End-to-end sanity test for the trained CNN.

Picks N labeled user captures, runs the full pipeline:
    panel → _find_label_rows → _find_value_crop → _segment_digits → CNN classify
and compares the assembled string to the user's ground-truth label.

Reports per-field accuracy and per-image diff so we can see exactly
where the pipeline breaks (segmentation, classification, or both).

Run with:
    OCR_TRAIN_SOURCE=path/to/glyphs python scripts/sanity_test_cnn.py
    python scripts/sanity_test_cnn.py --n 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
sys.path.insert(0, str(TOOL))

# Import the extractor's segmentation utilities
from scripts.extract_labeled_glyphs import (  # noqa: E402
    _find_label_rows, _find_value_crop, _segment_digits,
    _glyph_to_28x28, _upscale_to_ref,
)


def load_model(model_path: Path):
    """Load the trained PyTorch CNN."""
    import torch
    from ocr.train_torch import build_cnn
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    classes = ckpt["classes"]
    model = build_cnn(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


def predict_glyph(model, classes, glyph_28x28: np.ndarray) -> str:
    """Run a single 28x28 glyph through the CNN, return predicted character."""
    import torch
    arr = glyph_28x28.astype(np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        idx = int(logits.argmax(dim=1))
    return classes[idx]


def predict_field_string(model, classes, gray_crop: np.ndarray, expected_len: int) -> str:
    """Segment + classify each glyph in a value crop, return the string."""
    spans = _segment_digits(gray_crop, expected_count=expected_len)
    if len(spans) > expected_len:
        spans = spans[-expected_len:]
    out_chars = []
    for x1, x2 in spans:
        glyph = _glyph_to_28x28(gray_crop, x1, x2)
        if glyph is None:
            continue
        out_chars.append(predict_glyph(model, classes, glyph))
    return "".join(out_chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="Number of panels to test")
    ap.add_argument("--model",
                    default=str(TOOL / "ocr" / "models" / "torch_digit.pt"))
    args = ap.parse_args()

    print(f"Loading model from {args.model}...")
    model, classes = load_model(Path(args.model))
    print(f"Classes: {classes}\n")

    # Find all labeled captures
    captures = []
    panels_root = TOOL / "training_data_panels"
    for user_dir in panels_root.glob("user_*"):
        for jpath in (user_dir / "region1").glob("cap_*.json"):
            ipath = jpath.with_suffix(".png")
            if ipath.is_file():
                captures.append((ipath, jpath))
    print(f"Found {len(captures)} labeled panels.")
    if not captures:
        print("No labeled panels to test.")
        return

    sample = random.sample(captures, min(args.n, len(captures)))

    field_correct = {"mass": 0, "resistance": 0, "instability": 0}
    field_total = {"mass": 0, "resistance": 0, "instability": 0}
    char_correct = 0
    char_total = 0

    for img_path, json_path in sample:
        try:
            label = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        img, _scale = _upscale_to_ref(img)
        gray = np.asarray(img.convert("L"), dtype=np.uint8)

        try:
            rows = _find_label_rows(img)
        except Exception as exc:
            print(f"\n{img_path.name}: label rows failed: {exc}")
            continue
        if not rows:
            print(f"\n{img_path.name}: no label rows detected, skipping")
            continue

        print(f"\n--- {img_path.name} ---")
        for field in ("mass", "resistance", "instability"):
            truth = str(label.get(field, "")).strip()
            if not truth:
                continue
            chars = [c for c in truth if c.isdigit() or c in ".%"]
            if not chars:
                continue
            entry = rows.get(field)
            if entry is None:
                print(f"  {field}: truth={truth!r}  (label row not found)")
                continue
            y1, y2, lbl_right = entry
            x_min = max(0, lbl_right + 6)
            value_crop = _find_value_crop(img, gray, y1, y2, x_min=x_min)
            if value_crop is None:
                print(f"  {field}: truth={truth!r}  (value crop empty)")
                continue
            gray_crop = np.asarray(value_crop.convert("L"), dtype=np.uint8)
            pred = predict_field_string(model, classes, gray_crop, len(chars))
            ok = pred == "".join(chars)
            mark = "OK" if ok else "FAIL"
            print(f"  {field}: truth={''.join(chars)!r:>8}  pred={pred!r:>8}  {mark}")
            field_total[field] += 1
            if ok:
                field_correct[field] += 1
            # Per-character accuracy
            for i, ch in enumerate(chars):
                char_total += 1
                if i < len(pred) and pred[i] == ch:
                    char_correct += 1

    print("\n=== SUMMARY ===")
    for f in ("mass", "resistance", "instability"):
        n, t = field_correct[f], field_total[f]
        pct = (n / t * 100) if t else 0
        print(f"  {f:11s}: {n}/{t} fields correct  ({pct:.1f}%)")
    if char_total:
        print(f"  per-char    : {char_correct}/{char_total} correct  ({char_correct/char_total*100:.1f}%)")


if __name__ == "__main__":
    main()
