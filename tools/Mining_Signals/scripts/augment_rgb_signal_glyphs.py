"""Class-balanced RGB augmentation for signature glyphs.

Reads samples from ``training_data_user_sig_rgb/<class>/`` and writes
augmented variants ALONGSIDE the originals (as ``aug_<src>_NN.png``).
Per-class augmentation factor is chosen so each class lands near a
target sample count, addressing the heavy class imbalance from real
extraction (e.g. class ``9`` had only 1 real sample after the Tesseract-
verification gate).

Augmentations applied to each variant:
  * small rotation  (±3°)
  * translation     (±1 px)
  * brightness      (±10%)
  * contrast        (±10%)
  * Gaussian noise  (σ up to 4 of 255)

NOT applied (would distort the digit's geometry away from
realistic SC HUD captures):
  * scale jitter beyond the rotation's natural shrink
  * polarity flip (real RGB has fixed polarity)
  * heavy noise / blur

All augmentations preserve the 3-channel format and white-padding
ring.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
RGB_ROOT = TOOL / "training_data_user_sig_rgb"

# Per-class target counts. Underrepresented classes get more aggressive
# augmentation. Conservative numbers — enough for reasonable CNN
# learning, not so many that augmentation patterns dominate.
TARGET_PER_CLASS = 250


def _augment_one(img_rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply one randomized augmentation to a 28×28×3 uint8 sample."""
    pil = Image.fromarray(img_rgb, mode="RGB")
    # Rotate
    rot_deg = rng.uniform(-3.0, 3.0)
    if abs(rot_deg) > 0.1:
        pil = pil.rotate(
            rot_deg, resample=Image.BILINEAR, fillcolor=(255, 255, 255),
        )
    arr = np.asarray(pil, dtype=np.float32)
    # Translate
    dx = rng.randint(-1, 1)
    dy = rng.randint(-1, 1)
    if dx != 0 or dy != 0:
        arr = np.roll(arr, shift=(dy, dx, 0), axis=(0, 1, 2))
        if dy > 0:
            arr[:dy, :, :] = 255
        elif dy < 0:
            arr[dy:, :, :] = 255
        if dx > 0:
            arr[:, :dx, :] = 255
        elif dx < 0:
            arr[:, dx:, :] = 255
    # Brightness (multiplicative, per channel)
    brightness = rng.uniform(0.90, 1.10)
    arr = arr * brightness
    # Contrast (centered at 128)
    contrast = rng.uniform(0.90, 1.10)
    arr = (arr - 128.0) * contrast + 128.0
    # Noise (light Gaussian, per pixel per channel)
    sigma = rng.uniform(0.0, 4.0)
    if sigma > 0:
        arr = arr + np.random.normal(0, sigma, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=TARGET_PER_CLASS,
                   help="Target sample count per class.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clear", action="store_true",
                   help="Clear existing aug_*.png before generating.")
    args = p.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print(f"=== RGB augmentation (target {args.target}/class) ===")
    print(f"  Source dir: {RGB_ROOT}")
    if not RGB_ROOT.is_dir():
        print(f"  Source missing — run extract_rgb_signal_glyphs.py first")
        return 1

    total_added = 0
    for cls_dir in sorted(RGB_ROOT.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("_"):
            continue
        cls = cls_dir.name
        if args.clear:
            for f in cls_dir.glob("aug_*.png"):
                f.unlink()
        originals = [f for f in cls_dir.glob("*.png") if not f.name.startswith("aug_")]
        if not originals:
            print(f"  class {cls!r}: no originals — skipping")
            continue
        existing_aug = list(cls_dir.glob("aug_*.png"))
        current = len(originals) + len(existing_aug)
        needed = max(0, args.target - current)
        if needed == 0:
            print(f"  class {cls!r}: already at {current} (target {args.target}) — skipping")
            continue
        # Distribute the augmentations evenly across original samples
        per_orig = math.ceil(needed / len(originals))
        added = 0
        for orig_path in originals:
            try:
                img = np.asarray(
                    Image.open(orig_path).convert("RGB"), dtype=np.uint8,
                )
            except Exception:
                continue
            stem = orig_path.stem
            for i in range(per_orig):
                if added >= needed:
                    break
                aug = _augment_one(img, rng)
                out_path = cls_dir / f"aug_{stem}_{i:03d}.png"
                if out_path.exists():
                    continue
                Image.fromarray(aug, mode="RGB").save(out_path)
                added += 1
        total_added += added
        print(f"  class {cls!r}: {len(originals)} orig + {len(existing_aug)} prev + {added} new = {len(originals) + len(existing_aug) + added}")

    print(f"\n  Total augmentations added: {total_added}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
