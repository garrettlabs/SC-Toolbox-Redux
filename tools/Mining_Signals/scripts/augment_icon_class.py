"""Augment ``training_data_blacklist/`` icon PNG(s) into the signal
CNN's ``icon`` training class.

The signal CNN is trained on per-glyph 28×28 crops. With one source
PNG of the location-pin icon, we'd never get a robust 11th class —
the network would memorize the single crop and fail on every variant
the live segmenter actually produces. This script generates ~500
synthetic 28×28 icon crops by applying realistic transforms to the
source PNG(s):

  * scale jitter (60-120% of canonical size, mimics different HUD
    resolutions)
  * x/y translation jitter (±3 px in either axis, mimics segmenter
    framing variance)
  * rotation jitter (±6°, mimics anti-aliasing rotation artifacts)
  * polarity flip (50%) — segments hand the CNN a polarity-
    canonicalized crop, but the canonicalizer's choice depends on
    the local histogram, so the icon can arrive in either polarity
  * brightness/contrast jitter (±15%, mimics chromatic aberration
    and bubble-glow gradient)
  * Gaussian noise (σ up to 8 in [0, 255], mimics capture/encoder
    noise)
  * left/right edge artifacts (15% chance) — a thin bright stripe
    at one edge to simulate adjacent-digit ink bleeding into the
    icon's segmented bbox
  * background-fill jitter — pad the icon onto a constant-grey
    canvas chosen from the bubble-glow brightness range

Output: ``training_data_user_sig/icon/aug_<source_stem>_<i>.png``
where ``i`` runs 0..N-1. Files are non-clobbering (skipped if they
already exist) so re-running is idempotent — pass ``--clear`` to
wipe and regenerate.

Run with::

    python scripts/augment_icon_class.py
    python scripts/augment_icon_class.py --variants 800 --clear
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
sys.path.insert(0, str(TOOL))
from ocr import training_registry  # noqa: E402

KIND = "signal"
ICON_CLASS_DIRNAME = "icon"
AUG_PREFIX = "aug_"
TARGET_SIZE = 28


# ─────────────────────────────────────────────────────────────
# Augmentation primitives
# ─────────────────────────────────────────────────────────────

def _canonicalize_polarity(arr: np.ndarray) -> np.ndarray:
    """Match ``api._canonicalize_polarity`` — minority class is text,
    text ends up bright."""
    if arr.size == 0:
        return arr
    flat = arr.flatten()
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    total = arr.size
    sum_total = float(np.sum(np.arange(256) * hist))
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
    bright = int((arr > threshold).sum())
    dark = total - bright
    if dark < bright:
        return (255 - arr).astype(np.uint8)
    return arr.astype(np.uint8)


def _icon_to_28x28(
    pil_src: Image.Image,
    *,
    scale: float,
    rot_deg: float,
    dx: int, dy: int,
    bg_value: int,
    invert: bool,
    contrast: float,
    brightness: int,
    noise_sigma: float,
    edge_artifact: str | None,  # "left" / "right" / None
) -> np.ndarray:
    # Step 1: canonical-polarity icon, padded to a square.
    src = np.asarray(pil_src.convert("L"), dtype=np.uint8)
    canon = _canonicalize_polarity(src)
    h0, w0 = canon.shape
    side0 = max(h0, w0)
    sq = np.full((side0, side0), 0, dtype=np.uint8)
    yo = (side0 - h0) // 2
    xo = (side0 - w0) // 2
    sq[yo:yo + h0, xo:xo + w0] = canon

    # Step 2: scale to (TARGET_SIZE * scale) — leaving room for jitter.
    target_inner = max(8, int(round(TARGET_SIZE * scale)))
    target_inner = min(target_inner, TARGET_SIZE)
    pil_resized = Image.fromarray(sq).resize(
        (target_inner, target_inner), Image.BILINEAR,
    )

    # Step 3: rotate.
    if rot_deg:
        pil_resized = pil_resized.rotate(
            rot_deg, resample=Image.BILINEAR,
            fillcolor=0,  # rotation fills with bg (dark, since
                          # we're in canonical polarity = bright icon)
        )

    # Step 4: paste onto the 28×28 canvas at a jittered position.
    canvas = np.full(
        (TARGET_SIZE, TARGET_SIZE), bg_value, dtype=np.uint8,
    )
    pl_arr = np.asarray(pil_resized, dtype=np.uint8)
    inner_h, inner_w = pl_arr.shape
    base_y = (TARGET_SIZE - inner_h) // 2 + dy
    base_x = (TARGET_SIZE - inner_w) // 2 + dx
    base_y = max(0, min(TARGET_SIZE - inner_h, base_y))
    base_x = max(0, min(TARGET_SIZE - inner_w, base_x))
    # Paste with max-blend (icon is bright on dark canvas).
    region = canvas[base_y:base_y + inner_h, base_x:base_x + inner_w]
    canvas[base_y:base_y + inner_h, base_x:base_x + inner_w] = np.maximum(
        region, pl_arr,
    )

    # Step 5: contrast/brightness.
    f32 = canvas.astype(np.float32)
    f32 = f32 * contrast + brightness
    f32 = np.clip(f32, 0, 255)

    # Step 6: edge artifact (mimics a leftover digit/comma stripe).
    if edge_artifact == "left":
        stripe_w = random.randint(1, 3)
        stripe_v = random.randint(140, 240)
        f32[:, :stripe_w] = stripe_v
    elif edge_artifact == "right":
        stripe_w = random.randint(1, 3)
        stripe_v = random.randint(140, 240)
        f32[:, -stripe_w:] = stripe_v

    # Step 7: noise.
    if noise_sigma > 0:
        f32 += np.random.normal(0, noise_sigma, f32.shape)
        f32 = np.clip(f32, 0, 255)

    # Step 8: optional polarity flip — the live pipeline canonicalizes
    # input polarity but the canonicalizer's choice depends on the
    # local histogram, so the icon can arrive at the CNN in EITHER
    # polarity. Train on both.
    out = f32.astype(np.uint8)
    if invert:
        out = 255 - out
    return out


def _generate_one(
    src: Image.Image, rng: random.Random,
) -> np.ndarray:
    """Produce one randomized 28×28 icon variant."""
    return _icon_to_28x28(
        src,
        scale=rng.uniform(0.65, 1.0),
        rot_deg=rng.uniform(-6.0, 6.0),
        dx=rng.randint(-3, 3),
        dy=rng.randint(-3, 3),
        bg_value=rng.choice([0, rng.randint(20, 90)]),
        invert=rng.random() < 0.5,
        contrast=rng.uniform(0.85, 1.15),
        brightness=rng.randint(-15, 15),
        noise_sigma=rng.uniform(0.0, 8.0),
        edge_artifact=rng.choices(
            [None, "left", "right"], weights=[85, 8, 7], k=1,
        )[0],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, default=500,
                        help="Total icon variants to generate.")
    parser.add_argument("--clear", action="store_true",
                        help="Delete existing aug_* PNGs in the icon "
                             "class folder before generating.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility.")
    args = parser.parse_args()

    spec = training_registry.get(KIND)
    if "@" not in spec.label_set:
        print(
            f"WARNING: '{KIND}' label_set {spec.label_set!r} has no "
            "'@' icon class — augmentation will produce files but the "
            "next training run won't pick them up. Add '@' to "
            "training_registry.signal.label_set first."
        )

    bl_dir = TOOL / "training_data_blacklist"
    if not bl_dir.is_dir():
        print(f"ERROR: blacklist dir not found: {bl_dir}")
        return 1
    sources = sorted(bl_dir.rglob("*.png"))
    if not sources:
        print(f"ERROR: no PNGs in {bl_dir}")
        return 1

    icon_dir = spec.glyph_staging_dir / ICON_CLASS_DIRNAME
    icon_dir.mkdir(parents=True, exist_ok=True)

    if args.clear:
        n_cleared = 0
        for p in icon_dir.glob(f"{AUG_PREFIX}*.png"):
            p.unlink()
            n_cleared += 1
        print(f"  cleared {n_cleared} previous augmented icons")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    n_per_source = max(1, args.variants // len(sources))
    print(
        f"Generating {n_per_source} variants per source × "
        f"{len(sources)} source(s) = {n_per_source * len(sources)} "
        f"total icons"
    )

    n_written = 0
    for src_path in sources:
        try:
            src_img = Image.open(src_path)
        except Exception as exc:
            print(f"  skipped {src_path.name}: open failed ({exc})")
            continue
        stem_safe = "".join(
            c if c.isalnum() else "_" for c in src_path.stem
        )
        for i in range(n_per_source):
            arr = _generate_one(src_img, rng)
            out = icon_dir / f"{AUG_PREFIX}{stem_safe}_{i:04d}.png"
            try:
                Image.fromarray(arr, mode="L").save(out)
                n_written += 1
            except Exception as exc:
                print(f"  failed to write {out.name}: {exc}")

    print(f"  wrote {n_written} icon variants to {icon_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
