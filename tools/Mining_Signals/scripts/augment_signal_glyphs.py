"""Synthetic glyph augmentation for the signal-region CNN.

Closes the "training-vs-inference distribution gap":

  Training glyphs (Glyph Forge curated): pixel-perfect single digits,
  centered, no neighboring-digit pixels, no icon edge.

  Inference glyphs (live segmenter): off-by-N px shifts, leftover
  icon column on the left, adjacent-digit pixels bleeding in on
  the right, varying contrast.

This script reads every clean glyph from the staging dir, generates
N augmented variants per source glyph, and writes them alongside the
originals so the next ``train_for_region.py signal --no-extract``
trains on the union.

Augmentations applied per source glyph (random subset):

  * shift_h(±dx)         — horizontal translation, fills with bg
  * shift_v(±dy)         — vertical translation, fills with bg
  * left_artifact(w, v)  — bright/dark vertical stripe on left edge
                           (simulates leftover icon column)
  * right_ghost(other)   — paste a sliver of a different-class glyph's
                           leading edge on the right (simulates
                           adjacent-digit residue from kerning)
  * left_ghost(other)    — same on the left (simulates trailing
                           edge of previous digit creeping in)
  * contrast(s, b)       — multiplicative scale + brightness offset
  * gaussian_noise(sigma)— additive Gaussian noise

Filenames use prefix ``aug_<orig_stem>_<i>.png`` so they're trivially
distinguishable from human-curated ``user_<src>_<i>.png`` files.
Pass ``--clear`` to wipe previous augmented files before regenerating.

Run with:
    python scripts/augment_signal_glyphs.py
    python scripts/augment_signal_glyphs.py --variants 12 --clear
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
sys.path.insert(0, str(TOOL))
from ocr import training_registry  # noqa: E402

KIND = "signal"
AUG_PREFIX = "aug_"


# ─────────────────────────────────────────────────────────────
# Augmentation primitives — each takes uint8 (28,28), returns uint8
# ─────────────────────────────────────────────────────────────

def _bg_value(g: np.ndarray) -> int:
    """Return the most-common pixel value (background)."""
    # Quick mode-of-corners heuristic — corners are almost always
    # background in a centered-digit crop.
    corners = np.concatenate([
        g[:3, :3].flatten(), g[:3, -3:].flatten(),
        g[-3:, :3].flatten(), g[-3:, -3:].flatten(),
    ])
    return int(np.median(corners))


def shift_h(g: np.ndarray, dx: int) -> np.ndarray:
    out = np.full_like(g, _bg_value(g))
    h, w = g.shape
    if dx > 0:
        out[:, dx:] = g[:, :w - dx]
    elif dx < 0:
        out[:, :w + dx] = g[:, -dx:]
    else:
        out = g.copy()
    return out


def shift_v(g: np.ndarray, dy: int) -> np.ndarray:
    out = np.full_like(g, _bg_value(g))
    h, w = g.shape
    if dy > 0:
        out[dy:, :] = g[:h - dy, :]
    elif dy < 0:
        out[:h + dy, :] = g[-dy:, :]
    else:
        out = g.copy()
    return out


def left_artifact(g: np.ndarray, width: int, value: int) -> np.ndarray:
    """Paint a vertical bar on the left edge — simulates leftover
    icon column when the icon mask was too narrow."""
    out = g.copy()
    out[:, :width] = value
    return out


def right_ghost(g: np.ndarray, other: np.ndarray, sliver_w: int) -> np.ndarray:
    """Bring in a sliver of another digit's LEADING edge on the right
    side of `g`. Simulates the next digit's pixels bleeding into a
    too-wide segmentation crop."""
    if other is None or sliver_w <= 0 or sliver_w >= g.shape[1]:
        return g.copy()
    out = g.copy()
    sliver = other[:, :sliver_w].astype(np.int16)
    target = out[:, -sliver_w:].astype(np.int16)
    blended = ((target + sliver) // 2).clip(0, 255).astype(np.uint8)
    out[:, -sliver_w:] = blended
    return out


def left_ghost(g: np.ndarray, other: np.ndarray, sliver_w: int) -> np.ndarray:
    """Same as right_ghost but on the left side — simulates the PREVIOUS
    digit's trailing edge."""
    if other is None or sliver_w <= 0 or sliver_w >= g.shape[1]:
        return g.copy()
    out = g.copy()
    sliver = other[:, -sliver_w:].astype(np.int16)
    target = out[:, :sliver_w].astype(np.int16)
    blended = ((target + sliver) // 2).clip(0, 255).astype(np.uint8)
    out[:, :sliver_w] = blended
    return out


def contrast(g: np.ndarray, scale: float, offset: int) -> np.ndarray:
    out = g.astype(np.int16) * scale + offset
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_noise(g: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, sigma, g.shape)
    out = g.astype(np.int16) + noise.astype(np.int16)
    return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────

def _load_originals_by_class(staging_dir: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for cls in "0123456789":
        d = staging_dir / cls
        if not d.is_dir():
            continue
        # Originals only: the human-curated samples + any auto-extracted
        # ones (anything NOT prefixed with 'aug_').
        originals = [
            f for f in d.glob("*.png")
            if not f.name.startswith(AUG_PREFIX)
        ]
        if originals:
            out[cls] = originals
    return out


def _wipe_augmented(staging_dir: Path) -> int:
    n = 0
    for cls in "0123456789":
        d = staging_dir / cls
        if not d.is_dir():
            continue
        for f in d.glob(f"{AUG_PREFIX}*.png"):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    return n


def _make_variants(
    g: np.ndarray,
    other_class_samples: list[np.ndarray],
    n: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Produce ``n`` augmented variants of one source glyph, drawing
    'other digit' samples for ghosting from ``other_class_samples``."""
    out: list[np.ndarray] = []
    for _ in range(n):
        v = g.copy()
        # Apply 1-3 random transforms per variant
        n_ops = rng.integers(1, 4)
        ops = list(rng.permutation([
            "shift_h", "left_artifact", "right_ghost", "left_ghost",
            "contrast", "noise", "shift_v",
        ]))[:int(n_ops)]
        for op in ops:
            if op == "shift_h":
                dx = int(rng.integers(-3, 4))  # -3..+3
                v = shift_h(v, dx)
            elif op == "shift_v":
                dy = int(rng.integers(-2, 3))  # -2..+2
                v = shift_v(v, dy)
            elif op == "left_artifact":
                width = int(rng.integers(2, 6))
                # Bias toward bright (white) since icon edges are usually
                # bright after polarity canonicalization. ~70% bright.
                value = int(rng.integers(180, 256)) if rng.random() < 0.7 \
                    else int(rng.integers(0, 60))
                v = left_artifact(v, width, value)
            elif op == "right_ghost" and other_class_samples:
                other = other_class_samples[
                    int(rng.integers(0, len(other_class_samples)))
                ]
                sliver = int(rng.integers(2, 6))
                v = right_ghost(v, other, sliver)
            elif op == "left_ghost" and other_class_samples:
                other = other_class_samples[
                    int(rng.integers(0, len(other_class_samples)))
                ]
                sliver = int(rng.integers(2, 6))
                v = left_ghost(v, other, sliver)
            elif op == "contrast":
                scale = float(rng.uniform(0.75, 1.25))
                offset = int(rng.integers(-25, 26))
                v = contrast(v, scale, offset)
            elif op == "noise":
                sigma = float(rng.uniform(2.0, 12.0))
                v = gaussian_noise(v, sigma, rng)
        out.append(v)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variants", type=int, default=8,
        help="How many augmented variants to generate per original glyph.",
    )
    p.add_argument(
        "--clear", action="store_true",
        help="Wipe existing aug_*.png files before generating new ones.",
    )
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    spec = training_registry.get(KIND)
    staging = spec.glyph_staging_dir
    rng = np.random.default_rng(args.seed)

    if args.clear:
        n = _wipe_augmented(staging)
        print(f"[clear] removed {n} previous aug_*.png files")

    originals = _load_originals_by_class(staging)
    if not originals:
        print(f"[!] No originals found under {staging}")
        return 2

    # Pre-load every class's samples as numpy arrays so right_ghost /
    # left_ghost can sample neighbors cheaply.
    class_arrays: dict[str, list[np.ndarray]] = {}
    for cls, paths in originals.items():
        arrs = []
        for p in paths:
            try:
                arrs.append(
                    np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
                )
            except Exception:
                pass
        class_arrays[cls] = arrs

    # All "other" pool — every other class's glyphs (for ghosting)
    flat_other_per_cls: dict[str, list[np.ndarray]] = {}
    for cls in originals:
        pool = []
        for other_cls, arrs in class_arrays.items():
            if other_cls == cls:
                continue
            pool.extend(arrs)
        flat_other_per_cls[cls] = pool

    print(f"=== Augmenting {sum(len(v) for v in originals.values())} "
          f"originals → {args.variants}× variants each ===")
    print(f"    staging dir: {staging}")
    print()

    total_written = 0
    for cls in sorted(originals):
        d = staging / cls
        sources = class_arrays[cls]
        wrote = 0
        for src_path, src_arr in zip(originals[cls], sources):
            variants = _make_variants(
                src_arr, flat_other_per_cls[cls], args.variants, rng,
            )
            for i, v in enumerate(variants):
                out = d / f"{AUG_PREFIX}{src_path.stem}_{i}.png"
                try:
                    Image.fromarray(v, mode="L").save(out)
                    wrote += 1
                except Exception:
                    continue
        print(f"  {cls!r}: {len(sources)} originals → +{wrote} aug")
        total_written += wrote

    print()
    print(f"[done] wrote {total_written} augmented glyphs")
    print()
    print("Next: re-train on the combined original+augmented set:")
    print("  python scripts/train_for_region.py signal --no-extract --force")
    return 0


if __name__ == "__main__":
    sys.exit(main())
