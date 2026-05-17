"""Synthetic training-data generator for the value-crop CRNN.

The CRNN needs whole-value labeled crops like ``("0.61", image)``,
which we do not have — ``training_data/{0-9}/*.png`` only labels
single glyphs. This module fabricates sequence crops by
concatenating real single-glyph crops left-to-right with random
spacing and mild augmentation. ``.`` and ``%`` are absent from the
existing labeled set and are bootstrapped here via PIL rendering
of a sans-serif system font at matching pixel height; they are
augmented more aggressively to compensate for the single synthetic
source.

Polarity convention: runtime crops are polarity-corrected so text
is BRIGHT on a DARK background. Training glyphs are stored as-is
(bright digit strokes surrounded by white padding added by the
collector). We strip that padding, normalize every glyph to bright-
on-dark, and paste onto a dark canvas so the CRNN is trained on
the exact polarity it will see at inference time.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

log = logging.getLogger(__name__)

# Expanded alphabet: digits + punctuation + uppercase + space + parens.
# Covers HUD values (499, 0.61, 62%) AND mineral names ("IRON (ORE)",
# "RAW ICE", "TARANITE"), labels, and tooltip text. CTC blank sits
# at the end (index == len(alphabet)).
CHAR_CLASSES = "0123456789.-% ()ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BLANK_IDX = len(CHAR_CLASSES)

# Target canvas height (must match _crnn_recognize's runtime resize).
CANVAS_H = 32

_MODULE_DIR = Path(__file__).resolve().parent
_TRAINING_DIR = _MODULE_DIR.parent / "training_data"
_TEMPLATES_NPZ = _MODULE_DIR / "sc_templates" / "digits.npz"


# ── Glyph extraction ───────────────────────────────────────────────


def _tight_crop_bright(gray: np.ndarray, thr: int = 80) -> Optional[np.ndarray]:
    """Tight-crop to the digit strokes in a single-glyph training crop.

    Training samples are 28×28 with a ~3 px bright padding ring at
    the edges (added by ``training_collector._crop_to_28x28`` to
    center the digit in the resize). We strip that margin with a
    fixed inset, then bbox the bright content inside.
    """
    H, W = gray.shape
    if H <= 8 or W <= 8:
        return None
    # Strip a 3-px margin from all sides to discard the padding frame.
    m = 3
    inner = gray[m:H-m, m:W-m]
    mask = inner > thr
    ratio = float(mask.mean())
    if ratio < 0.05 or ratio > 0.65:
        return None
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    if len(ys) < 5 or len(xs) < 2:
        return None
    # Pad the bbox by 1 px for context, clamped to the inner region.
    y1 = max(0, ys[0] - 1)
    y2 = min(inner.shape[0], ys[-1] + 2)
    x1 = max(0, xs[0] - 1)
    x2 = min(inner.shape[1], xs[-1] + 2)
    crop = inner[y1:y2, x1:x2]
    h, w = crop.shape
    if h < 6 or w < 3 or w / max(1, h) > 2.0:
        return None
    return crop


def _normalize_polarity(gray: np.ndarray) -> np.ndarray:
    """Ensure bright-text-on-dark-bg (digit strokes >150, bg <100).

    Training glyphs from ``training_data/`` have bright padding
    (white) at the edges and bright digit strokes in the middle —
    polarity is already correct. This is a safety net for any
    alternate source.
    """
    if float(np.median(gray)) > 140:
        return 255 - gray
    return gray


def _load_sc_templates() -> dict[str, np.ndarray]:
    """Load the 10 clean digit templates from sc_templates/digits.npz.

    These are hand-curated, zero-mean unit-L2 normalized templates
    built from YouTube raw footage (see ``meta`` in the npz). We
    un-normalize them to uint8 display range, strip the 2-4 px bright
    padding frame, and return one glyph per character.
    """
    out: dict[str, np.ndarray] = {}
    if not _TEMPLATES_NPZ.is_file():
        log.warning("synth_data: %s missing", _TEMPLATES_NPZ)
        return out
    try:
        d = np.load(_TEMPLATES_NPZ)
        chars = d["chars"]
        images = d["images"]
    except Exception as exc:
        log.warning("synth_data: failed to load %s: %s", _TEMPLATES_NPZ, exc)
        return out
    for i in range(len(chars)):
        ch = chr(int(chars[i]))
        if ch not in "0123456789":
            continue
        arr = images[i]
        # Un-normalize to uint8 [0, 255]
        v_min, v_max = float(arr.min()), float(arr.max())
        if v_max - v_min < 1e-6:
            continue
        norm = ((arr - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
        # Strip the consistent padding frame (2 px top/bottom, up to 4 px
        # sides — observed empirically across all 10 templates).
        H, W = norm.shape
        trimmed = norm[2:H-2, 2:W-2]
        # Re-find the digit's inner bbox within the trimmed region
        mask = trimmed > 90
        if not mask.any():
            continue
        ys = np.where(mask.any(axis=1))[0]
        xs = np.where(mask.any(axis=0))[0]
        y1 = max(0, ys[0] - 1)
        y2 = min(trimmed.shape[0], ys[-1] + 2)
        x1 = max(0, xs[0] - 1)
        x2 = min(trimmed.shape[1], xs[-1] + 2)
        out[ch] = trimmed[y1:y2, x1:x2]
    log.info("synth_data: loaded %d sc_templates", len(out))
    return out


def _load_digit_bank() -> dict[str, list[np.ndarray]]:
    """Build the digit glyph bank from ``sc_templates/digits.npz``.

    These are 10 hand-curated templates built from real SC HUD
    YouTube footage; they're our cleanest source of per-digit shape
    truth. Augmentation at sequence-render time generates thousands
    of variations per template.

    ``training_data/{0-9}/*.png`` is not used here — the
    training_collector's segmentation often cuts digits mid-stroke
    or includes HUD chrome, and a clean automated filter is out of
    scope.
    """
    bank: dict[str, list[np.ndarray]] = {ch: [] for ch in "0123456789"}
    templates = _load_sc_templates()
    for ch, glyph in templates.items():
        bank[ch].append(glyph)
    counts = {ch: len(bank[ch]) for ch in bank}
    log.info("synth_data: digit bank (template counts) %s", counts)
    return bank


# ── Punctuation bootstrap (. and %) ────────────────────────────────


def _find_system_font(target_h: int) -> Optional[ImageFont.FreeTypeFont]:
    """Locate a sans-serif system font large enough to render cleanly.

    Tries common Windows fonts first. Returns None only if nothing
    truetype works — the caller falls back to the bitmap default.
    """
    candidates = ["arial.ttf", "consola.ttf", "tahoma.ttf", "segoeui.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=int(target_h * 1.2))
        except Exception:
            continue
    return None


def _render_char(ch: str, target_h: int) -> np.ndarray:
    """Render a single punctuation char to a bright-on-dark array."""
    font = _find_system_font(target_h)
    # Reference canvas — rendered white on black, then tight-cropped.
    W, H = target_h * 2, target_h * 2
    img = Image.new("L", (W, H), color=0)
    draw = ImageDraw.Draw(img)
    if font is not None:
        try:
            bbox = draw.textbbox((0, 0), ch, font=font)
            tx = (W - (bbox[2] - bbox[0])) // 2 - bbox[0]
            ty = (H - (bbox[3] - bbox[1])) // 2 - bbox[1]
            draw.text((tx, ty), ch, fill=255, font=font)
        except Exception:
            draw.text((W // 4, H // 4), ch, fill=255)
    else:
        draw.text((W // 4, H // 4), ch, fill=255)
    arr = np.array(img, dtype=np.uint8)
    # Tight-crop the rendered glyph
    mask = arr > 80
    if not mask.any():
        # Degenerate — return a small filled rect so the pipeline doesn't crash
        return np.full((max(3, target_h // 8), max(3, target_h // 8)), 255, dtype=np.uint8)
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    return arr[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]


def _build_punct_bank(target_h: int = 20, variants: int = 8) -> dict[str, list[np.ndarray]]:
    """Produce a small bank of crops for ALL non-digit alphabet chars.

    Expanded from just ``. %`` to also include uppercase/lowercase
    letters, space, and parens — the full mineral-name / label
    alphabet. Each char gets ``variants`` augmented copies so the
    CRNN doesn't memorize a single pixel pattern.
    """
    all_chars = ".-% ()ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    out: dict[str, list[np.ndarray]] = {c: [] for c in all_chars}
    rng = random.Random(17)
    for ch in all_chars:
        base = _render_char(ch, target_h)
        for _ in range(variants):
            arr = base.copy()
            # Slight size jitter
            h_j = max(3, int(arr.shape[0] * rng.uniform(0.85, 1.15)))
            w_j = max(3, int(arr.shape[1] * rng.uniform(0.85, 1.15)))
            arr = np.array(
                Image.fromarray(arr).resize((w_j, h_j), Image.BILINEAR),
                dtype=np.uint8,
            )
            # Slight blur
            if rng.random() < 0.6:
                arr = np.array(
                    Image.fromarray(arr).filter(
                        ImageFilter.GaussianBlur(rng.uniform(0.1, 0.6))
                    ),
                    dtype=np.uint8,
                )
            out[ch].append(arr)
    # `-` is in the alphabet for parity with the 28×28 classifier but
    # virtually never appears on the HUD. Provide a single thin bar
    # so CTC training doesn't see an unreachable class.
    bar_h = max(2, target_h // 8)
    bar_w = max(4, target_h // 3)
    out["-"] = [np.full((bar_h, bar_w), 255, dtype=np.uint8)]
    return out


# ── Label templates ────────────────────────────────────────────────


# Known SC mineral names + common HUD labels. These anchor the
# letter-training distribution to the actual text the OCR will see.
SC_VOCAB = [
    # Minerals (ore & gem)
    "IRON", "COPPER", "TITANIUM", "QUANTANIUM", "LARANITE", "BERYL",
    "TARANITE", "BORASE", "HEPHAESTANITE", "AGRICIUM", "GOLD", "TIN",
    "ALUMINUM", "TUNGSTEN", "CORUNDUM", "DIAMOND", "BEXALITE", "QUARTZ",
    # Forms
    "ORE", "RAW", "ICE", "CRYSTAL", "ROCK", "GEM",
    # Named combos
    "IRON (ORE)", "COPPER (ORE)", "QUANTANIUM (RAW)", "LARANITE (ORE)",
    "RAW ICE", "BERYL (GEM)", "TARANITE (ORE)", "TIN (ORE)", "GOLD (ORE)",
    "AGRICIUM (ORE)", "TITANIUM (ORE)", "ALUMINUM (ORE)",
    # HUD labels
    "MASS", "MASS:", "RESISTANCE", "RESISTANCE:", "INSTABILITY", "INSTABILITY:",
    "COMPOSITION", "SCAN RESULTS", "EASY", "MEDIUM", "HARD", "EXTREME",
    "IMPOSSIBLE", "STABLE", "UNSTABLE", "VOLATILE", "CRITICAL",
]


def _sample_label(rng: random.Random) -> str:
    """Weighted random label template.

    Covers three regimes:
    * Numeric (digits, decimals, percentages) — 55%
    * SC mineral names and HUD labels (letters) — 30%
    * Random letter+digit combos (labels like "MH1") — 15%
    """
    r = rng.random()
    if r < 0.25:
        # Plain integer 1–7 digits
        n = rng.choices([1, 2, 3, 4, 5, 6, 7], weights=[5, 12, 30, 25, 15, 8, 5])[0]
        return "".join(rng.choice("0123456789") for _ in range(n))
    if r < 0.40:
        # Decimal
        whole = rng.choices([1, 2, 3], weights=[50, 35, 15])[0]
        frac = rng.choices([1, 2], weights=[30, 70])[0]
        w = "".join(rng.choice("0123456789") for _ in range(whole))
        f = "".join(rng.choice("0123456789") for _ in range(frac))
        return f"{w}.{f}"
    if r < 0.55:
        # Percentage
        n = rng.choices([1, 2, 3], weights=[30, 60, 10])[0]
        if n == 3:
            return "100%"
        return "".join(rng.choice("0123456789") for _ in range(n)) + "%"
    if r < 0.85:
        # SC vocabulary token (mineral name, HUD label, etc.)
        return rng.choice(SC_VOCAB)
    # Random letter+digit mix (ship names like "MH1", "MK3", "MOLE", "ROC")
    pattern = rng.choice(["LL", "LLL", "LLLL", "LLD", "LLDD", "LD"])
    out = []
    for p in pattern:
        if p == "L":
            out.append(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        else:
            out.append(rng.choice("0123456789"))
    return "".join(out)


# ── Sequence rendering ─────────────────────────────────────────────


def _resize_glyph_to_h(g: np.ndarray, h_target: int) -> np.ndarray:
    h, w = g.shape
    if h == h_target:
        return g
    w_new = max(2, int(round(w * h_target / max(1, h))))
    return np.array(
        Image.fromarray(g).resize((w_new, h_target), Image.BILINEAR),
        dtype=np.uint8,
    )


def _render_sequence(
    label: str,
    digit_bank: dict[str, list[np.ndarray]],
    punct_bank: dict[str, list[np.ndarray]],
    rng: random.Random,
) -> np.ndarray:
    """Render one labeled value crop.

    Returns a uint8 grayscale array at height CANVAS_H, width variable,
    with bright text (255) on dark background (~20).
    """
    # Pick a glyph height that will fit on the canvas (with vertical pad).
    glyph_h = rng.randint(18, 24)
    # Horizontal pad on each side of the full sequence.
    pad_x = rng.randint(2, 5)
    # Vertical position jitter for the glyph row.
    pad_y_top = rng.randint(3, max(4, CANVAS_H - glyph_h - 3))

    # Sample a glyph image per character, resized to glyph_h.
    rendered: list[np.ndarray] = []
    for ch in label:
        if ch in digit_bank and digit_bank[ch]:
            src = rng.choice(digit_bank[ch])
        elif ch in punct_bank and punct_bank[ch]:
            src = rng.choice(punct_bank[ch])
        else:
            # Alphabet mismatch — should not happen given _sample_label
            continue
        g = _resize_glyph_to_h(src, glyph_h)
        # Punctuation (. and -) sits lower / on baseline — bias their y offset
        rendered.append(g)

    if not rendered:
        # Degenerate label
        return np.full((CANVAS_H, 8), 20, dtype=np.uint8)

    # Random inter-glyph gaps
    gaps = [rng.randint(0, 3) for _ in range(len(rendered) - 1)]

    total_w = pad_x * 2 + sum(g.shape[1] for g in rendered) + sum(gaps)
    # Dark background with small noise
    bg_base = rng.randint(15, 35)
    canvas = np.full((CANVAS_H, total_w), bg_base, dtype=np.float32)
    canvas += rng.uniform(0, 4.0) * np.random.randn(CANVAS_H, total_w)
    canvas = np.clip(canvas, 0, 255)

    # Paste each glyph, brightness-jittered, at a per-glyph vertical offset
    x = pad_x
    for i, g in enumerate(rendered):
        gh, gw = g.shape
        # Per-char vertical jitter on top of the row baseline
        y_jitter = rng.randint(-1, 1)
        y = max(0, min(CANVAS_H - gh, pad_y_top + y_jitter))

        # Brightness scale: treat g as a bright-on-dark mask, scale to 180-255
        glyph_f = g.astype(np.float32)
        peak = float(glyph_f.max()) if glyph_f.max() > 0 else 1.0
        scale = rng.uniform(0.75, 1.0) * 255.0 / peak
        bright = np.clip(glyph_f * scale, 0, 255)

        # Alpha-composite: wherever glyph is brighter than canvas, take glyph
        region = canvas[y:y + gh, x:x + gw]
        canvas[y:y + gh, x:x + gw] = np.maximum(region, bright)

        x += gw + (gaps[i] if i < len(gaps) else 0)

    # Global augmentations — kept MILD so the synthetic distribution
    # doesn't drift too far from the sharp real-HUD inference domain.
    # Half the samples stay completely unaugmented.
    img_u8 = canvas.astype(np.uint8)
    if rng.random() < 0.30:
        img_u8 = np.array(
            Image.fromarray(img_u8).filter(
                ImageFilter.GaussianBlur(rng.uniform(0.1, 0.4))
            ),
            dtype=np.uint8,
        )
    if rng.random() < 0.20:
        alpha = rng.uniform(0.92, 1.08)
        beta = rng.uniform(-4, 4)
        img_u8 = np.clip(img_u8.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img_u8


# ── Public API ─────────────────────────────────────────────────────


def generate_dataset(
    n: int = 20000,
    seed: int = 42,
    digit_bank: Optional[dict[str, list[np.ndarray]]] = None,
    punct_bank: Optional[dict[str, list[np.ndarray]]] = None,
) -> list[tuple[np.ndarray, str]]:
    """Generate `n` (image, label) pairs.

    Images are uint8 grayscale, height CANVAS_H, variable width.
    Labels are strings over ``CHAR_CLASSES``.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    if digit_bank is None:
        digit_bank = _load_digit_bank()
    if punct_bank is None:
        punct_bank = _build_punct_bank(target_h=20)

    # If any digit class is empty (user hasn't collected that digit yet),
    # oversample neighbors — refuse to crash, just warn.
    empty = [ch for ch in "0123456789" if not digit_bank.get(ch)]
    if empty:
        log.warning("synth_data: no glyphs for digits %s — CRNN recall on these will suffer", empty)

    samples: list[tuple[np.ndarray, str]] = []
    while len(samples) < n:
        label = _sample_label(rng)
        # Skip labels that reference digits we don't have any source for
        if any(ch in "0123456789" and not digit_bank.get(ch) for ch in label):
            continue
        img = _render_sequence(label, digit_bank, punct_bank, rng)
        samples.append((img, label))
    return samples


def label_to_indices(label: str) -> list[int]:
    """Convert a label string to CHAR_CLASSES indices (CTC targets)."""
    return [CHAR_CLASSES.index(ch) for ch in label if ch in CHAR_CLASSES]


if __name__ == "__main__":
    # Quick sanity check: generate 20 samples and save to disk as PNGs
    # alongside this file for eyeballing. Not part of the training flow.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="synth_samples", help="Output directory")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (img, label) in enumerate(generate_dataset(args.n)):
        safe = label.replace(".", "dot").replace("%", "pct")
        Image.fromarray(img).save(out_dir / f"{i:04d}_{safe}.png")
    print(f"Wrote {args.n} samples to {out_dir}/")
