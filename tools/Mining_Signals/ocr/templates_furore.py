"""Furore-font digit templates for NCC-based value reads.

A deterministic, CPU-only voter that runs when the neural engines
(CRNN + Tesseract) can't agree. The SC mining HUD is rendered in
exactly one font (Furore) at a knowable set of sizes — so matching
against pre-rendered templates is arguably the most reliable signal
available for the small-size extraction-mode cases that the neural
stack struggles with.

Usage:
    from ocr.templates_furore import match_value_crop
    text, confs = match_value_crop(value_crop_pil)

Templates are generated once from ``furore.otf`` at module first
use, cached as ``ocr/models/furore_templates.npz`` (~40 KB).
Subsequent loads are ~1 ms.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

_ALPHABET = "0123456789.%"
# Render sizes matching the pretrain / HUD render sizes (extraction
# mode runs 10-16 px; ship scan runs 18-48 px).
_SIZES = [10, 12, 14, 16, 18, 22, 26, 30, 34, 40, 48]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FURORE_CANDIDATES = [
    _REPO_ROOT / "furore.otf",
    _REPO_ROOT / "furore.ttf",
    _REPO_ROOT / "Furore.otf",
    _REPO_ROOT / "Furore.ttf",
]
_CACHE_PATH = _REPO_ROOT / "ocr" / "models" / "furore_templates.npz"

# dict[char][size] -> np.ndarray(H, W) uint8, dark-text-on-white.
_TEMPLATES: Optional[dict] = None


def _find_furore_path() -> Optional[Path]:
    for c in _FURORE_CANDIDATES:
        if c.is_file():
            return c
    return None


def _render_template(char: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """Render a single character, return tight dark-on-white uint8."""
    canvas_size = max(60, font.size * 3)
    canvas = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(canvas)
    draw.text((font.size, font.size), char, font=font, fill=0)
    arr = np.asarray(canvas, dtype=np.uint8)
    mask = arr < 200
    if not mask.any():
        return np.array([[255]], dtype=np.uint8)
    ys, xs = np.where(mask)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    # 1-px padding for shift tolerance
    y1 = max(0, y1 - 1)
    y2 = min(arr.shape[0], y2 + 1)
    x1 = max(0, x1 - 1)
    x2 = min(arr.shape[1], x2 + 1)
    return arr[y1:y2, x1:x2].copy()


def _generate_templates() -> dict[str, dict[int, np.ndarray]]:
    path = _find_furore_path()
    if path is None:
        log.warning("templates: Furore font not found, NCC voter disabled")
        return {}
    out: dict[str, dict[int, np.ndarray]] = {c: {} for c in _ALPHABET}
    for size in _SIZES:
        try:
            font = ImageFont.truetype(str(path), size=size)
        except Exception as exc:
            log.debug("templates: size=%d font load failed: %s", size, exc)
            continue
        for char in _ALPHABET:
            try:
                out[char][size] = _render_template(char, font)
            except Exception:
                pass
    return out


def _save_cache(templates: dict) -> None:
    flat: dict[str, np.ndarray] = {}
    for char, sizes in templates.items():
        for sz, arr in sizes.items():
            flat[f"{char}__{sz}"] = arr
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(_CACHE_PATH), **flat)


def _load_cache() -> Optional[dict]:
    if not _CACHE_PATH.is_file():
        return None
    try:
        with np.load(str(_CACHE_PATH)) as npz:
            out: dict[str, dict[int, np.ndarray]] = {c: {} for c in _ALPHABET}
            for key in npz.files:
                parts = key.split("__")
                if len(parts) != 2:
                    continue
                char, sz_s = parts
                if char in out:
                    out[char][int(sz_s)] = npz[key]
        return out
    except Exception as exc:
        log.debug("templates: cache load failed: %s", exc)
        return None


def _ensure_templates() -> dict:
    global _TEMPLATES
    if _TEMPLATES is not None:
        return _TEMPLATES
    cached = _load_cache()
    if cached is not None:
        _TEMPLATES = cached
        return _TEMPLATES
    log.info("templates: generating Furore templates (first-time)")
    _TEMPLATES = _generate_templates()
    try:
        if _TEMPLATES:
            _save_cache(_TEMPLATES)
    except Exception as exc:
        log.debug("templates: cache save failed: %s", exc)
    return _TEMPLATES


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation for same-shape arrays."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-6:
        return 0.0
    return float((a * b).sum() / denom)


def _resize_arr(arr: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    if new_h <= 0 or new_w <= 0:
        return arr
    return np.asarray(
        Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR),
        dtype=np.uint8,
    )


def _best_char_for_glyph(
    glyph: np.ndarray,
    templates: dict,
    focal_sizes: list[int],
) -> tuple[str, float]:
    """Match glyph against every char's templates at focal sizes.

    Strategy: resize each template to the glyph's exact height,
    then match widths (crop wider of the two or pad narrower with
    white). Report best-scoring (char, score).
    """
    gh, gw = glyph.shape
    if gh < 3 or gw < 2:
        return "", 0.0
    best_char = ""
    best_score = -1.0
    for char, size_map in templates.items():
        for sz, tmpl in size_map.items():
            if sz not in focal_sizes:
                continue
            th, tw = tmpl.shape
            if th < 2:
                continue
            # Match height; scale width proportionally
            tw_scaled = max(1, int(round(tw * gh / th)))
            tmpl_r = _resize_arr(tmpl, gh, tw_scaled)
            # Compare — handle width mismatch
            if tmpl_r.shape[1] == gw:
                score = _ncc(glyph, tmpl_r)
            elif tmpl_r.shape[1] < gw:
                # Slide template across glyph (±2 px search), take max
                score = -1.0
                for x in range(gw - tmpl_r.shape[1] + 1):
                    sub = glyph[:, x:x + tmpl_r.shape[1]]
                    s = _ncc(sub, tmpl_r)
                    if s > score:
                        score = s
            else:
                # Glyph narrower than resized template — pad glyph
                pad_w = tmpl_r.shape[1] - gw
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                glyph_padded = np.pad(
                    glyph,
                    ((0, 0), (pad_left, pad_right)),
                    mode="constant", constant_values=255,
                )
                score = _ncc(glyph_padded, tmpl_r)
            if score > best_score:
                best_score = score
                best_char = char
    return best_char, max(0.0, best_score)


def match_value_crop(value_crop: Image.Image) -> tuple[str, list[float]]:
    """Segment + NCC-match the value crop against Furore templates.

    Returns (decoded_text, per_glyph_scores). Scores are in [0, 1] —
    higher = better match. Caller should treat mean score < ~0.55
    as low confidence.

    Cheap on CPU: ~50 ms typical for a 4–5-digit value crop.
    """
    templates = _ensure_templates()
    if not any(templates.values()):
        return "", []

    try:
        gray = np.asarray(value_crop.convert("L"), dtype=np.uint8)
    except Exception:
        return "", []
    if gray.size == 0:
        return "", []

    # Polarity correction to match templates (dark text on white).
    if float(np.median(gray)) < 130:
        gray = 255 - gray

    # Otsu binarize for segmentation
    try:
        from .onnx_hud_reader import _otsu
        thr = _otsu(gray)
    except Exception:
        thr = 128
    binary = (gray < thr).astype(np.uint8)

    # Column projection — find glyph spans
    col_sum = binary.sum(axis=0)
    spans: list[tuple[int, int]] = []
    in_char = False
    start = 0
    for x in range(binary.shape[1]):
        if col_sum[x] > 0 and not in_char:
            in_char = True
            start = x
        elif col_sum[x] == 0 and in_char:
            in_char = False
            if x - start >= 2:
                spans.append((start, x))
    if in_char:
        spans.append((start, binary.shape[1]))

    if not spans:
        return "", []

    # Estimate median glyph height, pick 3 closest template sizes.
    heights: list[int] = []
    for x1, x2 in spans:
        rows = np.where(binary[:, x1:x2].any(axis=1))[0]
        if len(rows) > 0:
            heights.append(int(rows[-1] - rows[0]) + 1)
    if not heights:
        return "", []
    med_h = int(np.median(heights))
    focal_sizes = sorted(_SIZES, key=lambda s: abs(s - med_h))[:3]

    # Match each glyph
    chars: list[str] = []
    confs: list[float] = []
    for x1, x2 in spans:
        rows = np.where(binary[:, x1:x2].any(axis=1))[0]
        if len(rows) == 0:
            continue
        y1, y2 = int(rows[0]), int(rows[-1]) + 1
        glyph = gray[y1:y2, x1:x2]
        if glyph.size == 0:
            continue
        char, score = _best_char_for_glyph(glyph, templates, focal_sizes)
        if score < 0.35:  # reject junk
            continue
        chars.append(char)
        confs.append(score)

    if chars:
        log.debug(
            "templates: decoded=%r med_h=%d focal=%s mean_score=%.2f",
            "".join(chars), med_h, focal_sizes,
            sum(confs) / len(confs) if confs else 0.0,
        )
    return "".join(chars), confs
