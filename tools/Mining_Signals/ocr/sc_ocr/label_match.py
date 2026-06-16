"""NCC-based label finder for the SC mining HUD.

Replaces Tesseract for label-row detection. The SC HUD always renders
the labels "MASS:", "RESISTANCE:", "INSTABILITY:" in the same fixed
font. Once we have a canonicalized template of each, we can find their
exact pixel positions in any captured panel via normalized
cross-correlation (NCC) — fast (~5 ms total per scan), polarity-
independent (works on dark / light / colored backgrounds uniformly),
and gives us CONCRETE GROUND TRUTH per row instead of geometric
inference that compounds errors.

Templates live at ``ocr/sc_templates/labels.npz`` and are generated
offline by ``scripts/build_label_templates.py`` from a small handful
of clean panel captures.

Public API:

    from ocr.sc_ocr.label_match import find_label_positions

    matches = find_label_positions(img)
    # matches = {
    #     "mass":        {"x": 480, "y": 348, "w": 120, "h": 28, "score": 0.83},
    #     "resistance":  {"x": 480, "y": 392, "w": 230, "h": 28, "score": 0.81},
    #     "instability": {"x": 480, "y": 436, "w": 240, "h": 28, "score": 0.79},
    # }
    # Missing labels are absent from the dict (not None entries).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Templates path (next to the digit templates).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOL_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
_TEMPLATES_PATH = os.path.join(
    _TOOL_DIR, "ocr", "sc_templates", "labels.npz",
)
# Synthetic sharp RGB MASS template -- colorized from the grayscale
# MASS template with the measured mint-cyan foreground.  Used by the
# windowed RGB verifier in step 1b of find_label_positions to break
# ties between grayscale candidates that score similarly but land on
# DIFFERENTLY-COLORED text (e.g. mint MASS vs white UNKNOWN footer).
_RGB_TEMPLATES_PATH = os.path.join(
    _TOOL_DIR, "ocr", "sc_templates", "labels_rgb_sharp.npz",
)
_RGB_TEMPLATES_CACHE: Optional[dict] = None

# Per-channel weights for RGB NCC combine.  G is highest because the
# mint-cyan HUD text peaks in G (measured fg = (165, 200, 180) -- G
# 35 pts above R, 20 pts above B).  R is down-weighted because most
# white false-positive text scores high in R; B is balanced.
_RGB_WEIGHTS: tuple[float, float, float] = (0.20, 0.50, 0.30)

# Lazy-loaded template cache: {label_name: float32 array (H, W), zero-mean unit-variance}
_templates_cache: Optional[dict] = None
_templates_height: int = 28  # canonical height the templates are rendered at

# Acceptable NCC score for a confident match. Below this, the label
# is considered "not found" and the caller falls back.
_MIN_MATCH_SCORE = 0.45

# When MASS matches at or above this score we treat its position as
# ground truth and are willing to SYNTHESIZE the resistance /
# instability label positions from the HUD's rigid 3-row geometry
# (``mass + pitch`` / ``mass + 2·pitch``) even if their own NCC score
# falls below ``_MIN_MATCH_SCORE``.
#
# Why this exists: ``RESISTANCE:`` and ``INSTABILITY:`` are long words,
# so their templates span 160-230 px. On panels captured at a slight
# perspective skew the cumulative misalignment across that width tanks
# the NCC score (observed: 0.39 / 0.25 on a skewed panel where MASS —
# a short word — still scored 0.80). The three rows are ALWAYS present
# and ALWAYS evenly spaced in the SC mining HUD, so a confident MASS
# match plus the fixed pitch fully determines the other two rows.
# Without synthesis the finder drops to a single-anchor recovery that
# produces unusable row crops (value_crop is None / oversized-band).
#
# 0.65 is comfortably above ``_MIN_MATCH_SCORE`` (0.45) so synthesis
# only triggers off a genuinely confident MASS lock, never a marginal
# one.
_SYNTH_MASS_FLOOR = 0.65

# Multi-scale search: panels can be captured at different resolutions,
# so the rendered label height varies. Search at multiple scales and
# pick the best match. Scales relative to the template's native height.
_SCALE_FACTORS = (0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0)


def _load_templates() -> Optional[dict]:
    """Load the canonicalized label templates from disk, cached.

    Returns dict {"mass": np.ndarray, "resistance": np.ndarray,
    "instability": np.ndarray} or None if the file is missing.
    """
    global _templates_cache, _templates_height
    if _templates_cache is not None:
        return _templates_cache
    if not os.path.isfile(_TEMPLATES_PATH):
        log.warning(
            "label_match: templates not found at %s — run "
            "scripts/build_label_templates.py to bootstrap them",
            _TEMPLATES_PATH,
        )
        return None
    try:
        data = np.load(_TEMPLATES_PATH)
        templates = {}
        for key in ("mass", "resistance", "instability"):
            if key not in data:
                log.warning("label_match: template missing for %r", key)
                continue
            arr = data[key].astype(np.float32)
            # Pre-normalize to zero-mean unit-variance (NCC-ready).
            mean = float(arr.mean())
            std = float(arr.std())
            if std < 1e-3:
                log.warning(
                    "label_match: template for %r is degenerate", key,
                )
                continue
            templates[key] = (arr - mean) / std

            # ── Truncated-template fallbacks (RESI / INSTA) ──
            # The full RESISTANCE: / INSTABILITY: words span 160-230 px
            # at native scale.  On panels with slight perspective skew,
            # font-weight differences, or partial occlusion of the right
            # side of the word, the cumulative misalignment across that
            # width tanks the NCC score below threshold (observed: 0.25-
            # 0.39 in real captures where the same panel's MASS still
            # scored 0.80).
            #
            # Truncated templates -- the LEFT portion only ("RESI" of
            # "RESISTANCE", "INSTA" of "INSTABILITY") -- are shorter so
            # are less sensitive to right-side degradation while still
            # being long enough (~70 px) to be visually distinctive
            # versus other text in the panel.  We try BOTH the full and
            # truncated template per label and take whichever scores
            # higher.
            #
            # No new template assets: we slice these from the existing
            # labels.npz at load time.
            #
            # Slice widths chosen for DISAMBIGUATION, not just shortness:
            #
            #   RESISTANCE: -> "RESIST" (~60% = 103 px)
            #     Empirically the naive 4-letter "RESI" slice (42%) fails
            #     because "SCAN RESULTS" is always above the row labels
            #     and shares the "RES" prefix.  At 4 letters the only
            #     differentiator is letter 4 (I vs U) -- NCC's slack in
            #     a 71-px window scored RESULTS HIGHER than RESISTANCE
            #     in 8/9 measured panels.  "RESIST" (6 letters) vs
            #     "RESULT" differ at letters 4 (I/U) AND 5 (S/L) -- two
            #     differences in 6 letters is enough to peak on the
            #     correct row in every measured panel.
            #
            #   INSTABILITY: -> "INSTA" (~46% = 73 px)
            #     INSTA is unambiguous -- no other word in the panel
            #     starts with "INSTA".  Diagnostic confirmed the 5-letter
            #     slice peaks at the SAME position as the full template
            #     in every measured panel.
            raw_arr = data[key].astype(np.float32)
            if key == "resistance":
                # "RESIST" = 6 of 10 letters in "RESISTANCE"
                slice_w = max(16, int(raw_arr.shape[1] * 0.60))
                short = raw_arr[:, :slice_w]
            elif key == "instability":
                # "INSTA" = 5 of 11 letters in "INSTABILITY"
                slice_w = max(16, int(raw_arr.shape[1] * 0.46))
                short = raw_arr[:, :slice_w]
            else:
                short = None
            if short is not None:
                smean = float(short.mean())
                sstd = float(short.std())
                if sstd >= 1e-3:
                    templates[f"{key}_short"] = (short - smean) / sstd
        if "height" in data:
            _templates_height = int(data["height"])
        _templates_cache = templates
        log.info(
            "label_match: loaded %d templates from %s (canonical h=%d) "
            "[keys=%s]",
            len(templates), _TEMPLATES_PATH, _templates_height,
            sorted(templates.keys()),
        )
        return _templates_cache
    except Exception as exc:
        log.warning("label_match: template load failed: %s", exc)
        return None


def _canonicalize(gray: np.ndarray) -> np.ndarray:
    """Polarity-canonicalize a grayscale image so dark-text-on-light
    and bright-text-on-dark both end up looking the same to NCC.

    Returns a float32 array, zero-mean unit-variance, where text
    pixels are HIGH and background is LOW (independent of source
    polarity).
    """
    if gray.size == 0:
        return gray.astype(np.float32)
    # Decide polarity: in a small window of pixels, "text" is usually
    # the minority class. Run a quick Otsu-equivalent split.
    # Histogram-based 2-class threshold.
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

    bright = int((gray > threshold).sum())
    dark = total - bright
    # Want text to be BRIGHT (the minority class). If dark is minority,
    # invert.
    if dark < bright:
        out = (255 - gray).astype(np.float32)
    else:
        out = gray.astype(np.float32)

    mean = float(out.mean())
    std = float(out.std())
    if std < 1e-3:
        return out - mean
    return (out - mean) / std


def _ncc_search(
    target: np.ndarray, template: np.ndarray,
) -> tuple[float, int, int]:
    """Sliding-window NCC. Returns (best_score, best_x, best_y).

    Both ``target`` and ``template`` must be float32, zero-mean,
    unit-variance. Returns (-1.0, 0, 0) if template doesn't fit.

    Implementation note: uses ``scipy.signal.fftconvolve`` rather than
    ``correlate2d`` because for our typical sizes (target ~280×578,
    template ~28×75 across 8 scales) FFT-based correlation is ~70×
    faster — direct correlation is O(H·W·h·w) while FFT is
    O(H·W·log(H·W)). For 8-scale sweeps this is the difference
    between 3 seconds per scan and 50 ms per scan.
    """
    H, W = target.shape
    h, w = template.shape
    if h > H or w > W:
        return -1.0, 0, 0
    n = float(h * w)
    # FFT correlation = FFT convolve with the template flipped in both
    # axes. Both arrays are already zero-mean unit-variance so we just
    # divide by n to normalise the inner product.
    try:
        from scipy.signal import fftconvolve  # type: ignore
        c = fftconvolve(target, template[::-1, ::-1], mode="valid")
        c /= n
        idx = int(np.argmax(c))
        by, bx = divmod(idx, c.shape[1])
        return float(c[by, bx]), int(bx), int(by)
    except Exception:
        pass
    # Pure NumPy stride-trick fallback (only used if scipy unavailable).
    best_score = -2.0
    best_xy = (0, 0)
    for y in range(0, H - h + 1):
        row = target[y:y + h, :]
        for x in range(0, W - w + 1):
            window = row[:, x:x + w]
            score = float(np.sum(window * template) / n)
            if score > best_score:
                best_score = score
                best_xy = (x, y)
    return best_score, best_xy[0], best_xy[1]


def _resize_template(template: np.ndarray, scale: float) -> np.ndarray:
    """Resize a (H, W) float32 template to (round(H*scale), round(W*scale))."""
    h, w = template.shape
    new_h = max(8, int(round(h * scale)))
    new_w = max(8, int(round(w * scale)))
    # Use PIL for resize (handles float32 via 'F' mode).
    pil = Image.fromarray(template, mode="F").resize(
        (new_w, new_h), Image.BILINEAR,
    )
    arr = np.asarray(pil, dtype=np.float32)
    # Re-normalize after resize (resampling can shift the mean/std slightly).
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-3:
        return arr - mean
    return (arr - mean) / std


# Per-image cache key shape:
#   (id(img), size, mode, search_centers_key, search_radius)
# where search_centers_key is None for full-frame calls or a
# sorted tuple of (name, x, y) triples otherwise. Including the
# local-search arguments in the key prevents a local-search call
# from returning a stale full-frame result (and vice-versa) when
# both happen against the same Image instance in a single scan.
_LAST_CALL_CACHE: Optional[tuple[
    int,                                                      # id(img)
    tuple[int, int],                                          # img.size
    str,                                                      # img.mode
    Optional[tuple[tuple[str, int, int], ...]],               # search_centers
    int,                                                      # search_radius
    dict,                                                     # result
]] = None

# Default radius for label-local search. Per-label NCC against the
# canonicalized template is cheap inside a small window, so we err
# on the slightly-larger side: a few extra px of slack catches
# rendering jitter / sub-pixel resampling without exploding the
# cost.
_DEFAULT_LOCAL_RADIUS = 40


# Holds the matches dict produced by the most recent FULL-FRAME call to
# ``find_label_positions`` (i.e. ``search_centers is None``). Used by
# the auto-calibration hook in ``onnx_hud_reader._label_rows_from_anchor``
# to read the per-label observations without re-running the expensive
# multi-scale NCC sweep.
#
# Local-search calls do NOT update this — they target a narrow window
# around predicted positions, which would skew the calibration
# observations toward where the tracker thought the labels should be
# (defeating the purpose of using observed data to correct stale
# defaults).
#
# The dict stores image-relative coordinates from the call: when the
# caller cropped the image first (e.g. ``_label_rows_from_anchor``
# operating on a "below title" crop), the Y values are crop-relative
# and the caller is responsible for adding the crop offset before
# feeding them to the calibration learner.
_LAST_FULL_FRAME_MATCHES: dict[str, dict] = {}


def get_last_full_frame_matches() -> dict[str, dict]:
    """Return a shallow copy of the most-recent full-frame
    ``find_label_positions`` result.

    Empty dict if no full-frame call has been made yet, or the last
    full-frame call found nothing. Local-search calls do not update
    this state.

    Callers should be aware the returned coordinates are relative to
    whatever image was passed to that ``find_label_positions`` call;
    if the caller cropped before calling, they must add the crop
    offset themselves before interpreting Y values in the original
    image's coordinate frame.
    """
    return dict(_LAST_FULL_FRAME_MATCHES)


def _load_rgb_label_template(name: str) -> Optional[np.ndarray]:
    """Load the synthetic sharp RGB template for ``name``, cached.

    ``name`` is one of ``"mass" | "resistance" | "instability"``.
    Returns float32 (H, W, 3) or None if the file is missing.  RGB
    templates are OPTIONAL -- callers degrade gracefully to
    grayscale-only picking when the file isn't available.
    """
    global _RGB_TEMPLATES_CACHE
    if _RGB_TEMPLATES_CACHE is not None:
        return _RGB_TEMPLATES_CACHE.get(name)
    if not os.path.isfile(_RGB_TEMPLATES_PATH):
        log.info(
            "label_match: RGB template file not found at %s -- "
            "label picking will use grayscale only",
            _RGB_TEMPLATES_PATH,
        )
        _RGB_TEMPLATES_CACHE = {}
        return None
    try:
        data = np.load(_RGB_TEMPLATES_PATH)
        cache: dict = {}
        for key in ("mass", "resistance", "instability"):
            if key in data.files:
                cache[key] = data[key].astype(np.float32, copy=False)
        _RGB_TEMPLATES_CACHE = cache
        log.info(
            "label_match: loaded RGB templates from %s [keys=%s]",
            _RGB_TEMPLATES_PATH, sorted(cache.keys()),
        )
        return cache.get(name)
    except Exception as exc:
        log.warning("label_match: RGB template load failed: %s", exc)
        _RGB_TEMPLATES_CACHE = {}
        return None


def _load_rgb_mass_template() -> Optional[np.ndarray]:
    """Back-compat shim: ``_load_rgb_label_template("mass")``."""
    return _load_rgb_label_template("mass")


def _rgb_row_search(
    target_rgb: np.ndarray,
    template_rgb_native: np.ndarray,
    scale: float,
    y_window_top: int,
    y_window_bot: int,
) -> tuple[float, int, int]:
    """RGB NCC search for a label inside a horizontal y-window.

    The full-width RGB target is sliced to rows ``[y_window_top..y_window_bot]``
    and the scaled template is NCC-searched across that band.  Returns
    (best_combined_score, best_x, best_y_in_full_image) where the y is
    converted back to full-image coordinates by adding y_window_top.

    Mirrors the grayscale windowed search structure used for
    RESISTANCE/INSTABILITY rows -- runs once per row.
    """
    try:
        try:
            from hud_tracker.anchors.icon_rgb_ncc import _ncc_one_channel
        except ImportError:
            from ...hud_tracker.anchors.icon_rgb_ncc import _ncc_one_channel  # type: ignore
    except Exception:
        return -1.0, 0, 0
    scaled = _resize_rgb_template(template_rgb_native, scale)
    h, w = scaled.shape[:2]
    H, W = target_rgb.shape[:2]
    y_window_top = max(0, y_window_top)
    y_window_bot = min(H, y_window_bot)
    if y_window_bot - y_window_top < h or w > W:
        return -1.0, 0, 0
    band = target_rgb[y_window_top:y_window_bot, :]
    sR = _ncc_one_channel(band[..., 0], scaled[..., 0])
    sG = _ncc_one_channel(band[..., 1], scaled[..., 1])
    sB = _ncc_one_channel(band[..., 2], scaled[..., 2])
    if sR.size == 0:
        return -1.0, 0, 0
    wR, wG, wB = _RGB_WEIGHTS
    combined = wR * sR + wG * sG + wB * sB
    idx = int(np.argmax(combined))
    py, px = divmod(idx, combined.shape[1])
    return float(combined[py, px]), int(px), int(py + y_window_top)


def _resize_rgb_template(template_rgb: np.ndarray, scale: float) -> np.ndarray:
    """Resize an (H, W, 3) float32 RGB template by scale via PIL bilinear."""
    h, w = template_rgb.shape[:2]
    new_h = max(4, int(round(h * scale)))
    new_w = max(4, int(round(w * scale)))
    if new_h == h and new_w == w:
        return template_rgb
    pil = Image.fromarray(np.clip(template_rgb, 0, 255).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32)


def _rgb_mass_top_k(
    target_rgb: np.ndarray,
    template_rgb_native: np.ndarray,
    k: int = 8,
) -> list[dict]:
    """Full-frame RGB NCC of MASS template across all scales.

    Returns up to ``k`` candidate dicts ``{x, y, score, scale, w, h}``
    sorted by descending score.  Used by the voting step to produce
    independent RGB candidates that vote against the grayscale ones.
    """
    try:
        try:
            from hud_tracker.anchors.icon_rgb_ncc import _ncc_one_channel
        except ImportError:
            from ...hud_tracker.anchors.icon_rgb_ncc import _ncc_one_channel  # type: ignore
    except Exception:
        return []
    out: list[dict] = []
    H, W = target_rgb.shape[:2]
    wR, wG, wB = _RGB_WEIGHTS
    for scale in _SCALE_FACTORS:
        t = _resize_rgb_template(template_rgb_native, scale)
        if t.shape[0] > H or t.shape[1] > W:
            continue
        sR = _ncc_one_channel(target_rgb[..., 0], t[..., 0])
        sG = _ncc_one_channel(target_rgb[..., 1], t[..., 1])
        sB = _ncc_one_channel(target_rgb[..., 2], t[..., 2])
        if sR.size == 0:
            continue
        combined = wR * sR + wG * sG + wB * sB
        idx = int(np.argmax(combined))
        py, px = divmod(idx, combined.shape[1])
        out.append({
            "x": int(px), "y": int(py),
            "score": float(combined[py, px]),
            "scale": float(scale),
            "w": int(t.shape[1]), "h": int(t.shape[0]),
        })
    out.sort(key=lambda c: -c["score"])
    return out[:k]


def _rgb_score_windowed(
    target_rgb: np.ndarray,
    template_rgb_native: np.ndarray,
    scale: float,
    gx: int, gy: int,
    window_px: int = 30,
) -> float:
    """Best RGB NCC score within +/- ``window_px`` of (gx, gy).

    The grayscale matcher gives a coarse candidate position; the actual
    RGB peak may sit a few pixels off due to (a) the synthetic
    template's idealized colours differing slightly from the real
    rendered pixels and (b) sub-pixel alignment differences between
    grayscale-Otsu and per-channel NCC.  Searching a small window
    around the candidate accommodates both.

    Combines per-channel NCC with ``_RGB_WEIGHTS``.  Returns the best
    combined score in the window, or -1.0 if the template doesn't fit.
    """
    try:
        # Local import keeps icon_rgb_ncc out of the import chain for
        # callers that don't need RGB verification.
        try:
            from hud_tracker.anchors.icon_rgb_ncc import _ncc_one_channel
        except ImportError:
            from ...hud_tracker.anchors.icon_rgb_ncc import _ncc_one_channel  # type: ignore
    except Exception as exc:
        log.debug("label_match: RGB NCC helper unavailable: %s", exc)
        return -1.0
    scaled = _resize_rgb_template(template_rgb_native, scale)
    h, w = scaled.shape[:2]
    H, W = target_rgb.shape[:2]
    x1 = max(0, gx - window_px)
    y1 = max(0, gy - window_px)
    x2 = min(W, gx + window_px + w)
    y2 = min(H, gy + window_px + h)
    if x2 - x1 < w or y2 - y1 < h:
        return -1.0
    crop = target_rgb[y1:y2, x1:x2]
    sR = _ncc_one_channel(crop[..., 0], scaled[..., 0])
    sG = _ncc_one_channel(crop[..., 1], scaled[..., 1])
    sB = _ncc_one_channel(crop[..., 2], scaled[..., 2])
    if sR.size == 0:
        return -1.0
    wR, wG, wB = _RGB_WEIGHTS
    combined = wR * sR + wG * sG + wB * sB
    return float(combined.max())


def _normalize_search_centers(
    search_centers: Optional[dict],
) -> Optional[tuple[tuple[str, int, int], ...]]:
    """Convert a search_centers dict into a hashable, order-stable
    cache key. Returns None for None input.
    """
    if search_centers is None:
        return None
    items: list[tuple[str, int, int]] = []
    for name, ctr in search_centers.items():
        if ctr is None:
            continue
        items.append((str(name), int(ctr[0]), int(ctr[1])))
    items.sort()
    return tuple(items)


def find_label_positions(
    img: Image.Image,
    *,
    search_centers: Optional[dict] = None,
    search_radius: int = _DEFAULT_LOCAL_RADIUS,
    y_min: int = 0,
) -> dict[str, dict]:
    """NCC-match the three label templates against the panel image.

    ``y_min`` restricts the search to rows at or below that image-Y
    (e.g. "labels live below the title"). The restriction is applied
    INTERNALLY and the returned coordinates are STILL image-absolute.
    This replaces the old pattern where callers cropped the image
    themselves and re-added the offset at every consumer — each
    consumer was a chance for a missed or doubled conversion, and the
    per-label log lines printed crop-frame Y's next to other modules'
    image-frame Y's (the live-log "two coordinate spaces" confusion).
    The crop and the conversion now live in exactly one place: here.

    Returns a dict mapping label name → match info::

        {
            "mass":        {"x": 480, "y": 348, "w": 120, "h": 28, "score": 0.83},
            "resistance":  {"x": 480, "y": 392, "w": 230, "h": 28, "score": 0.81},
            "instability": {"x": 480, "y": 436, "w": 230, "h": 28, "score": 0.79},
        }

    Missing or low-confidence matches are simply absent from the dict.
    Caller should treat an empty dict as "no labels found, fall back".
    Coordinates are always image-absolute pixels.

    Local search mode
    -----------------
    When ``search_centers`` is provided it must be a dict mapping
    label name → expected (x, y) image-absolute center::

        find_label_positions(
            img,
            search_centers={
                "mass":        (480, 350),
                "resistance":  (480, 395),
                "instability": (480, 440),
            },
        )

    Each label's NCC search is restricted to a window
    ``[cx - search_radius, cx + search_radius] ×
    [cy - search_radius, cy + search_radius]`` around its expected
    center, clamped to image bounds. The MASS-first anchor logic is
    skipped — the tracker already knows where each row should be, so
    we just confirm each one independently.

    Labels absent from ``search_centers`` are NOT searched at all in
    local-search mode (the tracker said it doesn't know where they
    are). Pass ``None`` (the default) for full-frame behaviour.

    Caching: a single-entry "last image" cache keyed on
    (id(img), img.size, img.mode, search_centers, search_radius)
    returns the cached result when the same image is passed multiple
    times. The HUD pipeline calls this via several entry points per
    scan (the drift-correction block, ``_find_label_rows_by_ncc``,
    the lock-validation fast path…) — without this cache each call
    re-runs the 8-scale NCC sweep (~100-150 ms each), tripling the
    per-scan latency for no benefit. Local and full-frame requests
    for the same image have DIFFERENT cache entries so they can't
    return stale data to each other.
    """
    global _LAST_CALL_CACHE, _LAST_FULL_FRAME_MATCHES
    if y_min:
        _y0 = max(0, int(y_min))
        if img.height - _y0 < 40:
            return {}
        _sub = img.crop((0, _y0, img.width, img.height))
        # Shift any caller-supplied absolute centers into the crop
        # frame for the inner search; results are shifted back below.
        _sc_shift = None
        if search_centers:
            _sc_shift = {
                k: (int(c[0]), int(c[1]) - _y0)
                for k, c in search_centers.items() if c is not None
            }
        _out = find_label_positions(
            _sub, search_centers=_sc_shift, search_radius=search_radius,
        )
        _shifted: dict[str, dict] = {}
        for _k, _m in _out.items():
            _mm = dict(_m)
            _mm["y"] = int(_mm["y"]) + _y0
            _shifted[_k] = _mm
        if _shifted:
            log.info(
                "label_match: y_min=%d rows (image-absolute): %s",
                _y0,
                {k: (m["x"], m["y"]) for k, m in _shifted.items()},
            )
        # Re-stamp the shared full-frame state in ABSOLUTE
        # coordinates so get_last_full_frame_matches() consumers
        # never see a crop frame.
        if search_centers is None:
            _LAST_FULL_FRAME_MATCHES = dict(_shifted)
        return _shifted
    sc_key = _normalize_search_centers(search_centers)
    sr_norm = int(search_radius)
    cache_key = (id(img), img.size, img.mode, sc_key, sr_norm)
    if _LAST_CALL_CACHE is not None and _LAST_CALL_CACHE[:5] == cache_key:
        return _LAST_CALL_CACHE[5]

    templates = _load_templates()
    if not templates:
        _LAST_CALL_CACHE = (*cache_key, {})
        return {}

    W, H = img.size
    if W < 40 or H < 40:
        return {}

    # Local-search path: confirm each requested label independently
    # inside its own ±radius window. Doesn't use the MASS-anchor
    # logic — the rigid-body tracker already knows the expected
    # positions and would only fight the anchor refinement if we ran
    # it.
    #
    # Crop padding: the window is expanded by the template's max
    # dimension at the largest scale we'll test, so the full
    # template still fits inside the crop even when its center
    # lands at the ±r tolerance boundary. Without this, narrow
    # windows (e.g. ±40 px) would silently skip large templates
    # (resistance/instability are ~170 / 160 px wide at native
    # scale, multi-scale up to 2.0×) and miss legitimate matches.
    # The result is then filtered to keep only candidates whose
    # CENTER falls within ±r of the requested ``ctr``.
    if sc_key is not None:
        full_gray = np.asarray(img.convert("L"), dtype=np.uint8)
        results: dict[str, dict] = {}
        for name, ctr in (search_centers or {}).items():
            if name not in templates or ctr is None:
                continue
            cx = int(ctr[0])
            cy = int(ctr[1])
            r = max(1, sr_norm)
            # Pre-compute the resize-scaled templates so we can also
            # work out the crop pad and the per-scale fit check.
            scaled_templates: list[tuple[float, np.ndarray]] = []
            for scale in _SCALE_FACTORS:
                scaled = _resize_template(templates[name], scale)
                scaled_templates.append((scale, scaled))
            if not scaled_templates:
                continue
            max_tw = max(s.shape[1] for _, s in scaled_templates)
            max_th = max(s.shape[0] for _, s in scaled_templates)
            pad_x = max_tw // 2 + 4
            pad_y = max_th // 2 + 4
            wx1 = max(0, cx - r - pad_x)
            wy1 = max(0, cy - r - pad_y)
            wx2 = min(W, cx + r + pad_x)
            wy2 = min(H, cy + r + pad_y)
            if wx2 - wx1 < 8 or wy2 - wy1 < 8:
                continue
            window_gray = full_gray[wy1:wy2, wx1:wx2]
            window_target = _canonicalize(window_gray)
            best: Optional[dict] = None
            for scale, scaled in scaled_templates:
                if (
                    scaled.shape[0] > window_target.shape[0]
                    or scaled.shape[1] > window_target.shape[1]
                ):
                    continue
                score, x, y = _ncc_search(window_target, scaled)
                if score < _MIN_MATCH_SCORE:
                    continue
                # Image-absolute center of this candidate.
                abs_cx = int(x) + wx1 + scaled.shape[1] // 2
                abs_cy = int(y) + wy1 + scaled.shape[0] // 2
                if abs(abs_cx - cx) > r or abs(abs_cy - cy) > r:
                    # Outside the requested ±r tolerance — drop it
                    # even though NCC was satisfied. The pad let
                    # NCC see this candidate; the tolerance gate
                    # enforces the contract.
                    continue
                if best is None or score > best["score"]:
                    best = {
                        "x": int(x) + wx1,  # image-absolute
                        "y": int(y) + wy1,
                        "w": int(scaled.shape[1]),
                        "h": int(scaled.shape[0]),
                        "score": float(score),
                        "scale": float(scale),
                    }
            if best is not None:
                results[name] = best
                log.debug(
                    "label_match[local]: %s matched at (x=%d, y=%d, "
                    "w=%d, h=%d) score=%.2f scale=%.2f (center=%d,%d "
                    "r=%d)",
                    name, best["x"], best["y"], best["w"], best["h"],
                    best["score"], best["scale"], cx, cy, r,
                )
            else:
                log.debug(
                    "label_match[local]: %s not found in window "
                    "(center=%d,%d r=%d)", name, cx, cy, r,
                )
        _LAST_CALL_CACHE = (*cache_key, results)
        return results

    # Restrict search to the LEFT 65% of the image (labels never sit
    # in the right half — that's the value column).
    search_w = int(W * 0.65)
    crop = img.crop((0, 0, search_w, H))
    target_gray = np.asarray(crop.convert("L"), dtype=np.uint8)

    # ── DUAL-POLARITY targets (bypass Otsu heuristic) ──
    # The Otsu heuristic in _canonicalize makes ONE polarity decision
    # per image by counting which class is the minority.  On complex
    # backgrounds (planet-side scans with bright terrain behind the
    # HUD) that heuristic mis-flips the image -- dropping MASS NCC
    # from 0.92 at the CORRECT position to 0.59 at a SPURIOUS one
    # (measured on cap_20260418_160136_959).
    #
    # Fix: build BOTH polarities directly from the raw gray (skip
    # Otsu), normalize each, and run MASS NCC on both.  Take whichever
    # scores higher.  Same idea as comma_finder.py (signature
    # pipeline) and the inverted CNN voter -- polarity decided AFTER
    # matching, not before.
    #
    # IMPORTANT: don't call _canonicalize for either polarity, because
    # it might flip one of them again via Otsu and we'd lose the
    # opposite-polarity branch.  We just normalize each raw form
    # independently.
    def _norm(arr_f32: np.ndarray) -> np.ndarray:
        m = float(arr_f32.mean())
        s = float(arr_f32.std())
        if s >= 1e-3:
            return (arr_f32 - m) / s
        return arr_f32 - m
    target_pos = _norm(target_gray.astype(np.float32))           # text-bright case
    target_neg = _norm((255 - target_gray).astype(np.float32))   # text-dark case

    # ── MASS-FIRST search ──
    # MASS is the anchor. Find the best MASS match across all scales
    # AND both polarities. That scale + position + polarity defines
    # the panel's geometry. Then search for RESISTANCE / INSTABILITY
    # at THE SAME SCALE and polarity within narrow Y windows.
    if "mass" not in templates:
        log.warning("label_match: no MASS template — can't anchor")
        _LAST_CALL_CACHE = (*cache_key, {})
        _LAST_FULL_FRAME_MATCHES = {}
        return {}

    # Step 1: Find candidate MASS matches across all (scale, polarity)
    # pairs.  We collect the TOP grayscale candidate per (polarity, scale)
    # so the RGB verifier (step 1b) has multiple candidates to choose
    # from -- previously a confidently-wrong grayscale peak (panel
    # 155912_020 scored 0.86 at +205 px off-truth) was picked over the
    # actual MASS position because the wrong one had higher grayscale
    # NCC.  RGB color matching breaks ties between same-luminance
    # different-color regions (the MASS row is mint-cyan; spurious
    # peaks land on white "UNKNOWN" footer text, mineral name in
    # different color, etc.).
    mass_candidates: list[dict] = []
    for polarity_name, polarity_target in (
        ("pos", target_pos),
        ("neg", target_neg),
    ):
        for scale in _SCALE_FACTORS:
            scaled = _resize_template(templates["mass"], scale)
            if scaled.shape[0] > polarity_target.shape[0] or scaled.shape[1] > polarity_target.shape[1]:
                continue
            score, x, y = _ncc_search(polarity_target, scaled)
            if score < _MIN_MATCH_SCORE:
                continue
            mass_candidates.append({
                "x": x, "y": y,
                "w": scaled.shape[1], "h": scaled.shape[0],
                "score": float(score), "scale": float(scale),
                "polarity": polarity_name,
                "_target": polarity_target,
            })

    if not mass_candidates:
        log.debug("label_match: MASS not found at any (scale, polarity)")
        _LAST_CALL_CACHE = (*cache_key, {})
        _LAST_FULL_FRAME_MATCHES = {}
        return {}

    # Step 1b: VOTING between grayscale and RGB MASS detectors.
    # Conservative wiring:
    #   - Compute RGB top-K independently.
    #   - Find nearest RGB candidate to GRAYSCALE TOP-1.
    #   - If they AGREE (within AGREEMENT_PX): keep grayscale top-1
    #     and mark agreement=True (boosts synthesis confidence).
    #   - If they DISAGREE: scan further grayscale candidates for
    #     one whose nearest RGB candidate AGREES, and SWAP only if
    #     such an alternative exists (otherwise stick with top-1
    #     to preserve the proven dual-polarity result).
    #
    # Why not always pick max voting score?  Tested empirically: full
    # voting with agreement-bonus 0.20 + disagreement-penalty 0.10
    # regressed 2 ANNOTATED panels because a high-grayscale top-1 in
    # "disagreement" got beaten by a low-grayscale top-2 in "agreement"
    # even when top-1 was actually correct (RGB just peaked nearby).
    # The conservative gate "swap only if top-1 disagrees AND
    # alternative agrees" is the safer pattern.
    mass_candidates.sort(key=lambda c: -c["score"])
    _rgb_t = _load_rgb_mass_template()
    voting_used = False
    agreement = None
    if _rgb_t is not None:
        rgb_img = np.asarray(
            img.crop((0, 0, search_w, H)), dtype=np.float32,
        )
        rgb_candidates = _rgb_mass_top_k(rgb_img, _rgb_t, k=8)
        if rgb_candidates:
            AGREEMENT_PX = 15

            def _nearest_rgb_dist(g):
                nearest = min(
                    rgb_candidates,
                    key=lambda r: (r["x"] - g["x"]) ** 2 + (r["y"] - g["y"]) ** 2,
                )
                dist = ((nearest["x"] - g["x"]) ** 2
                        + (nearest["y"] - g["y"]) ** 2) ** 0.5
                return dist, nearest

            top1 = mass_candidates[0]
            top1_dist, top1_rgb = _nearest_rgb_dist(top1)
            voting_used = True
            if top1_dist <= AGREEMENT_PX:
                # Top-1 confirmed by RGB.
                agreement = True
                log.info(
                    "label_match: MASS top-1 AGREES with RGB (gray=%.2f, "
                    "rgb=%.2f, dist=%.1fpx) -- keeping top-1",
                    top1["score"], top1_rgb["score"], top1_dist,
                )
            else:
                # Top-1 disagrees with RGB.  Look for a grayscale
                # alternative that DOES agree.
                rescue = None
                for alt in mass_candidates[1:8]:
                    alt_dist, alt_rgb = _nearest_rgb_dist(alt)
                    if alt_dist <= AGREEMENT_PX:
                        rescue = (alt, alt_rgb, alt_dist)
                        break
                if rescue is not None:
                    alt, alt_rgb, alt_dist = rescue
                    log.warning(
                        "label_match: MASS top-1 DISAGREES with RGB "
                        "(gray=%.2f at (%d,%d), nearest RGB %.1fpx away); "
                        "swapping to candidate (gray=%.2f, rgb=%.2f, "
                        "dist=%.1fpx) at (%d,%d)",
                        top1["score"], top1["x"], top1["y"], top1_dist,
                        alt["score"], alt_rgb["score"], alt_dist,
                        alt["x"], alt["y"],
                    )
                    mass_candidates[0] = alt
                    agreement = True
                else:
                    # No alternative agrees -- keep top-1, mark as
                    # disagreement (lowers downstream confidence).
                    agreement = False
                    log.info(
                        "label_match: MASS top-1 DISAGREES with RGB and "
                        "no alternative agrees -- keeping top-1, marking "
                        "low confidence",
                    )

    winner = mass_candidates[0]
    target: np.ndarray = winner["_target"]
    best_mass = {
        "x": winner["x"], "y": winner["y"],
        "w": winner["w"], "h": winner["h"],
        "score": winner["score"],
        "scale": winner["scale"],
        "polarity": winner["polarity"],
    }
    if voting_used and agreement is not None:
        best_mass["agreement"] = agreement
    log.info(
        "label_match: MASS won at polarity=%s scale=%.2f score=%.3f pos=(%d,%d)",
        best_mass["polarity"], best_mass["scale"], best_mass["score"],
        best_mass["x"], best_mass["y"],
    )

    # Step 2: Pitch (vertical row spacing) ≈ 1.4× MASS-template height
    # at the matched scale. Used to predict where RESISTANCE and
    # INSTABILITY should appear.
    mass_h = best_mass["h"]
    pitch = int(round(mass_h * 1.4))
    mass_cy = best_mass["y"] + mass_h // 2

    results: dict[str, dict] = {"mass": best_mass}

    # Step 3: Search for RESISTANCE / INSTABILITY at the MASS scale,
    # WITHIN A NARROW Y WINDOW around their expected position.
    expected = {
        "resistance":  mass_cy + pitch,
        "instability": mass_cy + 2 * pitch,
    }
    # Allow ±35% of pitch slack — covers font-rendering variations
    # without admitting matches from far-off areas like the
    # COMPOSITION list.
    y_tolerance = int(pitch * 0.35)

    for name, expected_cy in expected.items():
        if name not in templates:
            continue
        # ── Full template FIRST; truncated only as fallback ──
        # We initially tried "use whichever scores higher" but the
        # short template's smaller area means it scores high on many
        # false-positive positions (random text fragments, edge
        # artefacts, neighbouring rows).  Running both as alternatives
        # caused a measured regression of -6 to -12 ANNOTATED locks.
        #
        # Correct discipline: trust the FULL template when it
        # actually meets threshold.  The short template only fires
        # when the full template scored below threshold AND the
        # short scored above it -- in that case the full word
        # genuinely failed and the truncated match is our best (only)
        # signal for the row.
        #
        # We further require that the short match's left-edge X be
        # within ±2× the short template width of mass_x, since all
        # three label rows are left-aligned at the same column.  This
        # kills false positives that match short-template glyphs
        # somewhere outside the label column.
        full_t = templates[name]
        short_t = templates.get(f"{name}_short")
        h_scaled_full = _resize_template(full_t, best_mass["scale"]).shape[0]
        # Use the full template's height to size the search window;
        # both variants have the same height since we slice only in W.
        win_top = max(0, expected_cy - h_scaled_full // 2 - y_tolerance)
        win_bot = min(target.shape[0], expected_cy + h_scaled_full // 2 + y_tolerance)
        if win_bot - win_top < h_scaled_full:
            continue
        windowed = target[win_top:win_bot, :]

        scaled = _resize_template(full_t, best_mass["scale"])
        if scaled.shape[0] > windowed.shape[0] or scaled.shape[1] > windowed.shape[1]:
            score, x, y = -1.0, 0, 0
        else:
            score, x, y = _ncc_search(windowed, scaled)
        variant_used = "full"

        # Truncated template runs ALONGSIDE the full template; whichever
        # has the higher NCC wins as long as it's column-aligned with
        # MASS (within 10 px -- all three numeric labels are
        # left-aligned at the same X).
        #
        # The slice widths (RESIST = 60%, INSTA = 46%) were chosen so
        # the short template's PEAK lands at the same (x,y) as the full
        # template's peak in every measured panel.  Without the column
        # gate the short template could still false-positive on stray
        # text (early attempts with a 4-letter "RESI" slice peaked
        # inside SCAN RESULTS due to the RES/RES prefix collision).
        #
        # Net effect: when the full template degrades below threshold
        # (e.g. right-side occlusion, perspective skew), the short
        # template can rescue the row anchor without giving up the
        # full template's specificity in normal conditions.
        if short_t is not None:
            scaled_short = _resize_template(short_t, best_mass["scale"])
            if (
                scaled_short.shape[0] <= windowed.shape[0]
                and scaled_short.shape[1] <= windowed.shape[1]
            ):
                s_score, s_x, s_y = _ncc_search(windowed, scaled_short)
                col_tol = 10
                if (
                    s_score > score
                    and s_score >= _MIN_MATCH_SCORE
                    and abs(s_x - best_mass["x"]) <= col_tol
                ):
                    log.info(
                        "label_match: %s short (%s, NCC=%.2f) "
                        "beats full (NCC=%.2f); using short",
                        name, name.upper()[:6 if name == "resistance" else 5],
                        s_score, score,
                    )
                    score, x, y, scaled = s_score, s_x, s_y, scaled_short
                    variant_used = "short"

        # ── RGB voting for RESIST/INSTA ──
        # If grayscale failed (or scored marginally), an independent
        # RGB NCC sweep within the same y-window may rescue the row.
        # Pattern mirrors the MASS-voting step 1b: agreement between
        # grayscale and RGB on position raises confidence, RGB can
        # rescue when grayscale alone falls below threshold.
        #
        # The y-window is already narrow (centered on mass_cy +
        # N*pitch with +/- pitch*0.35 slack), so we don't need a
        # separate column gate -- the window itself eliminates the
        # cross-panel false positives.
        rgb_row_t = _load_rgb_label_template(name)
        if rgb_row_t is not None:
            # Build RGB target once per call (lazy).
            try:
                _rgb_full_target  # type: ignore[name-defined]
            except NameError:
                _rgb_full_target = np.asarray(
                    img.crop((0, 0, search_w, H)), dtype=np.float32,
                )
            rgb_score, rgb_x, rgb_y_full = _rgb_row_search(
                _rgb_full_target, rgb_row_t, best_mass["scale"],
                win_top, win_bot,
            )
            # Convert grayscale (window-relative) position to full-image
            # coords for comparison.
            gray_y_full = (y + win_top) if score >= _MIN_MATCH_SCORE else -10_000
            same_row_px = 12  # rows are ~28 px tall; agreement within row
            if score >= _MIN_MATCH_SCORE and rgb_score >= _MIN_MATCH_SCORE:
                # Both fired -- check agreement on y.
                if abs(rgb_y_full - gray_y_full) <= same_row_px:
                    # Agreement; keep grayscale position but raise
                    # confidence by recording rgb_score.
                    log.info(
                        "label_match: %s gray (NCC=%.2f at y=%d) and "
                        "RGB (NCC=%.2f at y=%d) AGREE",
                        name, score, gray_y_full, rgb_score, rgb_y_full,
                    )
                # else: disagree silently; keep grayscale (already trusted)
            elif rgb_score >= _MIN_MATCH_SCORE and score < _MIN_MATCH_SCORE:
                # Grayscale failed but RGB found a row in the window.
                # Use the RGB-derived position; this rescues panels
                # where the rendered label is too dim/skewed for
                # grayscale NCC but the mint-cyan color is still
                # detectable.
                col_tol = 12
                if abs(rgb_x - best_mass["x"]) <= col_tol:
                    log.warning(
                        "label_match: %s grayscale failed (NCC=%.2f) but "
                        "RGB rescued (NCC=%.2f at x=%d, y=%d)",
                        name, score, rgb_score, rgb_x, rgb_y_full,
                    )
                    score = rgb_score
                    x = rgb_x
                    y = rgb_y_full - win_top  # back to window-relative
                    # scaled stays as grayscale template (used for w/h
                    # reporting downstream); the value column starts
                    # at x + scaled.shape[1] regardless.
                    variant_used = "rgb-rescue"

        if score < _MIN_MATCH_SCORE:
            # NCC failed for this label. Before dropping it, check
            # whether MASS matched confidently enough to synthesize
            # the position from the HUD's rigid 3-row geometry.
            #
            # The window we just searched is already centered on the
            # geometric expected_cy (``mass_cy + N·pitch``), so the
            # synthesized box is simply that expected position at the
            # MASS scale. Left edge is shared with MASS — all three
            # numeric labels are left-aligned in the panel.
            # Synthesis fires when MASS is CONFIDENT.  Two paths to
            # confidence:
            #   (a) grayscale score alone above _SYNTH_MASS_FLOOR (0.65)
            #   (b) voting AGREEMENT between grayscale and RGB
            #       detectors -- the voting test showed agree=100%
            #       correct on 66/66 panels with truth, so an
            #       agreement is at least as reliable as a high-only-
            #       grayscale match.
            mass_confident = (
                best_mass["score"] >= _SYNTH_MASS_FLOOR
                or best_mass.get("agreement") is True
            )
            if mass_confident:
                synth_y = expected_cy - h_scaled_full // 2
                synth_y = max(0, min(synth_y, target.shape[0] - h_scaled_full))
                results[name] = {
                    "x": int(best_mass["x"]),
                    "y": int(synth_y),
                    "w": int(scaled.shape[1]),
                    "h": int(scaled.shape[0]),
                    # Keep the REAL (low) NCC score so downstream
                    # consumers can tell this was geometry-derived,
                    # not template-confirmed.
                    "score": float(score),
                    "scale": float(best_mass["scale"]),
                    "synthesized": True,
                }
                log.info(
                    "label_match: %s NCC=%.2f below %.2f but MASS "
                    "confident (%.2f >= %.2f) — synthesizing position "
                    "from rigid 3-row geometry at expected_cy=%d "
                    "(skew-degraded wide template, position is "
                    "layout-guaranteed)",
                    name, score, _MIN_MATCH_SCORE, best_mass["score"],
                    _SYNTH_MASS_FLOOR, expected_cy,
                )
                continue
            log.debug(
                "label_match: %s rejected score=%.2f (expected_cy=%d, "
                "window=[%d,%d])",
                name, score, expected_cy, win_top, win_bot,
            )
            continue
        # When the truncated variant wins, REPORT THE FULL TEMPLATE'S
        # w/h, not the slice's.  Downstream consumers compute the
        # value-crop's left edge as ``m["x"] + m["w"]`` -- if w were
        # only the truncated width (e.g. RESI ~70 px vs RESISTANCE
        # ~170 px), the value crop would start mid-word and produce
        # garbage reads.  Since both templates share the same left
        # edge (the word's first letter), x is unchanged.
        full_scaled_for_report = _resize_template(full_t, best_mass["scale"])
        results[name] = {
            "x": x,
            "y": y + win_top,  # convert window-local back to image coord
            "w": int(full_scaled_for_report.shape[1]),
            "h": int(full_scaled_for_report.shape[0]),
            "score": float(score),
            "scale": float(best_mass["scale"]),
            "variant_used": variant_used,  # "full" | "short" (telemetry)
        }

    for name, m in results.items():
        log.debug(
            "label_match: %s matched at (x=%d, y=%d, w=%d, h=%d) "
            "score=%.2f scale=%.2f variant=%s",
            name, m["x"], m["y"], m["w"], m["h"], m["score"], m["scale"],
            m.get("variant_used", "full"),
        )

    _LAST_CALL_CACHE = (*cache_key, results)
    _LAST_FULL_FRAME_MATCHES = dict(results)
    return results


def reset_template_cache() -> None:
    """Clear the in-memory template cache. Call after rebuilding the
    templates file so the new templates are picked up without a
    process restart."""
    global _templates_cache, _LAST_CALL_CACHE, _LAST_FULL_FRAME_MATCHES
    _templates_cache = None
    _LAST_CALL_CACHE = None
    _LAST_FULL_FRAME_MATCHES = {}
