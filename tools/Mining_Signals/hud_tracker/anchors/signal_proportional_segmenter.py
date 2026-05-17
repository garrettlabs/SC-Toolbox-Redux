"""Proportional segmenter for SC mining-signature digit crops.

The SC mining-signature panel renders values in exactly two layouts:

  4-digit form:   D , D D D     →  4 digit slots + 1 comma slot
  5-digit form:   D D , D D D   →  5 digit slots + 1 comma slot

That deterministic structure is the point: we don't need column-projection
valleys to localize digits. Once we know the crop's pixel width and which
hypothesis (4 vs 5) holds, we can place digit bboxes by simple proportion.

Why this matters
----------------
The legacy column-projection segmenter (``_segment_glyphs``) finds spurious
valleys WITHIN digits — chromatic aberration cuts vertical streaks through
otherwise-bright glyphs — and misses real valleys BETWEEN digits because
aberration smears bright pixels across the gap. The end-result is per-span
crops that fragment one digit and merge another, and the CNN classifies
the resulting half-glyphs as garbage. In the user's screenshots, "3,400"
read as ``'47777'`` and ``'675?7'`` because the spans were misplaced.

Algorithm
---------
The SC signal font is *proportional* — `1` is much narrower than `5`/`8`
— so equal-width slots over the full crop width misalign whenever a `1`
is present. Instead we run a smoothed column-density projection and find
**ink-blob centers**, then bind those centers to the structural prior:

    4-digit hypothesis: 4 digit centers + 1 comma center, in left-to-right order
    5-digit hypothesis: 5 digit centers + 1 comma center, in left-to-right order

Comma centers are detected by looking for a low-density column gap between
two adjacent digit centers (commas are short — they only ink the bottom
~30% of the line height — so projection-by-height-coverage discriminates
commas from digits cleanly).

For each hypothesis we:
1. Find ``n_digits + 1`` blob centers in the crop.
2. Decide which one is the comma (the SHORTEST blob, by max ink-row).
3. Bind the rest as digits, preserving left-to-right order.
4. Generate digit bboxes from each digit's projection extent.
5. Run the CNN. Score = mean conf + lexicon bonus.

Both hypotheses are evaluated; the one with higher score wins. The
lexicon bonus tips the comparison toward known-valid signature reads
when CNN confidence is similar.

If the column projection finds the wrong number of blobs, we fall back
to a pure proportional layout over the ink-bearing column range — that
gives the segmenter a graceful degradation when chromatic aberration
fuses adjacent digits at the projection level.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

log = logging.getLogger("hud_tracker.signal_proportional_segmenter")


# Tunable constants. Each is documented with the empirical reasoning
# behind it so future maintainers know what to perturb if the
# segmenter starts mis-aligning on a new render scale.

# Width of a digit's bbox as a fraction of its measured projection
# extent. When we have a real ink blob we expand it slightly outward
# so the CNN sees a small gutter — the network was trained on
# bbox-extracted-then-padded crops, not tightly-cropped strokes.
_DIGIT_BBOX_INFLATE = 1.10

# Column-ink-density threshold (as a fraction of crop height) above
# which a column counts as ink-bearing. The SC signal font's stroke
# coverage along a digit's mid-row is consistently > 0.30 even on
# narrow `1`s; the inter-digit kerning gap drops to < 0.10. The 0.18
# threshold sits squarely in that valley.
_INK_COL_FRAC = 0.18

# Minimum gap (in columns of zero-ink projection) between two
# distinct ink runs for them to count as separate blobs. Smaller
# gaps (1-2 px) are typically chromatic-aberration kerning splits
# within a single glyph; ≥3 px is a real inter-digit gap.
_MIN_BLOB_GAP_PX = 2

# Width of a digit slot relative to a comma slot in the full-
# proportional fallback layout. SC commas are ~3-5 px wide vs
# ~12-20 px for digits (depends on render scale), so
# digit_pitch ≈ 2.5 × comma_pitch. Encoded as the comma slot's
# fraction of a digit slot.
_FALLBACK_COMMA_PITCH_FRAC = 0.40

# Width of a digit's bbox as a fraction of its slot width in the
# pure-proportional fallback. ~85% leaves room on each side of the
# bbox for chromatic-aberration halos to NOT bleed into the
# neighbouring digit's CNN crop.
_FALLBACK_DIGIT_BBOX_FRAC = 0.85
_FALLBACK_COMMA_BBOX_FRAC = 0.85

# Minimum crop width that can plausibly hold a 4-digit signature.
# Smaller than this and we return None — the caller falls back to
# the legacy column-projection segmenter which has better recovery
# on fragments. Empirical lower bound for typical 1080p captures.
_MIN_CROP_W_FOR_4DIGIT = 20

# Lexicon-membership bonus added to a hypothesis's mean CNN
# confidence when the composed integer is in the known-signature
# set. Large enough to flip a hypothesis at any practical confidence
# delta (mean CNN confs typically span 0.6-0.95) without being so
# large that a spurious lexicon hit on noise overrides a clean read.
_LEXICON_BONUS = 0.20

# Default mean confidence for the no-CNN case. Surfaced in the
# result so callers can detect the "I didn't actually evaluate this"
# path.
_NO_CNN_DEFAULT_CONF = 0.50

# Empty-slot detection thresholds. A digit slot whose ink mass is
# implausibly low for a populated digit position is almost always
# pill-background being treated as a digit — the CNN will read it as
# the icon class '@' or, depending on stride pattern, lock onto a
# specific digit at high confidence (uniform bg patches happen to
# align well with certain class prototypes; '7' is a common winner
# in the SC signal CNN's softmax landscape). We measure ink mass per
# slot post polarity-canonicalization and penalize hypotheses with
# slots whose mass is far below the median digit's mass.
#
# Penalty magnitudes were tuned against the 7,200 / 3,400 / 11,520
# / 16,960 captures: with mean-CNN-confs typically 0.85-0.95, a
# -0.30 penalty per empty slot is enough to flip a hypothesis (e.g.
# a 5-digit hypothesis with one empty slot gets net 0.55-0.65 vs a
# 4-digit hypothesis with 4 populated slots at 0.85 — 4-digit wins).
# A -0.5 penalty is reserved for the "all slots empty" corner case
# where the hypothesis should be dropped outright.
_EMPTY_SLOT_PENALTY_PER_SLOT = 0.30  # subtracted per slot below mass threshold
_EMPTY_SLOT_PENALTY_DEGENERATE = 0.50  # all-low-mass hypothesis penalty
_EMPTY_SLOT_MASS_FRACTION = 0.25      # slot mass < this × median = "empty"
_EMPTY_SLOT_MEDIAN_FLOOR = 4          # if median mass < this → degenerate

# ── Kerning-locked layout ─────────────────────────────────────────────
# When the comma is detected, slot positions are deterministic: each
# digit sits at a known fractional offset from the comma center, in
# units of row height. The offsets were measured empirically across
# 14 clean captures (see calibrate_kerning.py + kerning_model.json).
# Slot[0] = leftmost digit (5-digit values only), slot[1] = digit
# immediately left of comma, slots [2..4] = digits right of comma.
#
# The kerning model bypasses ink-extent / blob-detection entirely:
# we project all 5 candidate slot centers from the comma and let the
# CNN + empty-slot penalty decide which slots are populated. This
# directly attacks Stage 6 (wrong n_digits) and Stage 7 (digit
# position) in the failure profile, since slot count + position are
# now metric-driven rather than blob-derived.
#
# When the kerning model isn't loaded (missing JSON, schema mismatch)
# the segmenter falls through to the existing tiered layout, so the
# kerning path is purely additive.
import json as _kerning_json  # local alias to avoid colliding with logging
_KERNING_MODEL: Optional[dict[str, Any]] = None
_KERNING_MODEL_PATH = Path(__file__).resolve().parent / "kerning_model.json"


def _load_kerning_model() -> Optional[dict[str, Any]]:
    """Lazy-load + cache the kerning model. Returns ``None`` if the
    file is missing, the schema doesn't match, or required slot
    entries are absent. Cached after the first call.
    """
    global _KERNING_MODEL
    if _KERNING_MODEL is not None:
        return _KERNING_MODEL
    if not _KERNING_MODEL_PATH.is_file():
        return None
    try:
        doc = _kerning_json.loads(
            _KERNING_MODEL_PATH.read_text(encoding="utf-8"),
        )
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    if doc.get("schema") != "kerning_v1":
        return None
    slots = doc.get("slots") or {}
    # Slots 1..4 are mandatory (always present in any signature
    # value). Slot 0 is optional (only 5-digit values use it). If
    # any of 1..4 is missing the model is unusable.
    for s in ("1", "2", "3", "4"):
        if s not in slots:
            return None
    _KERNING_MODEL = doc
    return doc

# Comma detection: a comma's ink rows are concentrated in the bottom
# 35% of the line height (in canonical bright-text-on-dark polarity).
# Digits ink the full vertical extent. A blob whose ink-row centroid
# is in the bottom 35% AND whose top-30% rows are mostly empty is a
# comma.
_COMMA_VERT_CENTROID_FRAC = 0.65  # blobs with centroid below this y-frac are commas
_COMMA_TOP_EMPTY_FRAC = 0.30  # comma blobs have <X% ink in top 30% of rows


@dataclass
class _Hypothesis:
    """Container for one (n_digits) hypothesis and its score."""
    n_digits: int
    comma_position: int
    digit_bboxes: list[tuple[int, int, int, int]]  # (x, y, w, h)
    comma_bbox: tuple[int, int, int, int]
    digit_width_px: float
    comma_width_px: float
    composed: str
    classifications: list[tuple[str, float]]
    mean_conf: float
    in_lexicon: bool
    score: float
    used_blob_centers: bool  # True if blob-center binding succeeded; False = fallback
    # ── Empty-slot diagnostics (Fix 1) ─────────────────────────────
    slot_masses: list[int]      # per-digit-bbox bright-pixel counts
    n_empty_slots: int           # how many slots are below the median-mass threshold
    empty_slot_penalty: float    # negative number subtracted into score
    # ── Lexicon backtracking diagnostics (Fix 2) ───────────────────
    backtracked: bool            # True iff top-1 was rewritten by lexicon backtrack
    backtrack_from: str           # original top-1 composition (only set when backtracked)


def _canonicalize_polarity_local(gray: np.ndarray) -> np.ndarray:
    """Force the input to bright-text-on-dark-background polarity.

    The signature digit area can come in either polarity depending
    on the SC HUD's pill render state. The CNN was trained on bright-
    text-on-dark crops, so we must canonicalize before classification.

    Border-median rule: the value-bbox crop has a margin of pure
    background on the top and bottom edges (digit baselines and
    ascender heights don't reach the bbox corners — the bbox always
    has at least 1-3 px of padding). Sampling the median pixel value
    along those borders gives a robust estimate of the BACKGROUND
    luminance.

        bg_median > center_median  →  background is BRIGHT  →  invert
        bg_median ≤ center_median  →  background is DARK    →  no-op

    More reliable than a fixed threshold (e.g. > 128) because the
    contrast-stretched value crop has its dynamic range amplified to
    [0, 255] regardless of the original mean luminance — a fixed
    cutoff at 128 breaks down whenever the "bright" and "dark" classes
    in the source were both above or both below 128 before stretch.

    Note (DETERMINISM 2026-05-10): this function is INTENTIONALLY
    distinct from the production ``ocr.sc_ocr.api._canonicalize_polarity``
    (Otsu minority-class rule). The border-median rule is what the
    blob/ink-extent detection logic in this module was tuned against;
    swapping in the production function regresses captures where the
    polarity test happens to disagree (e.g. when digit ink dominates
    the center sample). The fix for cross-path divergence is at the
    api level: ``_signal_recognize_pil`` re-extracts crops using THIS
    function's output (the segmenter's gray_canon) rather than the
    production canonicalization. See the api-side comment around the
    ``_seg_gray_canon`` thread-through for details.
    """
    if gray.size == 0:
        return gray.astype(np.uint8)

    h, w = gray.shape[:2]
    # Sample border medians (top + bottom rows). Skip the leftmost
    # and rightmost columns of those rows — corner cells can pick up
    # edge artifacts from neighbouring panel chrome that aren't
    # representative of the actual background.
    border_strip = np.concatenate([
        gray[0, max(1, w // 8):w - max(1, w // 8)],
        gray[h - 1, max(1, w // 8):w - max(1, w // 8)],
    ])
    bg_median = float(np.median(border_strip))

    # Center sample — the middle 60% of the crop, in both dimensions.
    # This is where the digits live, so its median represents the
    # foreground-or-background mix at the line center.
    cx0 = max(1, w // 5)
    cx1 = max(cx0 + 1, w - w // 5)
    cy0 = max(1, h // 5)
    cy1 = max(cy0 + 1, h - h // 5)
    center_median = float(np.median(gray[cy0:cy1, cx0:cx1]))

    # If the background is BRIGHTER than the center (text), invert.
    # When equal (uniform crop), leave alone — there's no information.
    if bg_median > center_median + 5:
        return (255 - gray).astype(np.uint8)
    return gray.astype(np.uint8)


def _measure_ink_in_bbox(
    gray_canon: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> int:
    """Sum of bright pixels inside ``bbox`` after polarity-canonicalization.

    Used to detect empty-slot hypotheses where one of the digit
    positions is centered on pill background instead of an actual
    digit. The threshold is the per-bbox midpoint of dynamic range
    (same recipe :func:`_find_blobs` uses), so the count is robust
    against pill-color drift.

    Returns ``0`` for bboxes that are entirely uniform (dynamic range
    too small to distinguish ink from background) — those slots have
    no information and the empty-slot penalty should treat them as
    empty.
    """
    x, y, w, h = bbox
    h_full, w_full = gray_canon.shape[:2]
    x = max(0, min(int(x), w_full - 1))
    y = max(0, min(int(y), h_full - 1))
    w = max(1, min(int(w), w_full - x))
    h = max(1, min(int(h), h_full - y))
    crop = gray_canon[y:y + h, x:x + w]
    if crop.size == 0:
        return 0
    g_min = int(crop.min())
    g_max = int(crop.max())
    if g_max - g_min < 16:
        return 0
    thr = g_min + (g_max - g_min) // 2
    return int((crop > thr).sum())


def _empty_slot_penalty(
    gray_canon: np.ndarray,
    digit_bboxes: list[tuple[int, int, int, int]],
) -> tuple[float, list[int], int]:
    """Compute a penalty proportional to the number of empty digit slots.

    Returns ``(penalty, masses, n_empty)`` where:

      * ``penalty`` is a non-positive float to be ADDED to the
        hypothesis's score (so it reduces the score).
      * ``masses`` is the per-slot ink-mass list (debug surface).
      * ``n_empty`` is the count of slots flagged as empty.

    Logic:
      * If no slot has any ink at all → penalty
        ``-_EMPTY_SLOT_PENALTY_DEGENERATE`` (drop hypothesis).
      * If the median mass is below ``_EMPTY_SLOT_MEDIAN_FLOOR`` →
        same degenerate penalty (the whole crop is degenerate).
      * Otherwise, each slot whose mass is below
        ``_EMPTY_SLOT_MASS_FRACTION × median`` costs
        ``_EMPTY_SLOT_PENALTY_PER_SLOT``.

    Tuning rationale (see module-level constants for the empirical
    captures the thresholds were validated against). The 25%-of-median
    fraction is generous: even narrow `1` glyphs and aberration-thinned
    digits sit at 30-50% of the median mass, so legitimate slots clear
    the bar comfortably while empty pill-background patches (mass < 5%
    of median) are unambiguously flagged.
    """
    if not digit_bboxes:
        return (0.0, [], 0)
    masses = [_measure_ink_in_bbox(gray_canon, bb) for bb in digit_bboxes]
    if max(masses) == 0:
        return (-_EMPTY_SLOT_PENALTY_DEGENERATE, masses, len(masses))
    median_mass = float(np.median(masses))
    if median_mass < _EMPTY_SLOT_MEDIAN_FLOOR:
        return (-_EMPTY_SLOT_PENALTY_DEGENERATE, masses, len(masses))
    threshold = median_mass * _EMPTY_SLOT_MASS_FRACTION
    n_empty = sum(1 for m in masses if m < threshold)
    return (
        -_EMPTY_SLOT_PENALTY_PER_SLOT * n_empty,
        masses,
        n_empty,
    )


def _compose_with_comma(
    digit_chars: list[str],
    comma_position: int,
) -> str:
    """Compose a digit-only string by sandwiching the comma's position.

    The segmenter stores the comma's position in slot-index terms
    (1 for 4-digit, 2 for 5-digit). For lexicon checks we want the
    raw digit string with no comma — this helper builds it from the
    per-position digit chars by ignoring slot indexing entirely (the
    list is already digit-only since commas live in their own slot).

    Kept as a separate helper so :func:`_backtrack_with_lexicon`
    composes the same way the main scoring path does. Currently a
    pure concatenation; the ``comma_position`` parameter is preserved
    for symmetry with the scoring path's signature should we need it.
    """
    del comma_position  # unused — kept for parity with main path
    return "".join(digit_chars)


def _backtrack_with_lexicon(
    cnn_topk: list[list[tuple[str, float]]],
    comma_position: int,
    lexicon: set[int],
    *,
    n_digits_expected: int,
) -> Optional[tuple[str, float]]:
    """Try single-position swaps to reach a lexicon-valid composition.

    Parameters
    ----------
    cnn_topk : list of lists
        For each digit position, the CNN's top-K classifications as
        ``[(char, conf), (char, conf), ...]`` ordered by descending
        confidence. Top-2 is sufficient for almost all real cases —
        the CNN's softmax landscape rarely puts the truth at rank 3+.
    comma_position : int
        Comma slot index (1 for 4-digit, 2 for 5-digit). Used only to
        keep the helper signature ergonomic if a caller wants the
        composed-with-comma form back; the actual lexicon check is on
        the digit-only integer.
    lexicon : set of int
        Known signature integers. Empty / None means "no lexicon" —
        the helper returns None in that case.
    n_digits_expected : int
        Expected digit count (4 or 5). Compositions of a different
        length are rejected before lexicon lookup.

    Strategy
    --------
    1. If the top-1 composition is already lexicon-valid, return it.
    2. Otherwise, try EACH single-position swap — replace one
       position's top-1 with that position's top-2 char. N positions
       × 1 swap each = N candidates.
    3. Among lexicon-matching swap candidates, return the one with
       the highest mean confidence (the swap that minimally hurts
       overall confidence).
    4. Returns None when no single swap produces a lexicon match.

    The scope is intentionally limited to single swaps. Multi-swap
    backtracking opens an exponential search and risks "cleverly"
    composing an unrelated lexicon value out of two coincidentally-
    plausible alternatives. Single swaps are the conservative
    correction the failure-mode demands: ONE leading or trailing
    digit was wrong; the rest read fine.
    """
    if not cnn_topk or not lexicon:
        return None
    if len(cnn_topk) != n_digits_expected:
        return None
    # Each position must have a non-empty top list. Skip
    # backtracking entirely if any position is missing data.
    for pos in cnn_topk:
        if not pos:
            return None

    top1_chars = [pos[0][0] for pos in cnn_topk]
    top1_confs = [pos[0][1] for pos in cnn_topk]

    def _digits_only(chars: list[str]) -> Optional[str]:
        if all(c.isdigit() for c in chars):
            return "".join(chars)
        return None

    # Quick path: top-1 already lexicon-valid.
    top1_str = _digits_only(top1_chars)
    if top1_str is not None and len(top1_str) == n_digits_expected:
        try:
            if int(top1_str) in lexicon:
                return (top1_str, float(np.mean(top1_confs)))
        except Exception:
            pass

    # Single-position swap candidates.
    # For each position, walk top-K alternatives starting at rank 2
    # and use the FIRST DIGIT alternative encountered. Earlier
    # behaviour always took rank-2 (``cnn_topk[swap_idx][1][0]``) —
    # but on Stage 8 misclassifications the rank-2 character is
    # often the icon class ``@``, which fails ``_digits_only`` and
    # disqualifies the candidate before it can hit the lexicon.
    # The profiler showed this pattern dominates Stage 8 failures
    # (38.6% of all failures): the CNN's rank-1 was wrong, rank-2
    # was ``@``, real digit truth was at rank 3+. Iterating through
    # the full top-K and skipping non-digits unlocks rank-3+
    # candidates and lets the lexicon backtracker actually work
    # on those captures.
    candidates: list[tuple[str, float]] = []
    for swap_idx in range(len(cnn_topk)):
        topk_pos = cnn_topk[swap_idx]
        if len(topk_pos) < 2:
            continue
        # Find the first DIGIT alternative at rank 2+.
        alt_char: Optional[str] = None
        alt_conf: float = 0.0
        for rank in range(1, len(topk_pos)):
            ch, cf = topk_pos[rank]
            if ch.isdigit():
                alt_char = ch
                alt_conf = float(cf)
                break
        if alt_char is None:
            continue
        alt_chars = list(top1_chars)
        alt_chars[swap_idx] = alt_char
        alt_str = _digits_only(alt_chars)
        if alt_str is None or len(alt_str) != n_digits_expected:
            continue
        try:
            if int(alt_str) not in lexicon:
                continue
        except Exception:
            continue
        confs = list(top1_confs)
        confs[swap_idx] = alt_conf
        avg = float(np.mean(confs))
        candidates.append((alt_str, avg))

    if not candidates:
        return None
    # Pick highest mean confidence among lexicon-matching swaps.
    candidates.sort(key=lambda x: -x[1])
    del comma_position  # unused parameter retained for caller ergonomics
    return candidates[0]


def _detect_comma_extent(
    gray_canon: np.ndarray,
) -> Optional[tuple[int, int]]:
    """Locate the comma's column extent by its bottom-only ink signature.

    A comma renders only in the bottom ~30% of the line height. So a
    column with non-trivial ink in the bottom-third rows AND zero ink
    in the top-third + middle-third rows is uniquely a comma column.
    Digits, even narrow ones like ``1``, always ink the top + middle
    rows.

    This is the most reliable structural anchor in the entire crop:
    once we know where the comma is, the structural prior (4-digit
    has comma after position 0; 5-digit has comma after position 1)
    tells us how the rest of the digits are laid out.

    Returns ``(start_col, end_col_exclusive)`` of the comma's column
    extent, or ``None`` if no comma-shaped column run was detected
    (handles degenerate captures where the comma was cropped off or
    the polarity detection went wrong).
    """
    h, w = gray_canon.shape[:2]
    if h < 6 or w < 8:
        return None

    g_min = int(gray_canon.min())
    g_max = int(gray_canon.max())
    if g_max - g_min < 16:
        return None
    thr = g_min + (g_max - g_min) // 2
    bright = gray_canon > thr

    # Top third / middle third / bottom third partitions.
    top_end = h // 3
    mid_end = 2 * h // 3

    # A column is "comma-like" iff:
    #   (a) bottom-third has ≥ 2 bright pixels (real ink), AND
    #   (b) top-third has zero bright pixels, AND
    #   (c) middle-third has ≤ 1 bright pixel (the comma's tail can
    #       reach into the very bottom of the middle band).
    is_comma_col = np.zeros(w, dtype=bool)
    for x in range(w):
        col = bright[:, x]
        bot = col[mid_end:].sum()
        top = col[:top_end].sum()
        mid = col[top_end:mid_end].sum()
        if bot >= 2 and top == 0 and mid <= 1:
            is_comma_col[x] = True

    # Find the longest contiguous run of comma columns. Allowing 0-px
    # gaps gives us tolerance for chromatic-aberration's tendency to
    # punch a single-column hole through the comma's centroid.
    best: Optional[tuple[int, int]] = None
    best_len = 0
    s = -1
    for x in range(w):
        if is_comma_col[x]:
            if s < 0:
                s = x
            e = x + 1
        else:
            if s >= 0:
                if e - s > best_len:
                    best_len = e - s
                    best = (s, e)
                s = -1
    if s >= 0 and (e - s > best_len):
        best = (s, e)

    # Reject if the run is implausibly narrow (< 2 px) — likely speckle.
    if best is None or (best[1] - best[0]) < 2:
        return None
    return best


def _find_blobs(
    gray_canon: np.ndarray,
) -> tuple[list[tuple[int, int, int]], list[float]]:
    """Find ink blob extents in column-projection.

    Returns ``(blobs, vert_centroids)`` where:
      * ``blobs`` is a list of ``(start_col, end_col_exclusive, peak_density)``
      * ``vert_centroids`` is a parallel list of the y-centroid (as a
        fraction of crop height) of each blob's ink mass — used to
        discriminate commas (low-only ink) from digits (full-height).

    The detection threshold is ``_INK_COL_FRAC`` of the crop height.
    Adjacent ink columns separated by < ``_MIN_BLOB_GAP_PX`` are merged
    into a single blob (handles chromatic-aberration vertical streaks
    that briefly drop a column's stroke density inside a digit).
    """
    h, w = gray_canon.shape[:2]
    if h < 4 or w < 4:
        return [], []

    # Threshold the canonicalized grayscale at the midpoint of its
    # dynamic range. Per-crop adaptive — works on any pill colour.
    g_min = int(gray_canon.min())
    g_max = int(gray_canon.max())
    if g_max - g_min < 16:
        return [], []
    thr = g_min + (g_max - g_min) // 2
    bright = gray_canon > thr  # bright = text in canonical polarity

    col_counts = bright.sum(axis=0)
    col_thr = max(1, int(h * _INK_COL_FRAC))

    # Walk through columns merging spans with gaps < MIN_BLOB_GAP_PX.
    blobs: list[tuple[int, int, int]] = []
    in_run = False
    run_start = 0
    last_inked_col = -10
    for x in range(w):
        if col_counts[x] >= col_thr:
            if not in_run:
                # Maybe extend the previous blob if the gap is small.
                if blobs and (x - last_inked_col) <= _MIN_BLOB_GAP_PX:
                    # Merge: pop and restart from prev start.
                    prev_s, prev_e, prev_peak = blobs.pop()
                    run_start = prev_s
                else:
                    run_start = x
                in_run = True
            last_inked_col = x
        else:
            if in_run and (x - last_inked_col) > _MIN_BLOB_GAP_PX:
                end = last_inked_col + 1
                peak = int(col_counts[run_start:end].max())
                blobs.append((run_start, end, peak))
                in_run = False
    if in_run:
        end = last_inked_col + 1
        peak = int(col_counts[run_start:end].max())
        blobs.append((run_start, end, peak))

    # Compute y-centroid for each blob (fraction of crop height).
    vert_centroids: list[float] = []
    for s, e, _peak in blobs:
        slice_ink = bright[:, s:e]
        if slice_ink.sum() == 0:
            vert_centroids.append(0.5)
            continue
        ys = np.arange(h).reshape(-1, 1)
        y_centroid = float((slice_ink * ys).sum() / slice_ink.sum())
        vert_centroids.append(y_centroid / max(1, h - 1))

    return blobs, vert_centroids


def _identify_comma_blob(
    blobs: list[tuple[int, int, int]],
    vert_centroids: list[float],
    expected_comma_pos: int,
) -> Optional[int]:
    """Pick which blob is the comma.

    A comma's ink mass sits in the bottom of the line — its y-centroid
    is well below the digits' ~0.5 centroid. Combined with width (commas
    are narrower than digits), this gives a reliable discriminator.

    Returns the blob index judged most likely to be the comma, or
    ``None`` if no blob looks comma-like.

    The ``expected_comma_pos`` parameter is the comma's expected
    left-to-right position (1 for 4-digit, 2 for 5-digit). When two
    blobs both have comma-like properties (rare but possible on
    aberration-fused crops), we prefer the one closest to the
    expected position. This biases ambiguous reads toward the
    structural prior without requiring it.
    """
    if not blobs:
        return None

    n_blobs = len(blobs)
    # Score each blob's "comma-ness". Higher = more likely a comma.
    scores: list[float] = []
    widths = [e - s for s, e, _ in blobs]
    median_w = float(np.median(widths)) if widths else 1.0
    for i, ((s, e, peak), vc) in enumerate(zip(blobs, vert_centroids)):
        w = e - s
        # Narrower than the median digit width → more comma-like.
        narrow = max(0.0, (median_w - w) / max(1.0, median_w))
        # Y-centroid below 0.55 → more comma-like (commas live low).
        low = max(0.0, vc - 0.55) * 4.0  # vc=0.85 -> low=1.2
        # Position prior — closer to expected pos = +0.1 boost.
        pos_bonus = 0.1 if i == expected_comma_pos else 0.0
        scores.append(narrow + low + pos_bonus)

    best = int(np.argmax(scores))
    # Hard floor: a "comma" must be narrower than 0.6× median width
    # OR have y-centroid ≥ 0.6. Otherwise no comma was found.
    s, e, _ = blobs[best]
    w = e - s
    if (w >= 0.6 * median_w) and (vert_centroids[best] < 0.6):
        return None
    return best


def _bbox_from_blob(
    gray_canon: np.ndarray,
    blob: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    """Convert a column-span blob into a (x, y, w, h) bbox.

    Inflates the column extent by ``_DIGIT_BBOX_INFLATE`` to give the
    CNN's 28×28 padding a small margin. Y is the full crop height
    (bbox row-isolation isn't necessary because the runtime preprocess
    already row-isolates).
    """
    h, w = gray_canon.shape[:2]
    s, e, _ = blob
    span_w = e - s
    inflated = int(round(span_w * _DIGIT_BBOX_INFLATE))
    margin = max(0, (inflated - span_w) // 2)
    new_s = max(0, s - margin)
    new_e = min(w, e + (inflated - span_w - margin))
    return (new_s, 0, max(1, new_e - new_s), h)


def _propose_layout_fallback(
    crop_w: int,
    crop_h: int,
    n_digits: int,
    *,
    ink_left: int = 0,
    ink_right: Optional[int] = None,
) -> tuple[
    list[tuple[int, int, int, int]],
    tuple[int, int, int, int],
    float,
    float,
]:
    """Pure-proportional layout used when blob detection fails.

    The ink-bearing range ``[ink_left, ink_right)`` is divided into
    ``n_digits`` digit pitches + 1 comma pitch (~0.40 × digit pitch).
    Each digit bbox is centered in its slot; same for the comma.

    This is the OLD layout, kept for the case where chromatic
    aberration fuses every digit into a single column-projection blob
    (zero blobs detected). It's less accurate than blob-binding but
    better than nothing.
    """
    if ink_right is None:
        ink_right = crop_w

    region_left = float(ink_left)
    region_w = max(1.0, float(ink_right - ink_left))

    digit_pitch = region_w / (n_digits + _FALLBACK_COMMA_PITCH_FRAC)
    comma_pitch = digit_pitch * _FALLBACK_COMMA_PITCH_FRAC
    comma_idx = 1 if n_digits == 4 else 2

    digit_w = digit_pitch * _FALLBACK_DIGIT_BBOX_FRAC
    comma_w = comma_pitch * _FALLBACK_COMMA_BBOX_FRAC

    y = 0
    h = crop_h

    digit_bboxes: list[tuple[int, int, int, int]] = []
    comma_bbox: Optional[tuple[int, int, int, int]] = None

    cursor = region_left
    for slot_idx in range(n_digits + 1):
        if slot_idx == comma_idx:
            slot_w = comma_pitch
            slot_center = cursor + slot_w * 0.5
            cx = int(round(slot_center - comma_w * 0.5))
            cw = int(round(comma_w))
            comma_bbox = (
                max(0, cx),
                int(y),
                max(1, min(cw, crop_w - max(0, cx))),
                int(h),
            )
            cursor += slot_w
        else:
            slot_w = digit_pitch
            slot_center = cursor + slot_w * 0.5
            dx = int(round(slot_center - digit_w * 0.5))
            dw = int(round(digit_w))
            digit_bboxes.append((
                max(0, dx),
                int(y),
                max(1, min(dw, crop_w - max(0, dx))),
                int(h),
            ))
            cursor += slot_w

    assert comma_bbox is not None
    return digit_bboxes, comma_bbox, digit_w, comma_w


def _find_ink_extent(
    gray_canon: np.ndarray,
) -> tuple[int, int]:
    """Find the leftmost and rightmost columns containing digit ink.

    Uses the ``_INK_COL_FRAC`` height threshold from the blob detector
    but doesn't try to split into runs — just measures the union
    extent. This becomes the layout's effective "drawing area" so
    proportional slot positions are computed over the actual digit
    cluster, not the bbox padding.
    """
    h, w = gray_canon.shape[:2]
    if h < 4 or w < 4:
        return (0, w)

    g_min = int(gray_canon.min())
    g_max = int(gray_canon.max())
    if g_max - g_min < 16:
        return (0, w)
    thr = g_min + (g_max - g_min) // 2
    bright = gray_canon > thr
    col_counts = bright.sum(axis=0)
    col_thr = max(1, int(h * _INK_COL_FRAC))
    ink_cols = np.where(col_counts >= col_thr)[0]
    if ink_cols.size == 0:
        return (0, w)
    left = int(ink_cols[0])
    right = int(ink_cols[-1]) + 1
    return (left, right)


def _refine_center_to_ink(
    gray_canon: np.ndarray,
    proposed_x_center: float,
    search_radius: int = 4,
    *,
    avoid_columns: Optional[tuple[int, int]] = None,
) -> int:
    """Snap a proportional digit center to the local ink centroid.

    Pure-proportional placement assumes evenly-spaced digits, but the
    SC signature font has variable-width glyphs (a `1` is ~3 px; a `0`
    is ~10-12 px) and inter-digit kerning isn't uniform. After
    proportional layout decides each digit's APPROXIMATE position, this
    helper looks at the polarity-canonical brightness within a small
    window around that position and returns the column-weighted centroid
    of bright pixels — i.e., the actual ink center.

    Returns the proportional x_center unchanged when:
      * the search window has < 2 px of ink (no signal to refine to)
      * the gray array is too low-contrast to threshold

    The ``avoid_columns`` arg masks out the comma's column extent so a
    digit's refinement window can't pull toward comma ink.
    """
    h, w = gray_canon.shape[:2]
    if h < 4 or w < 4:
        return int(proposed_x_center)

    cmin = max(0, int(proposed_x_center) - search_radius)
    cmax = min(w, int(proposed_x_center) + search_radius + 1)
    if cmin >= cmax:
        return int(proposed_x_center)

    g_min = int(gray_canon.min())
    g_max = int(gray_canon.max())
    if g_max - g_min < 16:
        return int(proposed_x_center)
    thr = g_min + (g_max - g_min) // 2
    bright = gray_canon > thr  # bright = ink (post polarity-canon)
    col_counts = bright.sum(axis=0).astype(np.float32)

    if avoid_columns is not None:
        cl, cr = avoid_columns
        cl = max(0, int(cl))
        cr = min(w, int(cr))
        if cl < cr:
            col_counts[cl:cr] = 0.0

    window = col_counts[cmin:cmax]
    if window.sum() < 2.0:
        return int(proposed_x_center)

    indices = np.arange(cmin, cmax, dtype=np.float32)
    refined = float((window * indices).sum() / window.sum())
    return int(round(refined))


def _refine_digit_bbox_to_ink(
    gray_canon: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    avoid_columns: Optional[tuple[int, int]] = None,
    search_radius: int = 4,
) -> tuple[int, int, int, int]:
    """Recenter a digit bbox to the local ink centroid.

    Keeps the bbox WIDTH unchanged (proportional segmenter already
    chose a sensible width based on the slot) but slides x so the
    bbox is centered on actual digit ink. Stays within the image
    bounds.
    """
    x, y, w_bb, h_bb = bbox
    h, w = gray_canon.shape[:2]
    proposed_center = x + w_bb / 2.0
    refined_center = _refine_center_to_ink(
        gray_canon,
        proposed_center,
        search_radius=search_radius,
        avoid_columns=avoid_columns,
    )
    new_x = int(round(refined_center - w_bb / 2.0))
    new_x = max(0, min(new_x, w - 1))
    new_w = max(1, min(w_bb, w - new_x))
    return (new_x, y, new_w, h_bb)


def _enforce_non_overlap(
    digit_bboxes: list[tuple[int, int, int, int]],
    comma_bbox: Optional[tuple[int, int, int, int]],
    crop_w: int,
) -> list[tuple[int, int, int, int]]:
    """After ink-refinement, ensure adjacent digit bboxes don't overlap.

    If two adjacent refined positions ended up too close (refinement
    pulled both toward the same ink mass), nudge them apart symmetrically
    around their midpoint. Also keep digits clear of the comma's column
    extent (refinement masks comma columns but a bbox's tail can still
    overlap them).
    """
    if not digit_bboxes:
        return digit_bboxes

    sorted_idx = sorted(range(len(digit_bboxes)), key=lambda i: digit_bboxes[i][0])
    bboxes = list(digit_bboxes)

    for k in range(1, len(sorted_idx)):
        prev_i = sorted_idx[k - 1]
        cur_i = sorted_idx[k]
        px, py, pw, ph = bboxes[prev_i]
        cx, cy, cw, ch = bboxes[cur_i]
        prev_right = px + pw
        if prev_right > cx:
            # Overlap — push them apart at the midpoint
            overlap = prev_right - cx
            shift = (overlap + 1) // 2
            new_px = max(0, px - shift)
            new_pw = pw  # keep width
            new_cx = min(crop_w - cw, cx + shift)
            bboxes[prev_i] = (new_px, py, new_pw, ph)
            bboxes[cur_i] = (new_cx, cy, cw, ch)

    if comma_bbox is not None:
        ccx, _, ccw, _ = comma_bbox
        comma_left = ccx
        comma_right = ccx + ccw
        for k in range(len(bboxes)):
            x, y, w_bb, h_bb = bboxes[k]
            digit_right = x + w_bb
            digit_left = x
            # If the digit overlaps the comma extent on its right side,
            # clip the bbox so it stops just before the comma. Same on
            # the left side. Width is reduced — the CNN can still read
            # a slightly narrower crop because the inflated digit slot
            # had margin.
            if digit_left < comma_right and digit_right > comma_left:
                # Determine which side of comma this digit is on by
                # which has more overlap
                if digit_left < comma_left:
                    # Digit is on the LEFT of comma; clip right side
                    new_w = max(1, comma_left - digit_left)
                    bboxes[k] = (digit_left, y, new_w, h_bb)
                else:
                    # Digit is on the RIGHT of comma; clip left side
                    new_left = comma_right
                    new_w = max(1, w_bb - (new_left - digit_left))
                    bboxes[k] = (new_left, y, new_w, h_bb)

    return bboxes


def _propose_layout_from_kerning(
    crop_w: int,
    crop_h: int,
    comma_extent: tuple[int, int],
    n_digits: int,
    kerning_model: dict[str, Any],
) -> Optional[tuple[
    list[tuple[int, int, int, int]],
    tuple[int, int, int, int],
    float,
    float,
]]:
    """Project digit + comma bboxes from kerning offsets.

    For each slot index k in {0..4} that's part of the hypothesis, the
    digit center is at::

        center_x = comma_center + slot[k].center_offset_median * row_h

    Slot 0 is included only when ``n_digits == 5`` (the leftmost-of-5
    digit position). Slots 1..4 are always included. Each digit bbox
    is sized by ``slot[k].digit_w_median * row_h * _FALLBACK_DIGIT_BBOX_FRAC``
    — same gutter convention as the existing equal-slots tier so the
    CNN's 28×28 crop has consistent padding regardless of which layout
    tier produced the bbox.

    Returns ``None`` if any required slot's projected bbox falls
    completely outside the crop (caller falls through to next tier).
    """
    slots = kerning_model.get("slots") or {}
    needed_slot_keys = ["1", "2", "3", "4"]
    if n_digits == 5:
        needed_slot_keys = ["0"] + needed_slot_keys
    for s in needed_slot_keys:
        if s not in slots:
            return None

    cl, cr = comma_extent
    comma_center = 0.5 * (float(cl) + float(cr))
    row_h = float(crop_h)
    if row_h <= 0:
        return None

    # Compute a single bbox width to use across ALL slots. We DON'T
    # use the kerning model's per-slot ``digit_w_median`` directly
    # because those measurements came from the legacy segmenter's
    # bboxes, which are tight on the ink — already narrower than the
    # underlying digit pitch. Combining tight per-slot widths with
    # the projected slot centers means a small ink-position drift
    # (within ±sigma of the projected center, ~6 px on a 40 px row)
    # pushes digit content outside the bbox; the CNN sees a half-
    # digit and reads garbage, killing the hypothesis's mean_conf.
    #
    # Instead we derive an effective digit pitch from the kerning
    # offsets themselves (slot[k+1] - slot[k] in row-height units),
    # take the median, and multiply by the standard gutter fraction.
    # This produces wider slot bboxes that comfortably accept the
    # ±sigma drift while still leaving enough gap between adjacent
    # slot bboxes to avoid neighbour-digit bleed.
    pitches: list[float] = []
    sorted_slot_keys = sorted(slots.keys())
    for i in range(len(sorted_slot_keys) - 1):
        k_i = sorted_slot_keys[i]
        k_j = sorted_slot_keys[i + 1]
        info_i = slots[k_i]
        info_j = slots[k_j]
        off_i = float(info_i.get("center_offset_median", 0.0))
        off_j = float(info_j.get("center_offset_median", 0.0))
        delta = off_j - off_i
        # Skip the pre-comma → post-comma pitch (slot 1 → slot 2):
        # it's inflated by the comma's width and isn't a digit-to-
        # digit pitch we want to average in.
        if k_i == "1" and k_j == "2":
            continue
        if delta > 0.05:  # sanity: ignore degenerate / inverted entries
            pitches.append(delta)
    if not pitches:
        return None
    pitches_sorted = sorted(pitches)
    pitch_unit = pitches_sorted[len(pitches_sorted) // 2]
    pitch_px = max(6.0, pitch_unit * row_h)
    # Use the standard gutter ratio for the bbox-vs-slot relationship,
    # consistent with Tier 1 (equal-slots) behaviour.
    bbox_w_px = max(6.0, pitch_px * _FALLBACK_DIGIT_BBOX_FRAC)

    digit_bboxes: list[tuple[int, int, int, int]] = []
    for s in needed_slot_keys:
        info = slots[s]
        offset_unit = float(
            info.get("center_offset_median",
                     info.get("center_offset_mean", 0.0))
        )
        center_x = comma_center + offset_unit * row_h
        dx = int(round(center_x - bbox_w_px * 0.5))
        bw = int(round(bbox_w_px))
        # Clamp to crop edges. Reject the whole layout if any slot
        # falls fully outside (kerning model implies the digit cluster
        # extends past the crop_box — caller should re-extend).
        if dx >= crop_w or dx + bw <= 0:
            return None
        dx_clamped = max(0, dx)
        bw_clamped = max(1, min(bw, crop_w - dx_clamped))
        digit_bboxes.append((dx_clamped, 0, bw_clamped, int(crop_h)))

    comma_bbox = (
        int(cl),
        0,
        int(max(1, cr - cl)),
        int(crop_h),
    )
    digit_widths = [bb[2] for bb in digit_bboxes]
    digit_w_avg = float(np.mean(digit_widths)) if digit_widths else 0.0
    comma_w_actual = float(comma_bbox[2])
    return digit_bboxes, comma_bbox, digit_w_avg, comma_w_actual


def _build_layout_for_hypothesis(
    gray_canon: np.ndarray,
    n_digits: int,
    *,
    comma_extent: Optional[tuple[int, int]],
    ink_extent: tuple[int, int],
) -> tuple[
    list[tuple[int, int, int, int]],
    tuple[int, int, int, int],
    float,
    float,
    bool,  # used_comma_anchor
]:
    """Layout digit + comma bboxes for one hypothesis (4 or 5 digits).

    Two-tier strategy:

    **Tier 1 (comma-anchored, accurate)**: When the comma extent has
    been reliably detected from the bottom-only-ink signature, we
    anchor the layout to it. The structural prior tells us:

        4-digit (D , DDD): 1 digit slot left of comma + 3 right
        5-digit (DD, DDD): 2 digit slots left of comma + 3 right

    Each side of the comma is divided into its expected digit count.
    The left-region width is ``comma_left - ink_left`` and is split
    into ``L`` equal slots; the right-region width is
    ``ink_right - comma_right`` and is split into 3 equal slots
    (always 3 digits to the right of the comma). Each digit bbox is
    centered in its slot and uses ``_FALLBACK_DIGIT_BBOX_FRAC`` of
    the slot width to give the CNN's 28×28 padding a small gutter.

    **Tier 2 (pure-proportional fallback)**: When no comma is
    detected (degenerate capture, polarity error), we fall back to
    the equal-slots-with-narrow-comma layout over the ink extent.
    Less accurate but never crashes.
    """
    h, w = gray_canon.shape[:2]
    ink_left, ink_right = ink_extent

    # NOTE: a kerning-locked Tier 0 was tried and reverted (see
    # ``_propose_layout_from_kerning`` + ``kerning_model.json``).
    # Empirically it regressed Stage 6 (wrong n_digits) on the 106-
    # capture profile (18 → 30 failures, accuracy 17.0% → 9.4%) —
    # because the kerning's slot[0] is metric-projected from the
    # comma but the actual leading-digit's center drifts within
    # ±sigma (~6 px on a 40 px row) of that projection. When the
    # leading digit drifts outside the projected bbox, the CNN reads
    # garbage on slot[0], the empty-slot penalty fires, and the 4-
    # digit hypothesis wins by default. The equal-slots Tier 1 below
    # has a hidden advantage: ``ink_left`` tracks the actual leading-
    # digit position, so slot[0] is anchored on real ink.
    #
    # The kerning model is preserved on disk for a future ink-snap
    # variant (project slot center from kerning, then snap to local
    # ink centroid within ±sigma) but is NOT a default tier today.
    if comma_extent is not None:
        left_n = 1 if n_digits == 4 else 2  # digits to the LEFT of comma
        right_n = 3                         # digits to the RIGHT of comma
        comma_l, comma_r = comma_extent
        left_region_w = float(max(0, comma_l - ink_left))
        right_region_w = float(max(0, ink_right - comma_r))

        if left_region_w >= max(4.0, left_n * 4.0) and right_region_w >= 12.0:
            digit_bboxes: list[tuple[int, int, int, int]] = []
            # Left side: ``left_n`` equal slots from ink_left to comma_l
            left_slot = left_region_w / left_n
            for k in range(left_n):
                slot_center = ink_left + (k + 0.5) * left_slot
                slot_w_used = left_slot
                dw = slot_w_used * _FALLBACK_DIGIT_BBOX_FRAC
                dx = int(round(slot_center - dw * 0.5))
                digit_bboxes.append((
                    max(0, dx),
                    0,
                    max(1, min(int(round(dw)), w - max(0, dx))),
                    h,
                ))
            # Right side: 3 equal slots from comma_r to ink_right
            right_slot = right_region_w / right_n
            for k in range(right_n):
                slot_center = comma_r + (k + 0.5) * right_slot
                slot_w_used = right_slot
                dw = slot_w_used * _FALLBACK_DIGIT_BBOX_FRAC
                dx = int(round(slot_center - dw * 0.5))
                digit_bboxes.append((
                    max(0, dx),
                    0,
                    max(1, min(int(round(dw)), w - max(0, dx))),
                    h,
                ))
            comma_bbox = (
                int(comma_l),
                0,
                int(max(1, comma_r - comma_l)),
                h,
            )
            # NOTE: ink-aware refinement helpers (_refine_center_to_ink,
            # _refine_digit_bbox_to_ink, _enforce_non_overlap) are
            # defined above for future use, but invocation was reverted
            # after empirical testing showed they regressed working
            # captures. The proportional layout's positions are
            # already accurate enough on captures where the upstream
            # crop_box and comma detection are correct; refinement was
            # over-correcting and pulling digit centers into adjacent
            # ink masses on visually-clean captures (e.g.
            # cap_20260418_160452_795 GT 16,960 went from correct read
            # to dropped-digit read). The pure proportional positions
            # (no snap-to-ink) are kept as-is.
            digit_widths = [bb[2] for bb in digit_bboxes]
            digit_w_avg = float(np.mean(digit_widths)) if digit_widths else 0.0
            comma_w_actual = float(comma_bbox[2])
            return digit_bboxes, comma_bbox, digit_w_avg, comma_w_actual, True

    # Tier 2: pure-proportional fallback over the ink extent.
    digit_bboxes, comma_bbox, digit_w, comma_w = _propose_layout_fallback(
        w, h, n_digits, ink_left=ink_left, ink_right=ink_right,
    )
    # NOTE: ink-aware refinement reverted here too; see comment above.
    return digit_bboxes, comma_bbox, digit_w, comma_w, False


def _tighten_bboxes_to_ink(
    gray_canon: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Contract each bbox horizontally to its actual ink extent.

    Mirrors Glyph Forge's Layer H tightening, ported into the
    production segmenter. For each bbox, compute the column-ink
    density within the bbox (using polarity-canonicalized work),
    find the leftmost/rightmost columns whose ink coverage clears
    25% of the row height, and shrink the bbox to that range +
    1-px margin. Skip tiles that:
      * have no ink columns within their bbox (no-op),
      * would shrink to less than 4 px wide (sliver guard),
      * produce no change (already tight).

    Returns ``(new_bboxes, n_tightened)`` where ``n_tightened`` is
    the count of bboxes that actually shrank.

    Why this matters: Tesseract / blob detection / kerning all
    produce bboxes that can include comma-edge halo, leading icon
    residue, or trailing pill-background pixels. The CNN was
    trained on tight digit crops; bboxes with extra background
    distort the resize-to-28×28 step and move inputs out of
    distribution. Tightening eliminates the contamination so the
    CNN sees clean digit crops. Empirically fixed the
    cap_20260418_085431_378 "5 -> 3" case in Glyph Forge by
    contracting tile[2]'s left edge from x=117 (comma halo) to
    x=122 (real "5" body).
    """
    if gray_canon is None or gray_canon.size == 0 or not bboxes:
        return list(bboxes), 0

    # Polarity already canonicalized upstream by _canonicalize_polarity_local
    # — bright glyphs on dark bg. So inky columns just need the
    # threshold applied directly.
    binary = (gray_canon > 128).astype(np.uint8)
    col_density = binary.sum(axis=0)
    H = gray_canon.shape[0]
    abs_thr = max(1, int(H * 0.25))

    new_bboxes: list[tuple[int, int, int, int]] = []
    n_tightened = 0
    for (x, y, w, h) in bboxes:
        x1 = max(0, int(x))
        x2 = min(gray_canon.shape[1], int(x + w))
        sub = col_density[x1:x2] if x2 > x1 else np.array([], dtype=np.int32)
        if sub.size == 0:
            new_bboxes.append((x, y, w, h))
            continue
        ink_mask = sub >= abs_thr
        if not ink_mask.any():
            new_bboxes.append((x, y, w, h))
            continue
        ink_cols = np.where(ink_mask)[0]
        first_ink = int(ink_cols[0])
        last_ink = int(ink_cols[-1])
        margin = 1
        nx1 = x1 + max(0, first_ink - margin)
        nx2 = x1 + min(sub.size, last_ink + 1 + margin)
        nw = nx2 - nx1
        if nw < 4:
            new_bboxes.append((x, y, w, h))
            continue
        if nx1 == x1 and nx2 == x2:
            new_bboxes.append((x, y, w, h))
            continue
        new_bboxes.append((nx1, y, nw, h))
        n_tightened += 1
    return new_bboxes, n_tightened


def _crop_to_28x28(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Convert a (x,y,w,h) crop of a polarity-canonicalized grayscale
    array into the 28x28 float32 [0,1] form the signal CNN expects.

    Mirrors the tail of ``_segment_glyphs``: pad with 255 (white,
    matching the bright-text-on-dark-then-padded training convention),
    bilinear-resize to 28x28, divide by 255.
    """
    x, y, w, h = bbox
    h_full, w_full = gray.shape[:2]
    x = max(0, min(x, w_full - 1))
    y = max(0, min(y, h_full - 1))
    w = max(1, min(w, w_full - x))
    h = max(1, min(h, h_full - y))
    crop = gray[y:y + h, x:x + w].astype(np.float32)
    pad = 2
    padded = np.full(
        (crop.shape[0] + pad * 2, crop.shape[1] + pad * 2),
        255.0, dtype=np.float32,
    )
    padded[pad:pad + crop.shape[0], pad:pad + crop.shape[1]] = crop
    pil = Image.fromarray(padded.astype(np.uint8)).resize(
        (28, 28), Image.BILINEAR,
    )
    return np.array(pil, dtype=np.float32) / 255.0


def _score_hypothesis(
    gray_canon: np.ndarray,
    comma_extent: Optional[tuple[int, int]],
    ink_extent: tuple[int, int],
    n_digits: int,
    classifier: Optional[Callable[[list[np.ndarray]], list[tuple[str, float]]]],
    lexicon: Optional[set[int]],
    *,
    classifier_topk: Optional[
        Callable[[list[np.ndarray]], list[list[tuple[str, float]]]]
    ] = None,
) -> _Hypothesis:
    """Build per-slot bboxes for a given digit count, run the CNN, and
    score the composed string.

    Score formula (post Fix 1 + Fix 2):

        score = mean_cnn_confidence
              + (LEXICON_BONUS if in_lexicon else 0)
              + empty_slot_penalty       # ≤ 0
              + (LEXICON_BONUS if backtracked else 0)

    The empty-slot penalty (Fix 1) is computed BEFORE running the CNN —
    it's a structural property of where the bboxes landed, not of what
    the CNN reads. Hypotheses with one or more pill-background slots
    get -0.30/slot penalty (or -0.50 if degenerate), which is large
    enough to flip a hypothesis when the competing one has all
    populated slots.

    The lexicon backtracking (Fix 2) runs AFTER the CNN top-K but
    BEFORE final composition. If the CNN's top-1 read produces an
    out-of-lexicon string, the backtracker probes single-position
    swaps to top-2 and adopts a lexicon-valid alternative if found.
    The adoption gets the same +LEXICON_BONUS the in-lexicon top-1
    would have received, so a backtracked-correct hypothesis competes
    fairly with a non-backtracked correct hypothesis.
    """
    digit_bboxes, comma_bbox, digit_w, comma_w, used_anchor = (
        _build_layout_for_hypothesis(
            gray_canon, n_digits,
            comma_extent=comma_extent,
            ink_extent=ink_extent,
        )
    )

    # NOTE: ``_tighten_bboxes_to_ink`` (defined above) is available
    # but NOT applied here. Empirically the Glyph Forge tighten-to-ink
    # pass that worked there (19/20 on test captures) regressed
    # production accuracy 17.0% -> 15.1% on the 106-capture profile,
    # because production's ``gray_canon`` has different polarity /
    # contrast characteristics than Glyph Forge's source gray, and
    # the 25% vertical-coverage threshold cuts into real digit ink
    # on borderline captures (Stage 6 jumped 18->21). The function
    # is kept on disk for a future, more carefully-thresholded port.

    # ── Fix 1: empty-slot penalty (structural, pre-CNN) ───────────
    # Compute against the polarity-canonical grayscale so the
    # threshold is independent of pill colour.
    empty_pen, slot_masses, n_empty = _empty_slot_penalty(
        gray_canon, digit_bboxes,
    )
    # An additional penalty bucket activated post-CNN: positions the
    # CNN tags as the icon class '@' at high confidence are virtually
    # always pill-background slots (the icon class is the CNN's
    # learned anchor for "this is not a digit"). The structural ink-
    # mass test misses these when the bbox happens to clip a halo
    # bright enough to clear the median-mass threshold but not
    # actually contain a digit. We add this as a secondary check
    # below after CNN runs.

    # ── CNN classification (top-K when available) ────────────────
    classifications: list[tuple[str, float]] = []
    cnn_topk: list[list[tuple[str, float]]] = []
    if (classifier is not None or classifier_topk is not None) and digit_bboxes:
        try:
            crops = [_crop_to_28x28(gray_canon, b) for b in digit_bboxes]
        except Exception:
            crops = []
        if crops:
            if classifier_topk is not None:
                try:
                    # Request 4 ranks (default is 2) so the lexicon
                    # backtracker has rank-3+ candidates to try when
                    # rank-2 is the icon class ``@`` and would
                    # otherwise be filtered out as non-digit. The
                    # profiler showed Stage 8 failures (38.6% of all
                    # failures) frequently land in this pattern.
                    try:
                        raw = classifier_topk(crops, k=4)
                    except TypeError:
                        # Older callers may not accept a ``k`` kwarg;
                        # fall back to the default k=2.
                        raw = classifier_topk(crops)
                    cnn_topk = [list(r) for r in raw]
                    # Synthesize the top-1 list for backwards compat.
                    classifications = [
                        (r[0][0], r[0][1]) if r else ("?", 0.0)
                        for r in cnn_topk
                    ]
                except Exception:
                    cnn_topk = []
                    classifications = []
            if not classifications and classifier is not None:
                try:
                    classifications = list(classifier(crops))
                except Exception:
                    classifications = []
        # Pad if the classifier dropped any.
        while len(classifications) < len(digit_bboxes):
            classifications.append(("?", 0.0))
        while len(cnn_topk) < len(digit_bboxes):
            # If only the top-1 classifier ran, build a degenerate
            # top-K list from it so the backtracker has something to
            # work with even if the only "alternative" is "?".
            i = len(cnn_topk)
            cnn_topk.append([classifications[i]])

    # Compose digit string. Treat any non-digit class (incl. the icon
    # '@' class, surfaced as '?' by classifier_signal when idx >= 10)
    # as a placeholder.
    def _compose_top1(pairs: list[tuple[str, float]]) -> str:
        out: list[str] = []
        for cls, _ in pairs:
            out.append(cls if cls.isdigit() else "?")
        return "".join(out)

    composed = _compose_top1(classifications)

    if classifications:
        mean_conf = float(np.mean([c for _, c in classifications]))
    else:
        mean_conf = _NO_CNN_DEFAULT_CONF

    # ── Fix 2: lexicon backtracking (post-CNN, pre-final-score) ──
    # Try if top-1 isn't already a known signature value. The
    # backtracker checks single-position top-1→top-2 swaps and picks
    # the highest-mean-conf candidate that lands in the lexicon.
    backtracked = False
    backtrack_from = ""
    if (
        lexicon
        and cnn_topk
        and len(cnn_topk) == n_digits
        and len(composed) == n_digits
    ):
        # Quick check: is top-1 already lexicon-valid? Skip backtrack
        # if so (avoids burning the search on already-correct reads).
        try:
            top1_in_lex = composed.isdigit() and int(composed) in lexicon
        except Exception:
            top1_in_lex = False
        if not top1_in_lex:
            bt = _backtrack_with_lexicon(
                cnn_topk, 1 if n_digits == 4 else 2, lexicon,
                n_digits_expected=n_digits,
            )
            if bt is not None:
                bt_str, bt_mean = bt
                backtrack_from = composed
                composed = bt_str
                log.info(
                    "signal_proportional_segmenter: lexicon backtrack "
                    "(n_digits=%d): %s -> %s (mean_conf=%.2f)",
                    n_digits, backtrack_from, bt_str, bt_mean,
                )
                # Replace classifications at the swapped position(s)
                # with the chosen alternative so downstream consumers
                # see the corrected per-position confidences. We
                # detect which positions were swapped by comparing
                # top-1 vs the backtracked composition character by
                # character.
                new_classifications = list(classifications)
                for i, (orig, new) in enumerate(zip(backtrack_from, bt_str)):
                    if orig == new:
                        continue
                    if i < len(cnn_topk) and len(cnn_topk[i]) >= 2:
                        new_classifications[i] = cnn_topk[i][1]
                classifications = new_classifications
                mean_conf = bt_mean
                backtracked = True

    in_lexicon = False
    if (
        lexicon
        and len(composed) == n_digits
        and composed.isdigit()
    ):
        try:
            in_lexicon = int(composed) in lexicon
        except Exception:
            in_lexicon = False

    # Post-CNN empty-slot signal: any position the CNN tagged with a
    # non-digit class (e.g. ``@``, the icon class) is almost always
    # pill background that the structural mass-threshold check missed.
    # We fold this into the empty-slot penalty so a hypothesis where
    # the CNN's read ITSELF declares "this position is not a digit"
    # gets the same per-slot cost. After backtracking, the chosen
    # composition's chars are the ones we count — if backtracking
    # rewrote the position to a digit, the slot doesn't count as empty.
    n_empty_cnn = sum(
        1 for ch, _ in classifications if not ch.isdigit()
    )
    if n_empty_cnn:
        # Use the same per-slot penalty magnitude as the structural
        # check so the two reinforce each other on overlap and
        # complement each other on partial coverage. Avoid double-
        # counting: take the max of structural and CNN-flagged
        # empty counts — a slot is empty for one reason, not both.
        cnn_pen = -_EMPTY_SLOT_PENALTY_PER_SLOT * n_empty_cnn
        if cnn_pen < empty_pen:
            empty_pen = cnn_pen
            n_empty = max(n_empty, n_empty_cnn)

    score = (
        mean_conf
        + (_LEXICON_BONUS if in_lexicon else 0.0)
        + empty_pen
    )

    return _Hypothesis(
        n_digits=n_digits,
        comma_position=1 if n_digits == 4 else 2,
        digit_bboxes=digit_bboxes,
        comma_bbox=comma_bbox,
        digit_width_px=float(digit_w),
        comma_width_px=float(comma_w),
        composed=composed,
        classifications=classifications,
        mean_conf=mean_conf,
        in_lexicon=in_lexicon,
        score=score,
        used_blob_centers=used_anchor,
        slot_masses=list(slot_masses),
        n_empty_slots=int(n_empty),
        empty_slot_penalty=float(empty_pen),
        backtracked=backtracked,
        backtrack_from=backtrack_from,
    )


def segment_signal_proportional(
    rgb_crop_image: Image.Image,
    expected_digits: Optional[int] = None,
    *,
    classifier: Optional[Callable[[list[np.ndarray]], list[tuple[str, float]]]] = None,
    classifier_topk: Optional[
        Callable[[list[np.ndarray]], list[list[tuple[str, float]]]]
    ] = None,
    lexicon: Optional[set[int]] = None,
) -> Optional[dict[str, Any]]:
    """Proportional segmentation of a signature digit crop.

    Parameters
    ----------
    rgb_crop_image : PIL.Image.Image
        RGB image of the digit-area crop. Expected to come from the
        runtime's value bbox (``world_model_region2`` proportional
        crop, or the legacy ``find_digit_cluster`` bbox).
    expected_digits : int, optional
        4 or 5. If supplied, only that hypothesis is evaluated. If
        None (the typical call), both hypotheses are scored and the
        higher-scoring one is returned.
    classifier : callable, optional
        Function that takes a list of 28×28 float32 [0,1] grayscale
        crops and returns ``[(class_str, confidence_float), ...]``.
        Typically ``ocr.sc_ocr.api._classify_crops_signal``.
    classifier_topk : callable, optional
        Top-K variant — same signature as ``classifier`` but each
        per-crop entry is itself a list of ``(char, conf)`` ordered
        by descending probability. Used by lexicon backtracking
        (Fix 2) to consult a digit position's second-best alternative
        when the top-1 composition isn't in the known-signature set.
        When omitted, the segmenter still works — it just can't
        backtrack and falls back to top-1 only.
    lexicon : set[int], optional
        Set of known signature integers. Tips ambiguous cases toward
        the in-lexicon reading. Typically
        ``ocr.sc_ocr.api._KNOWN_SIGNAL_VALUES``.

    Returns
    -------
    dict or None
        ``None`` if the crop is too small to plausibly hold even a
        4-digit signature. Otherwise a dict with keys ``digits``,
        ``n_digits``, ``comma_position``, ``confidence``, ``details``.
    """
    if rgb_crop_image is None:
        return None

    pil_rgb = rgb_crop_image.convert("RGB")
    rgb_array = np.asarray(pil_rgb, dtype=np.uint8)
    if rgb_array.size == 0:
        return None

    gray = rgb_array.max(axis=2).astype(np.uint8)

    # ── Match runtime preprocessing for the CNN ──
    # The signal CNN was trained on glyphs rendered at ~28-32 px tall.
    # Native runtime captures come in at ~14-22 px tall, so the CNN
    # sees upscale-blur unless we pre-upscale. Lanczos to ~32 px.
    h0 = gray.shape[0]
    upscale = 1
    rgb_upscaled = rgb_array
    if h0 < 28:
        scale = max(2, 32 // max(1, h0))
        try:
            gray = np.asarray(
                Image.fromarray(gray, mode="L").resize(
                    (gray.shape[1] * scale, h0 * scale), Image.LANCZOS,
                ),
                dtype=np.uint8,
            )
            # Apply the SAME Lanczos upscale to the RGB array so the
            # downstream comma_finder, which expects RGB, sees an
            # image at the same coordinate scale as ``gray``.
            rgb_upscaled = np.asarray(
                Image.fromarray(rgb_array, mode="RGB").resize(
                    (rgb_array.shape[1] * scale, rgb_array.shape[0] * scale),
                    Image.LANCZOS,
                ),
                dtype=np.uint8,
            )
            upscale = scale
        except Exception:
            pass

    # (1) Min-max contrast stretch — matches runtime's pre-segmentation
    # remap. Training samples have full 0-255 dynamic range; runtime's
    # max-of-channels grayscale arrives compressed.
    g32 = gray.astype(np.float32)
    mn, mx = float(g32.min()), float(g32.max())
    if mx - mn > 8:
        g32 = (g32 - mn) * (255.0 / (mx - mn))
        gray = np.clip(g32, 0, 255).astype(np.uint8)

    # (2) Polarity canonicalize — bright text on dark bg.
    gray_canon = _canonicalize_polarity_local(gray)

    crop_h, crop_w = gray_canon.shape[:2]
    if crop_w < _MIN_CROP_W_FOR_4DIGIT or crop_h < 4:
        return None

    # ── Comma anchor (PRIMARY) ───────────────────────────────────────
    # Use the dedicated RGB comma detector (find_comma_voted) as the
    # primary X-axis anchor. The voted detector runs both polarities
    # of the structural finder and combines: when both polarities
    # agree we get a high-confidence comma column; when they disagree
    # we get the higher-confidence single-polarity result.
    #
    # If the detector doesn't return a result we fall back to the
    # legacy inline ``_detect_comma_extent`` heuristic — keeping the
    # existing safety net while promoting the new detector to the
    # primary role.
    comma_anchor: Optional[dict[str, Any]] = None
    try:
        from hud_tracker.anchors.comma_finder import find_comma_voted
        comma_anchor = find_comma_voted(rgb_upscaled)
    except Exception:
        comma_anchor = None

    if comma_anchor is not None:
        cb = comma_anchor["bbox"]
        comma_extent = (int(cb[0]), int(cb[0] + cb[2]))
    else:
        comma_extent = _detect_comma_extent(gray_canon)
    ink_extent = _find_ink_extent(gray_canon)

    if expected_digits in (4, 5):
        hypotheses_to_try = [expected_digits]
    else:
        hypotheses_to_try = [4, 5]

    hypotheses: list[_Hypothesis] = [
        _score_hypothesis(
            gray_canon, comma_extent, ink_extent, n,
            classifier, lexicon,
            classifier_topk=classifier_topk,
        )
        for n in hypotheses_to_try
    ]

    # Pick the winner. Highest score wins; on a tie, prefer the one
    # whose composed string contains no '?' placeholders. On a
    # remaining tie, prefer the hypothesis whose blob-count matches
    # exactly (used_blob_centers=True), then prefer 5-digit.
    def _winner_key(h: _Hypothesis) -> tuple[float, int, int, int]:
        no_q = 0 if "?" in h.composed else 1
        used = 1 if h.used_blob_centers else 0
        return (h.score, no_q, used, h.n_digits)

    winner = max(hypotheses, key=_winner_key)

    digits_out: list[dict[str, Any]] = []
    digit_iter = iter(zip(
        winner.digit_bboxes,
        winner.classifications or [("?", 0.0)] * len(winner.digit_bboxes),
    ))
    n_slots = winner.n_digits + 1
    comma_idx = 1 if winner.n_digits == 4 else 2

    for slot_idx in range(n_slots):
        if slot_idx == comma_idx:
            digits_out.append({
                "bbox": winner.comma_bbox,
                "is_comma": True,
            })
        else:
            try:
                bbox, (cls, conf) = next(digit_iter)
            except StopIteration:
                break
            digits_out.append({
                "bbox": bbox,
                "is_comma": False,
                "classification": str(cls),
                "confidence": float(conf),
            })

    return {
        "digits": digits_out,
        "n_digits": winner.n_digits,
        "comma_position": winner.comma_position,
        "confidence": float(winner.score),
        # CANONICAL GRAY exposed for the api-side determinism fix
        # (DETERMINISM 2026-05-10): the production ``_signal_recognize_pil``
        # used to re-extract per-glyph crops from its OWN
        # ``_canonicalize_polarity`` output, which is computed by an
        # Otsu minority-class rule that disagrees with this module's
        # border-median rule on a subset of captures (digit-ink-
        # dominated centers, atypical pill backgrounds). When the two
        # rules disagreed the segmenter and the api would feed
        # OPPOSITE polarities to the same gray CNN at the same
        # bboxes, producing wildly divergent reads — exactly the
        # "jumping" behavior the user reported. The fix: api uses
        # ``gray_canon_used`` as the source for re-extracting crops,
        # guaranteeing byte-identical inputs to the gray CNN
        # regardless of polarity disagreement.
        "gray_canon_used": gray_canon,
        "details": {
            "digit_width_px": winner.digit_width_px,
            "comma_width_px": winner.comma_width_px,
            "crop_w": int(crop_w),
            "crop_h": int(crop_h),
            "comma_extent": (
                tuple(int(v) for v in comma_extent)
                if comma_extent is not None else None
            ),
            "ink_extent": (int(ink_extent[0]), int(ink_extent[1])),
            "hypotheses": [
                {
                    "n_digits": h.n_digits,
                    "composed": h.composed,
                    "mean_conf": h.mean_conf,
                    "in_lexicon": h.in_lexicon,
                    "score": h.score,
                    "comma_position": h.comma_position,
                    "used_blob_centers": h.used_blob_centers,
                    "slot_masses": h.slot_masses,
                    "n_empty_slots": h.n_empty_slots,
                    "empty_slot_penalty": h.empty_slot_penalty,
                    "backtracked": h.backtracked,
                    "backtrack_from": h.backtrack_from,
                }
                for h in hypotheses
            ],
            "winner_n_digits": winner.n_digits,
            "winner_used_blob_centers": winner.used_blob_centers,
            "string_composed": winner.composed,
            "evaluated_with_classifier": classifier is not None,
            "comma_anchor_used": comma_anchor is not None,
            "comma_anchor": comma_anchor,
            "upscale_used": int(upscale),
        },
    }
