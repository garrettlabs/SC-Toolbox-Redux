"""NCC-based icon anchor for the signal scanner.

The location-pin icon ALWAYS appears to the left of the signal
number in every signature scan, regardless of resource type or
signal value. Only its color changes (cyan / orange / etc); its
shape is identical pixel-for-pixel after polarity canonicalization.

This module wraps the same NCC machinery sc_ocr's label-row finder
uses for HUD labels, but trains the templates from the blacklisted
icon image. Returns the icon's pixel position so callers can crop
the digit cluster at a known offset.

The blacklist directory was originally created so glyph extraction
could REJECT icon-shaped tiles. We re-purpose those same icon PNGs
as positive ANCHOR templates here. One-stop registration.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from . import label_match as _lm

log = logging.getLogger(__name__)

# Blacklist directory holds the icon template(s).
_TOOL_DIR = Path(__file__).resolve().parent.parent.parent
_ICON_TEMPLATES_DIR = _TOOL_DIR / "training_data_blacklist"

# Cache the canonicalized icon templates (one per blacklist entry,
# pre-resized to a few candidate scales) so NCC search per scan only
# pays the correlation cost, not the load+canonicalize cost.
_TEMPLATE_CACHE: Optional[list[tuple[np.ndarray, int, int]]] = None
# Each entry is (canonical_template, src_w, src_h)

# ── Temporal anchor smoothing ──────────────────────────────────────
# Frame-to-frame NCC scores wobble by ~0.05-0.15 on chromatically-
# aberrated HUDs. When the icon's score drops below the
# high-confidence threshold on a single scan, the runtime can land
# the match on a digit-pair false positive (e.g. ``11`` against the
# icon template) — propagating a wrong work crop downstream and
# producing an empty/garbage digit read. We smooth this by caching
# recent high-confidence matches and falling back to the cache
# median whenever the current scan's match is borderline or absent.
#
# Stability rule: cache must contain ≥ 3 entries that agree within
# ``_ANCHOR_STABILITY_TOLERANCE_PX`` to be considered locked.
# Self-heal rule: any high-confidence match that disagrees with the
# cache median by more than ``_ANCHOR_RESET_TOLERANCE_PX`` resets
# the cache (panel position changed — start fresh).
_RECENT_ANCHOR_CACHE: deque = deque(maxlen=8)
# Each entry: (timestamp_monotonic, x1, y1, x2, y2, score)
_ANCHOR_HIGH_CONF_THR = 0.60
_ANCHOR_CACHE_MAX_AGE_S = 5.0
_ANCHOR_STABILITY_TOLERANCE_PX = 6
_ANCHOR_RESET_TOLERANCE_PX = 25

# ── Pending-reset confirmation state ──
# Previously, ANY single high-confidence match disagreeing with the
# cache median by >RESET_TOLERANCE wiped the cache immediately. In
# practice the icon NCC's per-frame score variance (different template
# scales winning different frames) caused the picked x1 to wobble
# wildly: 11 → 110 → 65 → 110 → 56 → 111 → 69, never the same wrong
# position twice. Each disagreement wiped the cache, so the cache
# never accumulated the 3 entries it needs to be "stable" — the
# smoother was permanently neutralized and downstream consumers
# always saw the noisy current-frame match.
#
# New rule: require the SAME new position for ``_ANCHOR_RESET_CONFIRMATIONS``
# consecutive frames before wiping. A single noisy frame that returns
# to the cached position next frame is rejected as transient.
_ANCHOR_RESET_CONFIRMATIONS = 3
_PENDING_RESET_X1: Optional[int] = None
_PENDING_RESET_COUNT: int = 0


def reset_anchor_cache() -> None:
    """Clear the temporal anchor smoothing cache.

    Call this when the scan region changes, when the user moves to a
    new rock, or in tests where you need a clean baseline."""
    global _RECENT_ANCHOR_CACHE, _PENDING_RESET_X1, _PENDING_RESET_COUNT
    _RECENT_ANCHOR_CACHE.clear()
    _PENDING_RESET_X1 = None
    _PENDING_RESET_COUNT = 0


def _cnn_filter_icon_candidates(
    gray: np.ndarray,
    candidates: list[tuple[float, int, int, int, int, int]],
    rgb_image: Optional[np.ndarray] = None,
) -> list[tuple[float, int, int, int, int, int]]:
    """Filter NCC candidates to those the multi-voter accepts as the
    location-pin icon.

    History (old strict-AND path):
      The previous implementation chained the gray CNN re-rank, a
      leftmost-only filter, and the geometric validator as a strict
      AND-gate. When the gray CNN was uncertain (which happens
      whenever a candidate is off the training distribution — a digit
      cluster, a partial icon match, or a chromatic-aberrated icon
      crop), the entire pipeline rejected every candidate, including
      the real icon, and the anchor would fall back to its temporal
      cache (or to nothing).

    New voter path:
      Each NCC candidate (after the leftmost-only positional filter)
      runs through ``vote_on_icon_candidate`` from
      ``hud_tracker.anchors.icon_voter``. The voter consults FOUR
      detectors arranged in three tiers:

        Primary tier (decorrelated structural detectors):
          1. ``find_icon_by_geometry`` — HSV warm-color +
             teardrop / oval / notch checks.
          2. ``find_icon_by_contour`` — luma + edge contour matched
             against the canonical icon silhouette. Built by a
             parallel agent; the voter imports it defensively.
        Secondary tier (RGB CNN with @ class):
          ``model_signal_rgb_cnn_v2.onnx``. Trained by a parallel
          agent; loaded defensively.
        Tertiary tier (existing grayscale CNN with @ class):
          ``model_signal_cnn.onnx``. The classifier that used to be
          the single AND-gate now gets ONE vote of four — no longer
          a precondition.

      When the parallel-agent outputs are missing, the voter degrades
      gracefully: contour abstains, RGB v2 abstains, and the gray
      CNN's vote stands alone in the lone-primary fall-through path.

    Calling-convention notes:

      * The voter receives ALL NCC candidates, not just the ones the
        gray CNN accepts. The gray CNN's classification is now an
        input to the voter, not a gate.
      * The leftmost-only positional filter (a pure precision guard
        that drops digit-cluster false positives at the right of the
        panel) still runs FIRST, because position is unrelated to
        class identity and applies regardless of voter outcome.
      * When ``rgb_image`` is None (the auto-annotator's
        ``detect_icon`` adapter passes only gray), the voter operates
        in gray-only mode: only the tertiary tier votes, mirroring
        the legacy single-gate behavior so existing callers don't
        regress.
      * If the voter import fails (shouldn't happen, but be safe) we
        return the input list unchanged.
    """
    if not candidates:
        return candidates

    # ── Leftmost-only positional filter (precision guard) ──
    # The icon is always the leftmost UI element in the signal panel,
    # so any candidate whose left edge is significantly to the right
    # of the leftmost candidate cannot be the icon — it's some other
    # @-classifying shape (digit cluster, comma-followed-by-zero, etc.).
    # Apply this BEFORE the voter so the voter only burns CPU on
    # plausibly-positioned candidates.
    #
    # We keep the filter conservative: only drop candidates whose
    # x1 is more than a small margin to the right of the leftmost
    # candidate's x1. Identical-x1 candidates (multi-scale matches at
    # the same position) all pass through.
    if len(candidates) > 1:
        x1_min = min(int(c[1]) for c in candidates)
        # Tolerance: a single-icon-width is plenty; anything past that
        # is a different shape further right.
        x1_tol = 8
        before = list(candidates)
        candidates = [c for c in candidates if int(c[1]) <= x1_min + x1_tol]
        rejected = [c for c in before if c not in candidates]
        if rejected:
            log.warning(
                "[ANCHOR-DIAG] _cnn_filter_icon_candidates: leftmost-only "
                "filter kept x1<=%d (tol=%d), rejected %d cands at x1=%s",
                x1_min, x1_tol, len(rejected),
                [int(c[1]) for c in rejected],
            )

    # ── Multi-voter gate ──
    try:
        from hud_tracker.anchors.icon_voter import vote_on_icon_candidate
    except Exception as exc:
        log.debug(
            "_cnn_filter_icon_candidates: icon_voter import failed "
            "(%s) — returning candidates unchanged", exc,
        )
        return candidates

    # Optionally promote a PIL image of ``rgb_image`` so the voter
    # can derive crops in its own padding regime. We pass it through
    # and let the voter coerce.
    pil_image = None
    if rgb_image is not None:
        try:
            if isinstance(rgb_image, np.ndarray) and rgb_image.ndim == 3:
                pil_image = Image.fromarray(
                    rgb_image[..., :3].astype(np.uint8)
                ).convert("RGB")
        except Exception:
            pil_image = None

    # Pre-build a 28x28 gray crop per candidate so the gray CNN vote
    # uses exactly the same crop as the legacy path (matches the
    # training distribution: dark-ink-on-light, no polarity flip).
    def _gray_crop_28(cand) -> np.ndarray:
        cx1, cy1, cx2, cy2 = int(cand[1]), int(cand[2]), int(cand[3]), int(cand[4])
        pad_x = max(2, (cx2 - cx1) // 4)
        pad_y = max(2, (cy2 - cy1) // 4)
        rx1 = max(0, cx1 - pad_x)
        ry1 = max(0, cy1 - pad_y)
        rx2 = min(gray.shape[1], cx2 + pad_x)
        ry2 = min(gray.shape[0], cy2 + pad_y)
        region = gray[ry1:ry2, rx1:rx2]
        if region.size == 0:
            return np.zeros((28, 28), dtype=np.float32)
        try:
            pil = Image.fromarray(region.astype(np.uint8)).resize(
                (28, 28), Image.BILINEAR,
            )
            return np.asarray(pil, dtype=np.float32) / 255.0
        except Exception:
            return np.zeros((28, 28), dtype=np.float32)

    kept_after_voter: list[tuple[float, int, int, int, int, int]] = []
    for cand in candidates:
        x1, y1, x2, y2 = int(cand[1]), int(cand[2]), int(cand[3]), int(cand[4])
        bbox = (x1, y1, x2, y2)
        gray_crop = _gray_crop_28(cand)
        try:
            vote = vote_on_icon_candidate(
                rgb_image=pil_image if pil_image is not None else rgb_image,
                candidate_bbox=bbox,
                gray_cnn=None,  # let the voter lazy-import the helper
                rgb_cnn=None,   # let the voter lazy-load v2 if present
                gray_crop=gray_crop,
            )
        except Exception as exc:
            log.debug(
                "_cnn_filter_icon_candidates: voter raised (%s) — "
                "keeping candidate %s", exc, bbox,
            )
            kept_after_voter.append(cand)
            continue

        if vote.get("accepted"):
            _votes = vote.get("votes") or {}
            # Weak-accept guard. When the geometry structural detector
            # actively voted 'no' AND the grayscale CNN is unavailable,
            # the accept rests on the RGB CNN alone — one detector, no
            # decorrelated second opinion, overriding an *active*
            # structural rejection. That is the exact signature of the
            # location-pin false positive: with the signature panel
            # NOT on screen, NCC locks onto a glint / green arrow /
            # digit cluster and the lone RGB CNN rubber-stamps it.
            # Treat such a candidate as rejected so the icon is not
            # "found" when it is not actually present. (A real icon
            # passes the warm-colour geometry detector, so geometry
            # voting 'no' is genuine evidence against the candidate.)
            _weak_accept = (
                _votes.get("geometry") == "no"
                and _votes.get("gray_cnn") in (None, "unavailable")
            )
            if _weak_accept:
                log.info(
                    "[VOTE] cand %s REJECTED-WEAK (geometry=no + gray_cnn "
                    "unavailable — lone rgb_cnn accept not trusted) "
                    "votes=%s path=%s",
                    bbox, _votes, vote.get("decision_path"),
                )
            else:
                kept_after_voter.append(cand)
                log.info(
                    "[VOTE] cand %s ACCEPTED via %s votes=%s",
                    bbox, vote.get("decision_path"), vote.get("votes"),
                )
        else:
            log.info(
                "[VOTE] cand %s REJECTED votes=%s (path=%s)",
                bbox, vote.get("votes"), vote.get("decision_path"),
            )

    return kept_after_voter


def _smooth_anchor_with_cache(
    current: Optional[tuple[int, int, int, int, float]],
) -> Optional[tuple[int, int, int, int, float]]:
    """Apply temporal smoothing on top of the NCC anchor result.

    Decisions, in order:
      1. **Drop expired entries** from the cache (older than
         ``_ANCHOR_CACHE_MAX_AGE_S``).
      2. **High-confidence current match** (score ≥ ``_ANCHOR_HIGH_CONF_THR``):
         * If it disagrees with the cache median by more than
           ``_ANCHOR_RESET_TOLERANCE_PX`` (panel moved), wipe the
           cache and seed it with the current match.
         * Otherwise append to the cache.
         The match is then surfaced ONLY when the cache is stable
         (see ``_stability_gated``); until then ``None`` is returned.
      3. **Low-confidence current match** (score < ``_ANCHOR_HIGH_CONF_THR``)
         OR **no match at all**: if the cache contains ≥ 3 entries
         that agree within ``_ANCHOR_STABILITY_TOLERANCE_PX``, return
         the cache median position with the cache's median score.
         Otherwise return the current match (which may be ``None``).

    The cache median is preferred over a borderline current match
    because high-confidence past matches are stronger evidence of the
    icon's true position than a noisy single-frame score that could
    just as easily be a digit-pair false positive.
    """
    global _RECENT_ANCHOR_CACHE
    now = time.monotonic()

    # 1. Expire stale entries.
    while (
        _RECENT_ANCHOR_CACHE
        and now - _RECENT_ANCHOR_CACHE[0][0] > _ANCHOR_CACHE_MAX_AGE_S
    ):
        _RECENT_ANCHOR_CACHE.popleft()

    # Cache median helper (over the current cache contents).
    def _cache_median() -> Optional[tuple[int, int, int, int, float]]:
        if len(_RECENT_ANCHOR_CACHE) == 0:
            return None
        xs1 = sorted(int(e[1]) for e in _RECENT_ANCHOR_CACHE)
        ys1 = sorted(int(e[2]) for e in _RECENT_ANCHOR_CACHE)
        xs2 = sorted(int(e[3]) for e in _RECENT_ANCHOR_CACHE)
        ys2 = sorted(int(e[4]) for e in _RECENT_ANCHOR_CACHE)
        scores = sorted(float(e[5]) for e in _RECENT_ANCHOR_CACHE)
        mid = len(_RECENT_ANCHOR_CACHE) // 2
        return (xs1[mid], ys1[mid], xs2[mid], ys2[mid], scores[mid])

    def _cache_is_stable() -> bool:
        if len(_RECENT_ANCHOR_CACHE) < 3:
            return False
        xs1 = [int(e[1]) for e in _RECENT_ANCHOR_CACHE]
        xs2 = [int(e[3]) for e in _RECENT_ANCHOR_CACHE]
        return (
            (max(xs1) - min(xs1)) <= _ANCHOR_STABILITY_TOLERANCE_PX
            and (max(xs2) - min(xs2)) <= _ANCHOR_STABILITY_TOLERANCE_PX
        )

    def _stability_gated(
        result: Optional[tuple[int, int, int, int, float]],
    ) -> Optional[tuple[int, int, int, int, float]]:
        """Emit an icon position ONLY once the cache holds a stable run.

        The real location-pin icon is stationary, so its cache
        stabilises (``_cache_is_stable()`` — ≥3 recent entries agreeing
        within ``_ANCHOR_STABILITY_TOLERANCE_PX``) within ~3 scans. A
        false positive — NCC locking onto a glint or the green
        scan-direction arrow when the signature panel is NOT on screen
        — jumps position every scan and never stabilises. Returning
        ``None`` in that case keeps the signature scanner in its
        'scanning' state instead of hallucinating a value off a
        non-icon shape.
        """
        if _cache_is_stable():
            return result
        log.info(
            "_smooth_anchor_with_cache: icon emission suppressed — "
            "cache not stable yet (%d entr%s, need 3 within %dpx) — "
            "signature panel treated as NOT present",
            len(_RECENT_ANCHOR_CACHE),
            "y" if len(_RECENT_ANCHOR_CACHE) == 1 else "ies",
            _ANCHOR_STABILITY_TOLERANCE_PX,
        )
        return None

    # 2. High-confidence current match.
    if current is not None and current[4] >= _ANCHOR_HIGH_CONF_THR:
        global _PENDING_RESET_X1, _PENDING_RESET_COUNT
        med = _cache_median()
        cur_x1 = int(current[0])

        def _entry() -> tuple[float, int, int, int, int, float]:
            return (
                now,
                int(current[0]), int(current[1]),
                int(current[2]), int(current[3]),
                float(current[4]),
            )

        if med is None:
            _RECENT_ANCHOR_CACHE.append(_entry())
            _PENDING_RESET_X1 = None
            _PENDING_RESET_COUNT = 0
            return _stability_gated(current)

        disagrees = abs(cur_x1 - med[0]) > _ANCHOR_RESET_TOLERANCE_PX
        if not disagrees:
            # Agrees with cache — append and clear any pending reset
            # (the wobble corrected itself).
            _RECENT_ANCHOR_CACHE.append(_entry())
            _PENDING_RESET_X1 = None
            _PENDING_RESET_COUNT = 0
            return _stability_gated(current)

        # Disagrees with cache. Track whether this is a recurring new
        # position (panel actually moved) or a single-frame noise spike.
        if (
            _PENDING_RESET_X1 is not None
            and abs(cur_x1 - _PENDING_RESET_X1) <= _ANCHOR_RESET_TOLERANCE_PX
        ):
            _PENDING_RESET_COUNT += 1
        else:
            _PENDING_RESET_X1 = cur_x1
            _PENDING_RESET_COUNT = 1

        if _PENDING_RESET_COUNT >= _ANCHOR_RESET_CONFIRMATIONS:
            log.info(
                "_smooth_anchor_with_cache: %d consecutive high-conf "
                "matches at x1=%d (last differed from cache median "
                "x1=%d by %d px) — confirming panel move, wiping cache",
                _PENDING_RESET_COUNT, cur_x1, med[0],
                abs(cur_x1 - med[0]),
            )
            _RECENT_ANCHOR_CACHE.clear()
            _RECENT_ANCHOR_CACHE.append(_entry())
            _PENDING_RESET_X1 = None
            _PENDING_RESET_COUNT = 0
            # Cache was just wiped to one entry — the gate suppresses
            # until the new position re-confirms over the next ~3 scans.
            return _stability_gated(current)

        # Cache stable + transient disagreement — hold the cache median
        # so downstream consumers see the locked icon position rather
        # than a noisy single-frame jump that'll likely revert next tick.
        if _cache_is_stable():
            log.info(
                "_smooth_anchor_with_cache: high-conf match at x1=%d "
                "differs from cache median x1=%d by %d px (%d/%d "
                "confirmations) — holding cache median",
                cur_x1, med[0], abs(cur_x1 - med[0]),
                _PENDING_RESET_COUNT, _ANCHOR_RESET_CONFIRMATIONS,
            )
            return med

        # Cache not stable yet — the gate suppresses emission here by
        # definition; route through it for one consistent exit path.
        return _stability_gated(current)

    # 3. Low-confidence or absent current match — try the cache.
    if _cache_is_stable():
        med = _cache_median()
        if med is not None:
            if current is None:
                log.info(
                    "_smooth_anchor_with_cache: NCC found NO match "
                    "this frame — using cached median position "
                    "(%d,%d,%d,%d) from %d entries",
                    med[0], med[1], med[2], med[3],
                    len(_RECENT_ANCHOR_CACHE),
                )
            else:
                log.info(
                    "_smooth_anchor_with_cache: low-conf match "
                    "(score=%.2f < %.2f at x1=%d) overridden by "
                    "cached median (%d,%d,%d,%d) from %d entries",
                    float(current[4]), _ANCHOR_HIGH_CONF_THR,
                    int(current[0]), med[0], med[1], med[2], med[3],
                    len(_RECENT_ANCHOR_CACHE),
                )
            return med

    # No useful cache to fall back to — return whatever NCC produced
    # (may be None or a low-confidence match the caller will treat
    # the same way it always did).
    return current


def _load_icon_templates() -> list[np.ndarray]:
    """Load every PNG in the blacklist dir, canonicalize polarity
    (text/icon BRIGHT on dark bg), normalize to zero-mean unit-
    variance. Returns templates as float32 arrays."""
    if not _ICON_TEMPLATES_DIR.is_dir():
        return []
    out: list[np.ndarray] = []
    for f in sorted(_ICON_TEMPLATES_DIR.glob("*.png")):
        try:
            img = Image.open(f).convert("L")
            arr = np.asarray(img, dtype=np.uint8)
        except Exception:
            continue
        # Use sc_ocr's polarity canonicalization (label_match uses
        # the same convention: text/foreground is BRIGHT after
        # canonicalization).
        canonical = _lm._canonicalize(arr).astype(np.float32)
        if canonical.size == 0 or canonical.shape[0] < 8 or canonical.shape[1] < 8:
            continue
        # Zero-mean, unit-variance — match _ncc_search's input contract.
        mean = float(canonical.mean())
        canonical = canonical - mean
        std = float(canonical.std())
        if std < 1e-6:
            continue
        canonical = canonical / std
        out.append(canonical)
    return out


def _build_template_cache() -> list[tuple[np.ndarray, int, int]]:
    """Generate (template, w, h) triples at candidate scales for each
    blacklist icon. Multi-scale lets us match icons rendered at
    different sizes across capture resolutions.

    Scale range covers tw=12 px (small / 4K-downsampled captures) up
    through tw=72 px (large / 1080p-native captures). Earlier the
    range capped at tw=40 px because larger scales were producing
    matched boxes that wrapped around the icon AND part of the
    leading digit — but with the post-NCC ``_expand_to_icon_extent``
    pass clamping the matched extent to the icon's actual ink (and
    the structural-trim helper dropping anything that isn't), the
    top of the range is now safe to extend.

    Why this matters for false positives: when the actual icon in the
    capture is ~50-60 px wide and our largest template is 40 px,
    NCC scales the icon DOWN inside the matched box (low score) but
    the small templates still fit nicely on adjacent narrow shapes
    like the digit pair "11" (high score). NCC then picks "11" as
    the icon. Adding 48/56/64/72 px templates lets the big icon
    match at its native scale and outscore any digit-pair false
    positive."""
    raw = _load_icon_templates()
    cache: list[tuple[np.ndarray, int, int]] = []
    target_widths = (
        # Small captures / downsampled HUDs:
        12, 16, 20, 24, 28,
        # Mid-range:
        32, 36, 40, 44, 48,
        # Large captures / native 1440p+ HUDs:
        56, 64, 72,
    )
    for tmpl in raw:
        h0, w0 = tmpl.shape
        if w0 < 4:
            continue
        for tw in target_widths:
            s = float(tw) / float(w0)
            if s <= 0 or s > 1.5:  # skip absurd scales
                continue
            try:
                resized = _lm._resize_template(tmpl, s)
                if resized.shape[0] >= 6 and resized.shape[1] >= 6:
                    cache.append((resized, resized.shape[1], resized.shape[0]))
            except Exception:
                continue
    return cache


def _ensure_cache() -> list[tuple[np.ndarray, int, int]]:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = _build_template_cache()
        log.info(
            "signal_anchor: loaded %d icon templates from %s",
            len(_TEMPLATE_CACHE), _ICON_TEMPLATES_DIR,
        )
    return _TEMPLATE_CACHE


def reset_cache() -> None:
    """Force the next find_icon() call to rebuild from disk. Call
    after adding/removing icons from the blacklist dir."""
    global _TEMPLATE_CACHE
    _TEMPLATE_CACHE = None


def find_icon(
    gray: np.ndarray, min_score: float = 0.55,
    search_left_fraction: float = 0.55,
    rgb_image: Optional[np.ndarray] = None,
) -> Optional[tuple[int, int, int, int, float]]:
    """Locate the location-pin icon in ``gray`` via multi-scale NCC.

    Returns the **LEFTMOST** position whose NCC score crosses
    ``min_score``, NOT the global maximum. The SC font's digits
    (particularly '9' and the comma-followed-by-'0' pair) have a
    circle-on-stick shape that NCC scores higher than the actual
    icon. Since the icon is ALWAYS the leftmost UI element in the
    capture, "leftmost above threshold" finds it reliably while
    "global max" picks digits.

    Returns ``(x_left, y_top, x_right, y_bot, score)`` or ``None``.

    The optional ``rgb_image`` parameter, when supplied, enables the
    geometric structural validator inside ``_cnn_filter_icon_candidates``
    — see that function's docstring for details. Existing callers that
    pass only ``gray`` (e.g. the auto-annotator's ``detect_icon``
    adapter) keep their current behavior because ``rgb_image`` defaults
    to ``None`` and the validator is then skipped.
    """
    cache = _ensure_cache()
    if not cache:
        return _smooth_anchor_with_cache(None)
    # Canonicalize input to match template normalization.
    target_full = _lm._canonicalize(gray).astype(np.float32)
    if target_full.size == 0 or target_full.shape[0] < 8 or target_full.shape[1] < 8:
        return _smooth_anchor_with_cache(None)
    mean = float(target_full.mean())
    target_full = target_full - mean
    std = float(target_full.std())
    if std < 1e-6:
        return _smooth_anchor_with_cache(None)
    target_full = target_full / std

    # Search window: leftmost portion only.
    search_w = max(20, int(target_full.shape[1] * search_left_fraction))
    target = target_full[:, :search_w]

    # Run NCC for every template scale, gather a 2D score map per
    # template. Take the LEFTMOST position whose score crosses
    # min_score — across all scales, the leftmost over-threshold
    # position is the icon.
    # Vertical band constraint: the location-pin icon NEVER renders in
    # the extreme top or bottom of the captured signal panel — the panel
    # UI is centered. Without this, low-confidence NCC matches on noise
    # / dust specks in the corners can win as "leftmost above threshold"
    # and steal the anchor from the real icon further right but in the
    # mid-frame band. Restrict candidate match top-left y to [10%, 90%)
    # of the target height.
    H_target = target.shape[0]
    y_min_band = int(H_target * 0.10)
    y_max_band = int(H_target * 0.90)

    # Selection rule: prefer the HIGHEST-scoring match across all
    # template scales, with leftmost-right-edge as a tiebreaker only
    # when scores are within ``CLOSE_SCORE_DELTA`` of the max.
    #
    # Why this beats "leftmost above threshold":
    # The previous rule picked the leftmost column whose NCC score
    # crossed ``min_score``. A noise speck at the far left scoring 0.42
    # would beat the actual icon at score 0.75 further right — the
    # symptom in user screenshots was a tiny RED anchor box on a dust
    # speck while the real orange location-pin icon stood ignored. The
    # icon is in ``training_data_blacklist/`` precisely so we know what
    # to look for, so a strong template match should ALWAYS win.
    #
    # Why we still consider leftmost as a tiebreaker:
    # The original concern was that SC font's '9' / comma+'0' have a
    # circle-on-stick shape that NCC scores comparably to the icon at
    # SOME scales. When the icon and a digit both score similarly
    # (within 0.10), leftmost-right-edge correctly picks the icon
    # because the icon is the leftmost UI element by panel layout.
    CLOSE_SCORE_DELTA = 0.10
    # Per-scale best candidate: (score, x_left, y_top, x_right, y_bot, tw)
    per_scale_best: list[tuple[float, int, int, int, int, int]] = []
    for tmpl, tw, th in cache:
        if tw > target.shape[1] or th > target.shape[0]:
            continue
        y_lo = y_min_band
        y_hi = max(y_lo + 1, y_max_band - th + 1)
        try:
            from scipy.signal import correlate2d  # type: ignore
            n = float(th * tw)
            c = correlate2d(target, tmpl, mode="valid", boundary="fill", fillvalue=0)
            c = c / n
            y_hi_clamped = min(y_hi, c.shape[0])
            if y_hi_clamped <= y_lo:
                continue
            c_band = c[y_lo:y_hi_clamped]
            if c_band.size == 0:
                continue
            # Argmax across the band — this template's STRONGEST match,
            # not its leftmost weak one.
            flat_idx = int(np.argmax(c_band))
            yy, xx = np.unravel_index(flat_idx, c_band.shape)
            score = float(c_band[yy, xx])
            if score < min_score:
                continue
            best_x = int(xx)
            best_y = int(yy + y_lo)
        except Exception:
            # Fallback: full per-position scan, track the maximum
            # score (NOT leftmost-above-threshold).
            best_x = best_y = None
            score = -2.0
            H, W = target.shape
            y_hi_fb = min(y_hi, H - th + 1)
            for x in range(0, W - tw + 1):
                for y in range(y_lo, y_hi_fb):
                    s = float(np.sum(
                        target[y:y + th, x:x + tw] * tmpl
                    )) / float(th * tw)
                    if s > score:
                        score = s
                        best_x = x
                        best_y = y
            if best_x is None or score < min_score:
                continue

        per_scale_best.append((
            score, best_x, best_y, best_x + tw, best_y + th, tw,
        ))

    if not per_scale_best:
        return _smooth_anchor_with_cache(None)

    # ── Reject digit-pair false positives via connected-component check ──
    # The location-pin icon is ONE connected shape (head + tail joined).
    # A digit pair like "11" is TWO disconnected vertical bars with a
    # gap between them. NCC happens to score "11" comparably to a
    # partially-matched icon when an undersized template wins, so we
    # need an explicit shape filter to reject those false positives.
    # Filter the per-scale candidates: keep only those whose binary
    # mask (within the matched box) collapses to ≤ 1 connected
    # component after light dilation. Anything with 2+ components is
    # almost certainly a digit pair, comma+digit, or similar
    # non-icon shape.
    def _is_single_shape(cand_x1, cand_y1, cand_x2, cand_y2) -> bool:
        h_pad = max(2, (cand_y2 - cand_y1) // 6)
        w_pad = max(2, (cand_x2 - cand_x1) // 6)
        ry1 = max(0, cand_y1 - h_pad)
        ry2 = min(gray.shape[0], cand_y2 + h_pad)
        rx1 = max(0, cand_x1 - w_pad)
        rx2 = min(gray.shape[1], cand_x2 + w_pad)
        region = gray[ry1:ry2, rx1:rx2].astype(np.uint8)
        if region.size == 0:
            return True
        # Polarity-canonicalize so icon shows as bright.
        if float(np.median(region)) > 140:
            region = (255 - region.astype(np.int16)).clip(0, 255).astype(np.uint8)
        rmin = float(region.min())
        rmax = float(region.max())
        if rmax - rmin < 25:
            return True  # flat region, can't distinguish
        thr = rmin + 0.55 * (rmax - rmin)
        binary = (region > thr).astype(np.uint8)
        try:
            from scipy.ndimage import label as _scipy_label
            from scipy.ndimage import binary_dilation as _scipy_dilation
            # Light dilation reconnects icon parts that AA / chromatic
            # aberration may have separated. Adjacent digits ("11")
            # have a real gap that even dilation won't bridge.
            dilated = _scipy_dilation(binary, iterations=1)
            _labels, n_components = _scipy_label(dilated)
            return n_components <= 1
        except Exception:
            return True  # scipy unavailable, skip the check

    # Apply shape filter to candidates (keep candidates that pass).
    shape_filtered = [c for c in per_scale_best if _is_single_shape(
        c[1], c[2], c[3], c[4],
    )]
    n_rejected_shape = len(per_scale_best) - len(shape_filtered)
    if shape_filtered:
        candidates_for_pick = shape_filtered
    else:
        candidates_for_pick = per_scale_best
        log.warning(
            "signal_anchor.find_icon: ALL %d NCC candidates failed "
            "the single-shape check — falling back to all candidates",
            len(per_scale_best),
        )

    # ── CNN re-rank: validate candidates via the signal CNN's '@' class ──
    # NCC alone confuses the icon with digit clusters that share its
    # gross silhouette ("10," looks like a tail+head+base sequence).
    # We re-run each candidate's region through ``_classify_crops_signal``
    # — the model trained on 600 augmented icons + 5000+ digits already
    # knows the difference. Candidates the CNN classifies as ``@``
    # are real-icon-like; candidates classified as a digit are not.
    #
    # When at least one candidate is CNN-classified as ``@``, restrict
    # the cross-scale pick to those. When NONE are, fall back to the
    # NCC+shape-filtered set so the pipeline still produces something
    # on captures where the CNN is uncertain (the temporal smoothing
    # downstream catches those cases via cache fallback anyway).
    cnn_validated = _cnn_filter_icon_candidates(
        gray, candidates_for_pick, rgb_image=rgb_image,
    )
    n_cnn_validated = len(cnn_validated)
    if cnn_validated:
        candidates_for_pick = cnn_validated
    elif rgb_image is not None:
        # The voter ran with full RGB inputs and accepted NONE of the
        # NCC candidates — every one was either CNN-rejected or only
        # weakly accepted (geometry=no with no grayscale CNN to back
        # the lone RGB CNN). Previously the code fell through here and
        # force-picked the best raw-NCC candidate anyway — which is
        # what makes the signature scanner hallucinate a value when
        # the location-pin panel is NOT on screen (NCC locks onto a
        # glint / arrow / digit cluster). Return no-icon instead.
        # Temporal smoothing still replays a genuinely-recent cached
        # icon position if one exists, so a transient one-frame voter
        # glitch on a real icon does not drop the anchor.
        #
        # Scoped to ``rgb_image is not None`` so the gray-only caller
        # (the auto-annotator's detect_icon adapter) keeps its legacy
        # fallback — the voter cannot run its RGB detectors there.
        log.info(
            "signal_anchor.find_icon: voter accepted 0 of %d candidates "
            "— treating the signature panel as not present (no icon)",
            len(candidates_for_pick),
        )
        return _smooth_anchor_with_cache(None)

    # Cross-scale selection: highest score wins. When multiple
    # candidates land within CLOSE_SCORE_DELTA of the max, prefer
    # leftmost-right-edge (icon is the leftmost UI element).
    max_score = max(c[0] for c in candidates_for_pick)
    close_to_best = [
        c for c in candidates_for_pick if c[0] >= max_score - CLOSE_SCORE_DELTA
    ]
    # Sort by right-edge ascending, then by score descending, then by
    # tw ascending (tighter box around the icon).
    close_to_best.sort(key=lambda c: (c[3], -c[0], c[5]))
    chosen = close_to_best[0]
    score = chosen[0]
    x1, y1, x2, y2, best_tw = (
        chosen[1], chosen[2], chosen[3], chosen[4], chosen[5],
    )
    # Temporarily WARNING-level so it appears in the deployed log
    # file (default-filtered to WARNING). Once anchor stability is
    # confirmed live, this can drop back to log.info.
    log.warning(
        "[ANCHOR-DIAG] signal_anchor.find_icon: NCC pick "
        "ix=(%d,%d,%d,%d) score=%.2f tw=%d (n_scales_hit=%d, "
        "n_rejected_shape=%d, n_cnn_validated=%d, max_score=%.2f, "
        "n_within_delta=%d, target_W=%d)",
        x1, y1, x2, y2, score, best_tw, len(per_scale_best),
        n_rejected_shape, n_cnn_validated, max_score,
        len(close_to_best), target.shape[1],
    )
    # Expand the matched box to encompass the full icon's ink. NCC
    # usually locks onto a sub-portion (the bottom-tip when an
    # undersized template wins), but the user-facing RED anchor box
    # AND the digit-crop calculation both want the FULL icon extent.
    # Doing the expansion HERE (rather than only inside
    # find_digit_crop_box) means the live viewer's RED box reflects
    # the same boundary the digit crop respects — so the user can
    # visually verify the icon is fully contained within the anchor.
    x1, y1, x2, y2 = _expand_to_icon_extent(gray, x1, y1, x2, y2)

    # Apply temporal smoothing across recent scans. High-confidence
    # matches update the cache and pass through; low-confidence
    # matches get overridden by the cache's stable median position
    # if the cache is locked. This kills the failure mode where a
    # single-frame NCC wobble makes the anchor jump to a digit-pair
    # false positive (e.g. ``11``), which would otherwise cascade
    # into a wrong work crop and an empty digit read.
    return _smooth_anchor_with_cache((x1, y1, x2, y2, score))


def _expand_to_icon_extent(
    gray: np.ndarray,
    ix1: int, iy1: int, ix2: int, iy2: int,
    max_expansion_factor: float = 3.0,
) -> tuple[int, int, int, int]:
    """Grow the NCC-matched icon box to encompass the icon's full ink.

    The NCC anchor often matches only a sub-portion of the location-pin
    icon — typically the bottom-tip — when an undersized template scale
    wins the score race. The icon's body (round head + tail) extends
    well beyond the matched rectangle; if we don't expand, the digit
    crop starts past only the matched fragment and the icon's body
    falls into the digit region.

    Two-stage expansion:
      1. **Column/row scan** in canonical polarity (icon = bright).
         From the matched box edges, walk outward column by column
         (or row by row). A column counts as "icon ink" if its bright-
         pixel count is at least 12% of the match height (low enough
         to catch the icon's thin tail, high enough to ignore noise).
         Stop expanding when we hit ``MAX_GAP`` consecutive columns
         (rows) below threshold — that's the gap between the icon and
         the leading digit.
      2. **Connected-component** check (scipy if available) to confirm
         the expanded box only contains ONE component. If it absorbs
         the leading digit (component span > 4× match), bail and
         return the original.

    Capped at ``max_expansion_factor`` × match dimensions per side so
    a runaway can't consume the leading digit even when both stages
    fail to bound the expansion.
    """
    H, W = gray.shape
    iw = max(1, ix2 - ix1)
    ih = max(1, iy2 - iy1)

    # Polarity-canonicalize the WHOLE row band so column scans see the
    # icon as bright regardless of rendering polarity. We do our OWN
    # polarity check (median > 140 → invert) rather than calling
    # ``_lm._canonicalize`` because that helper returns float32 zero-
    # mean unit-variance data — we want raw uint8 [0, 255] so the
    # ink-density threshold below is meaningful in absolute brightness.
    band_y1 = max(0, iy1 - ih)
    band_y2 = min(H, iy2 + ih)
    band = gray[band_y1:band_y2, :].astype(np.uint8)
    if band.size == 0:
        return ix1, iy1, ix2, iy2
    if float(np.median(band)) > 140:
        canon_band = (255 - band.astype(np.int16)).clip(0, 255).astype(np.uint8)
    else:
        canon_band = band

    band_min = float(canon_band.min())
    band_max = float(canon_band.max())
    if band_max - band_min < 25:
        return ix1, iy1, ix2, iy2

    # Bright threshold — anything above this is "icon ink". Use Otsu
    # so the threshold ADAPTS to the histogram and naturally separates
    # ink (icon strokes + digit strokes) from any bubble/panel glow
    # behind them. A fixed-percentage threshold (0.45 of range) was
    # falling below the bubble glow on signal panels — the glow
    # itself counted as "ink columns" and the column scan walked
    # right through the gap between icon and the leading digit,
    # absorbing the entire digit cluster into the icon's anchor box.
    # Floor: at least 0.55 of range, so even a flat glow-heavy band
    # without a clear bimodal histogram doesn't fall below the
    # bubble brightness.
    hist, _ = np.histogram(canon_band.flatten(), bins=256, range=(0, 256))
    total = int(canon_band.size)
    sum_total = float(np.sum(np.arange(256) * hist))
    sum_bg = 0.0
    w_bg = 0
    max_var = 0.0
    otsu_thr = 127
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
            otsu_thr = t
    thr = max(float(otsu_thr), band_min + 0.55 * (band_max - band_min))
    bin_band = (canon_band > thr).astype(np.uint8)

    # Column-wise ink count within the row band.
    col_ink = bin_band.sum(axis=0)
    band_h = bin_band.shape[0]
    # A column is "ink-bearing" if at least ``min_ink_per_col`` rows
    # have bright pixels in it.  12% chosen to catch the icon's thin
    # tail without firing on chromatic-aberration speckle.
    min_ink_per_col = max(2, int(band_h * 0.12))

    cx = (ix1 + ix2) // 2
    cy = (iy1 + iy2) // 2
    max_w_each = int(iw * max_expansion_factor)
    max_h_each = int(ih * max_expansion_factor)
    left_cap = max(0, cx - max_w_each)
    right_cap = min(W, cx + max_w_each)

    # ── RIGHT expansion ──
    # Walk from ix2 rightward. Allow up to MAX_GAP consecutive empty
    # columns before declaring the icon ended (chromatic-aberration
    # halos can leave 1-2 px of dim columns within a single icon).
    # MAX_GAP is small and FIXED-relative-to-row-height (NOT to match
    # width) — scaling it with the matched template width caused real
    # captures to walk past the icon-to-digit gap when a large
    # template won the NCC race. The actual gap between the icon and
    # the leading digit is typically ~half a digit's width regardless
    # of template scale.
    MAX_GAP = max(2, band_h // 12)
    right = ix2
    consecutive_empty = 0
    while right < right_cap:
        if col_ink[right] >= min_ink_per_col:
            consecutive_empty = 0
            right += 1
        else:
            consecutive_empty += 1
            if consecutive_empty > MAX_GAP:
                right -= consecutive_empty
                break
            right += 1
    full_x2 = max(ix2, right + 1)

    # ── LEFT expansion ──
    left = ix1 - 1
    consecutive_empty = 0
    while left >= left_cap:
        if col_ink[left] >= min_ink_per_col:
            consecutive_empty = 0
            left -= 1
        else:
            consecutive_empty += 1
            if consecutive_empty > MAX_GAP:
                left += consecutive_empty
                break
            left -= 1
    full_x1 = min(ix1, left + 1)

    # ── ROW-band expansion (vertical) ──
    # Within the now-known horizontal extent, find the actual top/
    # bottom of icon ink.
    icon_band = bin_band[:, full_x1:full_x2]
    row_ink = icon_band.sum(axis=1)
    icon_w = max(1, full_x2 - full_x1)
    min_ink_per_row = max(2, int(icon_w * 0.12))
    inky_rows = np.where(row_ink >= min_ink_per_row)[0]
    if len(inky_rows) > 0:
        full_y1_band = int(inky_rows[0])
        full_y2_band = int(inky_rows[-1] + 1)
        full_y1 = band_y1 + full_y1_band
        full_y2 = band_y1 + full_y2_band
        full_y1 = min(full_y1, iy1)
        full_y2 = max(full_y2, iy2)
        # Vertical cap.
        full_y1 = max(full_y1, cy - max_h_each)
        full_y2 = min(full_y2, cy + max_h_each)
    else:
        full_y1, full_y2 = iy1, iy2

    # Sanity cap: if expansion produced something more than ~2x of the
    # icon's HEIGHT (which is a stable anchor — the icon is roughly
    # square in the SC HUD font), the expansion almost certainly
    # absorbed the leading digit. Width-relative caps (4× match width)
    # were unreliable because larger NCC template scales pushed the
    # cap past the bubble width on signal panels — leaving room for
    # the expansion to absorb the entire bubble.
    expanded_w = full_x2 - full_x1
    expanded_h = full_y2 - full_y1
    icon_height = max(ih, expanded_h)
    if expanded_w > icon_height * 2:
        log.warning(
            "_expand_to_icon_extent: expanded width=%d exceeds 2x "
            "icon height=%d — likely absorbed the leading digit. "
            "Falling back to original match (%d,%d,%d,%d).",
            expanded_w, icon_height, ix1, iy1, ix2, iy2,
        )
        return ix1, iy1, ix2, iy2

    if (full_x2 - full_x1) > iw or (full_y2 - full_y1) > ih:
        log.info(
            "_expand_to_icon_extent: grew matched box "
            "(%d,%d,%d,%d) -> (%d,%d,%d,%d) via column/row scan "
            "(was %dx%d, now %dx%d)",
            ix1, iy1, ix2, iy2,
            full_x1, full_y1, full_x2, full_y2,
            iw, ih, full_x2 - full_x1, full_y2 - full_y1,
        )
    return full_x1, full_y1, full_x2, full_y2


def find_digit_cluster(
    gray: np.ndarray,
) -> Optional[tuple[int, int, int, int]]:
    """Locate the signature's digit cluster directly via shape pattern.

    A complementary anchor to ``find_icon``. Where ``find_icon`` does
    NCC template matching for the location-pin icon, this function
    finds the DIGITS by their structural pattern: a leftmost cluster
    of 4–6 digit-shaped column-spans within a single horizontal row.

    The signal CNN reports 99.47% val accuracy on individual digits
    against ~5,000 real samples per class, so digit segmentation is
    a structurally stronger signal than icon NCC at small template
    scales (where the icon's silhouette is confusable with cramped
    multi-character clusters like ``16,``).

    Robustness over icon NCC:
      * No template-scaling needed — column projection adapts to the
        font size in the captured panel automatically.
      * No polarity-mismatch edge cases — adaptive binarization works
        on either polarity once canonicalized.
      * Multiple redundant glyphs offset single-frame NCC noise.
      * The 4–6-glyph + comma pattern is structurally unique in the
        signal panel; nothing else looks like it.

    Algorithm:
      1. Polarity-canonicalize input so text is bright.
      2. Adaptive-binarize.
      3. Row projection → find the contiguous Y-band with most ink
         (the digit row).
      4. Within that band, column projection → segment into
         (x1, x2) spans.
      5. For each span, compute its actual bounding box.
      6. Filter by digit-typical dimensions (height 8–25, width 2–16).
      7. Cluster spans by horizontal adjacency (gap ≤ icon_h px).
      8. Find the LEFTMOST cluster with 4–7 members and return its
         bounding box (4 = ``DDDD``; 7 = ``DDD,DDD`` worst case).

    Returns ``(x1, y1, x2, y2)`` in original-image coords, or None
    if no valid digit cluster pattern is detected.
    """
    if gray.size == 0 or gray.shape[0] < 10 or gray.shape[1] < 30:
        return None
    try:
        from .api import _canonicalize_polarity, _adaptive_binarize
    except Exception as exc:
        log.debug("find_digit_cluster: import failed: %s", exc)
        return None

    try:
        canon = _canonicalize_polarity(gray.astype(np.uint8))
        binary = _adaptive_binarize(canon)
    except Exception as exc:
        log.debug("find_digit_cluster: canon/binarize failed: %s", exc)
        return None
    if binary.size == 0 or int(binary.max()) == 0:
        return None

    # ── Step 3: find the digit row via row projection ──
    row_proj = np.sum(binary > 0, axis=1).astype(np.int32)
    if int(row_proj.max()) < 4:
        return None
    active_thr = max(2, int(row_proj.max() * 0.30))
    active_rows = row_proj > active_thr
    runs: list[tuple[int, int]] = []
    in_run = False
    run_start = 0
    H = active_rows.shape[0]
    for y in range(H + 1):
        is_active = bool(active_rows[y]) if y < H else False
        if is_active and not in_run:
            in_run = True
            run_start = y
        elif (not is_active) and in_run:
            in_run = False
            runs.append((run_start, y))
    if not runs:
        return None
    # Pick the tallest run (the digit row). Sanity-bound the height.
    y_top, y_bot = max(runs, key=lambda r: r[1] - r[0])
    band_h = y_bot - y_top
    if band_h < 8 or band_h > 50:
        return None

    # ── Step 4: column-segment within the digit band ──
    band_binary = binary[y_top:y_bot]
    col_proj = np.sum(band_binary > 0, axis=0).astype(np.int32)
    if int(col_proj.max()) == 0:
        return None
    col_active = col_proj > 0
    spans: list[tuple[int, int]] = []
    in_span = False
    span_start = 0
    W = col_active.shape[0]
    for x in range(W + 1):
        is_active = bool(col_active[x]) if x < W else False
        if is_active and not in_span:
            in_span = True
            span_start = x
        elif (not is_active) and in_span:
            in_span = False
            if x - span_start >= 1:
                spans.append((span_start, x))
    if not spans:
        return None

    # ── Step 5: per-span actual bbox ──
    span_bboxes: list[tuple[int, int, int, int]] = []
    for sx1, sx2 in spans:
        col_slice = band_binary[:, sx1:sx2]
        if int(col_slice.max()) == 0:
            continue
        rows_with_ink = np.where(col_slice.sum(axis=1) > 0)[0]
        if rows_with_ink.size == 0:
            continue
        sy1 = int(rows_with_ink[0])
        sy2 = int(rows_with_ink[-1] + 1)
        # Lift to full-image coords.
        span_bboxes.append((sx1, y_top + sy1, sx2, y_top + sy2))

    # ── Step 6: filter to digit-typical dimensions ──
    digit_spans: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in span_bboxes:
        sw = x2 - x1
        sh = y2 - y1
        if 8 <= sh <= 25 and 2 <= sw <= 16:
            digit_spans.append((x1, y1, x2, y2))
    if len(digit_spans) < 4:
        return None
    # Sort left-to-right.
    digit_spans.sort(key=lambda b: b[0])

    # ── Step 7: cluster by horizontal adjacency ──
    median_h = int(np.median([b[3] - b[1] for b in digit_spans]))
    max_gap = max(8, median_h)  # icon-to-digit gap is wider than this
    clusters: list[list[tuple[int, int, int, int]]] = []
    current = [digit_spans[0]]
    for i in range(1, len(digit_spans)):
        prev_x2 = current[-1][2]
        cur_x1 = digit_spans[i][0]
        if cur_x1 - prev_x2 <= max_gap:
            current.append(digit_spans[i])
        else:
            clusters.append(current)
            current = [digit_spans[i]]
    clusters.append(current)

    # ── Step 8: find first cluster with 4–7 members ──
    for cluster in clusters:
        n = len(cluster)
        if 4 <= n <= 7:
            cx1 = min(s[0] for s in cluster)
            cy1 = min(s[1] for s in cluster)
            cx2 = max(s[2] for s in cluster)
            cy2 = max(s[3] for s in cluster)
            log.warning(
                "[ANCHOR-DIAG] find_digit_cluster: matched cluster "
                "n=%d bbox=(%d,%d,%d,%d) median_h=%d",
                n, cx1, cy1, cx2, cy2, median_h,
            )
            return (cx1, cy1, cx2, cy2)

    log.warning(
        "[ANCHOR-DIAG] find_digit_cluster: no 4–7-glyph cluster found "
        "(found %d clusters with sizes %s)",
        len(clusters), [len(c) for c in clusters],
    )
    return None


# Module-level flag recording the MODE the most recent
# ``find_digit_crop_box`` call resolved through. Read by callers that
# need to decide accept-threshold based on whether the crop was
# structurally validated (digit-cluster found) or just guessed from
# icon position. See ``last_crop_mode()`` for the documented values.
_LAST_CROP_MODE: str = "none"


def last_crop_mode() -> str:
    """Return the resolution mode of the most recent
    ``find_digit_crop_box`` call.

    Values:
      * ``"combo"``       — both icon NCC + digit-cluster anchors
                            succeeded and agree (gap 0-50 px). HIGH
                            confidence; crop is structurally validated.
      * ``"digit_only"``  — only digit-cluster anchor succeeded (icon
                            NCC missed). Still trustworthy because the
                            digit cluster is the more specific signal.
      * ``"icon_only"``   — only icon NCC succeeded; digit-cluster
                            pattern NOT matched (no 4-7 digit-shaped
                            spans found in the image). DEGENERATE —
                            the crop is guessed from icon position
                            and may contain no actual digits. Callers
                            should refuse lexicon-relaxed acceptance
                            on results from icon-only crops, to
                            prevent hallucinating known-lexicon
                            values from noise when the rock has
                            left view.
      * ``"none"``        — both anchors missed; no crop returned.

    Thread-safety: the SCAN_HUD_ONNX pipeline serializes signature
    scans per process tick (single-thread per region scan), so the
    module-level flag is consistent across the call chain within
    one scan. The flag is set EVERY call so a stale value from a
    previous scan can't leak into a current one's gating decision.
    """
    return _LAST_CROP_MODE


def find_digit_crop_box(
    gray: np.ndarray,
    gap_after_icon_px: int = 4,
    pad_top_px: int = 2,
    pad_bot_px: int = 2,
    min_score: float = 0.40,
    rgb_image: Optional[np.ndarray] = None,
) -> Optional[tuple[int, int, int, int]]:
    """Combo-anchor: cross-validate icon NCC + digit-cluster pattern.

    Returns the bounding box of the signature's digit cluster
    ``(x1, y1, x2, y2)`` in original-image coords, or None on miss.

    Strategy: run BOTH ``find_icon`` (template-matched icon NCC) AND
    ``find_digit_cluster`` (4–7 glyph spatial pattern) on the same
    panel. Combine their results:

      * **Both succeed AND agree** (icon's right edge sits 0–50 px
        before the digit cluster's left edge): high-confidence path,
        use the digit cluster's tight bbox as the crop. Most reliable.
      * **Both succeed but disagree** (icon found far from digit
        cluster, or vice versa): trust the digit cluster — its 4–7-
        glyph pattern is structurally more specific than icon NCC.
      * **Only digit cluster found**: use it. Icon NCC failed
        (chromatic aberration knocked the score below threshold) but
        digit pattern is intact.
      * **Only icon found**: fall back to the legacy icon-edge crop
        (icon's right edge + gap → image right edge).
      * **Neither**: return None.

    ``min_score`` defaults to 0.40 (matches the legacy fallback). The
    icon's NCC score wobbles 0.40–0.70 on chromatically-aberrated SC
    HUDs; a higher threshold rejected the icon ~half the time. With
    the digit-cluster anchor backing it up, even when NCC fails the
    crop is still produced.
    """
    global _LAST_CROP_MODE
    _LAST_CROP_MODE = "none"  # reset every call; one of the branches below
                              # will set the resolved mode before returning.
    icon_result = find_icon(gray, min_score=min_score, rgb_image=rgb_image)
    digit_cluster = find_digit_cluster(gray)
    H, W = gray.shape

    # ── Both anchors fired: cross-validate ──
    if digit_cluster is not None and icon_result is not None:
        dc_x1, dc_y1, dc_x2, dc_y2 = digit_cluster
        ic_x1, ic_y1, ic_x2, ic_y2, ic_score = icon_result
        gap = dc_x1 - ic_x2
        # Consistent layout: icon ends just before digits start.
        if -5 <= gap <= 50:
            log.info(
                "find_digit_crop_box: COMBO-AGREE icon=(%d,%d,%d,%d) "
                "score=%.2f digits=(%d,%d,%d,%d) gap=%d",
                ic_x1, ic_y1, ic_x2, ic_y2, ic_score,
                dc_x1, dc_y1, dc_x2, dc_y2, gap,
            )
        else:
            log.warning(
                "find_digit_crop_box: COMBO-DISAGREE icon=(%d,%d,%d,%d) "
                "score=%.2f digits=(%d,%d,%d,%d) gap=%d — trusting "
                "digit cluster (more specific signal)",
                ic_x1, ic_y1, ic_x2, ic_y2, ic_score,
                dc_x1, dc_y1, dc_x2, dc_y2, gap,
            )
        # Either way, use the digit-cluster bbox with a small pad —
        # it's the actual digit extent, no expansion guesswork needed.
        x1 = max(0, dc_x1 - 2)
        y1 = max(0, dc_y1 - pad_top_px)
        x2 = min(W, dc_x2 + 3)
        y2 = min(H, dc_y2 + pad_bot_px)
        _LAST_CROP_MODE = "combo"
        return (x1, y1, x2, y2)

    # ── Only digit cluster fired: use it directly ──
    if digit_cluster is not None:
        dc_x1, dc_y1, dc_x2, dc_y2 = digit_cluster
        log.info(
            "find_digit_crop_box: DIGIT-ONLY (icon NCC missed) "
            "digits=(%d,%d,%d,%d)",
            dc_x1, dc_y1, dc_x2, dc_y2,
        )
        x1 = max(0, dc_x1 - 2)
        y1 = max(0, dc_y1 - pad_top_px)
        x2 = min(W, dc_x2 + 3)
        y2 = min(H, dc_y2 + pad_bot_px)
        _LAST_CROP_MODE = "digit_only"
        return (x1, y1, x2, y2)

    # ── Only icon fired: fall back to legacy icon-edge crop ──
    if icon_result is None:
        log.info(
            "find_digit_crop_box: BOTH anchors missed — returning None"
        )
        _LAST_CROP_MODE = "none"
        return None
    log.info(
        "find_digit_crop_box: ICON-ONLY (digit-cluster pattern not "
        "matched) — using legacy icon-edge crop"
    )
    _LAST_CROP_MODE = "icon_only"
    found = icon_result  # noqa — preserve legacy variable name below
    if found is None:
        return None
    # ``find_icon`` now returns the EXPANDED icon extent (its tail
    # already calls _expand_to_icon_extent), so we use ``ix2`` directly
    # as the icon's right edge — no second expansion pass needed.
    ix1, iy1, ix2, iy2, score = found
    H, W = gray.shape
    x1 = min(W, ix2 + gap_after_icon_px)
    x2 = W

    # ── Adaptive vertical padding (v2.2.7) ──
    # Hardcoded ``pad_top_px = 2`` was failing on chromatically-
    # aberrated location-pin captures where the NCC template matched
    # only a portion of the pin (typically the bottom circle / dot,
    # because that part has the cleanest contrast against the dark
    # backdrop). The crop then used the partial-match's vertical
    # bounds directly and ended up positioned BELOW the actual digit
    # text — visible in the debug overlay as a green box drawn below
    # the value glyphs (user reported screenshot 2026-05-03).
    #
    # Fix: scale padding by detected icon height so partial matches
    # still produce a crop that fully encompasses the digit row.
    # The location-pin icon and adjacent digits are roughly the same
    # height in SC's HUD, so one icon-height of upward padding
    # recovers the digit baseline even when only the bottom 50% of
    # the pin matched.
    icon_h = max(1, iy2 - iy1)
    pad_top_dyn = max(pad_top_px, icon_h)
    pad_bot_dyn = max(pad_bot_px, max(2, icon_h // 4))
    y1 = max(0, iy1 - pad_top_dyn)
    y2 = min(H, iy2 + pad_bot_dyn)

    # ── Refine crop_y by scanning actual digit ink ──
    # After we have a generous y-window from the padded icon match,
    # tighten it to where the digits ACTUALLY live. Scan the columns
    # immediately after the icon for rows with high std-dev (digit
    # strokes alternate dark/bright within a row → high std; empty
    # rows → low std). This gives us the same alignment we'd get
    # from a perfect icon template — even when the template only
    # caught part of the pin.
    try:
        scan_x_end = min(W, x1 + 200)
        if scan_x_end > x1 + 12 and y2 > y1 + 8:
            band = gray[y1:y2, x1:scan_x_end].astype(np.float32)
            row_var = band.std(axis=1)
            if row_var.size > 4 and float(row_var.max()) > 5.0:
                thresh = float(row_var.max()) * 0.30
                active = np.where(row_var > thresh)[0]
                if active.size >= 4:
                    text_top = int(active[0])
                    text_bot = int(active[-1] + 1)
                    # Tiny safety pad so we don't clip glyph
                    # anti-aliasing.
                    y1_refined = max(0, y1 + text_top - 2)
                    y2_refined = min(H, y1 + text_bot + 2)
                    if y2_refined - y1_refined >= 8:
                        log.info(
                            "signal_anchor.find_digit_crop_box: "
                            "refined y from icon-aligned (%d,%d) "
                            "to ink-aligned (%d,%d) (icon_h=%d "
                            "row_var_peak=%.1f)",
                            y1, y2, y1_refined, y2_refined,
                            icon_h, float(row_var.max()),
                        )
                        y1, y2 = y1_refined, y2_refined
    except Exception as exc:
        # Refinement is best-effort — fall back to padded icon box.
        log.debug(
            "signal_anchor.find_digit_crop_box: ink-refine failed: %s",
            exc,
        )

    # ── Refine crop_x by scanning for digit-ink right edge (v2.2.7) ──
    # Hardcoded ``x2 = W`` was extending the crop to the right edge of
    # the search region, sweeping in empty space + sometimes adjacent
    # UI text (distance value, comma noise, etc.). Visible in user
    # screenshots as a green crop box stretching well past the actual
    # digit cluster.
    #
    # Trim x2 to the right edge of contiguous ink in the y-refined
    # band. Scan columns from x1 rightward; find the rightmost
    # column with stroke-density above background, then bound the
    # crop just past it (small pad for AA halo). When no clear
    # right edge can be detected (very short input, noise-only band)
    # fall back to a conservative max-width based on the icon's
    # diagonal — wide enough for 5-digit values like "17,200" but
    # not wide enough to drag in adjacent UI.
    try:
        if y2 > y1 + 6 and x2 > x1 + 12:
            band = gray[y1:y2, x1:x2].astype(np.float32)
            # Per-column std-dev: digit columns alternate stroke /
            # background vertically → high std; empty columns → low.
            col_var = band.std(axis=0)
            if col_var.size > 4 and float(col_var.max()) > 5.0:
                col_thresh = float(col_var.max()) * 0.20
                active_cols = col_var > col_thresh
                # Find the rightmost run of active cols starting near
                # the icon. Walk from x1 rightward, allowing small
                # gaps (digit-to-digit kerning) but breaking on
                # sustained empty columns (the post-text void).
                last_inky = -1
                empty_run = 0
                # Allow up to icon_h cols of inter-digit gap (commas
                # + kerning + chromatic-aberration smear). Anything
                # longer than that is the void after the value.
                gap_cap = max(8, icon_h)
                for i in range(int(active_cols.size)):
                    if active_cols[i]:
                        last_inky = i
                        empty_run = 0
                    else:
                        empty_run += 1
                        if last_inky >= 0 and empty_run > gap_cap:
                            break
                if last_inky >= 0:
                    # Trim with a small right pad so we don't clip
                    # the rightmost glyph's anti-aliasing halo.
                    x2_refined = min(W, x1 + last_inky + 1 + 3)
                    if x2_refined - x1 >= 12 and x2_refined < x2:
                        log.info(
                            "signal_anchor.find_digit_crop_box: "
                            "refined x_right from %d to %d "
                            "(was sweeping %d px of empty space; "
                            "trimmed to %d px digit extent)",
                            x2, x2_refined, x2 - x1, x2_refined - x1,
                        )
                        x2 = x2_refined
    except Exception as exc:
        log.debug(
            "signal_anchor.find_digit_crop_box: x-refine failed: %s",
            exc,
        )

    if x2 - x1 < 12 or y2 - y1 < 6:
        log.info(
            "signal_anchor.find_digit_crop_box: REJECTED crop too small "
            "x1=%d x2=%d (icon ix1=%d ix2=%d score=%.2f) imgW=%d",
            x1, x2, ix1, ix2, score, W,
        )
        return None
    log.info(
        "signal_anchor.find_digit_crop_box: crop_box=(%d,%d,%d,%d) "
        "icon_right=%d gap=%d crop_w=%d crop_h=%d (imgW=%d icon_h=%d "
        "pad_top=%d pad_bot=%d)",
        x1, y1, x2, y2, ix2, gap_after_icon_px, x2 - x1, y2 - y1,
        W, icon_h, pad_top_dyn, pad_bot_dyn,
    )
    return (x1, y1, x2, y2)
