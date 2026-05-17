"""Per-domain format / range / dictionary validation.

Post-classification layer that filters out segmentation errors and
OCR hallucinations before they reach the UI. Each domain has its
own set of validators; failed reads fall through to ONNX fallback
in ``api.py`` before being returned as None.

Refinery validators reuse the existing fuzzy-match infrastructure
in ``ocr/refinery_reader.py`` — no reinvention.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

log = logging.getLogger(__name__)


# ── Pre-compiled regex patterns (module scope) ─────────────────────
_RE_NON_DIGIT = re.compile(r"[^0-9]")
_RE_NON_DIGIT_DOT = re.compile(r"[^0-9.]")
_RE_MULTI_DOT = re.compile(r"\.+")
_RE_COST_NUMBER = re.compile(r"[\d,]+(?:\.\d{1,2})?")


# ── Signal scanner ─────────────────────────────────────────────────

SIGNAL_MIN = 1000
SIGNAL_MAX = 35000


def validate_signal(raw: str) -> Optional[int]:
    """Parse a digit-only string as a signal number in [1000, 35000]."""
    digits = _RE_NON_DIGIT.sub("", raw)
    if not digits:
        return None
    try:
        val = int(digits)
    except ValueError:
        return None
    if SIGNAL_MIN <= val <= SIGNAL_MAX:
        return val
    # Strip leading digit icons (HUD sometimes has a decorative
    # digit-like icon glued to the value). Try dropping one leading
    # char.
    if len(digits) >= 4:
        try:
            val2 = int(digits[1:])
            if SIGNAL_MIN <= val2 <= SIGNAL_MAX:
                return val2
        except ValueError:
            pass
    return None


# ── Mining HUD ─────────────────────────────────────────────────────

MASS_MAX = 10_000_000.0  # kg — large asteroids can exceed a million


def validate_mass(raw: str) -> Optional[float]:
    """Parse a mass read as a float in [0.0, MASS_MAX].

    Mass-zero IS a valid game state: when a rock first becomes
    visible on the scanner, the HUD shows ``MASS: 0.00`` until the
    rock is mined. The previous lower bound of 0.1 rejected these
    correctly-read zeros and the downstream template fallback then
    hallucinated nonzero values from the empty area (e.g. reading
    "0400" from a "MASS: 0.00" panel because the progress-bar
    region produced spurious digit-shaped pixels).
    """
    cleaned = _RE_NON_DIGIT_DOT.sub("", raw)
    if not cleaned:
        return None
    # Collapse accidental double dots
    cleaned = _RE_MULTI_DOT.sub(".", cleaned).strip(".")
    if not cleaned:
        return None
    try:
        val = float(cleaned)
    except ValueError:
        return None
    if 0.0 <= val <= MASS_MAX:
        return val
    return None


def validate_pct(raw: str) -> Optional[float]:
    """Parse a percentage read as a float in [0, 100]."""
    # Strip trailing % and inner whitespace; keep digits + dot
    cleaned = _RE_NON_DIGIT_DOT.sub("", raw.replace("%", ""))
    if not cleaned:
        return None
    cleaned = _RE_MULTI_DOT.sub(".", cleaned).strip(".")
    if not cleaned:
        return None
    try:
        val = float(cleaned)
    except ValueError:
        return None
    if 0.0 <= val <= 100.0:
        return val
    # Sometimes trailing digit is a misread '%' — try dropping
    # progressively from the right.
    for end in range(len(cleaned) - 1, 0, -1):
        try:
            v = float(cleaned[:end])
            if 0.0 <= v <= 100.0:
                return v
        except ValueError:
            continue
    return None


def validate_instability(
    raw: str,
    confidences: list[float] | None = None,
) -> Optional[float]:
    """Parse an instability read as a float.

    In-game instability values for mineable asteroids practically
    always live in [0.0, 200.0]. A raw read like '12318' that passes
    numeric validation but exceeds that band almost certainly dropped
    the decimal point. We always probe for a missed decimal when:
      * The raw string has no dot AND
      * There's a low-confidence character (< 0.55) that could be
        a misclassified dot AND
      * Inserting a dot at that position yields a value in the
        plausible [0.0, 200.0] band
    """
    cleaned = _RE_NON_DIGIT_DOT.sub("", raw)
    if not cleaned:
        return None
    cleaned = _RE_MULTI_DOT.sub(".", cleaned).strip(".")
    if not cleaned:
        return None
    try:
        val = float(cleaned)
    except ValueError:
        return None

    # Try decimal recovery if the raw string has no dot, confidences
    # are provided, and there's a low-confidence char that might be
    # a misread decimal. This probe runs BEFORE accepting the no-dot
    # value so we don't lock in '12318' for a 12.10 truth.
    def _try_recover() -> Optional[float]:
        if "." in cleaned or not confidences:
            return None
        if len(confidences) != len(raw):
            return None
        # Rank chars by ascending confidence — try the lowest first
        order = sorted(range(len(confidences)), key=lambda i: confidences[i])
        for idx in order[:2]:  # try the 2 least-confident positions
            if confidences[idx] > 0.60:
                break  # remaining are all reasonably confident
            attempt = raw[:idx] + "." + raw[idx + 1:]
            attempt_clean = _RE_NON_DIGIT_DOT.sub("", attempt)
            attempt_clean = _RE_MULTI_DOT.sub(".", attempt_clean).strip(".")
            try:
                v2 = float(attempt_clean)
            except ValueError:
                continue
            # Prefer recovery when result falls in typical instability
            # range (0-200). If it's totally implausible (> 10000),
            # keep looking.
            if 0.0 <= v2 <= 200.0:
                return v2
        return None

    recovered = _try_recover()
    if recovered is not None:
        return recovered

    # No recovery needed or possible — accept raw value if in broad range.
    if 0.0 <= val <= 100000.0:
        return val
    return None


# ── Refinery ───────────────────────────────────────────────────────
# Delegated to ocr/refinery_reader.py's existing fuzzy matchers.
# They operate on the raw OCR text (after glyph joins) and return
# the canonical form. We re-import lazily to avoid a hard circular
# dependency.

def validate_refinery_method(raw: str) -> Optional[str]:
    try:
        from .. import refinery_reader
    except Exception:
        return raw.strip() or None
    # refinery_reader has _fuzzy_method which does the matching.
    matcher = getattr(refinery_reader, "_fuzzy_method", None)
    if matcher is None:
        return raw.strip() or None
    try:
        return matcher(raw) or None
    except Exception:
        return raw.strip() or None


def validate_refinery_commodity(raw: str) -> Optional[str]:
    try:
        from .. import refinery_reader
    except Exception:
        return raw.strip() or None
    matcher = getattr(refinery_reader, "_fuzzy_mineral", None)
    if matcher is None:
        return raw.strip() or None
    try:
        return matcher(raw) or None
    except Exception:
        return raw.strip() or None


def validate_refinery_time(raw: str) -> Optional[int]:
    try:
        from .. import refinery_reader
    except Exception:
        return None
    parser = getattr(refinery_reader, "_parse_time_to_seconds", None)
    if parser is None:
        return None
    try:
        secs = parser(raw)
        if secs and secs > 0:
            return int(secs)
    except Exception:
        pass
    return None


def validate_refinery_cost(raw: str) -> Optional[float]:
    # Allow digits, commas, dots, keep the first number-looking thing
    match = _RE_COST_NUMBER.search(raw)
    if not match:
        return None
    s = match.group(0).replace(",", "")
    try:
        val = float(s)
    except ValueError:
        return None
    if 1.0 <= val <= 1_000_000_000.0:
        return val
    return None


# ── HUD structural-anchor checks ───────────────────────────────────
# Conservative sanity checks for HUD numeric fields whose punctuation
# acts as a structural anchor:
#
#   * resistance: ends in ``%`` (always the rightmost glyph). When
#     the per-glyph cascade returns a string with ``%`` mid-position
#     (e.g. ``"%4"``, ``"4%5"``), the segmentation has fused or
#     mis-ordered boxes and the read should be rejected to let
#     downstream fallbacks (sticky-consensus, frozen-panel) handle
#     the frame.
#
#   * instability: contains exactly one ``.`` and it is interior —
#     at least one digit before AND one digit after. A leading-dot
#     read like ``".09"`` means the per-glyph cascade dropped the
#     integer digit on the left (most commonly because the crop's
#     left edge was clipped — direct analogue of the signature
#     pipeline's comma-anchored crop extension).
#
# These mirror the comma-anchored sanity in the signature scanner
# (``,`` is always interior in a digit string like ``8,375``); a
# break in the anchor's position is reliable evidence of broken
# segmentation rather than a real reading.
#
# Both checks are CONSERVATIVE: they only reject when the structure
# is clearly broken, and they no-op on:
#   * empty strings (nothing to check)
#   * fields other than ``resistance`` / ``instability``
#   * strings that don't contain the relevant punctuation at all
#     (handled by the existing digit-only / lexicon paths)

def check_hud_anchors(
    text: str,
    field: str,
    boxes: Optional[Sequence[Sequence[int]]] = None,
) -> tuple[bool, str]:
    """Validate structural anchors for HUD ``resistance`` / ``instability``.

    Parameters
    ----------
    text:
        Per-glyph cascade output string (e.g. ``"50%"``, ``"12.09"``).
    field:
        HUD field name. Only ``"resistance"`` and ``"instability"``
        engage the check; other fields are a no-op (returns
        ``(True, "")``).
    boxes:
        Optional list of ``(x, y, w, h)`` glyph spans aligned 1:1 with
        ``text``. When provided, the resistance check also rejects
        the case where the ``%`` box has digit-shaped boxes to its
        RIGHT (which would indicate the crop extended into the wrong
        region). When omitted, only the string-position checks fire.

    Returns
    -------
    (ok, reason):
        ``ok`` is ``True`` when the structure is consistent (or the
        check doesn't apply). When ``False``, ``reason`` is a short
        machine-readable tag (``"pct_not_rightmost"``,
        ``"pct_right_of_digit_box"``, ``"dot_leading"``,
        ``"dot_multiple"``) suitable for a log line.
    """
    # No-op fields: not our concern.
    if field not in ("resistance", "instability"):
        return True, ""
    # Empty / single-char reads can't be structurally checked. Let
    # downstream length / lexicon validators handle those.
    if not text:
        return True, ""

    if field == "resistance":
        # The % must be rightmost. If there's no % at all, this check
        # doesn't apply — the read might still be a valid bare-digit
        # resistance read pre-% (e.g. for some HUD layouts) and the
        # downstream validate_pct will normalize.
        if "%" not in text:
            return True, ""
        pct_pos = text.index("%")
        # First-position % is structurally impossible (no integer to
        # the left). Strict reject.
        # Mid-string % like "4%5" — also a clear segmentation error.
        if pct_pos != len(text) - 1:
            return False, "pct_not_rightmost"
        # Box-level check (when spans are available): the box at the
        # %'s position should be the rightmost on the row. If any
        # box's x lies to the RIGHT of the % box, the crop extended
        # past the % into the wrong area and the per-glyph cascade
        # picked up bogus tiles.
        if boxes is not None and len(boxes) == len(text):
            try:
                pct_box = boxes[pct_pos]
                pct_x_right = int(pct_box[0]) + int(pct_box[2])
                for i, b in enumerate(boxes):
                    if i == pct_pos:
                        continue
                    bx = int(b[0])
                    # If any other box starts AFTER the right edge of
                    # the % box, it's positioned to the right of the
                    # anchor — bad.
                    if bx >= pct_x_right:
                        return False, "pct_right_of_digit_box"
            except (TypeError, IndexError, ValueError):
                # Mal-formed boxes — fall through silently rather than
                # rejecting on a check-helper bug.
                return True, ""
        return True, ""

    # Instability: at most one '.', must be interior.
    if field == "instability":
        # No dot means nothing to validate here — the read might be
        # a legit integer-only value, or it might have dropped its
        # decimal entirely (handled elsewhere in validate_instability).
        if "." not in text:
            return True, ""
        # Multiple dots is unambiguously broken.
        if text.count(".") > 1:
            return False, "dot_multiple"
        dot_pos = text.index(".")
        # Leading dot — the integer digit on the left was clipped.
        # Direct analogue of the signature scanner's comma-anchored
        # left-extension trigger: a glyph that MUST have a digit
        # before it does not.
        if dot_pos == 0:
            return False, "dot_leading"
        # Trailing dot — no fractional part. This is borderline (e.g.
        # could be a partial read where the trailing digit dropped),
        # but treat as structural break: the dot anchor requires a
        # digit on both sides for the read to be self-consistent.
        if dot_pos == len(text) - 1:
            return False, "dot_trailing"
        return True, ""

    return True, ""


def estimate_digit_pitch(
    boxes: Sequence[Sequence[int]],
) -> Optional[int]:
    """Estimate the per-digit pixel pitch from segmenter spans.

    Used by the instability ``.``-leading recovery path to extend
    the value crop leftward by one digit's width when the leading
    integer digit was clipped (mirror of the signature scanner's
    comma-anchored crop extension).

    The estimate is the MEDIAN box width — robust against a single
    out-of-distribution span (e.g. a misclassified-as-tall dot, or
    a width-fused tile that contains two glyphs). When fewer than
    two boxes are available, returns ``None``.
    """
    if not boxes or len(boxes) < 2:
        return None
    widths: list[int] = []
    for b in boxes:
        try:
            w = int(b[2])
        except (TypeError, IndexError, ValueError):
            continue
        if w > 0:
            widths.append(w)
    if not widths:
        return None
    widths.sort()
    mid = len(widths) // 2
    if len(widths) % 2 == 1:
        return widths[mid]
    return (widths[mid - 1] + widths[mid]) // 2
