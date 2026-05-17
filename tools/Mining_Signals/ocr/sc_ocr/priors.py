"""Game-logic priors for SC mining values.

Post-filters that reject OCR outputs incompatible with observed
game state. Runs after the voter, before returning a value. If the
primary vote fails a prior check, the caller tries the NCC template
voter as a fourth opinion.

The big win is difficulty-label ↔ instability consistency: the
game shows an ``EASY | MEDIUM | HARD | VERY HARD | IMPOSSIBLE`` label
on every rock, and each difficulty caps the rock's MAX instability.
An OCR read of ``278`` for an ``EASY`` rock is ruled out by prior
knowledge — an EASY rock can never reach instability 278, regardless
of mining state.

Important: difficulty bounds the MAX instability, not a fixed band.
A freshly-scanned IMPOSSIBLE rock can legitimately show ``2.24``
because instability rises during mining; the difficulty class only
constrains how HIGH it can climb before fracture. Earlier versions
of this module modelled instability as a fixed per-difficulty band
``[d_lo, d_hi]`` and rejected low values on hard rocks — that was
incorrect and produced spurious "prior-reject" log lines for valid
reads (which were then routed through the template fallback for no
reason). The current model is upper-bound only.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

# Difficulty → max-instability ceiling (from in-game observation +
# SC mining wiki). Soft upper bound: real reads tend to sit at or
# below these, but we allow a 50% overshoot in is_plausible to
# absorb patch-to-patch value drift and unusual rocks.
#
# Lower bound is always 0 — even an IMPOSSIBLE rock can show
# instability ~0 before mining begins. The difficulty class is a
# CEILING, not a band.
#
# Both "extreme" and "very hard" are accepted as the in-game label
# spelling has drifted historically; calibration UI exposes "VERY
# HARD" while older docs reference "EXTREME". They map to the same
# bucket (4th out of 5).
_DIFFICULTY_INSTAB: dict[str, tuple[float, float]] = {
    "easy":       (0.0,   25.0),
    "medium":     (0.0,   55.0),
    "hard":       (0.0,  100.0),
    "extreme":    (0.0,  160.0),
    "very hard":  (0.0,  160.0),
    "impossible": (0.0,  300.0),
}

# Difficulty ordering for proximity scoring. "very hard" maps to the
# same slot as "extreme" so callers can use either spelling.
_DIFFICULTY_ORDER = ["easy", "medium", "hard", "extreme", "impossible"]

# Universal hard-range checks (invariant of rock type).
#
# Mass-0 is a valid game state: when a rock first becomes visible on
# the scanner (just appeared / not yet mined), the HUD shows
# ``MASS: 0.00`` until the player has mined some of it. The previous
# lower bound of 0.1 rejected correctly-read zeros and the downstream
# template fallback then hallucinated nonzero values from the empty
# value area — see commit history for the labeled-capture benchmark
# data that quantified this failure mode.
_UNIVERSAL_RANGES: dict[str, tuple[float, float]] = {
    "mass":        (0.0, 10_000_000.0),
    "resistance":  (0.0, 100.0),
    "instability": (0.0, 500.0),
}

# Tier 4 fuzzy OCR-confusable patterns (compiled at module load).
# Tesseract often mis-reads letter pairs when the font is stylized:
# 'EASY' → 'EA8Y' / 'EAGY' / 'EA9Y'. Check for 2-char stem + any
# char + expected tail. "VERY HARD" is checked BEFORE plain "hard"
# in detect_difficulty so the two-word label wins when both match.
_FUZZY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bea\w*y\b"),       "easy"),       # EA?Y style
    (re.compile(r"\bmed\w{1,3}"),     "medium"),     # MED+
    (re.compile(r"\bver\w{0,2}\s+\w{0,2}ard\b"), "extreme"),  # VERY HARD style
    (re.compile(r"\bext\w{2,6}"),     "extreme"),    # EXT+
    (re.compile(r"\bhar[dcbg]?\b"),   "hard"),       # HAR[D/C/B/G]
    (re.compile(r"\bimp\w{2,7}"),     "impossible"), # IMP+
]


def detect_difficulty(panel_text: str) -> Optional[str]:
    """Scan a blob of panel OCR text for a difficulty label.

    Returns the canonical lowercase difficulty name or None. Uses
    progressively looser matching: exact word → 4-char prefix → fuzzy
    2-char OCR-confusable substrings (handles `EASY` read as `EAGY`,
    `EA8Y`, etc).

    "VERY HARD" is recognised as the two-word in-game label and
    folded into the ``extreme`` bucket (the 4th-of-5 difficulty
    tier). It MUST be checked before plain "hard" because "very hard"
    contains "hard" as a substring.
    """
    if not panel_text:
        return None
    t = panel_text.lower()

    # Tier 1: exact substring. "very hard" first so it doesn't lose
    # to plain "hard"; longest labels next; "hard" last for the same
    # reason.
    if "very hard" in t or "very  hard" in t:
        return "extreme"
    for label in ("impossible", "extreme", "medium", "easy", "hard"):
        if label in t:
            return label

    # Tier 2: 4-char prefix
    for label in ("impossible", "extreme", "medium", "easy", "hard"):
        if label[:4] in t:
            return label

    # Tier 3: 3-char prefix (no 'har' — ambiguous with 'harm', 'hard', etc.)
    prefix_map = {
        "eas": "easy",
        "med": "medium",
        "ext": "extreme",
        "imp": "impossible",
    }
    for pfx, lbl in prefix_map.items():
        if pfx in t:
            return lbl

    # Tier 4: fuzzy OCR-confusable patterns (see _FUZZY_PATTERNS at
    # module scope — compiled once per process instead of per call).
    for patt, lbl in _FUZZY_PATTERNS:
        if patt.search(t):
            return lbl
    return None


def is_plausible(
    field: str,
    value: float,
    context: Optional[dict] = None,
) -> tuple[bool, str]:
    """Return ``(is_plausible, reason)``.

    Context may contain: ``difficulty``, ``scu``, ``rock_name``. Any
    key may be missing; we apply only the checks we have data for.
    """
    context = context or {}

    # Hard universal range
    lo, hi = _UNIVERSAL_RANGES.get(field, (float("-inf"), float("inf")))
    if not (lo <= value <= hi):
        return False, f"{value} outside universal range [{lo}, {hi}]"

    # Difficulty → instability ceiling consistency
    #
    # Instability is the rock's CURRENT instability and starts near
    # 0 on freshly-scanned rocks, rising during mining. Difficulty
    # bounds only the MAX value it can reach — so a low read on a
    # hard rock is perfectly fine (the rock just hasn't been mined
    # much yet). Only the upper bound is meaningful here.
    if field == "instability":
        difficulty = context.get("difficulty")
        if difficulty and difficulty in _DIFFICULTY_INSTAB:
            _, d_hi = _DIFFICULTY_INSTAB[difficulty]
            # 50% overshoot tolerance for patch drift / unusual rocks.
            soft_hi = d_hi * 1.5
            if value > soft_hi:
                return False, (
                    f"instability {value} exceeds difficulty={difficulty} "
                    f"ceiling (~{d_hi})"
                )

    # Resistance is (almost) always an integer 0-100 for breakable
    # rocks. Already covered by universal range but flag noise.
    if field == "resistance":
        if value < 0 or value > 100:
            return False, f"resistance {value} outside 0-100"

    return True, "ok"


def try_decimal_recovery(
    field: str,
    raw_text: str,
    context: Optional[dict] = None,
) -> Optional[float]:
    """Try inserting a decimal point at every position, return the
    plausible one (if any).

    Rescues the common Tesseract failure where a small-size `.` is
    dropped or misread as punctuation — e.g. `0.76` decoded as `©78`
    → regex extracts `78` → priors reject as too large for EASY →
    this function tries `7.8` (plausible), `0.78` (implausible on
    instability), etc. and returns the first plausible candidate.
    """
    # Strip non-digit/% chars; we'll reinsert the dot
    digits = "".join(c for c in raw_text if c.isdigit())
    has_pct = "%" in raw_text
    if len(digits) < 2:
        return None

    best_val = None
    best_score = -1.0
    # Try decimal at positions 1..len-1 AND as leading "0.XX"
    candidates: list[str] = []
    for i in range(1, len(digits)):
        candidates.append(digits[:i] + "." + digits[i:])
    candidates.append("0." + digits)  # leading-zero case

    # SC HUD instability is ALWAYS rendered with exactly 2 decimal
    # places (e.g. "2.78", "27.80", "382.36"). When a digits-only OCR
    # read has the dot dropped, the dot's true position is therefore
    # uniquely determined by length: 3 digits → "X.XX", 4 → "XX.XX",
    # 5 → "XXX.XX", etc. Prefer this canonical placement first; only
    # if it's implausible per game-state priors do we consider other
    # positions. This stops the priors-driven scorer from "rescuing"
    # `278` as `27.80` for a MEDIUM rock when the truth is `2.78`.
    if field == "instability" and len(digits) >= 3:
        canonical = digits[:-2] + "." + digits[-2:]
        try:
            cv = float(canonical)
            # Universal range only — accept canonical placement even
            # if difficulty bounds reject it, since the HUD format
            # prior is more reliable than our hand-calibrated
            # difficulty bands (which have known edge-case misses).
            lo, hi = _UNIVERSAL_RANGES.get(field, (float("-inf"), float("inf")))
            if lo <= cv <= hi:
                return cv
        except ValueError:
            pass

    for c in candidates:
        try:
            v = float(c)
        except ValueError:
            continue
        ok, _ = is_plausible(field, v, context)
        if not ok:
            continue
        s = score(field, v, context)
        if s > best_score:
            best_score = s
            best_val = v
    return best_val


def score(field: str, value: float, context: Optional[dict] = None) -> float:
    """Return a [0, 1] plausibility score.

    Used to break ties between candidate values when multiple engines
    disagree and both pass hard plausibility. Higher = more plausible.
    """
    ok, _ = is_plausible(field, value, context)
    if not ok:
        return 0.0

    context = context or {}
    if field == "instability":
        difficulty = context.get("difficulty")
        if difficulty and difficulty in _DIFFICULTY_INSTAB:
            _, d_hi = _DIFFICULTY_INSTAB[difficulty]
            # Uniform within [0, d_hi] (any current instability up
            # to the ceiling is equally plausible); linearly drops
            # past d_hi out to the soft bound d_hi * 1.5 where it
            # hits 0. Tie-breaking for try_decimal_recovery is now
            # mostly driven by the universal-range check + OCR
            # confidence rather than this score — which is correct,
            # because within the legal range we have no game-state
            # signal favouring one value over another.
            if value <= d_hi:
                return 1.0
            overshoot = (value - d_hi) / max(d_hi * 0.5, 1e-6)
            return max(0.0, 1.0 - overshoot)

    return 1.0
