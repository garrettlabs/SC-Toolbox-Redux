"""Profile loader for the SC-OCR engine.

Each scan target (mining_hud, refinery, signature_scanner, commodity_terminal)
has a JSON profile in ``ocr/sc_ocr/profiles/`` describing which model to use,
which characters are valid, and per-field validation rules.

This module loads those profiles lazily, exposes them as typed dataclasses,
and provides small validation helpers used by the rest of the OCR engine.

Stdlib only -- no external dependencies.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Directory containing the profile JSON files.
_PROFILES_DIR = Path(__file__).parent / "profiles"

# Module-level cache: profile_name -> Profile.
# Single-threaded scan loop, so a plain dict is sufficient.
_CACHE: Dict[str, "Profile"] = {}


@dataclass(frozen=True)
class FieldRule:
    """Validation rules for a single field within a profile.

    All attributes are optional; only those present in the JSON will be
    enforced by :func:`validate_field`.
    """

    chars: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    must_be_int: bool = False
    must_end_with: Optional[str] = None
    pattern: Optional[str] = None
    vocabulary: Optional[str] = None
    min_confidence: Optional[float] = None


@dataclass(frozen=True)
class Profile:
    """A loaded OCR profile."""

    name: str
    model: Optional[str] = None
    model_purpose: Optional[str] = None
    global_char_whitelist: Optional[str] = None
    min_confidence: float = 0.0
    fields: Dict[str, FieldRule] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _build_field_rule(data: Any) -> FieldRule:
    """Construct a FieldRule from a JSON dict, ignoring unknown keys."""
    if not isinstance(data, dict):
        return FieldRule()
    return FieldRule(
        chars=data.get("chars"),
        min=data.get("min"),
        max=data.get("max"),
        must_be_int=bool(data.get("must_be_int", False)),
        must_end_with=data.get("must_end_with"),
        pattern=data.get("pattern"),
        vocabulary=data.get("vocabulary"),
        min_confidence=data.get("min_confidence"),
    )


def _build_profile(name: str, data: Dict[str, Any]) -> Profile:
    raw_fields = data.get("fields") or {}
    fields_dict: Dict[str, FieldRule] = {}
    if isinstance(raw_fields, dict):
        for fname, frule in raw_fields.items():
            fields_dict[fname] = _build_field_rule(frule)

    try:
        min_conf = float(data.get("min_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        min_conf = 0.0

    return Profile(
        name=str(data.get("name") or name),
        model=data.get("model"),
        model_purpose=data.get("model_purpose"),
        global_char_whitelist=data.get("global_char_whitelist"),
        min_confidence=min_conf,
        fields=fields_dict,
        raw=data if isinstance(data, dict) else {},
    )


def _default_profile(name: str) -> Profile:
    """Return a permissive default profile when JSON is missing/broken."""
    logger.warning("Returning default profile for %r (load failed or missing)", name)
    return Profile(name=name)


def _load_profile_from_disk(name: str) -> Profile:
    path = _PROFILES_DIR / f"{name}.json"
    if not path.is_file():
        logger.warning("Profile file not found: %s", path)
        return _default_profile(name)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load profile %s: %s", path, exc)
        return _default_profile(name)
    if not isinstance(data, dict):
        logger.warning("Profile %s has non-object root; using default", path)
        return _default_profile(name)
    try:
        return _build_profile(name, data)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to construct profile %s: %s", name, exc)
        return _default_profile(name)


def get_profile(name: str) -> Profile:
    """Return the profile with the given name, loading and caching as needed."""
    cached = _CACHE.get(name)
    if cached is not None:
        return cached
    profile = _load_profile_from_disk(name)
    _CACHE[name] = profile
    return profile


def clear_cache() -> None:
    """Drop all cached profiles. Mainly useful for tests."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def apply_char_whitelist(text: str, allowed: str) -> str:
    """Return ``text`` with any character not in ``allowed`` removed."""
    if not allowed:
        return text
    allowed_set = set(allowed)
    return "".join(ch for ch in text if ch in allowed_set)


def _coerce_number(text: str) -> Optional[float]:
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def validate_field(
    profile: Profile,
    field: str,
    text: str,
    confs: List[float],
) -> Tuple[Optional[str], Optional[str]]:
    """Validate an OCR'd field value against a profile.

    Returns a tuple ``(cleaned_text, reason_if_rejected)``. On success the
    second element is ``None``. On rejection the first element is ``None``
    and the second is a short human-readable reason string.
    """
    if text is None:
        return None, "no_text"

    rule = profile.fields.get(field, FieldRule())

    # Per-character minimum confidence (field rule overrides profile-level).
    min_conf = rule.min_confidence
    if min_conf is None:
        min_conf = profile.min_confidence
    if min_conf and confs:
        # Only inspect confidences for chars actually present in text.
        for i, ch in enumerate(text):
            if i >= len(confs):
                break
            if confs[i] < min_conf:
                return None, f"low_confidence:{ch}@{i}={confs[i]:.2f}<{min_conf:.2f}"

    # Character whitelist (field rule trumps the global one).
    allowed = rule.chars or profile.global_char_whitelist
    if allowed:
        cleaned = apply_char_whitelist(text, allowed)
        if cleaned != text:
            return None, f"bad_chars:{text!r}->{cleaned!r}"
    else:
        cleaned = text

    # must_end_with check (use original cleaned text before stripping suffix).
    if rule.must_end_with and not cleaned.endswith(rule.must_end_with):
        return None, f"missing_suffix:{rule.must_end_with!r}"

    # Numeric range checks.
    needs_numeric = (
        rule.min is not None
        or rule.max is not None
        or rule.must_be_int
    )
    if needs_numeric:
        numeric_text = cleaned
        if rule.must_end_with and numeric_text.endswith(rule.must_end_with):
            numeric_text = numeric_text[: -len(rule.must_end_with)]
        value = _coerce_number(numeric_text)
        if value is None:
            return None, f"not_numeric:{cleaned!r}"
        if rule.must_be_int and not float(value).is_integer():
            return None, f"not_int:{cleaned!r}"
        if rule.min is not None and value < rule.min:
            return None, f"below_min:{value}<{rule.min}"
        if rule.max is not None and value > rule.max:
            return None, f"above_max:{value}>{rule.max}"

    # Regex pattern.
    if rule.pattern:
        try:
            if not re.fullmatch(rule.pattern, cleaned):
                return None, f"pattern_mismatch:{rule.pattern!r}"
        except re.error as exc:
            logger.warning("Bad regex in profile %s field %s: %s",
                           profile.name, field, exc)

    # Vocabulary snap-to-closest. Vocabulary content lives elsewhere; here we
    # only know it as a name. If a list-of-strings vocabulary has been stashed
    # on the rule via raw JSON, support it too as a convenience.
    raw_vocab = None
    raw_field = profile.raw.get("fields", {}).get(field, {}) if isinstance(
        profile.raw.get("fields"), dict) else {}
    if isinstance(raw_field, dict):
        v = raw_field.get("vocabulary")
        if isinstance(v, list):
            raw_vocab = [str(x) for x in v]
    if raw_vocab:
        matches = difflib.get_close_matches(cleaned, raw_vocab, n=1, cutoff=0.6)
        if matches:
            cleaned = matches[0]
        else:
            return None, f"vocab_no_match:{cleaned!r}"

    return cleaned, None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    p = get_profile("mining_hud")
    print(f"profile.name   = {p.name}")
    print(f"profile.model  = {p.model}")
    for fname, frule in p.fields.items():
        print(f"  field {fname!r}: chars={frule.chars!r}")

    wl = apply_char_whitelist("AB12.3%X", "0123456789.%")
    print(f"apply_char_whitelist('AB12.3%X', '0123456789.%') -> {wl!r}")
    assert wl == "12.3%", f"expected '12.3%', got {wl!r}"

    ok = validate_field(p, "mass", "5912", [0.99, 0.98, 0.97, 0.96])
    print(f"validate_field mass '5912' -> {ok}")
    assert ok[0] == "5912" and ok[1] is None, f"unexpected: {ok}"

    bad = validate_field(p, "mass", "59A12", [0.99, 0.98, 0.40, 0.97, 0.96])
    print(f"validate_field mass '59A12' -> {bad}")
    assert bad[0] is None and bad[1] is not None, f"expected rejection, got: {bad}"

    print("self-test OK")
