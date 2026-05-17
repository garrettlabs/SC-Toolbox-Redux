"""Per-field rolling lexicon of confirmed HUD reads.

Mirrors ``_KNOWN_SIGNAL_VALUES`` for the signal pipeline, but
self-learned: values are added when ``frozen_panel``'s auto-freeze
trigger confirms a (mass, resistance, instability) triple as
high-confidence (all three pass structural validators AND
mass/instab are non-zero AND optionally the three-field plausibility
check). Persisted to ``hud_tracker/learned_lexicon.json`` so the
lexicon survives restarts and grows across sessions.

Consumed by:

* ``_ocr_value_crop`` → HUD-RGB CRNN acceptance gate. When the
  CRNN's read parses to a value in the field's lexicon, the
  confidence threshold relaxes from 0.55 strict / 0.30 plausible
  to 0.30 — same trick the signal pipeline uses to accept borderline
  CRNN reads of known signatures.

* (future) beam-search rerank inside
  ``_classify_hud_value_via_crnn_rgb``. ``get_values(field)`` exposes
  the current set so a beam-decode rerank can prefer in-lexicon
  candidates over higher-confidence-but-implausible ones.

The lexicon is a TRUE rolling set — bounded at ``_LEXICON_CAP``
entries per field with LRU eviction. The bound is intentional:
the goal is to capture the values the user actually encounters
in their current session / play style (mineral types they mine,
locations they scan), not to be an exhaustive registry. A 100-entry
cap covers the practical recurrence rate without bloating the disk
footprint.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Bounded buffer size per field. Keeps the on-disk JSON small
# (typically <5 KB after a long session) while covering the
# recurring-value population — most rocks have unique values, but
# the same mineral type at similar deposit sizes produces similar
# readings, so a small LRU buffer concentrates hits naturally.
_LEXICON_CAP = 100

# Module-level state.  Maps field → OrderedDict[str_key, float_value].
# OrderedDict gives us O(1) LRU semantics via ``move_to_end`` /
# ``popitem(last=False)``. String keys are the canonical-form
# (see :func:`_key_for`) so float drift can't create duplicates.
_lexicon: dict[str, "OrderedDict[str, float]"] = {
    "mass": OrderedDict(),
    "resistance": OrderedDict(),
    "instability": OrderedDict(),
}
_lock = threading.Lock()
_loaded = False
_disk_path: Optional[Path] = None


def _get_disk_path() -> Path:
    """Return the on-disk JSON path, lazily resolved on first use."""
    global _disk_path
    if _disk_path is not None:
        return _disk_path
    # ``api.py`` lives at ``<tree>/ocr/sc_ocr/api.py``; the
    # ``hud_tracker`` directory is a sibling of ``ocr/`` at the
    # tree root. Mirror of the world-model loader path.
    root = Path(__file__).resolve().parent.parent.parent
    _disk_path = root / "hud_tracker" / "learned_lexicon.json"
    return _disk_path


def _key_for(field: str, value: float) -> str:
    """Canonical-form key for a (field, value) pair.

    * ``mass`` / ``resistance`` are integer-valued in the HUD UI
      (mass kg, resistance %) — drop fractional drift via ``round``.
    * ``instability`` is decimal — round to 2 dp to match the
      ``XX.XX`` precision the HUD itself displays.
    """
    if field in ("mass", "resistance"):
        return str(int(round(float(value))))
    return f"{float(value):.2f}"


def _load_from_disk() -> None:
    """Best-effort load from the on-disk JSON.

    Called lazily on first access. Marks ``_loaded`` True even on
    parse failure so we don't keep retrying a corrupt file every
    scan.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    p = _get_disk_path()
    if not p.is_file():
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.debug("hud_lexicon: load failed (%s) — starting empty", exc)
        return
    if not isinstance(data, dict):
        return
    with _lock:
        for field in ("mass", "resistance", "instability"):
            arr = data.get(field)
            if not isinstance(arr, list):
                continue
            # On-disk list is in observed order (oldest first); only
            # keep the tail so we honor the cap after a manual edit
            # or a previous run with a larger cap.
            for v in arr[-_LEXICON_CAP:]:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                k = _key_for(field, f)
                _lexicon[field][k] = f
    log.info(
        "hud_lexicon: loaded from %s — mass=%d resistance=%d "
        "instability=%d",
        p, len(_lexicon["mass"]), len(_lexicon["resistance"]),
        len(_lexicon["instability"]),
    )


def _persist_to_disk() -> None:
    """Best-effort persist. Silent on any I/O failure."""
    p = _get_disk_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        snapshot: dict[str, list[float]] = {}
        with _lock:
            for field in ("mass", "resistance", "instability"):
                # Preserve insertion order so the next load gives
                # LRU-correct eviction.
                snapshot[field] = list(_lexicon[field].values())
        p.write_text(
            json.dumps(snapshot, separators=(",", ":")),
            encoding="utf-8",
        )
    except Exception as exc:
        log.debug("hud_lexicon: persist failed: %s", exc)


def observe(field: str, value: float) -> None:
    """Record a confirmed value for the given field.

    Designed to be called from the ``frozen_panel.freeze`` hook —
    the value has already passed structural validators (mass ≥ 2
    digits, instability has a ``.``, resistance ≤ 100) AND the
    three-field plausibility trigger (all three present, mass /
    instab non-zero), so it's high-confidence enough to learn from.

    LRU eviction keeps the buffer bounded. Repeat observations of
    the same value touch the entry (move to the end) so frequently-
    seen values stay alive even when new values flow in.

    Persists to disk after each new addition so a crash mid-session
    doesn't lose the lexicon. Touch-on-existing skips the disk write.
    """
    if field not in _lexicon:
        return
    try:
        v = float(value)
    except (TypeError, ValueError):
        return
    _load_from_disk()
    key = _key_for(field, v)
    is_new = False
    with _lock:
        d = _lexicon[field]
        if key in d:
            d.move_to_end(key)
        else:
            d[key] = v
            is_new = True
            while len(d) > _LEXICON_CAP:
                d.popitem(last=False)
    if is_new:
        _persist_to_disk()
        log.debug(
            "hud_lexicon: observed new %s=%s (size=%d)",
            field, key, len(_lexicon[field]),
        )


def is_known(field: str, value: float) -> bool:
    """Return True when ``value`` is in the lexicon for ``field``.

    Comparison uses the canonical-form key so float-rounding drift
    (e.g. ``12.090000001`` vs ``12.09``) doesn't cause a false
    negative.
    """
    if field not in _lexicon:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    _load_from_disk()
    key = _key_for(field, v)
    with _lock:
        return key in _lexicon[field]


def get_values(field: str) -> set[float]:
    """Snapshot of the current value set for ``field``.

    Returns a copy; safe to inspect without holding the module
    lock. Used by the (future) CRNN beam-search rerank to bias
    decoding toward in-lexicon candidates.
    """
    if field not in _lexicon:
        return set()
    _load_from_disk()
    with _lock:
        return set(_lexicon[field].values())


def size(field: str) -> int:
    """Return the current entry count for ``field``."""
    if field not in _lexicon:
        return 0
    _load_from_disk()
    with _lock:
        return len(_lexicon[field])


def reset() -> None:
    """Clear the in-memory lexicon. Intended for tests."""
    global _loaded
    with _lock:
        for field in _lexicon:
            _lexicon[field].clear()
        _loaded = False


def _set_disk_path_for_tests(path: Optional[Path]) -> None:
    """Override the on-disk path. Intended for tests."""
    global _disk_path, _loaded
    _disk_path = path
    _loaded = False
