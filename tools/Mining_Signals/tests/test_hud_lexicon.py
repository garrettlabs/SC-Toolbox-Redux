"""Tests for ocr.sc_ocr.hud_lexicon — the per-field learned lexicon
that backs the HUD-RGB CRNN gate's lexicon-confirmed relaxation.

Covers:
* round-trip persistence (observe → save → reload → is_known still True)
* LRU eviction once the bounded buffer fills past ``_LEXICON_CAP``
* canonical-form key handling (mass=10810 == 10810.0 == 10810.4)
* instability dot precision (12.09 != 12.10 even though they're close)
* fail-open on unknown / missing field / non-numeric input
* concurrent-write safety via the module lock
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from ocr.sc_ocr import hud_lexicon


@pytest.fixture(autouse=True)
def _isolated_disk(tmp_path: Path, monkeypatch):
    """Redirect the lexicon's on-disk path to a temp file and reset
    module state before each test so tests don't share lexicon state.
    """
    p = tmp_path / "learned_lexicon.json"
    hud_lexicon._set_disk_path_for_tests(p)
    hud_lexicon.reset()
    yield p
    hud_lexicon._set_disk_path_for_tests(None)
    hud_lexicon.reset()


def test_observe_then_is_known_round_trip():
    hud_lexicon.observe("mass", 10810)
    assert hud_lexicon.is_known("mass", 10810) is True
    assert hud_lexicon.is_known("mass", 10810.0) is True
    # Canonical key rounds, so float drift still hits.
    assert hud_lexicon.is_known("mass", 10810.4) is True
    assert hud_lexicon.is_known("mass", 99999) is False


def test_instability_decimal_precision():
    hud_lexicon.observe("instability", 12.09)
    assert hud_lexicon.is_known("instability", 12.09) is True
    # 2-dp canonical form: 12.09 vs 12.10 must NOT collide.
    assert hud_lexicon.is_known("instability", 12.10) is False
    # Float drift within the rounding window does hit.
    assert hud_lexicon.is_known("instability", 12.094) is True


def test_persistence_across_reload(_isolated_disk: Path):
    hud_lexicon.observe("mass", 10810)
    hud_lexicon.observe("resistance", 50)
    hud_lexicon.observe("instability", 11.01)

    assert _isolated_disk.is_file(), "should have persisted on observe"
    data = json.loads(_isolated_disk.read_text(encoding="utf-8"))
    assert data["mass"] == [10810.0]
    assert data["resistance"] == [50.0]
    assert data["instability"] == [11.01]

    # Simulate process restart: clear in-memory state and re-query.
    # The lazy loader should refill from disk on the next is_known.
    hud_lexicon.reset()
    assert hud_lexicon.is_known("mass", 10810) is True
    assert hud_lexicon.is_known("resistance", 50) is True
    assert hud_lexicon.is_known("instability", 11.01) is True


def test_lru_eviction_at_cap(monkeypatch):
    # Shrink the cap so the test is fast.
    monkeypatch.setattr(hud_lexicon, "_LEXICON_CAP", 5)
    for i in range(10):
        hud_lexicon.observe("mass", 1000 + i)
    # Only the last 5 should remain.
    for i in range(5):
        assert hud_lexicon.is_known("mass", 1000 + i) is False
    for i in range(5, 10):
        assert hud_lexicon.is_known("mass", 1000 + i) is True
    assert hud_lexicon.size("mass") == 5


def test_lru_touch_keeps_recently_seen_alive(monkeypatch):
    """Repeated observation of the same value should refresh its
    position so it isn't evicted when newer values pour in.
    """
    monkeypatch.setattr(hud_lexicon, "_LEXICON_CAP", 3)
    hud_lexicon.observe("mass", 100)  # will become oldest
    hud_lexicon.observe("mass", 200)
    hud_lexicon.observe("mass", 300)
    # Now 100 is the oldest. Touch it so it moves to the tail.
    hud_lexicon.observe("mass", 100)
    # Add a fourth value; 200 should evict (not 100, which we just touched).
    hud_lexicon.observe("mass", 400)
    assert hud_lexicon.is_known("mass", 100) is True
    assert hud_lexicon.is_known("mass", 200) is False
    assert hud_lexicon.is_known("mass", 300) is True
    assert hud_lexicon.is_known("mass", 400) is True


def test_fail_open_on_unknown_field():
    # Querying an unrecognized field returns False; observing it is a no-op.
    hud_lexicon.observe("foo", 42)
    assert hud_lexicon.is_known("foo", 42) is False


def test_fail_open_on_non_numeric_value():
    # Garbage input must not raise — gate code calls this on every
    # CRNN read including borderline / malformed ones.
    hud_lexicon.observe("mass", "not a number")  # type: ignore[arg-type]
    hud_lexicon.observe("mass", None)  # type: ignore[arg-type]
    assert hud_lexicon.is_known("mass", "not a number") is False  # type: ignore[arg-type]
    assert hud_lexicon.is_known("mass", None) is False  # type: ignore[arg-type]


def test_get_values_returns_snapshot():
    hud_lexicon.observe("instability", 11.01)
    hud_lexicon.observe("instability", 12.09)
    snap = hud_lexicon.get_values("instability")
    assert snap == {11.01, 12.09}
    # Mutating the snapshot must not affect the module state.
    snap.add(999.0)
    assert hud_lexicon.is_known("instability", 999.0) is False


def test_concurrent_observe_no_corruption():
    """Lock should serialize writes; values from concurrent threads
    end up in the buffer without races or duplicates."""

    def _worker(start: int):
        for i in range(20):
            hud_lexicon.observe("mass", start + i)

    threads = [
        threading.Thread(target=_worker, args=(s * 100,))
        for s in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 4 workers × 20 values = 80 unique values; well under the cap.
    assert hud_lexicon.size("mass") == 80


def test_persist_survives_corrupt_json(_isolated_disk: Path):
    """A garbage JSON file shouldn't crash the loader — just start
    with an empty lexicon and let new observes populate it."""
    _isolated_disk.parent.mkdir(parents=True, exist_ok=True)
    _isolated_disk.write_text("not valid json {{{", encoding="utf-8")
    hud_lexicon.reset()
    # No crash on first query.
    assert hud_lexicon.is_known("mass", 10810) is False
    # And subsequent observe still works.
    hud_lexicon.observe("mass", 10810)
    assert hud_lexicon.is_known("mass", 10810) is True


def test_resistance_integer_canonicalization():
    """resistance values like '0' / '1' / '50' / '100' should
    canonicalize as ints — float drift from CRNN-derived parsing
    shouldn't fragment the buffer."""
    hud_lexicon.observe("resistance", 50.0)
    assert hud_lexicon.is_known("resistance", 50) is True
    assert hud_lexicon.is_known("resistance", 50.49) is True
    # Python ``round()`` uses banker's rounding (round-half-to-even),
    # so 50.5 rounds to 50 (hits the lexicon entry) while 51.5 rounds
    # to 52 (misses). The exact tie-breaking direction matters less
    # than the fact that we collapse near-integers — which is the
    # whole point of canonicalizing.
    assert hud_lexicon.is_known("resistance", 50.5) is True
    assert hud_lexicon.is_known("resistance", 51.5) is False
    assert hud_lexicon.is_known("resistance", 49.5) is True
