"""Tests for the frozen panel reference lifecycle.

Covers:
  * Initial state — not frozen
  * freeze() snapshots values + image, sets timestamps
  * refresh_title_seen() bumps the title-seen timestamp
  * is_expired() respects the timeout
  * clear() returns to UNFROZEN
  * Per-region singletons isolate state between regions
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ocr.sc_ocr.frozen_panel import (  # noqa: E402
    FrozenPanelReference,
    get_frozen_ref,
    reset_all,
)


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Clear module-level state between tests so per-region isolation
    tests can't leak singletons across the suite.
    """
    reset_all()
    yield
    reset_all()


def _make_img(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(45, 253, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_values(mass=54663.0, resist=82.0, instab=15.47, mineral="Raw Ice"):
    return {
        "mass": mass,
        "resistance": resist,
        "instability": instab,
        "mineral_name": mineral,
        "panel_visible": True,
    }


# ─── Initial state ──────────────────────────────────────────────────


def test_construction_is_unfrozen() -> None:
    ref = FrozenPanelReference()
    assert ref.is_frozen is False
    assert ref.panel_image is None
    assert ref.values == {}
    assert ref.frozen_at == 0.0
    assert ref.last_title_seen_at == 0.0


def test_age_seconds_zero_when_unfrozen() -> None:
    ref = FrozenPanelReference()
    assert ref.age_seconds() == 0.0
    assert ref.time_since_title_seen() == 0.0


def test_is_expired_false_when_unfrozen() -> None:
    ref = FrozenPanelReference()
    # Even with a tiny timeout, unfrozen never expires (there's
    # nothing to expire).
    assert ref.is_expired(timeout_sec=0.001) is False


# ─── freeze() ───────────────────────────────────────────────────────


def test_freeze_snapshots_state() -> None:
    ref = FrozenPanelReference()
    img = _make_img()
    values = _make_values()

    ref.freeze(img, values)

    assert ref.is_frozen is True
    assert ref.panel_image is not None
    assert ref.values == values
    assert ref.frozen_at > 0.0
    assert ref.last_title_seen_at > 0.0
    # frozen_at and last_title_seen_at should be set to the same
    # moment at the moment of freeze.
    assert abs(ref.frozen_at - ref.last_title_seen_at) < 0.01


def test_freeze_copies_image_so_caller_mutation_is_isolated() -> None:
    """The frozen reference must hold its own pixel state so that
    downstream annotation of the live img doesn't corrupt the
    snapshot.
    """
    ref = FrozenPanelReference()
    img = _make_img(seed=1)
    ref.freeze(img, _make_values())

    # Mutate the caller's img.
    arr = np.array(img)
    arr[:, :, :] = 0
    mutated = Image.fromarray(arr, mode="RGB")
    img.paste(mutated)

    # The frozen image should NOT have changed.
    frozen_arr = np.array(ref.panel_image)
    assert frozen_arr.sum() > 0, "frozen image was mutated by caller"


def test_freeze_replaces_existing_freeze() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(seed=1), _make_values(mass=1.0))
    first_at = ref.frozen_at

    # Small delay so timestamps actually differ.
    time.sleep(0.01)
    ref.freeze(_make_img(seed=2), _make_values(mass=2.0))

    assert ref.values["mass"] == 2.0
    assert ref.frozen_at > first_at


# ─── refresh_title_seen() ───────────────────────────────────────────


def test_refresh_title_seen_bumps_timestamp() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values())
    initial = ref.last_title_seen_at

    time.sleep(0.02)
    ref.refresh_title_seen()

    assert ref.last_title_seen_at > initial


def test_refresh_title_seen_noop_when_unfrozen() -> None:
    ref = FrozenPanelReference()
    ref.refresh_title_seen()
    assert ref.last_title_seen_at == 0.0


# ─── is_expired() ───────────────────────────────────────────────────


def test_not_expired_immediately_after_freeze() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values())
    assert ref.is_expired(timeout_sec=3.0) is False


def test_expires_after_timeout(monkeypatch) -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values())

    # Fast-forward the monotonic clock by 5 seconds via patching.
    real_monotonic = time.monotonic
    base = real_monotonic()
    monkeypatch.setattr(
        time, "monotonic", lambda: base + 5.0,
    )
    # last_title_seen_at was set to the original base time. Now
    # monotonic() returns base+5, so 5s elapsed → expired at 3s
    # threshold.
    assert ref.is_expired(timeout_sec=3.0) is True
    assert ref.is_expired(timeout_sec=10.0) is False


def test_refresh_title_seen_resets_expiry(monkeypatch) -> None:
    """Calling refresh_title_seen() right before the timeout should
    push expiry back.
    """
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values())

    # 2.5s after freeze: not yet expired.
    real_monotonic = time.monotonic
    base = real_monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: base + 2.5)
    assert ref.is_expired(timeout_sec=3.0) is False

    # Refresh title-seen. Expiry timer resets to the new "now".
    ref.refresh_title_seen()

    # 4.5s after the original freeze (2s after refresh) — should
    # NOT be expired because refresh reset the clock.
    monkeypatch.setattr(time, "monotonic", lambda: base + 4.5)
    assert ref.is_expired(timeout_sec=3.0) is False

    # 6s after refresh: NOW it's expired.
    monkeypatch.setattr(time, "monotonic", lambda: base + 2.5 + 6.0)
    assert ref.is_expired(timeout_sec=3.0) is True


# ─── clear() ────────────────────────────────────────────────────────


def test_clear_returns_to_unfrozen() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values())
    assert ref.is_frozen

    ref.clear()

    assert ref.is_frozen is False
    assert ref.panel_image is None
    assert ref.values == {}
    assert ref.frozen_at == 0.0
    assert ref.last_title_seen_at == 0.0


def test_clear_on_unfrozen_is_noop() -> None:
    ref = FrozenPanelReference()
    ref.clear()  # Should not raise
    assert ref.is_frozen is False


# ─── Per-region singletons ──────────────────────────────────────────


def test_get_frozen_ref_returns_same_instance_for_same_region() -> None:
    region = {"x": 100, "y": 200, "w": 400, "h": 300}
    ref1 = get_frozen_ref(region)
    ref2 = get_frozen_ref(region)
    assert ref1 is ref2


def test_get_frozen_ref_isolates_distinct_regions() -> None:
    region_a = {"x": 100, "y": 200, "w": 400, "h": 300}
    region_b = {"x": 500, "y": 600, "w": 400, "h": 300}
    ref_a = get_frozen_ref(region_a)
    ref_b = get_frozen_ref(region_b)
    assert ref_a is not ref_b

    ref_a.freeze(_make_img(seed=1), _make_values(mass=11.0))
    assert ref_a.is_frozen is True
    assert ref_b.is_frozen is False  # B unaffected


def test_get_frozen_ref_none_region_uses_default_key() -> None:
    ref1 = get_frozen_ref(None)
    ref2 = get_frozen_ref(None)
    assert ref1 is ref2


def test_reset_all_clears_every_region() -> None:
    region_a = {"x": 1, "y": 2, "w": 3, "h": 4}
    region_b = {"x": 5, "y": 6, "w": 7, "h": 8}
    ref_a = get_frozen_ref(region_a)
    ref_b = get_frozen_ref(region_b)
    ref_a.freeze(_make_img(), _make_values())
    ref_b.freeze(_make_img(), _make_values())

    reset_all()

    # After reset_all, the singletons should be cleared. Re-fetching
    # gives fresh, unfrozen instances.
    assert get_frozen_ref(region_a).is_frozen is False
    assert get_frozen_ref(region_b).is_frozen is False


# ─── Divergence auto-clear ──────────────────────────────────────────


def test_record_live_reading_noop_when_unfrozen() -> None:
    ref = FrozenPanelReference()
    cleared = ref.record_live_reading("mass", 54663.0)
    assert cleared is False


def test_agreement_resets_divergence_counter() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(mass=54663.0))

    # Two consecutive disagreements...
    ref.record_live_reading("mass", 88.0)
    ref.record_live_reading("mass", 88.0)
    assert ref._divergence["mass"] == 2

    # ...then an agreement resets the counter.
    cleared = ref.record_live_reading("mass", 54663.0)
    assert cleared is False
    assert ref._divergence["mass"] == 0
    assert ref.is_frozen is True


def test_three_consecutive_disagreements_auto_clear() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(instab=15.46))

    # Live reads 340.17 three times in a row — clearly different.
    ref.record_live_reading("instability", 340.17)
    assert ref.is_frozen is True
    ref.record_live_reading("instability", 340.17)
    assert ref.is_frozen is True
    cleared = ref.record_live_reading("instability", 340.17)
    assert cleared is True
    assert ref.is_frozen is False


def test_instability_tolerance_accepts_minor_drift() -> None:
    """Instability has a ±0.5 tolerance so 15.46 vs 15.47 (OCR
    rounding on the last digit) doesn't count as disagreement.
    """
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(instab=15.46))

    # 15.47 is within tolerance.
    ref.record_live_reading("instability", 15.47)
    assert ref._divergence["instability"] == 0

    # 17.0 is well outside tolerance.
    ref.record_live_reading("instability", 17.0)
    assert ref._divergence["instability"] == 1


def test_none_live_reading_does_not_count_disagreement() -> None:
    """A live OCR that returned None (structural-validator rejected,
    etc.) is not evidence of disagreement and should not move the
    counter.
    """
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(mass=54663.0))

    ref.record_live_reading("mass", 88.0)
    assert ref._divergence["mass"] == 1
    # None should not reset or increment.
    ref.record_live_reading("mass", None)
    assert ref._divergence["mass"] == 1


def test_disagreement_on_one_field_clears_whole_freeze() -> None:
    """Even if mass + resistance still agree, three disagreements on
    instability are enough to clear the freeze — the panel changed.
    """
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(
        mass=54663.0, resist=82.0, instab=15.46,
    ))

    # Mass + resistance keep agreeing.
    for _ in range(3):
        ref.record_live_reading("mass", 54663.0)
        ref.record_live_reading("resistance", 82.0)
        # Instability disagrees.
        cleared = ref.record_live_reading("instability", 340.17)
        if cleared:
            break

    assert ref.is_frozen is False  # cleared on instability alone


def test_freeze_resets_divergence_counters() -> None:
    """A fresh freeze should zero out any leftover divergence
    counters from a previous freeze.
    """
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(mass=1.0))
    # Build up some divergence.
    ref.record_live_reading("mass", 999.0)
    ref.record_live_reading("mass", 999.0)
    assert ref._divergence["mass"] == 2

    # Re-freeze.
    ref.freeze(_make_img(), _make_values(mass=2.0))
    assert ref._divergence["mass"] == 0
    assert ref.is_frozen is True


# ── Late-arriving mineral_name back-fill ─────────────────────────────


def test_update_field_if_missing_fills_none_mineral_name() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(mineral=None))
    assert ref.values["mineral_name"] is None
    changed = ref.update_field_if_missing("mineral_name", "Iron")
    assert changed is True
    assert ref.values["mineral_name"] == "Iron"


def test_update_field_if_missing_no_overwrite() -> None:
    # Already-populated field is left alone.
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(mineral="Raw Ice"))
    changed = ref.update_field_if_missing("mineral_name", "Iron")
    assert changed is False
    assert ref.values["mineral_name"] == "Raw Ice"


def test_update_field_if_missing_rejects_numeric_fields() -> None:
    # Numeric fields are immutable — the whole point of the freeze.
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(mass=54663.0))
    changed = ref.update_field_if_missing("mass", 99.0)
    assert changed is False
    assert ref.values["mass"] == 54663.0


def test_update_field_if_missing_noop_when_unfrozen() -> None:
    ref = FrozenPanelReference()
    changed = ref.update_field_if_missing("mineral_name", "Iron")
    assert changed is False
    assert ref.is_frozen is False


def test_update_field_if_missing_ignores_none_value() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(mineral=None))
    changed = ref.update_field_if_missing("mineral_name", None)
    assert changed is False
    assert ref.values["mineral_name"] is None


# ── All-None auto-clear ──────────────────────────────────────────────


def test_record_scan_outcome_noop_when_unfrozen() -> None:
    ref = FrozenPanelReference()
    cleared = ref.record_scan_outcome(any_field_read=False)
    assert cleared is False


def test_record_scan_outcome_resets_streak_on_any_read() -> None:
    from ocr.sc_ocr.frozen_panel import _ALL_NONE_THRESHOLD_FRAMES
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values())
    # Build up a streak just below threshold.
    for _ in range(_ALL_NONE_THRESHOLD_FRAMES - 1):
        cleared = ref.record_scan_outcome(any_field_read=False)
        assert cleared is False
    # A single successful read resets it.
    ref.record_scan_outcome(any_field_read=True)
    # Now we should need a full fresh threshold to clear.
    for _ in range(_ALL_NONE_THRESHOLD_FRAMES - 1):
        assert ref.record_scan_outcome(any_field_read=False) is False
    assert ref.is_frozen is True


def test_record_scan_outcome_clears_after_threshold_all_none() -> None:
    from ocr.sc_ocr.frozen_panel import _ALL_NONE_THRESHOLD_FRAMES
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values())
    cleared = False
    for _ in range(_ALL_NONE_THRESHOLD_FRAMES):
        cleared = ref.record_scan_outcome(any_field_read=False)
        if cleared:
            break
    assert cleared is True
    assert ref.is_frozen is False


def test_record_scan_outcome_resets_on_re_freeze() -> None:
    from ocr.sc_ocr.frozen_panel import _ALL_NONE_THRESHOLD_FRAMES
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values())
    # Accumulate streak to within 1 of threshold.
    for _ in range(_ALL_NONE_THRESHOLD_FRAMES - 1):
        ref.record_scan_outcome(any_field_read=False)
    assert ref._all_none_streak == _ALL_NONE_THRESHOLD_FRAMES - 1
    # Fresh freeze zeros the counter.
    ref.freeze(_make_img(), _make_values())
    assert ref._all_none_streak == 0


# ── Snapshot re-OCR (raw_image + calibration version tracking) ──────


def test_freeze_stores_raw_image_separately() -> None:
    """``freeze()`` accepts a separate ``raw_img`` kwarg. ``raw_image``
    property returns that copy; ``panel_image`` returns the display
    (annotated) copy. They can be different.
    """
    raw = _make_img(seed=0)
    annotated = _make_img(seed=1)  # different bytes
    ref = FrozenPanelReference()
    ref.freeze(annotated, _make_values(), raw_img=raw)
    assert ref.panel_image is not None
    assert ref.raw_image is not None
    # The stored images are independent copies; mutating the caller's
    # raw shouldn't affect what's stored.
    assert ref.raw_image.size == raw.size
    assert ref.panel_image.size == annotated.size


def test_freeze_raw_image_falls_back_to_panel_image() -> None:
    """When the caller doesn't supply ``raw_img``, ``raw_image``
    property still returns something usable (the same display image).
    Keeps backward compat with callers that don't know about re-OCR.
    """
    img = _make_img()
    ref = FrozenPanelReference()
    ref.freeze(img, _make_values())
    assert ref.raw_image is not None  # fallback to panel_image


def test_freeze_stores_calibration_version() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(), calibration_version=42)
    assert ref.calibration_version_at_freeze == 42
    assert ref.calibration_version_at_last_reocr == 42


def test_needs_snapshot_reocr_false_when_unfrozen() -> None:
    ref = FrozenPanelReference()
    assert ref.needs_snapshot_reocr(current_cal_version=99) is False


def test_needs_snapshot_reocr_false_when_version_unchanged() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(), calibration_version=5)
    assert ref.needs_snapshot_reocr(current_cal_version=5) is False


def test_needs_snapshot_reocr_true_when_version_advanced() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(), calibration_version=5)
    assert ref.needs_snapshot_reocr(current_cal_version=6) is True


def test_mark_snapshot_reocr_done_blocks_further_reocr() -> None:
    """After ``mark_snapshot_reocr_done(v)``, ``needs_snapshot_reocr``
    returns False until the version advances past v.
    """
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(), calibration_version=5)
    assert ref.needs_snapshot_reocr(8) is True
    ref.mark_snapshot_reocr_done(8)
    assert ref.needs_snapshot_reocr(8) is False
    assert ref.needs_snapshot_reocr(9) is True


def test_replace_field_values_updates_frozen_numbers() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(mass=999.0, resist=99.0, instab=99.99))
    ref.replace_field_values({"mass": 54663.0, "resistance": 82.0, "instability": 15.47})
    assert ref.values["mass"] == 54663.0
    assert ref.values["resistance"] == 82.0
    assert ref.values["instability"] == 15.47
    # Freeze itself is still active.
    assert ref.is_frozen is True


def test_replace_field_values_ignores_none() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(mass=54663.0))
    ref.replace_field_values({"mass": None, "resistance": 50.0})
    # Mass keeps its original value (snapshot OCR returned None — keep
    # captured); resistance gets updated.
    assert ref.values["mass"] == 54663.0
    assert ref.values["resistance"] == 50.0


def test_replace_field_values_ignores_unknown_keys() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values())
    ref.replace_field_values({"mass": 1.0, "bogus_field": 999.0})
    assert "bogus_field" not in ref.values


def test_replace_field_values_noop_when_unfrozen() -> None:
    ref = FrozenPanelReference()
    ref.replace_field_values({"mass": 1.0})
    assert ref.is_frozen is False
    assert ref.values == {}


def test_replace_field_values_can_update_mineral_name() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(mineral=None))
    ref.replace_field_values({"mineral_name": "Iron"})
    assert ref.values["mineral_name"] == "Iron"


def test_clear_resets_calibration_version_tracking() -> None:
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values(), calibration_version=42)
    ref.mark_snapshot_reocr_done(99)
    ref.clear()
    assert ref.calibration_version_at_freeze == 0
    assert ref.calibration_version_at_last_reocr == 0
    assert ref.raw_image is None


# ─── UI-freshness gate (label-presence streak) ──────────────────────


def test_single_scan_with_zero_labels_is_noop() -> None:
    """One scan with 0 labels matched is below the clear threshold —
    no false positives from a transient single-frame anchor miss.
    """
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values())

    res = ref.record_label_match_count(0)

    assert res["action"] == "noop"
    assert res["zero_label_streak"] == 1
    assert res["low_label_streak"] == 1
    assert ref.is_frozen is True  # caller didn't clear; ref still frozen


def test_two_consecutive_zero_label_scans_request_clear() -> None:
    """Two scans in a row with 0/3 labels triggers the "panel
    definitely gone" action. The method returns the request; the
    caller is responsible for actually invoking ``clear()``.
    """
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values())

    first = ref.record_label_match_count(0)
    assert first["action"] == "noop"
    assert first["zero_label_streak"] == 1

    second = ref.record_label_match_count(0)
    assert second["action"] == "clear"
    assert second["zero_label_streak"] == 2
    assert "0/3" in second["reason"]

    # Caller acts on the request.
    ref.clear()
    assert ref.is_frozen is False
    assert ref.panel_image is None
    assert ref.values == {}


def test_three_consecutive_one_label_scans_shorten_tolerance() -> None:
    """Three consecutive ``count == 1`` scans → request to shorten the
    freeze age tolerance. Each individual scan would NOT trigger the
    full clear (zero-label streak resets when count >= 1).
    """
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values())

    # 1st scan
    r1 = ref.record_label_match_count(1)
    assert r1["action"] == "noop"
    assert r1["zero_label_streak"] == 0  # 1 label resets zero streak
    assert r1["low_label_streak"] == 1

    # 2nd scan
    r2 = ref.record_label_match_count(1)
    assert r2["action"] == "noop"
    assert r2["zero_label_streak"] == 0
    assert r2["low_label_streak"] == 2

    # 3rd scan — fires
    r3 = ref.record_label_match_count(1)
    assert r3["action"] == "shorten_tolerance"
    assert r3["zero_label_streak"] == 0
    assert r3["low_label_streak"] == 3
    assert r3["tolerance_sec"] is not None
    assert r3["tolerance_sec"] > 0


def test_shorten_tolerance_makes_old_freeze_expired(monkeypatch) -> None:
    """A 5s-old frozen reference is NOT expired at the default 3s
    timeout (because refresh_title_seen was called), but IS expired
    under the shortened tolerance reported by record_label_match_count.
    """
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values())

    # Fast-forward by 5 seconds — but bump last_title_seen along the
    # way (simulates the panel being visible enough to keep refreshing
    # but partially occluded so labels are only matching 1/3).
    base = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: base + 5.0)
    # Pretend the title was last seen 2 seconds ago (well under the
    # default 3s timeout, but over the 1s shortened tolerance).
    ref._last_title_seen_at = base + 3.0

    # At default 3s tolerance, NOT expired.
    assert ref.is_expired(timeout_sec=3.0) is False

    # Fire the 3-consecutive-low-label trigger to get the shortened
    # tolerance value.
    ref.record_label_match_count(1)
    ref.record_label_match_count(1)
    res = ref.record_label_match_count(1)
    assert res["action"] == "shorten_tolerance"

    short_tol = float(res["tolerance_sec"])
    # Under the shortened tolerance, 2 seconds since last title-seen
    # IS expired.
    assert ref.is_expired(timeout_sec=short_tol) is True


def test_streak_resets_when_count_recovers_to_two_or_more() -> None:
    """Two zero-label scans then a 3-label scan resets BOTH streak
    counters back to zero — no clear is requested.
    """
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values())

    r1 = ref.record_label_match_count(0)
    assert r1["zero_label_streak"] == 1
    assert r1["low_label_streak"] == 1

    # One more zero scan — note: this WOULD return "clear" but the
    # caller is exercising the reset-recovery path, so we don't call
    # ref.clear(). The internal streak state is what matters.
    r2 = ref.record_label_match_count(0)
    assert r2["action"] == "clear"
    assert r2["zero_label_streak"] == 2

    # Recovery: panel reappears with all three labels matched.
    r3 = ref.record_label_match_count(3)
    assert r3["action"] == "noop"
    assert r3["zero_label_streak"] == 0
    assert r3["low_label_streak"] == 0

    # Frozen reference was never cleared — caller chose not to act on
    # the request — and a subsequent zero-label scan is back to a
    # single (sub-threshold) hit.
    r4 = ref.record_label_match_count(0)
    assert r4["action"] == "noop"
    assert r4["zero_label_streak"] == 1


def test_count_two_resets_zero_streak_but_low_streak_unaffected_by_design() -> None:
    """A scan with count == 2 resets BOTH streaks. (The spec is "≥ 2
    labels matched = panel mostly visible", so both gates relax.)"""
    ref = FrozenPanelReference()
    ref.freeze(_make_img(), _make_values())

    ref.record_label_match_count(1)
    ref.record_label_match_count(1)
    assert ref.zero_label_streak == 0
    assert ref.low_label_streak == 2

    # count == 2 resets BOTH.
    r = ref.record_label_match_count(2)
    assert r["action"] == "noop"
    assert r["zero_label_streak"] == 0
    assert r["low_label_streak"] == 0


def test_record_label_match_count_works_when_unfrozen() -> None:
    """The streak counters must be callable on an unfrozen ref —
    the early-return path in api.py invokes them even before the
    first freeze, so that the streak history is in place for the
    moment a freeze does occur.
    """
    ref = FrozenPanelReference()
    assert ref.is_frozen is False

    r1 = ref.record_label_match_count(0)
    r2 = ref.record_label_match_count(0)
    assert r2["action"] == "clear"
    # Caller may still call clear() — should be a noop on unfrozen.
    ref.clear()
    assert ref.is_frozen is False


def test_freeze_resets_label_streak_counters() -> None:
    """After a fresh freeze, label-match streaks must reset — the
    panel was just visible enough to lock all three values, so any
    prior streak is stale.
    """
    ref = FrozenPanelReference()
    # Build up some streak history while unfrozen.
    ref.record_label_match_count(0)
    ref.record_label_match_count(1)
    assert ref.zero_label_streak == 0  # zero-streak was reset by the 1-count
    assert ref.low_label_streak == 2

    # Freeze.
    ref.freeze(_make_img(), _make_values())
    assert ref.zero_label_streak == 0
    assert ref.low_label_streak == 0


def test_negative_or_garbage_label_count_treated_as_zero() -> None:
    """Defensive: an out-of-range count (negative, non-int) is
    clamped to 0 rather than raising. The OCR pipeline upstream
    should never produce a negative count, but the gate must not
    crash if it does.
    """
    ref = FrozenPanelReference()
    r1 = ref.record_label_match_count(-5)
    r2 = ref.record_label_match_count("nonsense")  # type: ignore[arg-type]
    assert r1["label_match_count"] == 0
    assert r2["label_match_count"] == 0
    assert r2["action"] == "clear"  # two "zero" scans in a row


def test_per_region_streaks_are_isolated() -> None:
    """Two separate regions must not share streak state."""
    region_a = {"x": 100, "y": 200, "w": 400, "h": 300}
    region_b = {"x": 500, "y": 600, "w": 400, "h": 300}

    ref_a = get_frozen_ref(region_a)
    ref_b = get_frozen_ref(region_b)
    ref_a.freeze(_make_img(seed=1), _make_values())
    ref_b.freeze(_make_img(seed=2), _make_values())

    # Region A sees 2 consecutive zero-label scans.
    ref_a.record_label_match_count(0)
    res_a = ref_a.record_label_match_count(0)
    assert res_a["action"] == "clear"

    # Region B has had its own (non-zero) scan history in the
    # meantime — its streak must NOT be polluted by region A's.
    res_b = ref_b.record_label_match_count(3)
    assert res_b["action"] == "noop"
    assert ref_b.zero_label_streak == 0


def test_returned_dict_contains_expected_keys() -> None:
    """API contract: the return dict always has these keys regardless
    of action so callers can dict-access without KeyError."""
    ref = FrozenPanelReference()
    res = ref.record_label_match_count(3)
    for key in (
        "action", "reason",
        "zero_label_streak", "low_label_streak",
        "tolerance_sec", "label_match_count",
    ):
        assert key in res, f"missing key {key!r} in {res}"
