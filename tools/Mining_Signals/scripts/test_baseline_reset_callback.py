"""Smoke test for the anchor-baseline-reset callback hook.

Verifies the fix for the "baseline poison" failure mode: when the
NCC tracker's outlier hysteresis confirms a sustained position jump
(>= ``_TRACK_OUTLIER_HYSTERESIS`` consecutive large-distance raw
detections), it now fires a callback that invalidates the lock
cache in api.py — so locks built against the OLD anchor's crop
pixels don't survive the re-baselining.

Asserted behaviour:
  * Cold-start primes the tracker with a stable position.
  * A single large-distance detection is REJECTED (streak=1/2)
    and does NOT fire the callback (lock cache survives).
  * A SECOND consecutive large-distance detection is ACCEPTED
    and DOES fire the callback (lock cache cleared, value
    buffers cleared, persistence streaks reset).

Run with the project Python:
  %LOCALAPPDATA%\\Python\\pythoncore-3.14-64\\python.exe \\
      scripts/test_baseline_reset_callback.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the tool tree importable when run from any cwd.
_TOOL = Path(__file__).resolve().parent.parent
if str(_TOOL) not in sys.path:
    sys.path.insert(0, str(_TOOL))


def _make_anchor(x: int, y: int) -> dict:
    return {
        "title_x": x,
        "title_y": y,
        "title_w": 95,
        "title_h": 17,
        "score": 0.665,
    }


def main() -> int:
    # Importing api.py registers the production callback on import.
    from ocr.sc_ocr import api as _api
    from ocr.sc_ocr import scan_results_match as _srm

    # Ensure a clean slate — reset both the tracker AND the
    # consensus state, so we know what we're observing.
    _srm.reset_anchor_tracker()
    _api._reset_consensus_buffers()

    # Sanity: the production callback must be registered after
    # `import api`. If this fails the wiring at the bottom of
    # api.py is broken.
    assert len(_srm._baseline_reset_callbacks) >= 1, (
        "production callback should be registered on api import; "
        f"got {len(_srm._baseline_reset_callbacks)}"
    )

    # Layer 1: callback fires at all. Drop production callback so
    # we control the listener list; add a counter to observe firing.
    _srm._clear_baseline_reset_callbacks()
    fires: list[int] = []

    def _listener() -> None:
        fires.append(1)

    _srm.register_baseline_reset_callback(_listener)

    # Prime the tracker at (2, 18) — the stale-poison position
    # from the bug log.
    out = _srm._smooth_anchor(_make_anchor(2, 18), now=100.0)
    assert out is not None and out["title_x"] == 2 and out["title_y"] == 18, (
        f"baseline prime failed: {out}"
    )
    assert len(fires) == 0, "no callback should fire on baseline prime"

    # First outlier at (334, 45) — ~333 px away. Tracker should
    # REJECT (streak=1/2) and NOT fire the callback.
    out = _srm._smooth_anchor(_make_anchor(334, 45), now=100.1)
    assert out is not None and out["title_x"] == 2, (
        f"first outlier should be rejected, got {out}"
    )
    assert len(fires) == 0, (
        f"callback fired on first outlier (should be rejected): {len(fires)}"
    )

    # Second consecutive outlier at (334, 45) — tracker confirms
    # real motion, snaps to (334, 45), and DOES fire callback.
    out = _srm._smooth_anchor(_make_anchor(334, 45), now=100.2)
    assert out is not None and out["title_x"] == 334 and out["title_y"] == 45, (
        f"second outlier should be accepted, got {out}"
    )
    assert len(fires) == 1, (
        f"callback should fire exactly once on confirmed jump, got {len(fires)}"
    )

    # Layer 2: production callback actually clears state. Reset
    # the listener list and re-register the production callback
    # via the normal api hook. Then prime lock cache and assert
    # the confirmed jump clears it.
    _srm.reset_anchor_tracker()
    _api._reset_consensus_buffers()
    _srm._clear_baseline_reset_callbacks()
    _srm.register_baseline_reset_callback(_api._on_anchor_baseline_reset)

    # Fake a populated lock cache + populated consensus state.
    import numpy as _np
    fake_fingerprint = _np.zeros((_api._CROP_FP_H * _api._CROP_FP_W,), dtype=_np.float32)
    _api._field_lock_cache[(0, 0, 100, 200)] = {
        "mass": (15683.0, fake_fingerprint),
    }
    _api._STABLE_VALUE["mass"] = 15683.0
    _api._RECENT_READS["mass"].append(15683.0)
    _api._RECENT_CROPS["mass"].append(fake_fingerprint)
    _api._PERSIST_STREAK["mass"] = 5
    _api._PERSIST_LAST["mass"] = 15683.0

    # Pre-condition: lock cache is populated.
    assert len(_api._field_lock_cache) == 1
    assert _api._STABLE_VALUE["mass"] == 15683.0
    assert len(_api._RECENT_READS["mass"]) == 1
    assert _api._PERSIST_STREAK["mass"] == 5

    # Trigger the confirmed jump again: prime at (2,18), then
    # two consecutive outliers at (334, 45).
    _srm._smooth_anchor(_make_anchor(2, 18), now=200.0)
    _srm._smooth_anchor(_make_anchor(334, 45), now=200.1)   # rejected
    # Lock cache should still be populated after the rejected
    # first outlier (no callback).
    assert len(_api._field_lock_cache) == 1, (
        "lock cache should survive a rejected outlier"
    )

    _srm._smooth_anchor(_make_anchor(334, 45), now=200.2)   # accepted

    # Post-condition: lock cache + consensus state are cleared.
    assert len(_api._field_lock_cache) == 0, (
        f"lock cache should be cleared, got {len(_api._field_lock_cache)}"
    )
    assert _api._STABLE_VALUE["mass"] is None, (
        f"_STABLE_VALUE should be None, got {_api._STABLE_VALUE['mass']}"
    )
    assert len(_api._RECENT_READS["mass"]) == 0
    assert len(_api._RECENT_CROPS["mass"]) == 0
    assert _api._PERSIST_STREAK["mass"] == 0
    assert _api._PERSIST_LAST["mass"] is None

    # Layer 3: difficulty cache should NOT be cleared (per design).
    _api._difficulty_cache[(0, 0, 100, 200)] = "HARD"
    _srm.reset_anchor_tracker()
    _srm._smooth_anchor(_make_anchor(2, 18), now=300.0)
    _srm._smooth_anchor(_make_anchor(334, 45), now=300.1)
    _srm._smooth_anchor(_make_anchor(334, 45), now=300.2)   # accepted
    assert _api._difficulty_cache.get((0, 0, 100, 200)) == "HARD", (
        "difficulty cache must survive baseline reset"
    )

    # Layer 4: a bad listener doesn't break the tracker.
    _srm._clear_baseline_reset_callbacks()

    def _bad_listener() -> None:
        raise RuntimeError("intentional test failure")

    fires2: list[int] = []

    def _good_listener() -> None:
        fires2.append(1)

    _srm.register_baseline_reset_callback(_bad_listener)
    _srm.register_baseline_reset_callback(_good_listener)
    _srm.reset_anchor_tracker()
    _srm._smooth_anchor(_make_anchor(2, 18), now=400.0)
    _srm._smooth_anchor(_make_anchor(334, 45), now=400.1)
    _srm._smooth_anchor(_make_anchor(334, 45), now=400.2)
    assert len(fires2) == 1, (
        "good listener must still fire even when an earlier "
        "listener raises"
    )

    # Restore real production state for any code that imports
    # this module subsequently (best-effort housekeeping).
    _srm._clear_baseline_reset_callbacks()
    _srm.register_baseline_reset_callback(_api._on_anchor_baseline_reset)
    _srm.reset_anchor_tracker()
    _api._reset_consensus_buffers()

    print("baseline-reset callback smoke test: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
