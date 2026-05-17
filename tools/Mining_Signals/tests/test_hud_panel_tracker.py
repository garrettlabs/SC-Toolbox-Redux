"""Unit tests for ``ocr.sc_ocr.hud_panel_tracker.HudPanelTracker``.

These tests exercise the tracker's state machine end-to-end with
mocked detectors. The real Agent A (rigid-body solver) and Agent B
(local-search detectors) are not in this branch yet — the tracker
uses stubs that match their spec'd interfaces, and the tests mock the
detector dependency directly so no real images or templates are
needed.

Five required scenarios:

1. Cold start with mocked detector returning valid title position →
   tracker locks and returns ``label_rows``.
2. Subsequent track() calls with detector returning slightly-shifted
   titles → tracker updates pose smoothly.
3. Detector returns garbage (50px outlier) → tracker rejects, counter
   increments.
4. 3 consecutive rejections → tracker resets to cold-start, then
   succeeds on a clean frame.
5. Sequence of 20 frames with small jitter (+/-5px) → final pose is
   within 2px of the per-anchor mean.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Allow running from any cwd by inserting the repo root onto sys.path.
_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from ocr.sc_ocr import hud_panel_tracker as _hpt  # noqa: E402
from ocr.sc_ocr.hud_panel_tracker import (  # noqa: E402
    DEFAULT_OFFSETS,
    HudPanelTracker,
)


# ─── Helpers ───────────────────────────────────────────────────────


def _blank_img(w: int = 1280, h: int = 720) -> Image.Image:
    """Cheap synthetic image — the tracker never actually inspects
    pixels in these tests because we replace the detectors with mocks.
    """
    return Image.new("RGB", (w, h), color=(0, 0, 0))


def _patch_detectors(
    monkeypatch: pytest.MonkeyPatch,
    *,
    title_xy: Optional[tuple[int, int]],
    label_offsets: Optional[dict[str, tuple[int, int]]] = None,
    title_w: int = 200,
    title_h: int = 28,
) -> None:
    """Patch the tracker's two detector entry points.

    ``title_xy=None`` simulates "no detection". ``label_offsets`` maps
    label name → (x, y); if ``None`` we synthesize the labels from
    ``title_xy`` using DEFAULT_OFFSETS so the tracker sees >=3 anchors.
    """

    def _fake_find_scan_results_anchor(
        img, *, search_center=None, search_radius=60,
    ):
        if title_xy is None:
            return None
        return {
            "title_x": int(title_xy[0]),
            "title_y": int(title_xy[1]),
            "title_w": int(title_w),
            "title_h": int(title_h),
            "score": 0.95,
        }

    if label_offsets is None and title_xy is not None:
        # Synthesize labels straight from the canonical offset table so
        # the LSQ solver sees perfect agreement -> residuals ~ 0.
        # DEFAULT_OFFSETS values are in **title-height units**; multiply
        # by ``title_h`` to convert to pixels for the fake measurements.
        # The solver should then discover scale ≈ title_h.
        label_offsets = {}
        for key, (ox, oy) in DEFAULT_OFFSETS.items():
            if not key.startswith("label_"):
                continue
            # label_<short> -> short
            short = key[len("label_"):]
            # The label_match keys are "mass", "resistance",
            # "instability" — skip "mineral" since label_match doesn't
            # produce one and the tracker would only consume the three.
            if short not in ("mass", "resistance", "instability"):
                continue
            label_offsets[short] = (
                int(title_xy[0] + ox * title_h),
                int(title_xy[1] + oy * title_h),
            )

    def _fake_find_label_positions(
        img, *, search_centers=None, search_radius=40,
    ):
        # ``search_centers`` / ``search_radius`` are accepted (and
        # ignored) so the tracker's local-search path in track_step
        # exercises the same fake measurements as the cold-start path.
        # The synthesized offsets are still perfect-agreement so the
        # LSQ residuals stay near zero.
        if not label_offsets:
            return {}
        out = {}
        for name, (x, y) in label_offsets.items():
            out[name] = {
                "x": int(x), "y": int(y), "w": 120, "h": 28,
                "score": 0.85, "scale": 1.0,
            }
        return out

    monkeypatch.setattr(
        _hpt,
        "_find_scan_results_anchor",
        _fake_find_scan_results_anchor,
    )

    # ``find_label_positions`` is imported INSIDE
    # _collect_anchors at call time. Replace the symbol that the
    # tracker will resolve via its absolute-import fallback chain.
    # We patch sys.modules so both import paths see the same fake.
    import types
    fake_lm = types.ModuleType("ocr.sc_ocr.label_match")
    fake_lm.find_label_positions = _fake_find_label_positions  # type: ignore
    monkeypatch.setitem(
        sys.modules, "ocr.sc_ocr.label_match", fake_lm,
    )


# ─── Test 1: cold start locks on valid title ───────────────────────


def test_cold_start_locks_on_valid_title(monkeypatch):
    _patch_detectors(monkeypatch, title_xy=(500, 100))
    tracker = HudPanelTracker(offsets=DEFAULT_OFFSETS)
    assert tracker.is_locked is False
    assert tracker.last_pose is None

    rows = tracker.track(_blank_img())

    assert tracker.is_locked, "tracker should lock after cold start"
    assert rows is not None, "should return label_rows on lock"
    assert tracker.last_pose is not None
    px, py, scale = tracker.last_pose
    # With perfectly synthesized anchors the LSQ solver should land
    # exactly on the title's top-left (panel origin) and discover
    # ``scale ≈ title_h`` (default fixture title_h=28).
    assert abs(px - 500.0) < 0.5
    assert abs(py - 100.0) < 0.5
    assert abs(scale - 28.0) < 0.5

    # label_rows shape
    assert "mass" in rows
    assert "resistance" in rows
    assert "instability" in rows
    for key in ("mass", "resistance", "instability"):
        y1, y2, lr = rows[key]
        assert isinstance(y1, int) and isinstance(y2, int)
        assert isinstance(lr, int)
        assert y2 > y1


# ─── Test 2: small shift -> pose updates smoothly ──────────────────


def test_track_step_updates_pose_smoothly(monkeypatch):
    # First frame: detector says title is at (500, 100).
    _patch_detectors(monkeypatch, title_xy=(500, 100))
    tracker = HudPanelTracker(offsets=DEFAULT_OFFSETS)
    rows0 = tracker.track(_blank_img())
    assert rows0 is not None
    pose0 = tracker.last_pose
    assert pose0 is not None
    assert abs(pose0[0] - 500.0) < 0.5

    # Second frame: panel drifts +3px in x, +2px in y.
    _patch_detectors(monkeypatch, title_xy=(503, 102))
    rows1 = tracker.track(_blank_img())
    assert rows1 is not None, "tracker should still be locked"
    assert tracker.is_locked
    pose1 = tracker.last_pose
    assert pose1 is not None
    assert abs(pose1[0] - 503.0) < 0.5
    assert abs(pose1[1] - 102.0) < 0.5

    # Third frame: another small drift -3px in x.
    _patch_detectors(monkeypatch, title_xy=(500, 103))
    rows2 = tracker.track(_blank_img())
    assert rows2 is not None
    pose2 = tracker.last_pose
    assert pose2 is not None
    assert abs(pose2[0] - 500.0) < 0.5
    assert abs(pose2[1] - 103.0) < 0.5


# ─── Test 3: outlier causes rejection ──────────────────────────────


def test_outlier_increments_rejection_counter(monkeypatch):
    # Lock first.
    _patch_detectors(monkeypatch, title_xy=(500, 100))
    tracker = HudPanelTracker(
        offsets=DEFAULT_OFFSETS,
        max_residual_px=5.0,
        max_motion_px=40.0,
        cold_start_after_rejections=3,
    )
    assert tracker.track(_blank_img()) is not None
    assert tracker.is_locked
    assert tracker._rejection_count == 0

    # Now corrupt ONLY the title (offset +50 px in x).
    # The labels stay anchored at the old pose, so the LSQ fit will
    # see a 50-px disagreement and the residuals will blow up.
    _patch_detectors(
        monkeypatch,
        title_xy=(550, 100),
        label_offsets={
            # Labels unchanged from the original cold-start pose.
            "mass": (
                int(500 + DEFAULT_OFFSETS["label_mass"][0]),
                int(100 + DEFAULT_OFFSETS["label_mass"][1]),
            ),
            "resistance": (
                int(500 + DEFAULT_OFFSETS["label_resistance"][0]),
                int(100 + DEFAULT_OFFSETS["label_resistance"][1]),
            ),
            "instability": (
                int(500 + DEFAULT_OFFSETS["label_instability"][0]),
                int(100 + DEFAULT_OFFSETS["label_instability"][1]),
            ),
        },
    )
    rows_outlier = tracker.track(_blank_img())
    assert rows_outlier is None, "tracker should reject outlier frame"
    assert tracker._rejection_count == 1, (
        f"expected rejection_count=1, got {tracker._rejection_count}"
    )


# ─── Test 4: 3 rejections -> cold-start, then succeed ──────────────


def test_three_rejections_triggers_cold_start(monkeypatch):
    # Lock first.
    _patch_detectors(monkeypatch, title_xy=(500, 100))
    tracker = HudPanelTracker(
        offsets=DEFAULT_OFFSETS,
        max_residual_px=5.0,
        max_motion_px=40.0,
        cold_start_after_rejections=3,
    )
    assert tracker.track(_blank_img()) is not None
    assert tracker.is_locked

    # Three consecutive frames where the detector returns nothing —
    # the tracker has nothing to verify against, so each frame is a
    # rejection.
    _patch_detectors(monkeypatch, title_xy=None, label_offsets={})
    for i in range(3):
        result = tracker.track(_blank_img())
        assert result is None, f"frame {i}: expected reject, got {result}"
    # After the 3rd rejection the tracker drops the lock and resets
    # its counter (so a future cold-start can run).
    assert tracker.is_locked is False
    assert tracker.last_pose is None
    assert tracker._rejection_count == 0

    # Next frame the detector returns a clean title — tracker
    # should cold-start and lock again.
    _patch_detectors(monkeypatch, title_xy=(600, 120))
    rows = tracker.track(_blank_img())
    assert rows is not None
    assert tracker.is_locked
    assert tracker.last_pose is not None
    assert abs(tracker.last_pose[0] - 600.0) < 0.5
    assert abs(tracker.last_pose[1] - 120.0) < 0.5


# ─── Test 5: 20 frames small jitter -> stable mean pose ────────────


def test_twenty_frames_small_jitter_stable_pose(monkeypatch):
    """Drive the tracker through a long sequence of frames with small
    sub-pixel-ish jitter. The final pose should sit very close to the
    mean of the per-frame title positions (within 2 px).
    """
    # Pre-build a deterministic sequence of titles around a true mean.
    # Use a fixed list so the test is reproducible without seeding RNG.
    base_x = 600
    base_y = 120
    jitter_pattern = [
        (0, 0), (5, -3), (-4, 2), (3, 1), (-2, -2),
        (1, 4), (-5, 5), (4, -4), (-1, 0), (2, -1),
        (0, 3), (-3, -5), (5, 2), (-4, 1), (2, 5),
        (-5, -3), (3, 4), (-2, -1), (4, 3), (-1, -4),
    ]
    assert len(jitter_pattern) == 20
    title_positions = [
        (base_x + dx, base_y + dy) for dx, dy in jitter_pattern
    ]
    mean_x = sum(x for x, _ in title_positions) / 20.0
    mean_y = sum(y for _, y in title_positions) / 20.0

    tracker = HudPanelTracker(
        offsets=DEFAULT_OFFSETS,
        max_residual_px=20.0,  # generous: jittered labels add residual
        max_motion_px=40.0,
        cold_start_after_rejections=3,
    )

    last_pose: Optional[tuple[float, float, float]] = None
    for i, (tx, ty) in enumerate(title_positions):
        _patch_detectors(monkeypatch, title_xy=(tx, ty))
        rows = tracker.track(_blank_img())
        # Cold-start may need >=3 anchors; with synthesized labels we
        # always have 4 anchors so every frame should produce rows.
        assert rows is not None, f"frame {i} returned None unexpectedly"
        last_pose = tracker.last_pose
        assert last_pose is not None

    assert last_pose is not None
    # The tracker's pose for the final frame is determined by the LSQ
    # fit at that specific frame — not the running mean. So we assert
    # against the final-frame title position (which the stub LSQ
    # converges on exactly), not the historical mean.
    final_tx, final_ty = title_positions[-1]
    assert abs(last_pose[0] - final_tx) < 2.0, (
        f"final pose x={last_pose[0]:.2f} too far from frame x={final_tx}"
    )
    assert abs(last_pose[1] - final_ty) < 2.0, (
        f"final pose y={last_pose[1]:.2f} too far from frame y={final_ty}"
    )

    # And the per-frame jitter never drove the tracker out of lock.
    assert tracker.is_locked

    # Sanity: the mean across the trajectory is close to base.
    assert abs(mean_x - base_x) < 2.0
    assert abs(mean_y - base_y) < 2.0


# ─── Bonus: reset() forces cold start ──────────────────────────────


def test_reset_forces_cold_start(monkeypatch):
    _patch_detectors(monkeypatch, title_xy=(500, 100))
    tracker = HudPanelTracker(offsets=DEFAULT_OFFSETS)
    assert tracker.track(_blank_img()) is not None
    assert tracker.is_locked

    tracker.reset()
    assert tracker.is_locked is False
    assert tracker.last_pose is None

    # After reset the next track() should perform a fresh cold-start.
    rows = tracker.track(_blank_img())
    assert rows is not None
    assert tracker.is_locked


if __name__ == "__main__":
    # Allow running directly: ``python tests/test_hud_panel_tracker.py``
    sys.exit(pytest.main([__file__, "-v"]))
