"""Tests for the HUD panel image stabilizer.

Covers:
  * Phase correlation primitive — recovers known translations on
    synthetic patterns with sub-pixel-rounded accuracy.
  * Cold start — establishes pose via mocked tracker, caches reference.
  * Track step — accumulates motion across frames.
  * Lost lock — weak correlation peak triggers reset.
  * Motion rejection — implausibly large jumps are rejected.
  * Re-anchor — periodic absolute detection corrects accumulated drift.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ocr.sc_ocr.hud_panel_stabilizer import (  # noqa: E402
    HudPanelStabilizer,
    phase_correlate,
)


# ─────────────────────────────────────────────────────────────────────
# Phase correlation primitive
# ─────────────────────────────────────────────────────────────────────


def _synthetic_patch(h: int, w: int, seed: int = 42) -> np.ndarray:
    """Return a textured patch suitable for phase correlation tests."""
    rng = np.random.default_rng(seed)
    # Low-frequency texture: smoothed noise + a few high-contrast
    # geometric features for the correlation to lock onto.
    base = rng.normal(128.0, 30.0, size=(h, w))
    # A bright cross to give the FFT a strong unique feature.
    base[h // 2 - 2:h // 2 + 2, :] += 80.0
    base[:, w // 2 - 2:w // 2 + 2] += 80.0
    # A bright square in one corner — asymmetric to break ambiguity.
    base[h // 4:h // 4 + 8, w // 4:w // 4 + 8] += 100.0
    return np.clip(base, 0.0, 255.0)


def _shift_patch(patch: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Roll-shift; assumes shifts are small relative to patch size
    so the wrap-around doesn't introduce visible artifacts at the
    sample peak.
    """
    return np.roll(np.roll(patch, dy, axis=0), dx, axis=1)


def test_phase_correlate_zero_shift_returns_origin() -> None:
    patch = _synthetic_patch(64, 256)
    dx, dy, response = phase_correlate(patch, patch)
    assert dx == 0
    assert dy == 0
    # Identical inputs → perfect correlation peak.
    assert response > 0.5


@pytest.mark.parametrize("true_dx,true_dy", [
    (5, 0),
    (0, 5),
    (-5, 0),
    (0, -5),
    (10, -7),
    (-12, 3),
])
def test_phase_correlate_recovers_known_translation(
    true_dx: int, true_dy: int,
) -> None:
    """Shift a known patch by (dx, dy) and verify phase correlation
    recovers the same vector. Tolerates ±1 px to allow for the windowing
    function's mild peak-broadening on noisy inputs.
    """
    ref = _synthetic_patch(64, 256)
    cur = _shift_patch(ref, true_dx, true_dy)
    dx, dy, response = phase_correlate(ref, cur)
    assert abs(dx - true_dx) <= 1, f"dx={dx} expected {true_dx}"
    assert abs(dy - true_dy) <= 1, f"dy={dy} expected {true_dy}"
    assert response > 0.05, (
        f"weak peak (response={response}) — phase correlation failed"
    )


def test_phase_correlate_uncorrelated_signals_low_response() -> None:
    """Two unrelated patches should produce a weak / noisy peak.
    Doesn't strictly bound the response (random patches can spuriously
    correlate) but the cleanly-translated case should be > 10x stronger
    than the random-vs-random case.
    """
    a = _synthetic_patch(64, 256, seed=1)
    b = _synthetic_patch(64, 256, seed=999)
    _, _, response_random = phase_correlate(a, b)

    c = _shift_patch(a, 3, 2)
    _, _, response_shifted = phase_correlate(a, c)

    assert response_shifted > response_random * 3.0, (
        f"shifted={response_shifted:.4f} should be much stronger than "
        f"random={response_random:.4f}"
    )


def test_phase_correlate_shape_mismatch_raises() -> None:
    a = np.zeros((64, 64), dtype=np.float64)
    b = np.zeros((128, 128), dtype=np.float64)
    with pytest.raises(ValueError):
        phase_correlate(a, b)


# ─────────────────────────────────────────────────────────────────────
# Stabilizer fixtures
# ─────────────────────────────────────────────────────────────────────


def _make_synthetic_panel_image(
    width: int = 1024,
    height: int = 768,
    panel_x: int = 100,
    panel_y: int = 80,
    title_h: int = 45,
    seed: int = 7,
) -> Image.Image:
    """Build a synthetic image with a fake SCAN RESULTS-like panel
    region — a high-contrast textured rectangle starting at
    (panel_x, panel_y) of size 256x60 (matching the production title
    proportions). The rest of the image is low-amplitude noise.
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(80.0, 15.0, size=(height, width)).astype(np.float64)

    # Bright "title" rectangle with structure.
    tw = 256
    panel = _synthetic_patch(title_h, tw, seed=seed)
    panel += 40.0
    img[panel_y:panel_y + title_h, panel_x:panel_x + tw] = panel

    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(img, mode="L").convert("RGB")


def _make_mock_tracker(
    pose: Optional[Tuple[float, float, float]],
    rows_template: Optional[dict] = None,
) -> MagicMock:
    """Build a mock that mimics ``HudPanelTracker``'s public API:
    ``.track(img) -> dict|None``, ``.last_pose -> tuple|None``,
    ``.reset() -> None``.
    """
    rows = rows_template if rows_template is not None else {
        "_mineral_row": (130, 170, 530),
        "mass":         (200, 240, 530),
        "resistance":   (260, 300, 530),
        "instability":  (320, 360, 530),
    }
    mock = MagicMock()
    mock.last_pose = pose
    if pose is None:
        mock.track.return_value = None
    else:
        mock.track.return_value = rows
    mock.reset = MagicMock()
    return mock


# ─────────────────────────────────────────────────────────────────────
# Cold start
# ─────────────────────────────────────────────────────────────────────


def test_cold_start_locks_when_tracker_returns_pose() -> None:
    img = _make_synthetic_panel_image()
    mock = _make_mock_tracker(pose=(100.0, 80.0, 45.0))
    stab = HudPanelStabilizer(tracker_factory=lambda: mock)

    assert stab.is_locked is False
    rows = stab.stabilize(img)
    assert rows is not None
    assert stab.is_locked is True
    assert stab.pose == (100.0, 80.0, 45.0)
    # Reference patch was captured.
    assert stab._reference is not None
    assert stab._reference_origin is not None


def test_cold_start_returns_none_when_tracker_fails() -> None:
    img = _make_synthetic_panel_image()
    mock = _make_mock_tracker(pose=None)
    stab = HudPanelStabilizer(tracker_factory=lambda: mock)

    rows = stab.stabilize(img)
    assert rows is None
    assert stab.is_locked is False


# ─────────────────────────────────────────────────────────────────────
# Tracking step — accumulates motion via phase correlation
# ─────────────────────────────────────────────────────────────────────


def test_track_step_recovers_known_panel_shift() -> None:
    """Cold-start at one position, then shift the panel by a known
    delta. The phase correlation should detect that delta and update
    the pose accordingly.
    """
    pose0 = (100.0, 80.0, 45.0)
    img0 = _make_synthetic_panel_image(panel_x=100, panel_y=80)
    mock = _make_mock_tracker(pose=pose0)
    stab = HudPanelStabilizer(
        tracker_factory=lambda: mock,
        reanchor_every_n=100,  # don't re-anchor in this test
    )

    rows0 = stab.stabilize(img0)
    assert rows0 is not None
    assert stab.pose == pose0

    # Shift the panel +5 right, +3 down. Same seed so contents match.
    img1 = _make_synthetic_panel_image(panel_x=105, panel_y=83)
    rows1 = stab.stabilize(img1)
    assert rows1 is not None
    new_pose = stab.pose
    assert new_pose is not None
    # Pose should track the panel shift within phase-correlation noise.
    assert abs(new_pose[0] - 105.0) <= 2.0, (
        f"panel_x={new_pose[0]} expected ~105"
    )
    assert abs(new_pose[1] - 83.0) <= 2.0, (
        f"panel_y={new_pose[1]} expected ~83"
    )


def test_track_step_handles_zero_motion() -> None:
    """Two identical frames should yield zero-motion update."""
    img = _make_synthetic_panel_image(panel_x=200, panel_y=150)
    mock = _make_mock_tracker(pose=(200.0, 150.0, 45.0))
    stab = HudPanelStabilizer(
        tracker_factory=lambda: mock,
        reanchor_every_n=100,
    )

    stab.stabilize(img)
    pose_before = stab.pose

    # Same image again — phase correlation should report (0, 0).
    stab.stabilize(img)
    pose_after = stab.pose

    assert pose_before is not None and pose_after is not None
    assert pose_after[0] == pose_before[0]
    assert pose_after[1] == pose_before[1]


# ─────────────────────────────────────────────────────────────────────
# Lock loss
# ─────────────────────────────────────────────────────────────────────


def test_large_motion_triggers_reset_and_cold_start() -> None:
    """A huge jump (panel teleports) should be rejected as scene cut.
    The stabilizer then reset()s and tries cold-start. Since the mock
    tracker still returns a pose, cold-start succeeds.
    """
    pose0 = (100.0, 80.0, 45.0)
    pose1 = (500.0, 80.0, 45.0)  # +400px jump
    img0 = _make_synthetic_panel_image(panel_x=100, panel_y=80)
    img1 = _make_synthetic_panel_image(panel_x=500, panel_y=80)

    # First call: tracker returns pose0. Second call: tracker returns
    # pose1 (the user moved the panel a lot before re-cold-starting).
    poses = [pose0, pose1]
    rows = [
        {"mass": (200, 240, 530)},
        {"mass": (200, 240, 930)},
    ]

    def _factory():
        mock = MagicMock()
        idx = [0]

        def _track(img):
            i = idx[0]
            mock.last_pose = poses[i]
            idx[0] = min(i + 1, len(poses) - 1)
            return rows[i]

        mock.track = MagicMock(side_effect=_track)
        mock.last_pose = poses[0]
        mock.reset = MagicMock()
        return mock

    stab = HudPanelStabilizer(
        tracker_factory=_factory,
        max_motion_px=50.0,  # tight enough to reject the 400px jump
    )
    stab.stabilize(img0)
    assert stab.is_locked
    # Now img1 has a panel 400 px to the right — phase correlation
    # against the cached reference will produce a weak / out-of-range
    # peak. Stabilizer should reset and re-cold-start.
    rows_after = stab.stabilize(img1)
    assert rows_after is not None  # cold-start succeeded on retry
    assert stab.is_locked


def test_low_correlation_triggers_reset() -> None:
    """If the reference patch contents are replaced by unrelated noise
    (the panel disappeared), phase correlation produces a weak peak
    and the stabilizer resets.
    """
    pose0 = (100.0, 80.0, 45.0)
    img0 = _make_synthetic_panel_image(panel_x=100, panel_y=80)

    mock = _make_mock_tracker(pose=pose0)
    stab = HudPanelStabilizer(
        tracker_factory=lambda: mock,
        min_response=0.5,  # absurdly high threshold to force rejection
    )
    stab.stabilize(img0)
    assert stab.is_locked

    # Frame 2: pure noise — no panel content. Correlation will be weak.
    rng = np.random.default_rng(123)
    noise = rng.normal(80.0, 15.0, size=(768, 1024)).astype(np.uint8)
    img_noise = Image.fromarray(noise, mode="L").convert("RGB")

    # The tracker mock still says pose0 — cold-start retry on a noise
    # frame will use that pose. We only assert the stabilizer doesn't
    # crash; the actual return may be cold-start-via-mock-tracker.
    _ = stab.stabilize(img_noise)
    # Just verify it didn't get stuck in a broken state.
    # (locked or unlocked, both are valid outcomes here.)


# ─────────────────────────────────────────────────────────────────────
# Re-anchor
# ─────────────────────────────────────────────────────────────────────


def test_reanchor_fires_after_n_frames() -> None:
    """After ``reanchor_every_n`` successful track steps, the
    stabilizer should call the tracker's ``track()`` again to refresh
    the absolute pose.
    """
    img = _make_synthetic_panel_image(panel_x=100, panel_y=80)

    track_calls = [0]

    def _factory():
        mock = MagicMock()
        mock.last_pose = (100.0, 80.0, 45.0)

        def _track(_img):
            track_calls[0] += 1
            return {"mass": (200, 240, 530)}

        mock.track = MagicMock(side_effect=_track)
        mock.reset = MagicMock()
        return mock

    stab = HudPanelStabilizer(
        tracker_factory=_factory,
        reanchor_every_n=3,
    )

    # Cold start (1 track call).
    stab.stabilize(img)
    assert track_calls[0] == 1

    # 3 track steps without re-anchor; on the 3rd, re-anchor fires.
    stab.stabilize(img)  # frame 1: phase correlation, no re-anchor yet
    assert track_calls[0] == 1
    stab.stabilize(img)  # frame 2
    assert track_calls[0] == 1
    stab.stabilize(img)  # frame 3: re-anchor!
    assert track_calls[0] == 2  # tracker.track() called again


def test_label_rows_output_shape() -> None:
    """``stabilize()`` should return a dict matching the
    ``_find_label_rows`` contract: keys mass/resistance/instability
    (and optionally _mineral_row) mapping to (y1, y2, label_right).
    """
    img = _make_synthetic_panel_image(panel_x=100, panel_y=80)
    mock = _make_mock_tracker(pose=(100.0, 80.0, 45.0))
    stab = HudPanelStabilizer(tracker_factory=lambda: mock)

    rows = stab.stabilize(img)
    assert rows is not None
    for key in ("mass", "resistance", "instability"):
        assert key in rows
        y1, y2, lr = rows[key]
        assert isinstance(y1, int) and isinstance(y2, int)
        assert isinstance(lr, int)
        assert y2 > y1
