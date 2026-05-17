"""Unit tests for the auto-calibration learner in
``ocr.sc_ocr.hud_panel_tracker``.

The calibration system records (title_y, title_h, label positions)
samples from successful detections, computes per-field median ratios,
and publishes the learned multipliers as overrides for the solver's
default offset table.

Scenarios covered:

1. ``observe_calibration_sample`` records valid samples and skips
   degenerate ones (title_h<=0, fewer than 2 label matches).
2. After ``_CAL_MIN_SAMPLES`` consistent observations the learner
   publishes ``get_learned_offsets`` / ``get_learned_row_mults`` and
   the values are inside the median range of the samples.
3. Noisy observations (relative std > tolerance) are rejected — the
   learner stays in the unlearned state for that field.
4. ``HudPanelTracker._solve`` consults the learned offsets when
   available and falls back to defaults otherwise.
5. ``_pose_to_label_rows`` consults the learned row mults when
   available and falls back to defaults otherwise.
6. The user's reported HUD ratios (4.53 / 6.18 / 7.82) feed cleanly
   through the pipeline and the resulting solver offsets match.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from any cwd by inserting the repo root onto sys.path.
_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from ocr.sc_ocr import hud_panel_tracker as _hpt  # noqa: E402
from ocr.sc_ocr.hud_panel_tracker import (  # noqa: E402
    DEFAULT_OFFSETS,
    HudPanelTracker,
    _CAL_MIN_SAMPLES,
    _ROW_OFFSET_MULTS,
    get_calibration_state,
    get_learned_offsets,
    get_learned_row_mults,
    observe_calibration_sample,
    reset_calibration,
)


# ─── Helpers ───────────────────────────────────────────────────────


def _make_matches(
    title_y: float,
    title_h: float,
    ratios: dict[str, float],
    label_h: float = 28.0,
) -> dict[str, dict]:
    """Build a label_matches dict in the shape ``find_label_positions``
    would return: per-field ``{"x", "y", "w", "h"}`` with the y derived
    from the requested label-top-to-title ratio.
    """
    out: dict[str, dict] = {}
    for field, ratio in ratios.items():
        out[field] = {
            "x": 480,
            "y": int(title_y + ratio * title_h),
            "w": 120,
            "h": int(label_h),
        }
    return out


@pytest.fixture(autouse=True)
def _reset_calibration_between_tests():
    """The calibration buffer is module-level state; reset it around
    every test so tests don't bleed into each other."""
    reset_calibration()
    yield
    reset_calibration()


# ─── Test 1: sample recording and rejection of degenerate inputs ───


def test_observe_records_valid_sample():
    # Use integer-aligned title_h so the int-truncated label_y in
    # _make_matches reproduces the input ratio exactly (no quantization
    # noise — that's tested separately).
    matches = _make_matches(
        title_y=100.0, title_h=100.0,
        ratios={"mass": 3.33, "resistance": 4.98, "instability": 6.64},
    )
    observe_calibration_sample(
        title_y=100.0, title_h=100.0, label_matches=matches,
    )
    state = get_calibration_state()
    assert state["n_samples"] == 1
    sample = state["samples"][0]
    assert sample["title_y"] == 100.0
    assert sample["title_h"] == 100.0
    assert "mass" in sample and "resistance" in sample and "instability" in sample
    # Ratio reproduces what we put in (within float rounding).
    assert abs(sample["mass"]["ratio_top"] - 3.33) < 1e-9


def test_observe_skips_zero_title_height():
    matches = _make_matches(
        title_y=100.0, title_h=45.0,
        ratios={"mass": 3.33, "resistance": 4.98, "instability": 6.64},
    )
    observe_calibration_sample(
        title_y=100.0, title_h=0.0, label_matches=matches,
    )
    assert get_calibration_state()["n_samples"] == 0


def test_observe_skips_insufficient_label_matches():
    # Only one label matched — sample is dropped.
    matches = _make_matches(
        title_y=100.0, title_h=45.0,
        ratios={"mass": 3.33},
    )
    observe_calibration_sample(
        title_y=100.0, title_h=45.0, label_matches=matches,
    )
    assert get_calibration_state()["n_samples"] == 0


def test_observe_accepts_two_label_matches():
    # Two labels matched — sample IS recorded.
    matches = _make_matches(
        title_y=100.0, title_h=45.0,
        ratios={"mass": 3.33, "instability": 6.64},
    )
    observe_calibration_sample(
        title_y=100.0, title_h=45.0, label_matches=matches,
    )
    state = get_calibration_state()
    assert state["n_samples"] == 1
    assert "resistance" not in state["samples"][0]


# ─── Test 2: stable samples publish learned values ─────────────────


def test_stable_samples_publish_learned_values():
    # Feed exactly _CAL_MIN_SAMPLES consistent observations of the
    # user's reported HUD ratios (4.53 / 6.18 / 7.82).
    ratios = {"mass": 4.53, "resistance": 6.18, "instability": 7.82}
    for _ in range(_CAL_MIN_SAMPLES):
        observe_calibration_sample(
            title_y=100.0, title_h=45.0,
            label_matches=_make_matches(100.0, 45.0, ratios),
        )

    learned_offsets = get_learned_offsets()
    learned_row_mults = get_learned_row_mults()
    assert learned_offsets is not None, "calibration should have published"
    assert learned_row_mults is not None

    # Solver offsets: label-TOP multipliers should equal observed ratios
    # within integer-quantization tolerance (label_y is int-truncated so
    # the back-computed ratio drifts up to 1/title_h from the input).
    _Q = 1.0 / 45.0  # quantization noise from int(label_y)
    assert abs(learned_offsets["label_mass"][1] - 4.53) < _Q
    assert abs(learned_offsets["label_resistance"][1] - 6.18) < _Q
    assert abs(learned_offsets["label_instability"][1] - 7.82) < _Q
    # scan_results origin is fixed at (0, 0).
    assert learned_offsets["scan_results"] == (0.0, 0.0)
    # label_mineral is scaled proportionally with mass — should NOT
    # remain at the default 1.5.
    assert learned_offsets["label_mineral"][1] != DEFAULT_OFFSETS[
        "label_mineral"
    ][1]

    # Row-center mults: observed label CENTER ratio = (label_y +
    # label_h/2 - title_y) / title_h. label_y is int-truncated by
    # _make_matches so back-compute the expected value through the
    # same truncation:
    def _expected_center_mult(ratio_top, title_y=100.0, title_h=45.0, label_h=28):
        label_y = int(title_y + ratio_top * title_h)
        return (label_y + label_h / 2.0 - title_y) / title_h

    assert abs(
        learned_row_mults["mass"] - _expected_center_mult(4.53)
    ) < 1e-9
    assert abs(
        learned_row_mults["resistance"] - _expected_center_mult(6.18)
    ) < 1e-9
    assert abs(
        learned_row_mults["instability"] - _expected_center_mult(7.82)
    ) < 1e-9


def test_below_min_samples_no_publish():
    ratios = {"mass": 4.53, "resistance": 6.18, "instability": 7.82}
    for _ in range(_CAL_MIN_SAMPLES - 1):
        observe_calibration_sample(
            title_y=100.0, title_h=45.0,
            label_matches=_make_matches(100.0, 45.0, ratios),
        )
    assert get_learned_offsets() is None
    assert get_learned_row_mults() is None


# ─── Test 3: noisy samples are rejected per field ──────────────────


def test_noisy_samples_rejected():
    """Mass observations span a wide range -> relative-std blows the
    tolerance gate -> mass is NOT locked even though we have enough
    samples. Resistance and instability are consistent and lock in.

    Mass ratios are kept >1.0 away from resistance so the inter-label
    gap gate doesn't fire (which would drop the whole sample, not
    just the mass field). The wide span is still enough to push the
    relative-std past the 10% tolerance.
    """
    # Wide span but all gaps to resistance=6.18 stay above 1.0.
    mass_ratios = [2.0, 3.0, 4.0, 4.5, 5.0]
    for m in mass_ratios:
        ratios = {"mass": m, "resistance": 6.18, "instability": 7.82}
        observe_calibration_sample(
            title_y=100.0, title_h=100.0,
            label_matches=_make_matches(100.0, 100.0, ratios),
        )
    learned = get_learned_offsets()
    assert learned is not None
    # Mass: rejected, falls back to default (3.33).
    assert learned["label_mass"] == DEFAULT_OFFSETS["label_mass"]
    # Resistance / instability: locked in exactly (integer-aligned
    # title_h means no quantization noise).
    assert abs(learned["label_resistance"][1] - 6.18) < 1e-9
    assert abs(learned["label_instability"][1] - 7.82) < 1e-9


def test_no_fields_lock_no_publish():
    # All three fields wildly noisy — nothing publishes.
    for i, mass in enumerate([2.0, 3.5, 5.0, 7.5, 10.0]):
        ratios = {
            "mass": mass,
            "resistance": mass * 1.4,
            "instability": mass * 1.7,
        }
        observe_calibration_sample(
            title_y=100.0, title_h=45.0,
            label_matches=_make_matches(100.0, 45.0, ratios),
        )
    assert get_learned_offsets() is None


# ─── Test 4: tracker._solve consults learned offsets ───────────────


def test_solver_uses_learned_offsets(monkeypatch):
    """Once calibration publishes, ``HudPanelTracker._solve`` should
    pass the learned table to the LSQ solver instead of the defaults.
    """
    # Publish learned values.
    ratios = {"mass": 4.53, "resistance": 6.18, "instability": 7.82}
    for _ in range(_CAL_MIN_SAMPLES):
        observe_calibration_sample(
            title_y=100.0, title_h=45.0,
            label_matches=_make_matches(100.0, 45.0, ratios),
        )
    assert get_learned_offsets() is not None

    # Capture which offsets the solver receives.
    captured: dict = {}

    def _fake_solver(measurements, offsets, weights=None, *, fixed_scale=None):
        captured["offsets"] = dict(offsets)
        return ((500.0, 100.0, 45.0), {k: 0.5 for k in measurements})

    monkeypatch.setattr(_hpt, "_solve_panel_pose", _fake_solver)

    tracker = HudPanelTracker()
    tracker._solve({"scan_results": (500.0, 100.0)})
    assert "offsets" in captured
    # Learned mass offset should track 4.53 (within int-quantization),
    # not the default 3.33.
    _Q = 1.0 / 45.0
    assert abs(captured["offsets"]["label_mass"][1] - 4.53) < _Q
    assert abs(captured["offsets"]["label_resistance"][1] - 6.18) < _Q
    assert abs(captured["offsets"]["label_instability"][1] - 7.82) < _Q


def test_solver_falls_back_to_defaults_without_calibration(monkeypatch):
    """Without published calibration, the solver should see the
    tracker's defaults (the ``offsets`` passed to its constructor).
    """
    captured: dict = {}

    def _fake_solver(measurements, offsets, weights=None, *, fixed_scale=None):
        captured["offsets"] = dict(offsets)
        return ((0.0, 0.0, 1.0), {})

    monkeypatch.setattr(_hpt, "_solve_panel_pose", _fake_solver)

    # No samples observed - learned state is None.
    assert get_learned_offsets() is None
    tracker = HudPanelTracker()
    tracker._solve({"scan_results": (500.0, 100.0)})
    assert captured["offsets"]["label_mass"][1] == DEFAULT_OFFSETS[
        "label_mass"
    ][1]


# ─── Test 5: _pose_to_label_rows consults learned row mults ────────


def test_pose_to_label_rows_uses_learned_row_mults():
    # Publish learned values.
    ratios = {"mass": 4.53, "resistance": 6.18, "instability": 7.82}
    for _ in range(_CAL_MIN_SAMPLES):
        observe_calibration_sample(
            title_y=100.0, title_h=45.0,
            label_matches=_make_matches(100.0, 45.0, ratios),
        )
    learned_row_mults = get_learned_row_mults()
    assert learned_row_mults is not None

    tracker = HudPanelTracker()
    # panel_y=100, scale (≈title_h_px)=45.
    rows = tracker._pose_to_label_rows(
        pose=(500.0, 100.0, 45.0),
        img_width=1280,
        img_height=720,
    )

    # Expected center_y = panel_y + scale * learned_mult
    # mass: 100 + 45 * 4.20 = 289 -> band centered there.
    expected_mass_center = 100.0 + 45.0 * learned_row_mults["mass"]
    y1, y2, _lr = rows["mass"]
    actual_center = (y1 + y2) // 2
    assert abs(actual_center - expected_mass_center) < 2

    # Resistance and instability follow the same formula.
    expected_resist_center = 100.0 + 45.0 * learned_row_mults["resistance"]
    y1, y2, _lr = rows["resistance"]
    assert abs((y1 + y2) // 2 - expected_resist_center) < 2

    expected_instab_center = 100.0 + 45.0 * learned_row_mults["instability"]
    y1, y2, _lr = rows["instability"]
    assert abs((y1 + y2) // 2 - expected_instab_center) < 2


def test_pose_to_label_rows_falls_back_to_defaults():
    # No calibration: row centers use _ROW_OFFSET_MULTS defaults.
    assert get_learned_row_mults() is None
    tracker = HudPanelTracker()
    rows = tracker._pose_to_label_rows(
        pose=(500.0, 100.0, 28.0),
        img_width=1280,
        img_height=720,
    )
    expected_mass_center = 100.0 + 28.0 * _ROW_OFFSET_MULTS["mass"]
    y1, y2, _lr = rows["mass"]
    assert abs((y1 + y2) // 2 - expected_mass_center) < 2


# ─── Test 6: end-to-end through the user's reported ratios ─────────


def test_users_reported_ratios_lock_in_correct_solver_pose():
    """Replay the user's actual HUD ratios. With learned offsets,
    the LSQ solver fed synthesized measurements from those ratios
    should return ``scale ≈ title_h`` (real ≈ 45) instead of the
    inflated value (≈ 54) it produced with default offsets.
    """
    title_y, title_h = 100.0, 45.0
    user_ratios = {"mass": 4.53, "resistance": 6.18, "instability": 7.82}

    # Step 1: feed observations.
    for _ in range(_CAL_MIN_SAMPLES):
        observe_calibration_sample(
            title_y=title_y, title_h=title_h,
            label_matches=_make_matches(title_y, title_h, user_ratios),
        )

    # Step 2: build synthesized measurements at the SAME ratios and
    # run them through the solver via the tracker.
    measurements = {
        "scan_results": (500.0, title_y),
    }
    for fld, mult in user_ratios.items():
        measurements[f"label_{fld}"] = (
            500.0, title_y + mult * title_h,
        )

    tracker = HudPanelTracker(max_residual_px=5.0)
    pose, residuals = tracker._solve(measurements)
    panel_x, panel_y, scale = pose
    # With learned offsets matching the synthesized geometry, the LSQ
    # solver should recover the true title_h (45) — not the inflated
    # scale the defaults would force.
    assert abs(panel_x - 500.0) < 0.5
    assert abs(panel_y - title_y) < 0.5
    assert abs(scale - title_h) < 1.0
    # Residuals should be near zero (perfect synthesized fit).
    if residuals:
        assert max(residuals.values()) < 1.0


# ─── Test 7: reset_calibration wipes state ─────────────────────────


def test_reset_clears_samples_and_learned():
    ratios = {"mass": 4.53, "resistance": 6.18, "instability": 7.82}
    for _ in range(_CAL_MIN_SAMPLES):
        observe_calibration_sample(
            title_y=100.0, title_h=45.0,
            label_matches=_make_matches(100.0, 45.0, ratios),
        )
    assert get_learned_offsets() is not None
    assert get_calibration_state()["n_samples"] >= _CAL_MIN_SAMPLES

    reset_calibration()

    assert get_learned_offsets() is None
    assert get_learned_row_mults() is None
    assert get_calibration_state()["n_samples"] == 0


# ─── Test 8: bounded buffer doesn't grow past _CAL_MAX_SAMPLES ─────


def test_sample_buffer_bounded():
    from ocr.sc_ocr.hud_panel_tracker import _CAL_MAX_SAMPLES
    ratios = {"mass": 4.53, "resistance": 6.18, "instability": 7.82}
    for _ in range(_CAL_MAX_SAMPLES + 5):
        observe_calibration_sample(
            title_y=100.0, title_h=45.0,
            label_matches=_make_matches(100.0, 45.0, ratios),
        )
    assert get_calibration_state()["n_samples"] == _CAL_MAX_SAMPLES


# ─── Test 9: calibration version bumps on publish ──────────────────


def test_calibration_version_starts_at_zero():
    from ocr.sc_ocr.hud_panel_tracker import get_calibration_version
    assert get_calibration_version() == 0


def test_calibration_version_bumps_on_first_publish():
    from ocr.sc_ocr.hud_panel_tracker import get_calibration_version
    assert get_calibration_version() == 0
    ratios = {"mass": 4.53, "resistance": 6.18, "instability": 7.82}
    for _ in range(_CAL_MIN_SAMPLES):
        observe_calibration_sample(
            title_y=100.0, title_h=45.0,
            label_matches=_make_matches(100.0, 45.0, ratios),
        )
    assert get_calibration_version() == 1


def test_calibration_version_stable_when_values_unchanged():
    from ocr.sc_ocr.hud_panel_tracker import get_calibration_version
    ratios = {"mass": 4.53, "resistance": 6.18, "instability": 7.82}
    # Build up identical samples; once locked, more identical samples
    # should NOT bump the version because the published values haven't
    # changed (medians stay the same).
    for _ in range(_CAL_MIN_SAMPLES):
        observe_calibration_sample(
            title_y=100.0, title_h=45.0,
            label_matches=_make_matches(100.0, 45.0, ratios),
        )
    v1 = get_calibration_version()
    for _ in range(_CAL_MIN_SAMPLES):
        observe_calibration_sample(
            title_y=100.0, title_h=45.0,
            label_matches=_make_matches(100.0, 45.0, ratios),
        )
    assert get_calibration_version() == v1


def test_reset_calibration_resets_version():
    from ocr.sc_ocr.hud_panel_tracker import get_calibration_version
    ratios = {"mass": 4.53, "resistance": 6.18, "instability": 7.82}
    for _ in range(_CAL_MIN_SAMPLES):
        observe_calibration_sample(
            title_y=100.0, title_h=45.0,
            label_matches=_make_matches(100.0, 45.0, ratios),
        )
    assert get_calibration_version() == 1
    reset_calibration()
    assert get_calibration_version() == 0


# ─── Test 10: plausibility gates reject false-match samples ────────


def test_plausibility_gate_rejects_too_small_mass_ratio():
    """A mass ratio of 1.5 means the label is right up at the title —
    structurally impossible. The pre-anchor full-frame label_match
    can false-match here against SCAN RESULTS title pieces. The
    gate must reject this sample so the median isn't polluted.
    """
    # Mass at ratio 1.5 (false-match in title area) — should be rejected.
    matches = _make_matches(
        title_y=100.0, title_h=100.0,
        ratios={"mass": 1.5, "resistance": 6.18, "instability": 7.82},
    )
    observe_calibration_sample(
        title_y=100.0, title_h=100.0, label_matches=matches,
    )
    state = get_calibration_state()
    # The sample was recorded (resistance + instability passed) but
    # the bad mass entry should be missing.
    if state["n_samples"] == 1:
        sample = state["samples"][0]
        assert "mass" not in sample, (
            f"mass should have been rejected but got {sample.get('mass')}"
        )


def test_plausibility_gate_rejects_too_large_instability_ratio():
    # Instability at ratio 15 is way past the bottom of any plausible HUD.
    matches = _make_matches(
        title_y=100.0, title_h=100.0,
        ratios={"mass": 4.53, "resistance": 6.18, "instability": 15.0},
    )
    observe_calibration_sample(
        title_y=100.0, title_h=100.0, label_matches=matches,
    )
    state = get_calibration_state()
    if state["n_samples"] == 1:
        sample = state["samples"][0]
        assert "instability" not in sample


def test_plausibility_gate_rejects_whole_sample_when_only_one_passes():
    """If only one label passes the gate, the sample (n_labels<2 after
    gate) should be silently dropped — same as having too few label
    matches in the first place.
    """
    matches = _make_matches(
        title_y=100.0, title_h=100.0,
        # Only resistance is in bounds; mass and instability are not.
        ratios={"mass": 1.0, "resistance": 6.0, "instability": 15.0},
    )
    observe_calibration_sample(
        title_y=100.0, title_h=100.0, label_matches=matches,
    )
    assert get_calibration_state()["n_samples"] == 0


def test_ordering_gate_rejects_swapped_labels():
    """Labels must appear in increasing Y order (mass < resistance <
    instability). A sample where mass is BELOW resistance means at
    least one of them false-matched.
    """
    # mass at 6.0, resistance at 4.0 — out of order.
    matches = _make_matches(
        title_y=100.0, title_h=100.0,
        ratios={"mass": 6.0, "resistance": 4.5, "instability": 7.0},
    )
    observe_calibration_sample(
        title_y=100.0, title_h=100.0, label_matches=matches,
    )
    assert get_calibration_state()["n_samples"] == 0


def test_gap_gate_rejects_collided_labels():
    """Two labels at nearly the same ratio means at least one
    false-matched. The min-gap rule catches this.
    """
    matches = _make_matches(
        title_y=100.0, title_h=100.0,
        # mass and resistance at the same place — collision.
        ratios={"mass": 4.5, "resistance": 4.7, "instability": 7.0},
    )
    observe_calibration_sample(
        title_y=100.0, title_h=100.0, label_matches=matches,
    )
    assert get_calibration_state()["n_samples"] == 0


def test_well_formed_user_ratios_pass_all_gates():
    """The user's reported ratios (4.53/6.18/7.82) are well-spaced,
    in-bounds, and in-order — should pass cleanly.
    """
    ratios = {"mass": 4.53, "resistance": 6.18, "instability": 7.82}
    observe_calibration_sample(
        title_y=100.0, title_h=100.0,
        label_matches=_make_matches(100.0, 100.0, ratios),
    )
    state = get_calibration_state()
    assert state["n_samples"] == 1
    sample = state["samples"][0]
    assert "mass" in sample and "resistance" in sample and "instability" in sample


if __name__ == "__main__":
    # Allow running directly: ``python tests/test_calibration_learning.py``
    sys.exit(pytest.main([__file__, "-v"]))
