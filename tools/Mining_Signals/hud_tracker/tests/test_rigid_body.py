"""Tests for ``hud_tracker.rigid_body.solve_panel_pose``.

Run::

    python -m pytest hud_tracker/tests/test_rigid_body.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


# ── Path bootstrap so ``hud_tracker`` imports without ``pip install`` ──
# Mirrors the harness in ``hud_tracker/anchors/test_*`` files.
_THIS = Path(__file__).resolve()
_HUD_TRACKER_DIR = _THIS.parent.parent
_REPO_ROOT = _HUD_TRACKER_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hud_tracker.rigid_body import (  # noqa: E402
    DEFAULT_OFFSETS,
    DEFAULT_Y_ONLY_OFFSETS,
    solve_panel_pose,
)


# ── Test fixtures ────────────────────────────────────────────────────
# Canonical-ish offsets in title_h units. We use multiple anchors with
# enough X-diversity that the system is well-conditioned: pure-vertical
# offsets alone (rows at dx=0) leave the X-fit dependent on the two
# chrome anchors only, so an outlier on a chrome anchor would bias X
# unrealistically. Real per-row label/colon detectors live near
# x ≈ 2..5 * title_h (well inside the panel), so we give the test
# rows realistic X values rather than dx=0 stubs.
TEST_OFFSETS: dict[str, tuple[float, float]] = {
    "title":         (0.0, 0.0),
    "label_mass":    (0.8, 3.0),
    "label_resist":  (0.8, 4.4),
    "label_instab":  (0.8, 5.8),
    "colon_mass":    (3.5, 3.0),
    "colon_resist":  (3.5, 4.4),
    "colon_instab":  (3.5, 5.8),
    "chrome_top":    (4.47, 1.0),
    "chrome_bot":    (4.51, 7.92),
}


def _synthesize_measurements(
    panel_x: float, panel_y: float, scale: float,
    offsets: dict[str, tuple[float, float]],
    noise_sigma: float = 0.0,
    outlier: tuple[str, float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[tuple[str, float, float]]:
    """Generate noisy measurements consistent with a known pose.

    If ``outlier`` is ``(anchor_id, dx_err, dy_err)``, that anchor gets
    an additional (dx_err, dy_err) shift on top of any Gaussian noise.
    """
    if rng is None:
        rng = np.random.default_rng(0xDEADBEEF)
    out: list[tuple[str, float, float]] = []
    for aid, (dx, dy) in offsets.items():
        mx = panel_x + scale * dx
        my = panel_y + scale * dy
        if noise_sigma > 0.0:
            mx += float(rng.normal(0.0, noise_sigma))
            my += float(rng.normal(0.0, noise_sigma))
        if outlier is not None and outlier[0] == aid:
            mx += outlier[1]
            my += outlier[2]
        out.append((aid, mx, my))
    return out


# ── Tests ────────────────────────────────────────────────────────────


def test_noiseless_exact_recovery():
    """1. Synthetic noiseless measurements → exact recovery."""
    truth = (120.0, 47.0, 18.0)  # panel_x, panel_y, scale (title_h px)
    meas = _synthesize_measurements(*truth, TEST_OFFSETS, noise_sigma=0.0)

    result = solve_panel_pose(meas, TEST_OFFSETS)
    assert result is not None, "solver should not return None for 6 valid anchors"
    px, py, scale, residuals = result

    assert math.isclose(px, truth[0], abs_tol=1e-6), f"panel_x: got {px}"
    assert math.isclose(py, truth[1], abs_tol=1e-6), f"panel_y: got {py}"
    assert math.isclose(scale, truth[2], abs_tol=1e-6), f"scale: got {scale}"

    # All residuals should be zero (within FP noise) on noiseless input.
    for aid, r in residuals.items():
        assert r < 1e-6, f"residual for {aid} should be ~0, got {r}"


def test_gaussian_noise_2px_recovery_within_3px():
    """2. Synthetic + Gaussian noise (σ=2 px) → recovery within ~3 px."""
    truth = (200.0, 80.0, 25.0)
    # Use multiple seeds and check that the AVERAGE recovery error is
    # well within 3 px (any single seed could be unlucky).
    errs_xy = []
    errs_scale = []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        meas = _synthesize_measurements(
            *truth, TEST_OFFSETS, noise_sigma=2.0, rng=rng,
        )
        result = solve_panel_pose(meas, TEST_OFFSETS)
        assert result is not None
        px, py, scale, _ = result
        errs_xy.append(math.hypot(px - truth[0], py - truth[1]))
        errs_scale.append(abs(scale - truth[2]))

    # With 6 anchors, σ_x_recovery should be << 2 px because LS averages
    # the noise. Expected std on (px, py) is roughly σ/sqrt(N) ≈ 0.8 px.
    mean_xy_err = float(np.mean(errs_xy))
    max_xy_err = float(np.max(errs_xy))
    mean_scale_err = float(np.mean(errs_scale))

    assert mean_xy_err < 3.0, (
        f"mean XY recovery error {mean_xy_err:.3f} should be < 3 px"
    )
    assert max_xy_err < 5.0, (
        f"max XY recovery error {max_xy_err:.3f} suggests instability"
    )
    # Scale is estimated from the spread of offsets; with our offsets
    # the scale-recovery std should be ~0.2 px (in scale units).
    assert mean_scale_err < 1.0, (
        f"mean scale recovery error {mean_scale_err:.3f} too large"
    )


def test_single_outlier_is_flagged_via_residual():
    """3. One outlier (50 px off) → solver still produces reasonable pose,
    outlier's residual is conspicuously larger than other anchors'.
    """
    truth = (300.0, 100.0, 20.0)
    # Outlier on 'chrome_bot' offsetting by (50, 50).
    rng = np.random.default_rng(42)
    meas = _synthesize_measurements(
        *truth, TEST_OFFSETS,
        noise_sigma=1.0,
        outlier=("chrome_bot", 50.0, 50.0),
        rng=rng,
    )

    result = solve_panel_pose(meas, TEST_OFFSETS)
    assert result is not None
    px, py, scale, residuals = result

    # The outlier should be the LARGEST residual, and substantially
    # larger than the typical (median) anchor — that's what a real
    # outlier-detector watches for. We don't require the outlier to
    # dwarf the MAX other residual: ordinary LS spreads error to
    # geometrically-adjacent anchors (e.g. chrome_top here shares
    # similar X with chrome_bot, so its residual rises too). Robust
    # outlier rejection (IRLS / RANSAC) belongs in the tracker on top
    # of this solver, not in the pure-math core.
    assert "chrome_bot" in residuals
    outlier_r = residuals["chrome_bot"]
    other_rs = [r for aid, r in residuals.items() if aid != "chrome_bot"]
    max_other = max(other_rs)
    median_other = float(np.median(other_rs))

    assert outlier_r > 20.0, (
        f"outlier residual should be large; got {outlier_r:.2f}"
    )
    assert outlier_r == max(residuals.values()), (
        f"outlier should be the LARGEST residual; "
        f"got outlier={outlier_r:.2f}, max_other={max_other:.2f}"
    )
    assert outlier_r > 3.0 * median_other, (
        f"outlier ({outlier_r:.2f}) should be >>3x typical residual "
        f"(median other = {median_other:.2f})"
    )

    # The non-outlier pose recovery should remain reasonable. With
    # N-1 good anchors + 1 outlier (no robust LS), bias is bounded.
    err = math.hypot(px - truth[0], py - truth[1])
    assert err < 30.0, (
        f"pose error with one outlier {err:.2f} px exceeds 30 px"
    )


def test_two_anchors_fixed_scale_solvable():
    """4. Only 2 anchors with fixed_scale → solvable (2 unknowns,
    4 equations).
    """
    truth_xy = (150.0, 60.0)
    scale = 22.0
    offsets = {
        "title":     (0.0, 0.0),
        "row_mass":  (0.0, 3.0),
    }
    meas = _synthesize_measurements(
        truth_xy[0], truth_xy[1], scale, offsets, noise_sigma=0.0,
    )

    result = solve_panel_pose(meas, offsets, fixed_scale=scale)
    assert result is not None, "fixed-scale + 2 anchors should be solvable"
    px, py, ret_scale, residuals = result

    assert math.isclose(px, truth_xy[0], abs_tol=1e-6)
    assert math.isclose(py, truth_xy[1], abs_tol=1e-6)
    assert math.isclose(ret_scale, scale, abs_tol=1e-12), (
        "fixed_scale should be returned verbatim"
    )
    for aid, r in residuals.items():
        assert r < 1e-6, f"noiseless residual for {aid} should be ~0"


def test_single_anchor_returns_none_for_free_scale():
    """5. Only 1 anchor → returns None (free-scale needs >= 2)."""
    meas = [("title", 100.0, 50.0)]
    offsets = {"title": (0.0, 0.0)}
    result = solve_panel_pose(meas, offsets)
    assert result is None, "1 anchor + free scale should be under-determined"


def test_empty_measurements_returns_none():
    """6. Empty measurements → returns None."""
    assert solve_panel_pose([], {}) is None
    assert solve_panel_pose([], TEST_OFFSETS) is None
    # No measurements + fixed_scale should still be None.
    assert solve_panel_pose([], TEST_OFFSETS, fixed_scale=20.0) is None


def test_weights_downweight_noisy_anchor_improves_structure():
    """7. Weighted: down-weight the noisy anchor → residual structure
    improves AND the recovered pose is closer to truth than with the
    noisy anchor at equal weight.
    """
    truth = (250.0, 90.0, 24.0)
    # Inject a noisy anchor that biases the fit. We use a deterministic
    # bias rather than a stochastic noise so the test is fully
    # reproducible — the weighting effect doesn't depend on RNG luck.
    base_meas = _synthesize_measurements(
        *truth, TEST_OFFSETS, noise_sigma=0.0,
    )
    # Replace 'chrome_bot' with a large biased measurement.
    biased = []
    for aid, mx, my in base_meas:
        if aid == "chrome_bot":
            biased.append((aid, mx + 25.0, my + 25.0))
        else:
            biased.append((aid, mx, my))

    # Equal weights: bias from chrome_bot propagates into the fit.
    eq = solve_panel_pose(biased, TEST_OFFSETS)
    assert eq is not None
    eq_px, eq_py, eq_scale, _eq_res = eq
    err_eq = math.hypot(eq_px - truth[0], eq_py - truth[1])

    # Down-weight chrome_bot heavily — the solver should converge to
    # the 5-clean-anchor solution, which is exact.
    weights = {aid: 1.0 for aid in TEST_OFFSETS}
    weights["chrome_bot"] = 1e-6
    weighted = solve_panel_pose(biased, TEST_OFFSETS, weights=weights)
    assert weighted is not None
    w_px, w_py, w_scale, w_res = weighted
    err_w = math.hypot(w_px - truth[0], w_py - truth[1])

    assert err_w < err_eq, (
        f"down-weighting should reduce error: equal={err_eq:.2f} "
        f"weighted={err_w:.2f}"
    )
    # With chrome_bot effectively muted and 5 noiseless anchors, the
    # recovered pose should be essentially exact.
    assert err_w < 0.5, f"weighted pose should be near-exact; err={err_w:.4f}"
    assert math.isclose(w_scale, truth[2], abs_tol=0.05), (
        f"weighted scale should converge; got {w_scale}"
    )

    # Residual STRUCTURE: in the weighted solve, chrome_bot's
    # (geometric, un-weighted) residual is large (it's an outlier from
    # the fitted pose); the other anchors' residuals are tiny.
    chrome_bot_r = w_res["chrome_bot"]
    other_max = max(r for aid, r in w_res.items() if aid != "chrome_bot")
    assert chrome_bot_r > 5.0, (
        f"chrome_bot residual should remain large; got {chrome_bot_r:.2f}"
    )
    assert other_max < 1.0, (
        f"other anchors should fit cleanly; max other = {other_max:.4f}"
    )


# ── Extra sanity tests for robustness ────────────────────────────────


def test_missing_offset_is_skipped():
    """A measurement for an anchor not in offsets is silently skipped."""
    truth = (100.0, 50.0, 20.0)
    meas = [
        ("title", truth[0] + truth[2] * 0.0, truth[1] + truth[2] * 0.0),
        ("label_mass", truth[0] + truth[2] * 0.8, truth[1] + truth[2] * 3.0),
        ("colon_mass", truth[0] + truth[2] * 3.5, truth[1] + truth[2] * 3.0),
        ("unknown_anchor", 999.0, 999.0),  # should be skipped
    ]
    result = solve_panel_pose(meas, TEST_OFFSETS)
    assert result is not None
    px, py, scale, residuals = result
    assert "unknown_anchor" not in residuals
    assert math.isclose(px, truth[0], abs_tol=1e-6)
    assert math.isclose(py, truth[1], abs_tol=1e-6)
    assert math.isclose(scale, truth[2], abs_tol=1e-6)


def test_none_offset_is_skipped():
    """An offset of ``None`` in the offsets dict is silently skipped."""
    offsets = {
        "title":      (0.0, 0.0),
        "row_mass":   (0.0, 3.0),
        "row_resist": (0.0, 4.4),
        "label_mass": None,  # not yet calibrated
    }
    meas = [
        ("title", 100.0, 50.0),
        ("row_mass", 100.0, 110.0),
        ("row_resist", 100.0, 138.0),
        ("label_mass", 80.0, 110.0),  # should be silently skipped
    ]
    result = solve_panel_pose(meas, offsets)
    assert result is not None
    _, _, _, residuals = result
    assert "label_mass" not in residuals


def test_rank_deficient_offsets_return_none():
    """If all offsets are co-linear with identical (dx, dy) up to
    translation (only one DISTINCT offset), scale is not separable from
    translation and the solver returns None.
    """
    # Two measurements at the SAME offset — rank-deficient in 3D
    # (cannot tell where on the offset ray the scale puts us).
    offsets = {"a": (0.0, 0.0), "b": (0.0, 0.0)}
    meas = [("a", 10.0, 20.0), ("b", 10.5, 20.5)]
    result = solve_panel_pose(meas, offsets)
    assert result is None, "rank-deficient input should return None"


def test_default_offsets_table_shape():
    """``DEFAULT_OFFSETS`` has all the keys the spec requires."""
    required = {
        "title", "label_mass", "label_resist", "label_instab",
        "colon_mass", "colon_resist", "colon_instab",
        "chrome_top", "chrome_bot",
    }
    assert required.issubset(set(DEFAULT_OFFSETS.keys())), (
        f"missing keys: {required - set(DEFAULT_OFFSETS.keys())}"
    )
    # Title is the origin.
    assert DEFAULT_OFFSETS["title"] == (0.0, 0.0)
    # Chrome lines have BOTH x and y calibrated.
    for key in ("chrome_top", "chrome_bot"):
        v = DEFAULT_OFFSETS[key]
        assert v is not None, f"{key} should have a calibrated offset"
        assert isinstance(v, tuple) and len(v) == 2


def test_default_y_only_offsets():
    """Y-only offsets sourced from ``_ROW_OFFSET_MULTS`` are present
    and match the documented values.
    """
    assert math.isclose(DEFAULT_Y_ONLY_OFFSETS["row_mineral"], 1.6)
    assert math.isclose(DEFAULT_Y_ONLY_OFFSETS["row_mass"], 3.0)
    assert math.isclose(DEFAULT_Y_ONLY_OFFSETS["row_resistance"], 4.4)
    assert math.isclose(DEFAULT_Y_ONLY_OFFSETS["row_instab"], 5.8)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
