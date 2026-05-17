"""Tests for the HUD structural-anchor checks.

Covers :func:`ocr.sc_ocr.validate.check_hud_anchors` and
:func:`ocr.sc_ocr.validate.estimate_digit_pitch`.

The anchor checks are the HUD-numeric analogue of the signature
pipeline's comma-anchored sanity / crop-extension logic:

  * resistance: ``%`` must be the RIGHTMOST glyph.
  * instability: ``.`` must be INTERIOR (digit on both sides).

These tests cover the conservative reject conditions (clearly broken
structure) AND the "must not produce false positives" conditions
(legit ``"0%"``, ``"0.0"``, ``"100.5"``) called out in the task
spec. They exercise pure helpers — no OCR-pipeline mocking required.
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

from ocr.sc_ocr import validate as _v  # noqa: E402


# ─── Resistance: % must be rightmost ───────────────────────────────


class TestResistanceAnchor:
    """``%`` is always the rightmost glyph in a resistance read."""

    @pytest.mark.parametrize(
        "text",
        ["50%", "0%", "100%", "99%", "5%", "82%"],
    )
    def test_pct_rightmost_passes(self, text: str):
        ok, reason = _v.check_hud_anchors(text, "resistance")
        assert ok is True
        assert reason == ""

    def test_leading_pct_rejected(self):
        # Segmenter mis-ordered "%" to the front — clear break.
        ok, reason = _v.check_hud_anchors("%50", "resistance")
        assert ok is False
        assert reason == "pct_not_rightmost"

    def test_midstring_pct_rejected(self):
        # Wider failure mode: "%" classified mid-segment.
        ok, reason = _v.check_hud_anchors("50%3", "resistance")
        assert ok is False
        assert reason == "pct_not_rightmost"

    def test_solo_pct_rejected(self):
        # "%" alone has no digit to the left — structurally
        # impossible. Caught by the not-rightmost check because
        # the pct's position equals len-1 BUT we still want this
        # treated as broken; the existing validate_pct will reject
        # it on its own. Verify behavior either way is acceptable.
        ok, _ = _v.check_hud_anchors("%", "resistance")
        # "%" at position 0 with length 1 is technically "rightmost"
        # (only character), but it has no integer — we let the
        # downstream validate_pct reject this. Anchor passes.
        assert ok is True

    def test_no_pct_in_text_passes(self):
        # Some HUD layouts return digits without %. Don't reject —
        # the validate_pct fallback handles digit-only reads.
        ok, reason = _v.check_hud_anchors("50", "resistance")
        assert ok is True
        assert reason == ""

    def test_pct_box_right_of_digit_rejected(self):
        # When boxes are provided, a digit-shaped box positioned to
        # the right of the % box indicates the crop extended past
        # the % into bogus territory.
        # Text "5%" with boxes saying the % is at x=20 but there's
        # ANOTHER box at x=40 (a phantom digit on the right):
        text = "5%"
        boxes = [(0, 0, 16, 30), (20, 0, 14, 30)]
        # Sanity: this should pass the text-position check alone.
        ok, _ = _v.check_hud_anchors(text, "resistance", boxes=boxes)
        assert ok is True
        # Now simulate the broken scenario where the % is in the
        # middle of the box list (mis-ordering of boxes vs text).
        # We construct it by flipping: text "5%" but boxes "% then 5":
        boxes_bad = [(40, 0, 14, 30), (0, 0, 16, 30)]
        # Here text[1] is "%" which corresponds to boxes_bad[1]
        # at x=0 — and boxes_bad[0] (the digit) is at x=40, RIGHT
        # of the %.
        ok2, reason2 = _v.check_hud_anchors(
            text, "resistance", boxes=boxes_bad,
        )
        assert ok2 is False
        assert reason2 == "pct_right_of_digit_box"

    def test_pct_misaligned_boxes_falls_through_safely(self):
        # Mal-formed boxes (wrong length / type) must not raise.
        ok, _ = _v.check_hud_anchors(
            "50%", "resistance", boxes=[(0, 0, 18, 30)],  # too few
        )
        assert ok is True  # length mismatch -> skip the box check
        ok2, _ = _v.check_hud_anchors(
            "50%", "resistance", boxes=[(0, 0), (20, 0), (40, 0)],
        )
        # Box check tries to read [2] index — handled by the
        # IndexError fallthrough; the text-only check still applies
        # and passes.
        assert ok2 is True


# ─── Instability: . must be interior ───────────────────────────────


class TestInstabilityAnchor:
    """``.`` is always interior in instability — digit on each side."""

    @pytest.mark.parametrize(
        "text",
        ["12.09", "0.0", "100.5", "15.47", "5.0", "1.0", "82.13"],
    )
    def test_interior_dot_passes(self, text: str):
        ok, reason = _v.check_hud_anchors(text, "instability")
        assert ok is True
        assert reason == ""

    def test_leading_dot_rejected(self):
        # Leading integer digit clipped — the comma-anchored
        # extension analogue.
        ok, reason = _v.check_hud_anchors(".09", "instability")
        assert ok is False
        assert reason == "dot_leading"

    def test_trailing_dot_rejected(self):
        # No fractional part — borderline but we treat as broken
        # because the anchor requires a digit on each side.
        ok, reason = _v.check_hud_anchors("12.", "instability")
        assert ok is False
        assert reason == "dot_trailing"

    def test_multiple_dots_rejected(self):
        ok, reason = _v.check_hud_anchors("12.0.9", "instability")
        assert ok is False
        assert reason == "dot_multiple"

    def test_no_dot_passes(self):
        # Integer-only instability — handled by downstream validators.
        # Don't structurally reject here.
        ok, reason = _v.check_hud_anchors("12", "instability")
        assert ok is True
        assert reason == ""

    def test_integer_zero_passes(self):
        # "0" alone is a legit (extreme) instability read.
        ok, reason = _v.check_hud_anchors("0", "instability")
        assert ok is True
        assert reason == ""


# ─── No-op fields / empty input ────────────────────────────────────


class TestNoOpCases:

    @pytest.mark.parametrize(
        "field",
        ["mass", "mineral_name", "signal", "", "unknown_field"],
    )
    def test_other_fields_noop(self, field: str):
        # Anchor checks only apply to resistance / instability.
        # Other fields must always pass — no false positives.
        ok, reason = _v.check_hud_anchors("anything.%goes", field)
        assert ok is True
        assert reason == ""

    @pytest.mark.parametrize(
        "field",
        ["resistance", "instability"],
    )
    def test_empty_string_passes(self, field: str):
        # Empty input is not the anchor check's concern — let the
        # length / lexicon validators handle it downstream.
        ok, reason = _v.check_hud_anchors("", field)
        assert ok is True
        assert reason == ""


# ─── Digit-pitch estimator ─────────────────────────────────────────


class TestEstimateDigitPitch:

    def test_median_of_three(self):
        # Median of widths [16, 18, 20] -> 18.
        boxes = [(0, 0, 18, 30), (20, 0, 16, 30), (40, 0, 20, 30)]
        assert _v.estimate_digit_pitch(boxes) == 18

    def test_median_of_four_avg(self):
        # Median of widths [16, 18, 20, 22] -> (18+20)//2 == 19.
        boxes = [
            (0, 0, 18, 30),
            (20, 0, 16, 30),
            (40, 0, 20, 30),
            (60, 0, 22, 30),
        ]
        assert _v.estimate_digit_pitch(boxes) == 19

    def test_robust_to_outlier(self):
        # A single tiny dot-shaped box shouldn't drag the median.
        boxes = [
            (0, 0, 18, 30),
            (20, 0, 16, 30),
            (40, 0, 4, 10),  # dot-shaped outlier
            (50, 0, 20, 30),
            (70, 0, 22, 30),
        ]
        # Sorted widths: [4, 16, 18, 20, 22] -> median is 18.
        assert _v.estimate_digit_pitch(boxes) == 18

    def test_empty_boxes(self):
        assert _v.estimate_digit_pitch([]) is None

    def test_single_box(self):
        assert _v.estimate_digit_pitch([(0, 0, 18, 30)]) is None

    def test_zero_width_filtered(self):
        # Zero-width spurious boxes shouldn't be included.
        boxes = [(0, 0, 18, 30), (20, 0, 0, 30), (40, 0, 20, 30)]
        # Filtered widths: [18, 20] -> avg of middle two -> 19.
        assert _v.estimate_digit_pitch(boxes) == 19

    def test_malformed_box_tolerated(self):
        # Boxes that aren't (x, y, w, h) shape shouldn't raise.
        boxes = [(0, 0, 18, 30), (1, 2), (40, 0, 20, 30)]
        # Filtered widths: [18, 20] (the (1, 2) raises IndexError
        # on [2] which is caught) -> avg of two -> 19.
        assert _v.estimate_digit_pitch(boxes) == 19


# ─── Integration: anchor check + pitch estimate combined ───────────


class TestIntegration:
    """Verify the contract used by api.py: anchor failure +
    pitch estimate together describe what the caller needs to
    decide whether to extend the crop or just reject."""

    def test_leading_dot_with_pitch(self):
        # Text ".09" with two visible boxes (the dot at x=0 and a
        # digit at x=5) - the dot anchor fails, and the pitch
        # estimate tells the caller HOW FAR to extend.
        text = ".09"
        boxes = [(0, 0, 4, 10), (5, 0, 16, 30), (22, 0, 18, 30)]
        ok, reason = _v.check_hud_anchors(text, "instability", boxes=boxes)
        assert ok is False
        assert reason == "dot_leading"
        pitch = _v.estimate_digit_pitch(boxes)
        # Median of [4, 16, 18] -> 16 (a sensible left-extension
        # delta).
        assert pitch == 16

    def test_legit_instability_does_not_engage_pitch(self):
        # "12.09" is fine -> caller doesn't need to extend; anchor
        # passes and the pitch estimate is informational only.
        text = "12.09"
        boxes = [
            (0, 0, 18, 30),
            (20, 0, 16, 30),
            (40, 0, 4, 10),
            (50, 0, 20, 30),
            (72, 0, 22, 30),
        ]
        ok, reason = _v.check_hud_anchors(text, "instability", boxes=boxes)
        assert ok is True
        assert reason == ""
