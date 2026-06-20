"""Regression tests for Mining Redux OCR gate ordering.

These tests pin the two remembered gates that can suppress Mining Signals
scan results while broader smoke checks still pass:

* the public HUD reader must invoke SC-OCR before any optional legacy ONNX
  runtime/model gate, and
* the SCAN RESULTS title gate must still synthesize an anchor from label-row
  layout evidence when the NCC template path is unavailable.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

# Allow running from any cwd by inserting the Mining_Signals tool root.
_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ocr import onnx_hud_reader as _hud  # noqa: E402
from ocr.sc_ocr import api as _sc_api  # noqa: E402
from ocr.sc_ocr import scan_results_match as _srm  # noqa: E402


def test_scan_hud_onnx_calls_sc_ocr_before_optional_onnx_model(monkeypatch):
    """Missing legacy ONNX deps must not suppress the primary SC-OCR path."""
    calls: list[dict] = []
    expected = {
        "mass": 1234.5,
        "resistance": 67.8,
        "instability": 9.1,
        "panel_visible": True,
    }

    def fake_sc_ocr_scan(region):
        calls.append(dict(region))
        return expected

    monkeypatch.setattr(_hud, "_ensure_model", lambda: False)
    monkeypatch.setattr(_sc_api, "scan_hud_onnx", fake_sc_ocr_scan)

    region = {"x": 1, "y": 2, "w": 300, "h": 200}
    assert _hud.scan_hud_onnx(region) is expected
    assert calls == [region]


def test_scan_results_anchor_synthesizes_from_rows_when_template_unavailable(
    monkeypatch,
):
    """A missing NCC template should fall back to row/layout evidence."""
    _srm.reset_cache()
    _srm.reset_anchor_tracker()
    img = Image.new("RGB", (500, 300), "black")

    monkeypatch.setattr(_srm, "_load_template", lambda: None)
    monkeypatch.setattr(
        _srm._lm,
        "find_label_positions",
        lambda _img: {
            "mass": {"x": 80, "y": 120, "w": 70, "h": 20},
            "resistance": {"x": 80, "y": 160, "w": 110, "h": 20},
            "instability": {"x": 80, "y": 200, "w": 120, "h": 20},
        },
    )

    anchor = _srm.find_scan_results_anchor(img)

    assert anchor is not None
    assert anchor["score"] == 0.45
    assert 70 <= anchor["title_x"] <= 90
    assert 10 <= anchor["title_y"] <= 25
    assert anchor["title_w"] > 100
    assert anchor["title_h"] >= 20


def test_scan_results_anchor_missing_template_local_search_stays_none(
    monkeypatch,
):
    """Local-search misses stay safe; only full-frame acquisition synthesizes."""
    _srm.reset_cache()
    _srm.reset_anchor_tracker()
    img = Image.new("RGB", (500, 300), "black")

    monkeypatch.setattr(_srm, "_load_template", lambda: None)
    monkeypatch.setattr(
        _srm._lm,
        "find_label_positions",
        lambda _img: {
            "mass": {"x": 80, "y": 120, "w": 70, "h": 20},
            "resistance": {"x": 80, "y": 160, "w": 110, "h": 20},
        },
    )

    assert _srm.find_scan_results_anchor(
        img,
        search_center=(120, 30),
        search_radius=60,
    ) is None


def test_scan_results_anchor_template_fallback_failure_returns_none(
    monkeypatch,
):
    """Malformed row/layout evidence must fail closed rather than raising."""
    _srm.reset_cache()
    _srm.reset_anchor_tracker()
    img = Image.new("RGB", (500, 300), "black")

    monkeypatch.setattr(_srm, "_load_template", lambda: None)

    def boom(_img):
        raise RuntimeError("synthetic label detector failure")

    monkeypatch.setattr(_srm._lm, "find_label_positions", boom)

    assert _srm.find_scan_results_anchor(img) is None
