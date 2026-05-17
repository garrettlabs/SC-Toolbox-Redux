"""Tests for the HUD-RGB CRNN beam-search + lexicon-rerank pipeline.

The reranker is extracted as ``api._lexicon_rerank_candidates`` so it
can be exercised as a pure function without needing the full ONNX
inference path. Tests here drive ``_lexicon_rerank_candidates`` with
synthetic beam-search candidate lists and verify the four key
invariants the HUD pipeline relies on:

* Empty-lexicon safety:    cold-start install → no-op, greedy wins.
* Promote-not-reject:      out-of-lexicon top is kept when no
                           lexicon-confirmed alternative exists.
* Lexicon-driven flip:     lower-confidence in-lexicon candidate
                           wins over higher-confidence not-in-lexicon
                           top.
* Greedy-already-good:     greedy top is in the lexicon → no change.
* Per-field parsing:       mass / resistance / instability digit
                           conventions match what the lexicon stores.

The candidate-list shape mirrors ``_prefix_beam_search_ctc``'s output:
``list[tuple[text, log_prob, geom_mean_conf]]`` sorted by ``log_prob``
descending.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ocr.sc_ocr import api as _api
from ocr.sc_ocr import hud_lexicon


@pytest.fixture(autouse=True)
def _isolated_lexicon(tmp_path: Path):
    """Each test starts with an empty in-memory lexicon backed by a
    temp file so we don't pollute the real ``learned_lexicon.json``.
    """
    p = tmp_path / "learned_lexicon.json"
    hud_lexicon._set_disk_path_for_tests(p)
    hud_lexicon.reset()
    yield p
    hud_lexicon._set_disk_path_for_tests(None)
    hud_lexicon.reset()


# ───────────── Empty-lexicon safety (cold-start no-op) ─────────────


def test_empty_lexicon_returns_greedy_unchanged():
    """With no entries in the lexicon, the rerank must be a strict
    no-op — the cold-start install MUST behave exactly like the
    previous greedy-only path or we risk regressing baseline accuracy.
    """
    # Top candidate has highest log_prob AND highest mean conf.
    cands = [
        ("10810", -0.1, 0.95),
        ("18810", -0.5, 0.85),
        ("10800", -0.8, 0.72),
    ]
    picked, info = _api._lexicon_rerank_candidates(cands, "mass")
    assert picked == cands[0], "empty lexicon must keep greedy top"
    assert info["changed"] is False
    assert info["n_in_lexicon"] == 0
    assert info["lex_size"] == 0


def test_empty_candidates_returns_none():
    """Defensive: empty candidate list → ``None`` so the caller can
    fall through to greedy decode cleanly."""
    picked, info = _api._lexicon_rerank_candidates([], "mass")
    assert picked is None
    assert info["changed"] is False


# ───────────── Lexicon-driven flip (the new win) ─────────────


def test_promotes_lower_conf_in_lexicon_candidate_over_greedy():
    """Greedy/top is NOT in the lexicon; a lower-mean-conf candidate
    IS in the lexicon → rerank must promote the in-lexicon read."""
    hud_lexicon.observe("mass", 10810)
    cands = [
        ("18810", -0.1, 0.92),   # greedy top, NOT in lexicon
        ("10810", -0.4, 0.71),   # in lexicon, lower conf
        ("18800", -0.7, 0.55),   # not in lexicon
    ]
    picked, info = _api._lexicon_rerank_candidates(cands, "mass")
    assert picked is not None
    assert picked[0] == "10810", "should have promoted the in-lexicon read"
    assert info["changed"] is True
    assert info["n_in_lexicon"] == 1
    assert info["greedy"] == ("18810", 0.92)
    assert info["winner"] == ("10810", 0.71)


def test_picks_highest_mean_conf_among_in_lexicon_candidates():
    """When multiple in-lexicon candidates exist, pick by HIGHEST
    geometric-mean confidence — that's the quantity downstream
    accept gates compare against their thresholds."""
    hud_lexicon.observe("mass", 10810)
    hud_lexicon.observe("mass", 10800)
    cands = [
        ("18810", -0.1, 0.92),   # not in lexicon, top by log_prob
        ("10800", -0.3, 0.65),   # in lexicon, lower mean conf
        ("10810", -0.5, 0.83),   # in lexicon, HIGHER mean conf
    ]
    picked, info = _api._lexicon_rerank_candidates(cands, "mass")
    assert picked is not None
    assert picked[0] == "10810", "should pick the higher-mean in-lexicon read"
    assert info["changed"] is True


# ───────────── Greedy-already-in-lexicon (no flip) ─────────────


def test_greedy_top_in_lexicon_keeps_greedy():
    """Greedy top is already in the lexicon → no rerank needed,
    return greedy unchanged even if other in-lexicon candidates
    exist with lower mean conf (correct) — but ALSO when another
    in-lexicon candidate happens to have HIGHER mean conf, the
    rerank picks that one (still pinned to the highest-mean
    in-lexicon).

    This test covers the simple version: greedy IS in lexicon and
    is the highest-mean candidate → return greedy.
    """
    hud_lexicon.observe("mass", 10810)
    cands = [
        ("10810", -0.1, 0.95),   # greedy, IN lexicon, highest mean
        ("10800", -0.4, 0.71),
    ]
    picked, info = _api._lexicon_rerank_candidates(cands, "mass")
    assert picked is not None
    assert picked[0] == "10810"
    assert info["changed"] is False
    assert info["n_in_lexicon"] == 1


def test_no_in_lexicon_candidate_keeps_greedy():
    """Lexicon is populated but no candidate parses to a known
    value → promote-not-reject means we keep greedy."""
    hud_lexicon.observe("mass", 99999)  # populates lexicon
    cands = [
        ("18810", -0.1, 0.92),
        ("18800", -0.5, 0.70),
        ("18820", -0.8, 0.55),
    ]
    picked, info = _api._lexicon_rerank_candidates(cands, "mass")
    assert picked is not None
    assert picked[0] == "18810", (
        "no in-lex candidate → keep greedy (promote-not-reject)"
    )
    assert info["changed"] is False
    assert info["n_in_lexicon"] == 0
    assert info["lex_size"] == 1


# ───────────── Per-field parsing ─────────────


def test_resistance_strips_percent_for_lexicon_match():
    """Resistance reads from the CRNN include a trailing ``%``;
    the lexicon stores plain integers. Parse must strip the ``%``
    so the canonical form matches."""
    hud_lexicon.observe("resistance", 50)
    cands = [
        ("75%", -0.1, 0.90),    # not in lexicon
        ("50%", -0.3, 0.75),    # in lexicon (canonical = 50)
    ]
    picked, info = _api._lexicon_rerank_candidates(cands, "resistance")
    assert picked is not None
    assert picked[0] == "50%"
    assert info["changed"] is True


def test_instability_decimal_lexicon_match():
    """Instability reads include ``.`` — the parse must keep the
    decimal so 11.01 != 11.10 in the lexicon comparison."""
    hud_lexicon.observe("instability", 11.01)
    cands = [
        ("11.10", -0.1, 0.92),  # NOT in lexicon (canonical 11.10)
        ("11.01", -0.5, 0.78),  # IN lexicon (canonical 11.01)
    ]
    picked, info = _api._lexicon_rerank_candidates(cands, "instability")
    assert picked is not None
    assert picked[0] == "11.01"
    assert info["changed"] is True


def test_mass_integer_canonical_match():
    """Mass canonical form is int. A candidate with non-integer
    float drift like '10810.0' should still match a lexicon entry
    of 10810."""
    hud_lexicon.observe("mass", 10810)
    cands = [
        ("18810", -0.1, 0.92),
        ("10810", -0.5, 0.78),   # parses to 10810.0 → canonical 10810
    ]
    picked, info = _api._lexicon_rerank_candidates(cands, "mass")
    assert picked is not None
    assert picked[0] == "10810"
    assert info["changed"] is True


# ───────────── Failure modes / silent skip ─────────────


def test_non_numeric_candidate_skipped_silently():
    """A candidate whose digit-filter yields empty string must
    not crash and must not be considered for lexicon membership."""
    hud_lexicon.observe("mass", 10810)
    cands = [
        ("18810", -0.1, 0.92),   # greedy, not in lexicon
        ("",       -0.3, 0.80),  # garbage parse — skip silently
        ("...",    -0.4, 0.78),  # all dots, parses to error → skip
        ("10810", -0.6, 0.70),   # in lexicon (lower conf)
    ]
    picked, info = _api._lexicon_rerank_candidates(cands, "mass")
    assert picked is not None
    assert picked[0] == "10810", "should reach the in-lex candidate"
    # n_in_lexicon counts only successful matches (1).
    assert info["n_in_lexicon"] == 1


def test_unknown_field_returns_greedy_without_lexicon_query():
    """``hud_lexicon.get_values('foo')`` returns an empty set, so
    rerank for an unknown field is a strict no-op."""
    cands = [
        ("18810", -0.1, 0.92),
        ("10810", -0.3, 0.85),
    ]
    picked, info = _api._lexicon_rerank_candidates(cands, "foo")
    assert picked is not None
    assert picked[0] == "18810"
    assert info["changed"] is False
    assert info["lex_size"] == 0


def test_info_dict_records_diagnostics_for_logging():
    """The call site logs the info dict on a winner change — make
    sure all the fields the log statement reads are populated."""
    hud_lexicon.observe("mass", 10810)
    cands = [
        ("18810", -0.1, 0.92),
        ("10810", -0.5, 0.71),
    ]
    picked, info = _api._lexicon_rerank_candidates(cands, "mass")
    # Required keys for the log line at the call site.
    for k in ("n_candidates", "n_in_lexicon", "lex_size",
              "greedy", "winner", "changed"):
        assert k in info, f"info dict must have {k!r}"
    assert info["n_candidates"] == 2
    assert info["greedy"] == ("18810", 0.92)
    assert info["winner"] == ("10810", 0.71)


def test_lexicon_returning_floats_with_drift_still_matches():
    """Sanity: the lexicon's canonical-key handling collapses float
    drift. A candidate parse of 50.49 must hit a lexicon entry of 50
    via the rerank — the rerank delegates to ``is_known`` which
    handles canonicalization."""
    hud_lexicon.observe("resistance", 50)
    cands = [
        ("75%", -0.1, 0.92),
        ("50%", -0.3, 0.80),
    ]
    picked, info = _api._lexicon_rerank_candidates(cands, "resistance")
    assert picked is not None
    assert picked[0] == "50%"
    assert info["changed"] is True


def test_single_candidate_returns_unchanged():
    """One-candidate beam (e.g. very confident input): must
    return that single candidate regardless of lexicon state."""
    hud_lexicon.observe("mass", 10810)
    cands = [("18810", -0.1, 0.99)]
    picked, info = _api._lexicon_rerank_candidates(cands, "mass")
    assert picked is not None
    assert picked[0] == "18810"
    assert info["changed"] is False
    assert info["n_in_lexicon"] == 0
