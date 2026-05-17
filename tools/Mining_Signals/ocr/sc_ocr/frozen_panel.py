"""Frozen panel reference for the mining HUD.

Once the OCR pipeline reaches strong consensus on a SCAN RESULTS
panel — all three numeric fields (mass / resistance / instability)
have locked simultaneously via unanimous N-frame agreement — we
*snapshot* the captured panel image and lock the OCR'd values to it.
Subsequent live frames don't re-OCR for new values; they simply
verify the panel is still on-screen (via the SCAN RESULTS title
detector) and publish the frozen values.

Why this matters
----------------
The live OCR pipeline runs against jiggling frames, so even with
ideal row tracking the per-frame reads can fluctuate by ±1 digit
due to anti-aliasing, partial-pixel rendering, and segmentation
edge effects. The N-way consensus + 3-frame lock-cache already
suppress most of that noise, but the values still bounce when
freshly entering a panel or recovering from a lost lock.

A *frozen* reference flips the architecture:

    Live pipeline:  measure → measure → measure → consensus → publish
    Frozen path:    measure once → lock → publish locked value
                                    ↑
                              VERIFY (re-detect title)
                                    ↓
                          clear when title absent ≥ N seconds

Once frozen, the publish path is deterministic regardless of how
much the live OCR fluctuates. The published value is *the* value
we agreed on at the moment of lock. The only event that can change
the published value is a clear-then-refreeze cycle, which is gated
on title-absent-3s (the rock left the view, the user looked away).

State machine
-------------
* **UNFROZEN**: every scan runs the live pipeline normally. On
  every scan that locks all three numeric fields simultaneously,
  ``maybe_freeze(img, values)`` snapshots state → transition to
  FROZEN.
* **FROZEN**: every scan still runs the live pipeline (so per-glyph
  CNN training data continues to flow, and the UI can show a live
  vs. frozen comparison). But ``publish_values()`` returns the
  frozen tuple instead of the live one. Each scan that confirms the
  SCAN RESULTS title is on-screen calls ``refresh_title_seen()``,
  bumping a timestamp. If ``now - last_title_seen > timeout_sec``,
  ``maybe_clear()`` transitions back to UNFROZEN.

Per-region singletons
---------------------
Keyed by region tuple ``(x, y, w, h)`` so calibrated regions don't
share state with auto-detected ones. Same lifecycle as
``_field_lock_cache`` in api.py.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PIL import Image

log = logging.getLogger(__name__)


# Default title-absent timeout. Match the signature scanner's value
# so the two timeouts feel consistent to users.
_DEFAULT_TIMEOUT_SEC: float = 3.0

# Consecutive live-vs-frozen disagreement frames before auto-clear.
# Three frames of high-confidence disagreement is "the panel changed
# or the freeze was wrong." The structural validators in api.py
# already block misreads from entering the consensus buffer, so a
# disagreement that survives N frames means the live OCR is actually
# seeing a different value — not a one-off noise spike.
_DIVERGENCE_THRESHOLD_FRAMES: int = 3

# Consecutive frames where ALL three numeric fields produced ``None``
# raw reads before we conclude the freeze is stuck on stale values.
# 10 frames ≈ 10 seconds of continuous failure — long enough to let
# transient row-band glitches recover on their own without nuking a
# good freeze, short enough that a real scene change (where live OCR
# can't read the new panel because the row bands are wrong) clears
# the freeze within a usable timeframe. Pairs with the title-absent
# 3s timeout for the "panel disappeared" case; this one covers
# "panel changed but title-detector still latches onto something."
_ALL_NONE_THRESHOLD_FRAMES: int = 10

# Field-specific tolerance for "agreement". Mass is an integer count,
# so we require exact match. Resistance and instability are decimals
# where minor OCR rounding can flip the last digit — we accept matches
# within a small absolute tolerance.
_FIELD_AGREE_TOLERANCE: dict[str, float] = {
    "mass":        0.0,    # exact integer match
    "resistance":  0.0,    # exact integer match (resistance is X%)
    "instability": 0.5,    # within 0.5 (15.46 vs 15.47 OK)
}

# UI-freshness gate thresholds (label-match presence based).
#
# Complements the existing 3s title-absent timeout (``is_expired``)
# with a STRONGER signal: how many of the three numeric labels
# (mass / resistance / instability) matched on each scan. Title-absent
# (zero labels) for two scans in a row is much stronger evidence the
# panel left the screen than the 3-second wallclock timeout, so we
# can clear the freeze earlier and avoid publishing stale numbers
# for the leftover window.
#
# - ``_ZERO_LABEL_CLEAR_STREAK``: consecutive scans with 0/3 labels
#   matched → "panel definitely gone". Clear the freeze and surface
#   panel_visible=False to the UI.
# - ``_LOW_LABEL_TIGHTEN_STREAK``: consecutive scans with ≤1/3 labels
#   matched → "panel mostly gone". Shorten the freeze age tolerance
#   so we don't continue publishing 30-second-old values while the
#   panel is half-occluded.
# - ``_LOW_LABEL_TIGHTEN_TIMEOUT_SEC``: the shortened tolerance to
#   return when the LOW_LABEL trigger fires.
_ZERO_LABEL_CLEAR_STREAK: int = 2
_LOW_LABEL_TIGHTEN_STREAK: int = 3
_LOW_LABEL_TIGHTEN_TIMEOUT_SEC: float = 1.0


class FrozenPanelReference:
    """Snapshot of a panel + its OCR'd values, with expiry tracking.

    Lifecycle:
      * Construction → not frozen (``is_frozen == False``).
      * ``freeze(img, values)`` → sets the snapshot. ``is_frozen ==
        True``, ``last_title_seen`` updated to now.
      * ``refresh_title_seen()`` → bumps the title-seen timestamp.
        Call this every scan where the title was successfully
        detected, regardless of whether the panel is currently
        frozen — it lets the system "remember" that the panel is
        still on-screen.
      * ``is_expired(timeout_sec)`` → True if frozen but
        ``last_title_seen`` is older than the timeout.
      * ``clear()`` → drop everything, back to UNFROZEN.

    Thread safety: this class is NOT thread-safe. Callers are
    expected to serialize access (the OCR pipeline runs the
    relevant code path under the same per-scan execution).
    """

    def __init__(self) -> None:
        self._panel_image: Optional["Image.Image"] = None
        # Un-annotated raw snapshot — the same pixels the live OCR saw
        # at freeze time, without the cyan band overlay or FROZEN
        # watermark that ``_panel_image`` carries. Stored so the
        # snapshot re-OCR pass can run the full pipeline against a
        # clean image (running OCR on the annotated copy would let
        # cyan box pixels distort the underlying glyphs).
        self._raw_image: Optional["Image.Image"] = None
        # Calibration version that was in effect when this freeze was
        # captured / when the snapshot was most recently re-OCR'd.
        # The re-OCR pass compares the current learner version to this
        # field and only re-runs when they differ — same image with
        # same band geometry produces identical OCR (deterministic),
        # so re-running without a geometry change is wasted work.
        self._calibration_version_at_freeze: int = 0
        self._calibration_version_at_last_reocr: int = 0
        self._values: dict[str, object] = {}
        self._frozen_at: float = 0.0
        self._last_title_seen_at: float = 0.0
        # Divergence counters: per-field count of consecutive
        # high-confidence live readings that disagree with the frozen
        # value. Reset to 0 on every agreement. When ANY field's
        # counter reaches ``_DIVERGENCE_THRESHOLD_FRAMES`` we auto-
        # clear — the panel content has definitively changed.
        self._divergence: dict[str, int] = {
            "mass": 0,
            "resistance": 0,
            "instability": 0,
        }
        # Counter of consecutive scans where ALL three numeric live
        # reads were ``None``. Sustained None reads mean either the
        # panel is gone (title-absent will handle that), the live OCR
        # is broken (row-band drift), or the panel content changed in
        # a way that defeats the current row geometry. Beyond
        # ``_ALL_NONE_THRESHOLD_FRAMES`` we clear so a stale freeze
        # doesn't keep publishing forever.
        self._all_none_streak: int = 0
        # UI-freshness streak counters. Updated by
        # ``record_label_match_count`` once per scan with the number
        # of mass/resistance/instability labels that matched.
        #   - ``_zero_label_streak`` increments on label_count == 0
        #     and resets to 0 when ≥1 label matches.
        #   - ``_low_label_streak`` increments on label_count <= 1
        #     and resets to 0 when ≥2 labels match.
        # The two counters move independently because the thresholds
        # they feed into (full-clear vs tolerance-shorten) want
        # different sensitivities. They complement the all-None
        # streak above — that one watches OCR output failure, these
        # two watch UI-visibility failure (orthogonal signals).
        self._zero_label_streak: int = 0
        self._low_label_streak: int = 0

    # ── Public state ───────────────────────────────────────────────

    @property
    def is_frozen(self) -> bool:
        return self._panel_image is not None

    @property
    def panel_image(self) -> Optional["Image.Image"]:
        return self._panel_image

    @property
    def raw_image(self) -> Optional["Image.Image"]:
        """Un-annotated frozen snapshot, suitable for re-running OCR.
        Falls back to ``panel_image`` when the caller didn't supply a
        separate raw copy at freeze time.
        """
        return self._raw_image or self._panel_image

    @property
    def calibration_version_at_freeze(self) -> int:
        """Calibration learner version at the moment of freeze. Used
        by the snapshot re-OCR pass to decide whether geometry has
        changed since the values were captured.
        """
        return self._calibration_version_at_freeze

    @property
    def calibration_version_at_last_reocr(self) -> int:
        """Calibration learner version the snapshot was most recently
        re-OCR'd against. Equals
        ``calibration_version_at_freeze`` immediately after freeze;
        bumped each time the re-OCR pass runs against a newer
        version.
        """
        return self._calibration_version_at_last_reocr

    @property
    def values(self) -> dict[str, object]:
        return dict(self._values)

    @property
    def frozen_at(self) -> float:
        """``time.monotonic()`` value at the moment of freeze, or 0 if
        not frozen. Use for UI age display.
        """
        return self._frozen_at

    @property
    def last_title_seen_at(self) -> float:
        """``time.monotonic()`` of the most recent title-detection
        ack via ``refresh_title_seen()``. Drives the expiry check.
        """
        return self._last_title_seen_at

    def age_seconds(self) -> float:
        """Seconds since the freeze (regardless of title-seen). For UI
        labels. Returns 0 when not frozen.
        """
        if not self.is_frozen:
            return 0.0
        return max(0.0, time.monotonic() - self._frozen_at)

    def time_since_title_seen(self) -> float:
        """Seconds since the last title-detection ack. Returns 0 when
        not frozen (no expiry possible).
        """
        if not self.is_frozen:
            return 0.0
        return max(0.0, time.monotonic() - self._last_title_seen_at)

    # ── Mutators ───────────────────────────────────────────────────

    def freeze(
        self,
        img: "Image.Image",
        values: dict[str, object],
        *,
        raw_img: Optional["Image.Image"] = None,
        calibration_version: int = 0,
    ) -> None:
        """Snapshot ``img`` + ``values`` as the new frozen reference.
        Existing state (if any) is replaced.

        Parameters
        ----------
        img:
            Image used for the UI display. May be annotated (cyan
            bands, FROZEN watermark, etc.) — this is what gets shown
            in the panel-finder.
        values:
            Per-field OCR'd values captured at freeze time.
        raw_img:
            Optional un-annotated version of the same frame. When
            present, this is what the snapshot re-OCR pass consumes
            (the annotated ``img`` would have its glyphs distorted by
            the overlay graphics). When omitted, ``img`` is used for
            both display and re-OCR — fine for callers that aren't
            running snapshot re-OCR.
        calibration_version:
            The tracker's calibration version at the moment of freeze.
            The re-OCR pass compares this against the current version
            and only re-runs OCR when they differ.
        """
        # Defensive copy: the caller's img may be mutated by the
        # downstream OCR pipeline (overlay annotations, etc.). We need
        # the pristine pixel state for side-by-side display.
        try:
            self._panel_image = img.copy()
        except Exception as exc:
            log.warning(
                "FrozenPanelReference: failed to copy img (%s) — "
                "storing reference instead", exc,
            )
            self._panel_image = img
        # Raw image for re-OCR. Falls back to the same image as the
        # display copy when the caller didn't supply one.
        if raw_img is not None:
            try:
                self._raw_image = raw_img.copy()
            except Exception:
                self._raw_image = raw_img
        else:
            self._raw_image = self._panel_image
        self._calibration_version_at_freeze = int(calibration_version)
        self._calibration_version_at_last_reocr = int(calibration_version)
        self._values = dict(values)
        now = time.monotonic()
        self._frozen_at = now
        self._last_title_seen_at = now
        # Fresh freeze: clear all divergence counters so a previous
        # session's divergence noise doesn't trigger an immediate
        # auto-clear.
        self._divergence = {k: 0 for k in self._divergence}
        self._all_none_streak = 0
        # A successful freeze implies the panel was visible enough to
        # produce three locked values, so any UI-freshness streak we
        # might have accumulated from earlier scans is stale by
        # definition. Reset it to avoid spurious post-freeze clears.
        self._zero_label_streak = 0
        self._low_label_streak = 0
        log.info(
            "FrozenPanelReference: FROZE values=%s cal_v=%d",
            self._values, self._calibration_version_at_freeze,
        )

    def record_live_reading(
        self, field: str, live_value: Optional[float],
    ) -> bool:
        """Compare a live OCR reading against the frozen value for
        ``field`` and update the divergence counter.

        Returns ``True`` if this call caused an auto-clear, ``False``
        otherwise.

        ``live_value`` may be ``None`` (the live OCR couldn't read the
        field this frame — e.g. structural-validator rejected it).
        In that case we don't count divergence (no evidence either
        way), but we also don't reset the counter.

        When ANY field accumulates ``_DIVERGENCE_THRESHOLD_FRAMES``
        consecutive disagreements, the freeze auto-clears — strong
        evidence the panel changed (new rock, new scan) and the
        frozen values are stale.
        """
        if not self.is_frozen:
            return False
        if live_value is None:
            return False
        if field not in self._divergence:
            return False
        frozen_val = self._values.get(field)
        if frozen_val is None:
            # No frozen value for this field — treat as agreement
            # (don't bias toward clearing).
            self._divergence[field] = 0
            return False

        tol = _FIELD_AGREE_TOLERANCE.get(field, 0.0)
        try:
            agrees = abs(float(live_value) - float(frozen_val)) <= tol
        except (TypeError, ValueError):
            return False

        if agrees:
            self._divergence[field] = 0
            return False

        self._divergence[field] += 1
        log.info(
            "FrozenPanelReference: DIVERGENCE field=%s live=%s "
            "frozen=%s streak=%d/%d",
            field, live_value, frozen_val,
            self._divergence[field], _DIVERGENCE_THRESHOLD_FRAMES,
        )
        if self._divergence[field] >= _DIVERGENCE_THRESHOLD_FRAMES:
            log.warning(
                "FrozenPanelReference: AUTO-CLEAR — field=%s "
                "disagreed for %d consecutive frames (live=%s "
                "frozen=%s); freeze is stale",
                field, self._divergence[field],
                live_value, frozen_val,
            )
            self.clear()
            return True
        return False

    def record_scan_outcome(
        self,
        *,
        any_field_read: bool,
    ) -> bool:
        """Record whether the current scan produced any valid numeric
        read at all.

        Returns ``True`` if this call triggered an auto-clear,
        ``False`` otherwise.

        Auto-clears when ``any_field_read`` has been ``False`` for
        ``_ALL_NONE_THRESHOLD_FRAMES`` consecutive scans — a sustained
        live-pipeline failure is strong evidence the freeze no longer
        matches reality (panel changed and the geometry is now off,
        or the rock was replaced with one we can't read at the
        current calibration). The 3-frame divergence detector can't
        fire here because there's no value to disagree with; this
        catches the case it can't.

        Pass ``any_field_read=True`` whenever at least one of mass /
        resistance / instability produced a non-None structural-
        validator-passed raw value this scan. The streak resets
        on any non-None read.
        """
        if not self.is_frozen:
            return False
        if any_field_read:
            self._all_none_streak = 0
            return False
        self._all_none_streak += 1
        if self._all_none_streak >= _ALL_NONE_THRESHOLD_FRAMES:
            log.warning(
                "FrozenPanelReference: AUTO-CLEAR — live OCR produced no "
                "valid numeric reads for %d consecutive scans (threshold "
                "%d). The freeze likely doesn't reflect what's currently "
                "on-screen (geometry drift or panel changed).",
                self._all_none_streak, _ALL_NONE_THRESHOLD_FRAMES,
            )
            self.clear()
            return True
        return False

    @property
    def zero_label_streak(self) -> int:
        """Current count of consecutive scans with 0 of 3 numeric
        labels matched. Reset to 0 by ``record_label_match_count``
        on any scan with ≥1 label.
        """
        return self._zero_label_streak

    @property
    def low_label_streak(self) -> int:
        """Current count of consecutive scans with ≤1 of 3 numeric
        labels matched. Reset to 0 by ``record_label_match_count``
        on any scan with ≥2 labels.
        """
        return self._low_label_streak

    def record_label_match_count(self, count: int) -> dict:
        """Update the UI-freshness streak counters with this scan's
        label-match count (0–3 = how many of mass/resistance/
        instability matched a label row this scan) and return the
        recommended action.

        Returned dict shape:
            {
              "action": "clear" | "shorten_tolerance" | "noop",
              "reason": str,                    # human-readable
              "zero_label_streak": int,         # post-update
              "low_label_streak": int,          # post-update
              "tolerance_sec": float | None,    # for "shorten_tolerance"
              "label_match_count": int,         # echo of input
            }

        The streak math is monotonic-per-call:
          - count == 0 → ``_zero_label_streak`` += 1, ``_low_label_streak`` += 1
          - count == 1 → ``_zero_label_streak`` = 0, ``_low_label_streak`` += 1
          - count >= 2 → both counters reset to 0

        Trigger precedence (strongest signal wins):
          1. ``_zero_label_streak >= _ZERO_LABEL_CLEAR_STREAK``
             → action = "clear" — panel definitely gone.
          2. else ``_low_label_streak >= _LOW_LABEL_TIGHTEN_STREAK``
             → action = "shorten_tolerance" — panel mostly gone.
          3. else action = "noop".

        This method is callable when the ref is UNFROZEN — the streak
        counters still update so that if a freeze fires later we
        carry no stale label-presence history. It's the caller's
        responsibility to interpret "clear" on an already-unfrozen
        ref (typically still useful to surface ``panel_visible=False``
        to consumers).
        """
        try:
            n = int(count)
        except (TypeError, ValueError):
            n = 0
        if n < 0:
            n = 0

        # Update streak counters.
        if n == 0:
            self._zero_label_streak += 1
            self._low_label_streak += 1
        elif n == 1:
            self._zero_label_streak = 0
            self._low_label_streak += 1
        else:
            # n >= 2: any scan with ≥2 labels matched resets BOTH
            # streaks. The panel is "mostly visible" (transient
            # single-label miss can't fire either gate).
            self._zero_label_streak = 0
            self._low_label_streak = 0

        result: dict = {
            "action": "noop",
            "reason": "",
            "zero_label_streak": self._zero_label_streak,
            "low_label_streak": self._low_label_streak,
            "tolerance_sec": None,
            "label_match_count": n,
        }

        if self._zero_label_streak >= _ZERO_LABEL_CLEAR_STREAK:
            result["action"] = "clear"
            result["reason"] = (
                f"{self._zero_label_streak} consecutive scans with "
                f"0/3 labels matched"
            )
        elif self._low_label_streak >= _LOW_LABEL_TIGHTEN_STREAK:
            result["action"] = "shorten_tolerance"
            result["reason"] = (
                f"{self._low_label_streak} consecutive scans with "
                f"≤1/3 labels matched"
            )
            result["tolerance_sec"] = _LOW_LABEL_TIGHTEN_TIMEOUT_SEC

        return result

    def refresh_title_seen(self) -> None:
        """Mark the SCAN RESULTS title as detected this frame. Resets
        the expiry timer. No-op when not frozen (we only track title-
        seen state while a freeze is active; pre-freeze the live
        pipeline handles its own staleness).
        """
        if not self.is_frozen:
            return
        self._last_title_seen_at = time.monotonic()

    def needs_snapshot_reocr(self, current_cal_version: int) -> bool:
        """Whether the snapshot should be re-OCR'd at this version.

        Returns ``True`` when the current calibration learner version
        is newer than the version the snapshot was last OCR'd
        against — that means the band geometry the OCR pipeline will
        use has changed since the captured values were produced, so
        re-running OCR on the static snapshot can yield a cleaner
        read. Returns ``False`` when the version matches (re-running
        would produce the same output for the same image and
        geometry — wasted work) or when there's no active freeze.
        """
        if not self.is_frozen:
            return False
        return int(current_cal_version) > int(
            self._calibration_version_at_last_reocr
        )

    def mark_snapshot_reocr_done(self, at_version: int) -> None:
        """Record that the snapshot was re-OCR'd at this calibration
        version. ``needs_snapshot_reocr`` will then return ``False``
        until the version moves again.
        """
        self._calibration_version_at_last_reocr = int(at_version)

    def replace_field_values(self, values: dict) -> None:
        """Overwrite numeric field values without changing the freeze
        state. Used by the snapshot re-OCR pass to refresh captured
        values when the band geometry has improved.

        Only ``mass``, ``resistance``, ``instability``, and
        ``mineral_name`` can be replaced. Other keys are ignored.
        ``None`` values are also ignored (the snapshot OCR may
        legitimately fail on a field — we want to keep the previously
        captured value in that case, not nuke it).
        """
        if not self.is_frozen:
            return
        for k, v in (values or {}).items():
            if k not in ("mass", "resistance", "instability", "mineral_name"):
                continue
            if v is None:
                continue
            old = self._values.get(k)
            self._values[k] = v
            if old != v:
                log.info(
                    "FrozenPanelReference: snapshot re-OCR replaced "
                    "%s: %r -> %r", k, old, v,
                )

    def update_field_if_missing(
        self, field: str, value: object,
    ) -> bool:
        """Fill in a field that was ``None`` at freeze time.

        Returns ``True`` when the field was actually populated by this
        call, ``False`` otherwise. The numeric fields (``mass``,
        ``resistance``, ``instability``) are immutable once frozen —
        the whole point of the freeze is to lock those down — so
        this method silently no-ops on them and is intended for
        ancillary metadata like ``mineral_name`` that the live OCR
        may take several scans to resolve.

        Use case: the live mineral-name OCR (CRNN + Tesseract +
        fuzzy lexicon) is slower and less reliable than the numeric
        OCR, so the freeze trigger can fire on a clean numeric scan
        while ``mineral_name`` is still ``None``. Once a later scan
        resolves the mineral name we want it to show up in the UI
        without un-freezing the panel.
        """
        if not self.is_frozen:
            return False
        # Numeric fields are locked. Only ancillary metadata can be
        # back-filled.
        if field in ("mass", "resistance", "instability"):
            return False
        if value is None:
            return False
        if self._values.get(field) is not None:
            return False
        self._values[field] = value
        log.info(
            "FrozenPanelReference: back-filled %s=%r (was None at freeze)",
            field, value,
        )
        return True

    def is_expired(
        self, timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    ) -> bool:
        """True if frozen AND ``last_title_seen`` is older than
        ``timeout_sec``. False otherwise (including not-frozen).
        """
        if not self.is_frozen:
            return False
        return self.time_since_title_seen() > float(timeout_sec)

    def clear(self) -> None:
        """Drop the frozen state. Transitions back to UNFROZEN."""
        had_freeze = self.is_frozen
        if had_freeze:
            log.info(
                "FrozenPanelReference: CLEARED "
                "(frozen for %.1fs, title last seen %.1fs ago)",
                self.age_seconds(),
                self.time_since_title_seen(),
            )
        self._panel_image = None
        self._raw_image = None
        self._calibration_version_at_freeze = 0
        self._calibration_version_at_last_reocr = 0
        self._values = {}
        self._frozen_at = 0.0
        self._last_title_seen_at = 0.0
        self._divergence = {k: 0 for k in self._divergence}
        self._all_none_streak = 0
        # UI-freshness streaks are intentionally NOT reset here.
        # ``clear()`` is called from many paths (timeout, divergence,
        # zero-label gate itself); zeroing the streak from inside the
        # zero-label trigger would make the trigger oscillate. The
        # counters re-converge naturally on the next scan's
        # ``record_label_match_count`` call.


# ── Module-level singletons ──────────────────────────────────────────
# One reference per region (calibrated regions and auto-detected
# regions should not share state).

_frozen_refs: dict[str, FrozenPanelReference] = {}


def _region_key(region: Optional[dict]) -> str:
    """Build a deterministic key for ``region``. Mirrors the
    ``_region_key`` pattern used elsewhere in the pipeline.
    """
    if region is None:
        return "default"
    try:
        return (
            f"{int(region.get('x', 0))}_{int(region.get('y', 0))}_"
            f"{int(region.get('w', 0))}_{int(region.get('h', 0))}"
        )
    except Exception:
        return "default"


def get_frozen_ref(region: Optional[dict]) -> FrozenPanelReference:
    """Return the FrozenPanelReference singleton for ``region``.
    Creates one on first access; subsequent calls return the same
    instance.
    """
    key = _region_key(region)
    ref = _frozen_refs.get(key)
    if ref is None:
        ref = FrozenPanelReference()
        _frozen_refs[key] = ref
    return ref


def reset_all() -> None:
    """Clear every region's frozen state. For tests."""
    for ref in _frozen_refs.values():
        ref.clear()
    _frozen_refs.clear()


def get_active_frozen() -> Optional[FrozenPanelReference]:
    """Return the first currently-frozen reference across all regions,
    or ``None`` if no region has an active freeze.

    Used by the panel-finder UI to show a side-by-side view without
    needing to know which region is being scanned. In normal use only
    one region is active at a time, so "first frozen" == "the one
    the user cares about". If multiple regions happen to be frozen
    simultaneously, callers should iterate ``_frozen_refs`` themselves.
    """
    for ref in _frozen_refs.values():
        if ref.is_frozen:
            return ref
    return None


__all__ = [
    "FrozenPanelReference",
    "get_frozen_ref",
    "get_active_frozen",
    "reset_all",
]
