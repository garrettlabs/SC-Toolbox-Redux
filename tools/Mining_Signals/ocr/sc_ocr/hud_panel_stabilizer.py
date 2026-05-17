"""HUD panel image stabilizer via phase correlation.

Once an initial absolute panel position is established (via the
rigid-body tracker), per-frame tracking reduces to *"how did the
pixels move between this frame and the last?"* — a question phase
correlation answers in O(N log N) FFT operations with sub-pixel
accuracy and zero risk of false matches against distant lookalikes.

This is the same architecture surveillance cameras and astrophotography
stacks use: detect-once, accumulate inter-frame motion vectors, and
periodically re-anchor against absolute detection to correct any
sub-pixel drift.

Why a stabilizer in addition to the rigid-body tracker?
--------------------------------------------------------
The rigid-body tracker re-runs template NCC every frame (in a local
window when locked) to find each anchor's absolute pixel position.
That's robust to large jumps but still susceptible to the
"two-strong-matches" problem (e.g. SCAN RESULTS-NCC firing both at
the real title AND at a similar-looking COMPOSITION row glyph).

Phase correlation between *consecutive* frames bypasses the matching
problem entirely. Between two consecutive frames the panel moves at
most a few pixels — and the only signal that *consistently* moves by
that delta across the whole image patch is the panel itself. The
cross-power spectrum has a unique peak at the true motion vector,
and lookalike features elsewhere in the image don't perturb it
because they're either stationary (background) or moving by a
different vector (parallax-distant world content).

Information-theoretically, "find the panel" is a much harder
problem than "find how the panel moved between two frames where it
was already known." The stabilizer exploits that gap.

Architecture
------------
* **Cold start**: delegate to ``HudPanelTracker`` for absolute pose.
  Cache the panel region as the reference patch.
* **Track step**: phase-correlate current frame's patch against the
  reference. Apply the delta to the cached pose. Update the reference
  with the current patch so each frame's reference stays fresh
  (panel content can change subtly — different rock, different
  difficulty bar, etc.).
* **Re-anchor**: every ``reanchor_every_n`` frames, run absolute
  detection again and snap the pose. Corrects sub-pixel drift
  accumulation from many small phase-correlation steps.
* **Loss of lock**: if the correlation peak is too weak
  (``response < min_response``) or implies a jump larger than
  ``max_motion_px``, reset and cold-start next frame.

No new dependencies — uses ``numpy.fft`` only.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Phase correlation primitive
# ─────────────────────────────────────────────────────────────────────


def phase_correlate(
    ref: np.ndarray,
    cur: np.ndarray,
    *,
    apply_window: bool = True,
) -> Tuple[float, float, float]:
    """Compute the translation (dx, dy) that maps ``ref`` onto ``cur``.

    Phase correlation: compute the cross-power spectrum of two image
    patches, normalize by its magnitude (phase-only), and inverse-FFT
    to a delta function whose peak is at the translation between the
    two patches.

    Returns ``(dx, dy, response)``:
      * ``dx`` is positive when ``cur`` is shifted RIGHT relative to ``ref``
        (i.e. the contents of ``cur`` appear ``dx`` pixels further right
        than in ``ref``).
      * ``dy`` is positive when ``cur`` is shifted DOWN.
      * ``response`` is the normalized peak magnitude — values near
        1.0 mean a clean unique peak; values near 0 mean the inputs
        are uncorrelated and the peak is noise.

    Inputs must be 2D arrays of identical shape. A Hanning window is
    applied to both before FFT to suppress edge artifacts (the FFT
    treats inputs as periodic, which creates spurious correlation at
    image boundaries on non-tileable patches).
    """
    if ref.shape != cur.shape:
        raise ValueError(
            f"phase_correlate: shape mismatch ref={ref.shape} cur={cur.shape}",
        )
    if ref.ndim != 2:
        raise ValueError(
            f"phase_correlate: expected 2D arrays, got {ref.ndim}D",
        )

    h, w = ref.shape

    ref_f = ref.astype(np.float64, copy=True)
    cur_f = cur.astype(np.float64, copy=True)

    # De-mean to suppress the DC term which would otherwise dominate.
    ref_f -= ref_f.mean()
    cur_f -= cur_f.mean()

    if apply_window:
        # Outer-product Hanning window. The window goes to zero at the
        # patch edges, killing edge-induced spurious peaks at (0,0).
        win = np.outer(np.hanning(h), np.hanning(w))
        ref_f *= win
        cur_f *= win

    # Cross-power spectrum.
    #
    # Sign convention: we want positive dx/dy when ``cur`` is shifted
    # right/down relative to ``ref``. With numpy's FFT convention
    # (forward FFT uses ``exp(-2πi k n / N)``), the math is:
    #
    #     cur[n] = ref[n - d]
    #     F(cur)[k] = exp(-2πi k d / N) * F(ref)[k]
    #     F(cur) * conj(F(ref)) = exp(-2πi k d / N) * |F(ref)|²
    #     iFFT(phase-norm) → peak at +d
    #
    # If we instead used F(ref) * conj(F(cur)), the peak lands at -d.
    Fa = np.fft.fft2(ref_f)
    Fb = np.fft.fft2(cur_f)
    R = Fb * np.conj(Fa)
    eps = 1e-10
    R /= np.abs(R) + eps  # phase-only normalization

    r = np.fft.ifft2(R).real

    # Locate the peak.
    flat_idx = int(np.argmax(r))
    py, px = np.unravel_index(flat_idx, r.shape)

    # FFT wraps around — negative shifts appear as large positive
    # indices. Fold them back to signed values.
    if py > h // 2:
        py -= h
    if px > w // 2:
        px -= w

    response = float(r.flat[flat_idx])
    return float(px), float(py), response


# ─────────────────────────────────────────────────────────────────────
# Stabilizer state machine
# ─────────────────────────────────────────────────────────────────────


class HudPanelStabilizer:
    """Phase-correlation panel tracker with periodic absolute re-anchor.

    Public API:
      * ``stabilize(img)`` — main entry; returns ``label_rows`` dict
        in the same shape as ``onnx_hud_reader._find_label_rows``.
      * ``reset()`` — drop the lock; next call cold-starts.
      * ``is_locked`` — whether we have a valid pose + reference patch.
      * ``pose`` — current ``(panel_x, panel_y, title_h_px)`` or None.

    Construction:
      * ``reanchor_every_n`` — frames between absolute re-detection
        (default 30 ≈ once per second at 30 FPS).
      * ``max_motion_px`` — reject correlation peaks implying a larger
        jump (likely the panel disappeared / scene cut).
      * ``min_response`` — reject peaks with low confidence (likely
        the patch no longer contains the panel).
      * ``tracker_factory`` — callable returning a ``HudPanelTracker``
        used for cold-start and re-anchor. Defaults to constructing
        a fresh tracker on demand.
    """

    # Reference-patch dimensions. Power-of-2 makes the FFT optimal.
    # 256×64 covers the SCAN RESULTS title (typ. ~250×45 in production
    # captures) with a small margin.
    _REFERENCE_W: int = 256
    _REFERENCE_H: int = 64

    def __init__(
        self,
        *,
        reanchor_every_n: int = 30,
        max_motion_px: float = 60.0,
        min_response: float = 0.02,
        tracker_factory: Optional[Callable] = None,
    ) -> None:
        self.reanchor_every_n = int(reanchor_every_n)
        self.max_motion_px = float(max_motion_px)
        self.min_response = float(min_response)
        self._tracker_factory = tracker_factory

        # Lazy-init: not all callers will exercise the cold-start path.
        self._tracker = None

        # State.
        self._pose: Optional[Tuple[float, float, float]] = None
        self._reference: Optional[np.ndarray] = None
        self._reference_origin: Optional[Tuple[int, int]] = None
        self._frames_since_anchor: int = 0
        # Human-readable reason for the most recent ``stabilize()`` call
        # that returned ``None``. Cleared when a lock succeeds. Lets the
        # integration layer surface diagnostics even when this module's
        # logger is filtered out of the viewer.
        self._last_failure_reason: Optional[str] = None
        # Snapshot of the tracker's calibration version at the moment
        # we cold-started. Phase correlation only measures translation
        # — the pose's ``scale`` carries forward from the cold-start
        # solver. When auto-calibration publishes new offsets, the
        # solver would produce a different scale, but our cached pose
        # doesn't update. Comparing version at every stabilize() call
        # lets us detect the staleness and force a fresh cold-start.
        self._calibration_version_at_cold_start: int = 0

    # ── Public state ───────────────────────────────────────────────

    @property
    def is_locked(self) -> bool:
        return self._pose is not None and self._reference is not None

    @property
    def pose(self) -> Optional[Tuple[float, float, float]]:
        return self._pose

    @property
    def last_failure_reason(self) -> Optional[str]:
        """Human-readable reason for the most recent ``stabilize()`` call
        that returned ``None``. Cleared when a lock succeeds.
        """
        return self._last_failure_reason

    def reset(self) -> None:
        """Drop all state. Next ``stabilize()`` call will cold-start."""
        self._pose = None
        self._reference = None
        self._reference_origin = None
        self._frames_since_anchor = 0

    # ── Main entry ─────────────────────────────────────────────────

    def stabilize(self, img: "Image.Image") -> Optional[dict]:
        """Return ``label_rows`` dict or ``None`` if not locked."""
        # Stale-baseline check: if auto-calibration has published a new
        # set of offsets since we last cold-started, our cached pose
        # was computed with the wrong scale and every phase-correlation
        # update inherits that wrong scale. Reset so the next call
        # cold-starts via the tracker with the current (learned) offsets.
        if self.is_locked:
            try:
                from ocr.sc_ocr.hud_panel_tracker import (
                    get_calibration_version,
                )
                _cur_ver = get_calibration_version()
                if _cur_ver != self._calibration_version_at_cold_start:
                    log.info(
                        "HudPanelStabilizer: calibration version moved "
                        "%d -> %d while locked — resetting so the next "
                        "cold-start uses the learned offsets",
                        self._calibration_version_at_cold_start, _cur_ver,
                    )
                    self.reset()
            except Exception:
                pass

        if not self.is_locked:
            return self._cold_start(img)

        new_pose = self._track_step(img)
        if new_pose is None:
            log.info("HudPanelStabilizer: lost lock, retrying cold-start")
            self.reset()
            return self._cold_start(img)

        self._pose = new_pose
        self._frames_since_anchor += 1

        if self._frames_since_anchor >= self.reanchor_every_n:
            self._reanchor(img)

        return self._pose_to_label_rows(self._pose, img.width, img.height)

    # ── Cold start ─────────────────────────────────────────────────

    def _get_tracker(self):
        if self._tracker is None:
            if self._tracker_factory is not None:
                self._tracker = self._tracker_factory()
            else:
                from ocr.sc_ocr.hud_panel_tracker import HudPanelTracker
                self._tracker = HudPanelTracker()
        return self._tracker

    def _cold_start(self, img: "Image.Image") -> Optional[dict]:
        """Establish absolute pose via the rigid-body tracker, then
        cache the current frame's panel region as the reference patch.
        """
        tracker = self._get_tracker()
        rows = tracker.track(img)
        if rows is None or tracker.last_pose is None:
            tracker_reason = getattr(tracker, "last_failure_reason", None)
            reason = (
                f"tracker cold-start failed: {tracker_reason}"
                if tracker_reason
                else "tracker cold-start failed (no reason reported)"
            )
            log.warning("HudPanelStabilizer: cold-start — %s", reason)
            self._last_failure_reason = reason
            return None

        self._pose = tracker.last_pose
        if not self._capture_reference(img):
            reason = (
                f"reference-patch capture failed (image too small or "
                f"pose out of bounds; img={img.width}x{img.height} "
                f"pose={self._pose})"
            )
            log.warning("HudPanelStabilizer: cold-start — %s", reason)
            self._last_failure_reason = reason
            self.reset()
            return None
        self._frames_since_anchor = 0
        self._last_failure_reason = None
        # Snapshot the calibration version this cold-start was solved
        # against. ``stabilize()`` compares against it on every call to
        # detect a stale baseline.
        try:
            from ocr.sc_ocr.hud_panel_tracker import (
                get_calibration_version,
            )
            self._calibration_version_at_cold_start = (
                get_calibration_version()
            )
        except Exception:
            self._calibration_version_at_cold_start = 0
        log.info(
            "HudPanelStabilizer: COLD-START lock @ pose=(%.1f,%.1f,scale=%.1f) "
            "cal_ver=%d",
            self._pose[0], self._pose[1], self._pose[2],
            self._calibration_version_at_cold_start,
        )
        return rows

    def _capture_reference(self, img: "Image.Image") -> bool:
        """Cache the panel region as the reference patch for future
        phase correlation. Centered on the SCAN RESULTS title
        (the panel origin).
        """
        if self._pose is None:
            return False
        px, py, scale = self._pose

        # Patch center: title roughly spans (panel_x, panel_y) to
        # (panel_x + title_w, panel_y + title_h). The title is
        # typically 250 wide × 45 tall. Center the patch on the
        # title's middle.
        title_cx = px + 125.0  # half-width of typical title
        title_cy = py + (scale / 2.0)

        x0 = int(round(title_cx - self._REFERENCE_W / 2))
        y0 = int(round(title_cy - self._REFERENCE_H / 2))

        # Clamp to image bounds while preserving patch size.
        x0 = max(0, min(x0, img.width - self._REFERENCE_W))
        y0 = max(0, min(y0, img.height - self._REFERENCE_H))

        if (
            x0 < 0
            or y0 < 0
            or x0 + self._REFERENCE_W > img.width
            or y0 + self._REFERENCE_H > img.height
        ):
            log.debug(
                "HudPanelStabilizer: image too small for reference patch "
                "(%dx%d, need %dx%d)",
                img.width, img.height,
                self._REFERENCE_W, self._REFERENCE_H,
            )
            return False

        gray = np.asarray(img.convert("L"), dtype=np.float64)
        patch = gray[y0:y0 + self._REFERENCE_H,
                     x0:x0 + self._REFERENCE_W].copy()
        if patch.shape != (self._REFERENCE_H, self._REFERENCE_W):
            return False
        self._reference = patch
        self._reference_origin = (x0, y0)
        return True

    # ── Per-frame tracking ─────────────────────────────────────────

    def _track_step(
        self, img: "Image.Image",
    ) -> Optional[Tuple[float, float, float]]:
        """Phase-correlate this frame's patch against the reference.
        Return the updated pose or None on lost lock.
        """
        if (
            self._reference is None
            or self._reference_origin is None
            or self._pose is None
        ):
            return None

        x0, y0 = self._reference_origin
        if (
            x0 + self._REFERENCE_W > img.width
            or y0 + self._REFERENCE_H > img.height
            or x0 < 0
            or y0 < 0
        ):
            log.debug(
                "HudPanelStabilizer: reference origin (%d,%d) out of bounds "
                "for image %dx%d", x0, y0, img.width, img.height,
            )
            return None

        gray = np.asarray(img.convert("L"), dtype=np.float64)
        cur_patch = gray[y0:y0 + self._REFERENCE_H,
                         x0:x0 + self._REFERENCE_W]
        if cur_patch.shape != self._reference.shape:
            return None

        dx, dy, response = phase_correlate(self._reference, cur_patch)
        log.debug(
            "HudPanelStabilizer: phase_corr dx=%+.2f dy=%+.2f response=%.4f",
            dx, dy, response,
        )

        if response < self.min_response:
            reason = (
                f"weak phase-correlation peak (response={response:.4f} < "
                f"{self.min_response:.4f} threshold) — likely lost panel"
            )
            log.warning("HudPanelStabilizer: %s", reason)
            self._last_failure_reason = reason
            return None

        motion = float(np.hypot(dx, dy))
        if motion > self.max_motion_px:
            reason = (
                f"motion {motion:.1f}px > {self.max_motion_px:.1f}px "
                f"(dx={dx:+.1f}, dy={dy:+.1f}) — scene cut / lost lock"
            )
            log.warning("HudPanelStabilizer: %s", reason)
            self._last_failure_reason = reason
            return None

        # The phase correlation tells us how the reference content
        # appears in the current frame. If dx=+5, the panel pixels
        # in the current frame are 5 px to the right of where they
        # were in the reference frame.
        px, py, scale = self._pose
        new_pose = (px + dx, py + dy, scale)

        # Update the reference patch with this frame so each frame's
        # delta is computed against the immediately-prior frame, not
        # against the cold-start frame. This keeps the reference fresh
        # against gradual content changes (different rock, slowly
        # changing difficulty bar, etc.) and prevents the correlation
        # peak from drifting toward zero over many frames.
        new_x0 = int(round(x0 + dx))
        new_y0 = int(round(y0 + dy))
        new_x0 = max(0, min(new_x0, img.width - self._REFERENCE_W))
        new_y0 = max(0, min(new_y0, img.height - self._REFERENCE_H))
        new_patch = gray[
            new_y0:new_y0 + self._REFERENCE_H,
            new_x0:new_x0 + self._REFERENCE_W,
        ].copy()
        if new_patch.shape == self._reference.shape:
            self._reference = new_patch
            self._reference_origin = (new_x0, new_y0)

        return new_pose

    # ── Periodic re-anchor ─────────────────────────────────────────

    def _reanchor(self, img: "Image.Image") -> None:
        """Run absolute detection to correct sub-pixel drift.

        The phase correlation step uses integer-pixel motion vectors
        (no sub-pixel refinement in this implementation), so after
        many frames a sub-pixel drift can accumulate. Running absolute
        detection periodically snaps the pose back to ground truth.
        """
        tracker = self._get_tracker()
        # Force a fresh cold-start in the tracker. Its own lock state
        # would otherwise just do local-window search.
        tracker.reset()
        rows = tracker.track(img)
        if rows is None or tracker.last_pose is None:
            log.debug(
                "HudPanelStabilizer: re-anchor detection failed, "
                "keeping current pose"
            )
            self._frames_since_anchor = 0
            return

        abs_pose = tracker.last_pose
        drift = float(
            np.hypot(
                abs_pose[0] - self._pose[0],
                abs_pose[1] - self._pose[1],
            )
        )
        log.info(
            "HudPanelStabilizer: RE-ANCHOR drift=%.2fpx — snapping "
            "pose (%.1f,%.1f) -> (%.1f,%.1f)",
            drift,
            self._pose[0], self._pose[1],
            abs_pose[0], abs_pose[1],
        )
        self._pose = abs_pose
        self._capture_reference(img)
        self._frames_since_anchor = 0

    # ── Output shaping ─────────────────────────────────────────────

    def _pose_to_label_rows(
        self,
        pose: Tuple[float, float, float],
        img_width: int,
        img_height: int,
    ) -> dict:
        """Mirrors ``HudPanelTracker._pose_to_label_rows`` so the
        stabilizer's output is byte-identical to the tracker's when
        the poses agree.
        """
        from ocr.sc_ocr.hud_panel_tracker import (
            _ROW_OFFSET_MULTS,
            _VALUE_COL_LEFT_FRAC,
            get_learned_row_mults,
        )

        panel_x, panel_y, scale = pose
        title_h_px = max(1.0, float(scale))
        # Match the tracker + EARLY-DIRECT band half-height. 0.5 ×
        # title_h keeps the crop under CRNN's 50-px ceiling so the
        # downstream segmenter doesn't get force-fed a too-tall crop.
        half_h = max(8, int(title_h_px * 0.5))
        label_right = int(img_width * _VALUE_COL_LEFT_FRAC)

        # Mirror the tracker's auto-cal consult: prefer learned
        # row-center mults when calibration has published them,
        # otherwise fall back to the production defaults.
        row_mults = get_learned_row_mults() or _ROW_OFFSET_MULTS

        rows: dict[str, tuple[int, int, int]] = {}
        for key, mult in row_mults.items():
            center_y = panel_y + title_h_px * mult
            y1 = max(0, int(center_y - half_h))
            y2 = min(int(img_height), int(center_y + half_h))
            if y2 - y1 < 4:
                continue
            rows[key] = (y1, y2, label_right)
        return rows


__all__ = ["HudPanelStabilizer", "phase_correlate"]
