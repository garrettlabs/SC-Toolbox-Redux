"""Panel Finder popout window — embedded version of the standalone
live_panel_finder_viewer.py script.

Same job: poll ``debug_panel_overlay.png`` every 400 ms and display it
with proper centering + aspect-fit. Lives as a separate top-level
window opened from the calibration dialog so the user can:

  * Drag it freely around the screen
  * Resize from very small (~ 200 x 200) up to fullscreen
  * Close it without affecting the calibration dialog
  * Keep it open as a persistent reference while calibrating

The image is ALWAYS centered both horizontally and vertically inside
the viewer using a stretch-flanked QLabel layout, regardless of
window size or image aspect ratio.
"""
from __future__ import annotations

import logging
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QTimer, QMimeData
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

log = logging.getLogger(__name__)

# Default to the same overlay path the OCR pipeline writes to
_THIS_DIR = Path(__file__).resolve().parent
TOOL_DIR = _THIS_DIR.parent
DEFAULT_OVERLAY_PATH = TOOL_DIR / "debug_panel_overlay.png"

POLL_MS = 400

# Extra window height added when Debug Mode is switched ON. The debug
# panel otherwise just steals half of a fixed-size window from the
# overlay image — an easy change to miss, and on a small window the
# log widget gets squeezed below its 120 px minimum. Growing the
# window makes the toggle an obvious, visible action and guarantees
# the panel room. Removed again on OFF.
_DEBUG_PANEL_EXTRA_H = 240

# Loggers captured by the panel-finder Debug Mode panel. Keep this list
# in sync with the equivalent set in scripts/signature_finder_viewer.py
# (both viewers ship their own list because the two GUIs cover different
# pipelines).
_DEBUG_LOGGERS = (
    "ocr.sc_ocr.api",
    "ocr.sc_ocr.scan_results_match",
    "ocr.sc_ocr.label_match",
    "ocr.sc_ocr.hud_panel_tracker",
    "ocr.sc_ocr.hud_panel_stabilizer",
    "ocr.onnx_hud_reader",
    "hud_tracker.anchors.chrome_lines",
    "hud_tracker.anchors.hud_color_finder",
    "hud_tracker.anchors.mineral_name_color",
)

# Cap the captured-log buffer at this many lines. Older lines drop off
# the top so we never let a leak-prone diagnostic pile up unbounded.
_DEBUG_LOG_MAX_LINES = 500


class _QueueLogHandler(logging.Handler):
    """Thread-safe ``logging.Handler`` that pushes each formatted record
    into a ``queue.Queue``. Loggers may emit from worker threads; the
    GUI's poll timer drains the queue on the main thread, so the only
    cross-thread contract is the queue itself (which is already
    thread-safe).
    """

    def __init__(self, sink: queue.Queue):
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            line = f"{ts} {record.name}  {record.levelname}  {record.getMessage()}"
            self._sink.put_nowait(line)
        except Exception:
            # Never let a logging path raise — that would propagate back
            # into whichever worker emitted the record.
            pass


class PanelFinderPopout(QWidget):
    """Standalone window showing the live panel finder overlay."""

    def __init__(self, overlay_path: Optional[Path] = None, parent=None):
        # Top-level window, NOT a child of parent (so it doesn't get
        # locked to the calibration dialog's z-order). Keep parent
        # only for clean shutdown.
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("SC-OCR Panel Finder")
        self._overlay_path = overlay_path or DEFAULT_OVERLAY_PATH
        self._cached_pil: Optional[Image.Image] = None
        self._last_mtime = 0.0

        # Start SMALL — user requested. They can resize up.
        self.resize(360, 360)
        self.setMinimumSize(180, 180)

        # ── Layout ──
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        # Compact header strip
        header = QWidget()
        hl = QHBoxLayout(header)
        hl.setContentsMargins(4, 2, 4, 2)
        hl.setSpacing(6)
        self._meta = QLabel("waiting…")
        self._meta.setStyleSheet(
            "color: #888; font-family: Consolas; font-size: 8pt;"
        )
        hl.addWidget(self._meta, 1)
        # Debug Mode checkbox — toggles the log-capture panel below the
        # image. Default OFF so existing users see no behavior change.
        self._debug_checkbox = QCheckBox("Debug Mode")
        self._debug_checkbox.setStyleSheet(
            "QCheckBox { color: #ccc; font-family: Consolas; font-size: 9pt; }"
        )
        self._debug_checkbox.toggled.connect(self._on_debug_toggled)
        hl.addWidget(self._debug_checkbox)
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedSize(22, 22)
        refresh_btn.setToolTip("Force refresh now")
        refresh_btn.clicked.connect(self._tick_force)
        refresh_btn.setStyleSheet(
            "QPushButton { background: #333; color: white; border: none; "
            "font-size: 11pt; }"
            "QPushButton:hover { background: #555; }"
        )
        hl.addWidget(refresh_btn)
        v.addWidget(header)

        # ── Side-by-side image area ──
        # Left: frozen reference snapshot (only visible when a freeze
        # is active). Right: live OCR overlay. Two stretch-flanked
        # columns inside a single dark-background wrapper.
        wrap = QWidget()
        wrap.setStyleSheet("background: #111;")
        wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        wrap_h_root = QHBoxLayout(wrap)
        wrap_h_root.setContentsMargins(0, 0, 0, 0)
        wrap_h_root.setSpacing(0)

        # ── Frozen pane (left) ──
        # Hidden when no frozen reference exists. Shows the snapshot
        # image plus a small status strip with the locked values and
        # age. Stays visible until the OCR pipeline clears its frozen
        # reference (icon-absent-3s).
        frozen_col = QWidget()
        frozen_col.setStyleSheet("background: #0c1a0c;")  # subtle green tint
        frozen_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        frozen_v = QVBoxLayout(frozen_col)
        frozen_v.setContentsMargins(2, 2, 2, 2)
        frozen_v.setSpacing(2)

        frozen_header = QLabel("FROZEN")
        frozen_header.setAlignment(Qt.AlignCenter)
        frozen_header.setStyleSheet(
            "color: #6f6; background: transparent; "
            "font-family: Consolas; font-size: 8pt; font-weight: bold;"
        )
        frozen_v.addWidget(frozen_header)

        # Stretch-flanked image — same centering technique as the
        # live pane below.
        frozen_img_wrap = QWidget()
        frozen_img_wrap.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding,
        )
        frozen_img_wrap_v = QVBoxLayout(frozen_img_wrap)
        frozen_img_wrap_v.setContentsMargins(0, 0, 0, 0)
        frozen_img_wrap_v.setSpacing(0)
        frozen_img_wrap_v.addStretch(1)
        frozen_img_wrap_h = QHBoxLayout()
        frozen_img_wrap_h.setContentsMargins(0, 0, 0, 0)
        frozen_img_wrap_h.setSpacing(0)
        frozen_img_wrap_h.addStretch(1)
        self._frozen_img = QLabel("(no freeze)")
        self._frozen_img.setAlignment(Qt.AlignCenter)
        self._frozen_img.setStyleSheet(
            "background: transparent; color: #444; "
            "font-family: Consolas; font-size: 8pt;"
        )
        self._frozen_img.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        frozen_img_wrap_h.addWidget(self._frozen_img)
        frozen_img_wrap_h.addStretch(1)
        frozen_img_wrap_v.addLayout(frozen_img_wrap_h)
        frozen_img_wrap_v.addStretch(1)
        frozen_v.addWidget(frozen_img_wrap, 1)

        # Status strip — mass / resistance / instability / mineral
        # + age in seconds. Updated by _tick.
        self._frozen_status = QLabel("")
        self._frozen_status.setAlignment(Qt.AlignCenter)
        self._frozen_status.setWordWrap(True)
        self._frozen_status.setStyleSheet(
            "color: #afa; background: transparent; "
            "font-family: Consolas; font-size: 8pt;"
        )
        frozen_v.addWidget(self._frozen_status)

        # Wire the column into the root layout. Start it hidden;
        # _tick will show it when a freeze becomes active.
        self._frozen_col = frozen_col
        self._frozen_col.setVisible(False)
        wrap_h_root.addWidget(frozen_col, 1)

        # ── Live pane (right) ──
        # Existing layout, just wrapped in its own column so both panes
        # get equal width when frozen pane is visible.
        live_col = QWidget()
        live_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        live_v = QVBoxLayout(live_col)
        live_v.setContentsMargins(0, 0, 0, 0)
        live_v.setSpacing(0)
        live_v.addStretch(1)
        wrap_h = QHBoxLayout()
        wrap_h.setContentsMargins(0, 0, 0, 0)
        wrap_h.setSpacing(0)
        wrap_h.addStretch(1)
        self._img = QLabel("(no overlay yet)")
        self._img.setAlignment(Qt.AlignCenter)
        self._img.setStyleSheet(
            "background: transparent; color: #555; "
            "font-family: Consolas; font-size: 9pt;"
        )
        # Critical: SizePolicy must NOT expand, so the surrounding
        # stretches actually push the label to the center.
        self._img.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        wrap_h.addWidget(self._img)
        wrap_h.addStretch(1)
        live_v.addLayout(wrap_h)
        live_v.addStretch(1)
        wrap_h_root.addWidget(live_col, 1)

        v.addWidget(wrap, 1)

        # Cached frozen state for change detection. Avoids re-rendering
        # the (potentially large) frozen image every 400 ms when
        # nothing has changed about the freeze.
        self._cached_frozen_id: Optional[int] = None
        self._cached_frozen_pix: Optional[QPixmap] = None

        # ── Debug Mode panel (hidden until checkbox toggled on) ──
        # State: a queue.Queue between the (possibly worker-thread) log
        # handler and the main-thread poll-timer drain. The handler is
        # only attached to loggers while Debug Mode is on; toggling off
        # tears it down completely so we never leak handlers across
        # toggles.
        self._debug_log_queue: queue.Queue = queue.Queue()
        self._debug_log_handler: Optional[_QueueLogHandler] = None
        self._debug_log_lines: list[str] = []
        # Saved per-logger levels captured on toggle-ON so toggle-OFF
        # / closeEvent can restore the prior configuration. Without
        # this, ``_on_debug_toggled``'s ``setLevel(DEBUG)`` line
        # permanently mutates the shared logger registry — every
        # scan after the first toggle pays DEBUG-level overhead and
        # any other attached handler (file logger, console root)
        # sees verbose chatter forever.
        self._saved_logger_levels: dict[str, int] = {}
        # Window height captured the moment Debug Mode is switched ON
        # so toggling it OFF can restore the prior size exactly.
        self._pre_debug_height: Optional[int] = None

        self._debug_panel = QWidget()
        dp_v = QVBoxLayout(self._debug_panel)
        dp_v.setContentsMargins(4, 2, 4, 2)
        dp_v.setSpacing(3)

        # Buttons row above the log (Copy + Clear + status notification)
        dp_btns = QHBoxLayout()
        dp_btns.setContentsMargins(0, 0, 0, 0)
        dp_btns.setSpacing(6)
        self._copy_button = QPushButton("Copy log + image")
        self._copy_button.setToolTip(
            "Copy the captured log text AND the current overlay image to "
            "the clipboard. Paste into chat/email/Word for diagnostic reports."
        )
        self._copy_button.setStyleSheet(
            "QPushButton { background: #335577; color: #cce0ff; "
            "padding: 3px 10px; border: none; font-size: 9pt; }"
            "QPushButton:hover { background: #4477aa; }"
        )
        self._copy_button.clicked.connect(self._on_copy_clicked)
        dp_btns.addWidget(self._copy_button)

        self._clear_button = QPushButton("Clear")
        self._clear_button.setToolTip("Empty the captured log buffer.")
        self._clear_button.setStyleSheet(
            "QPushButton { background: #444; color: #ccc; padding: 3px 10px; "
            "border: none; font-size: 9pt; }"
            "QPushButton:hover { background: #666; }"
        )
        self._clear_button.clicked.connect(self._on_clear_clicked)
        dp_btns.addWidget(self._clear_button)

        # Inline status notification (transient — cleared after ~2.5 s)
        self._debug_status_lbl = QLabel("")
        self._debug_status_lbl.setStyleSheet(
            "color: #5fff9c; font-family: Consolas; font-size: 9pt;"
        )
        dp_btns.addWidget(self._debug_status_lbl, 1)
        dp_v.addLayout(dp_btns)

        self._debug_log_widget = QPlainTextEdit()
        self._debug_log_widget.setReadOnly(True)
        self._debug_log_widget.setMaximumBlockCount(_DEBUG_LOG_MAX_LINES)
        self._debug_log_widget.setStyleSheet(
            "QPlainTextEdit { background: #181818; color: #cccccc; "
            "border: 1px solid #333; font-family: Consolas; font-size: 9pt; }"
        )
        self._debug_log_widget.setMinimumHeight(120)
        dp_v.addWidget(self._debug_log_widget)

        self._debug_panel.setVisible(False)
        v.addWidget(self._debug_panel, 1)

        # Timer used to clear the transient "copied" status notification
        self._debug_status_timer = QTimer(self)
        self._debug_status_timer.setSingleShot(True)
        self._debug_status_timer.timeout.connect(
            lambda: self._debug_status_lbl.setText("")
        )

        # Pause-on-move: see SignatureFinderViewer for full rationale.
        # During a title-bar drag, Qt's QMoveEvent and our polling
        # tick queue on the same main thread; the tick blocks the
        # drag until it finishes. Skipping ticks while the window is
        # actively moving makes the drag feel instant.
        self._move_pause_until = 0.0
        self._move_pause_seconds = 0.35
        import time as _time
        self._time = _time

        # Polling timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(POLL_MS)

        self._tick()

    def moveEvent(self, event):
        super().moveEvent(event)
        self._move_pause_until = self._time.monotonic() + self._move_pause_seconds

    # ──────────────────────────────────────────
    # Polling + render
    # ──────────────────────────────────────────

    def _tick_force(self) -> None:
        self._last_mtime = 0.0
        self._tick()

    def _tick(self) -> None:
        # Drain captured log lines into the debug panel BEFORE the
        # move-pause check — log lines accumulate in the queue
        # regardless of whether the user is dragging, and the cap on
        # _DEBUG_LOG_MAX_LINES means we never grow unbounded. We only
        # touch widgets when Debug Mode is on.
        if self._debug_log_handler is not None:
            self._drain_debug_log_queue()
        # Update the frozen-reference pane on every tick (regardless
        # of the move-pause check below — we want the frozen pane to
        # appear/disappear instantly when the OCR pipeline freezes or
        # clears, not delayed by dragging).
        self._update_frozen_pane()
        # Skip while the window is being dragged.
        if self._time.monotonic() < self._move_pause_until:
            return
        # Heartbeat so the OCR pipeline keeps writing
        # debug_panel_overlay.png. If no viewer touches this file
        # within HEARTBEAT_TTL_SEC the pipeline skips the write
        # entirely.
        try:
            from ocr.sc_ocr import debug_overlay as _dbg
            _dbg.viewer_heartbeat_tag("overlay")
        except Exception:
            pass
        if not self._overlay_path.is_file():
            self._meta.setText(f"(missing: {self._overlay_path.name})")
            self._img.setText("Waiting for OCR pipeline…")
            return
        mtime = self._overlay_path.stat().st_mtime
        size = self._overlay_path.stat().st_size
        ts = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
        delta = max(0, int(datetime.now().timestamp() - mtime))
        self._meta.setText(f"{ts}  ({delta}s ago)  {size:,} B")

        if mtime == self._last_mtime:
            return
        self._last_mtime = mtime
        try:
            pil = Image.open(self._overlay_path).convert("RGB")
        except Exception as exc:
            self._img.setText(f"open failed: {exc}")
            return
        self._cached_pil = pil
        self._render()

    def _render(self) -> None:
        pil = self._cached_pil
        if pil is None:
            return
        # Use the wrapper's available area, NOT the QLabel's (the
        # QLabel sizes to its content under our layout).
        wrap = self._img.parent()
        avail_w = max(40, wrap.width() - 8) if wrap else 360
        avail_h = max(40, wrap.height() - 8) if wrap else 360
        ratio = min(avail_w / pil.width, avail_h / pil.height)
        new_w = max(20, int(pil.width * ratio))
        new_h = max(20, int(pil.height * ratio))
        if new_w == pil.width and new_h == pil.height:
            scaled = pil
        else:
            scaled = pil.resize((new_w, new_h), Image.LANCZOS)
        self._img.setPixmap(QPixmap.fromImage(ImageQt(scaled)))
        # Force the QLabel to size to the new pixmap so the
        # surrounding stretch flanks center it correctly.
        self._img.setFixedSize(new_w, new_h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    # ──────────────────────────────────────────
    # Frozen-reference pane
    # ──────────────────────────────────────────

    def _update_frozen_pane(self) -> None:
        """Show or hide the left-side frozen reference pane based on
        whether the OCR pipeline currently has a freeze active.

        Runs every tick (400 ms). Cheap when nothing changes — uses
        ``id()`` of the cached image to detect new freezes without
        re-rendering.
        """
        try:
            from ocr.sc_ocr.frozen_panel import get_active_frozen
            frozen = get_active_frozen()
        except Exception:
            frozen = None

        if frozen is None or not frozen.is_frozen:
            # Hide the pane if there's no active freeze.
            if self._frozen_col.isVisible():
                self._frozen_col.setVisible(False)
                self._cached_frozen_id = None
                self._cached_frozen_pix = None
            return

        # Pane visible.
        if not self._frozen_col.isVisible():
            self._frozen_col.setVisible(True)

        pil_img = frozen.panel_image
        if pil_img is None:
            return

        # Render the frozen image only when the underlying object has
        # changed (cheap identity check). The image itself is immutable
        # under our usage so id() is a reliable change signal.
        img_id = id(pil_img)
        if img_id != self._cached_frozen_id:
            try:
                # Resize to fit the available width while preserving
                # aspect ratio. Cap at the column's current width
                # (post-layout) to avoid horizontal overflow.
                target_w = max(120, min(640, self._frozen_col.width() - 8))
                w, h = pil_img.size
                if w > 0:
                    scale = target_w / float(w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    rendered = pil_img.resize(
                        (max(1, new_w), max(1, new_h)),
                        Image.LANCZOS,
                    )
                else:
                    rendered = pil_img
                qim = ImageQt(rendered.convert("RGBA"))
                pix = QPixmap.fromImage(QImage(qim))
                self._frozen_img.setPixmap(pix)
                self._frozen_img.setFixedSize(pix.size())
                self._cached_frozen_pix = pix
                self._cached_frozen_id = img_id
            except Exception as exc:
                log.debug("frozen-pane render failed: %s", exc)
                self._frozen_img.setText("(render error)")

        # Status strip always updates (age changes every tick).
        vals = frozen.values
        try:
            mass = vals.get("mass")
            resist = vals.get("resistance")
            instab = vals.get("instability")
            mineral = vals.get("mineral_name")
            age_s = frozen.age_seconds()
            since_title = frozen.time_since_title_seen()
            self._frozen_status.setText(
                f"mass={mass}  resist={resist}%  instab={instab}\n"
                f"mineral={mineral or '?'}\n"
                f"age={age_s:.1f}s  title_seen={since_title:.1f}s ago"
            )
        except Exception as exc:
            log.debug("frozen-pane status update failed: %s", exc)
            self._frozen_status.setText("(status error)")

    # ──────────────────────────────────────────
    # Debug Mode: log capture + clipboard payload
    # ──────────────────────────────────────────

    def _on_debug_toggled(self, on: bool) -> None:
        """Show/hide the debug panel. When toggling ON we attach a
        single ``_QueueLogHandler`` to every logger in
        ``_DEBUG_LOGGERS``; when toggling OFF we remove the handler
        from each one so we don't leak handlers across toggles. The
        panel widget itself is kept alive (lazy hide) so subsequent
        toggles are instantaneous."""
        if on:
            # Wire up a fresh queue + handler each time. Re-toggling
            # always starts with an empty buffer.
            self._debug_log_queue = queue.Queue()
            self._debug_log_lines = []
            self._debug_log_widget.clear()
            self._debug_log_handler = _QueueLogHandler(self._debug_log_queue)
            # Snapshot each logger's pre-toggle level BEFORE we mutate
            # anything, so the toggle-off branch can restore the
            # prior configuration verbatim. Captures even NOTSET (0).
            self._saved_logger_levels = {}
            for name in _DEBUG_LOGGERS:
                lg = logging.getLogger(name)
                self._saved_logger_levels[name] = lg.level
                lg.addHandler(self._debug_log_handler)
                # Make sure DEBUG-level records actually reach the
                # handler — many of these loggers default to WARNING.
                # We don't lower the propagation level on the root
                # logger though; only the per-source loggers we care
                # about.
                if lg.level == logging.NOTSET or lg.level > logging.DEBUG:
                    lg.setLevel(logging.DEBUG)
            self._debug_panel.setVisible(True)
            # Seed the log view with a confirmation line. Without this
            # the panel stays blank until the next scan emits a record
            # — which, combined with the panel quietly appearing in a
            # fixed-size window, reads as "the toggle did nothing".
            seed = (
                f"[Debug Mode ON — capturing {len(_DEBUG_LOGGERS)} "
                f"pipeline loggers at DEBUG. Pipeline log lines appear "
                f"here while a scan is running.]"
            )
            self._debug_log_lines.append(seed)
            self._debug_log_widget.appendPlainText(seed)
            # Grow the window so switching Debug Mode on is an obvious,
            # visible change and the log widget gets its full room.
            self._grow_window_for_debug()
        else:
            if self._debug_log_handler is not None:
                for name in _DEBUG_LOGGERS:
                    try:
                        logging.getLogger(name).removeHandler(
                            self._debug_log_handler,
                        )
                    except Exception:
                        pass
                self._debug_log_handler = None
            # Restore each logger to its pre-toggle level. Without
            # this, every logger we lifted to DEBUG above stays at
            # DEBUG forever — production scans pay verbose-logging
            # overhead and other handlers (file, root) see chatter
            # they shouldn't. Iterating ``_saved_logger_levels``
            # rather than ``_DEBUG_LOGGERS`` means a future addition
            # to the constant won't accidentally restore a logger we
            # never touched.
            for name, prior_level in self._saved_logger_levels.items():
                try:
                    logging.getLogger(name).setLevel(prior_level)
                except Exception:
                    pass
            self._saved_logger_levels = {}
            # Drop the captured buffer + queue so toggling off truly
            # releases memory.
            self._debug_log_queue = queue.Queue()
            self._debug_log_lines = []
            self._debug_log_widget.clear()
            self._debug_panel.setVisible(False)
            self._restore_window_after_debug()

    def _grow_window_for_debug(self) -> None:
        """Add height for the Debug Mode panel so toggling it on is an
        obvious, visible change. Skipped when maximised / fullscreen
        (the panel just shares the existing space then). Clamped to the
        screen's available height."""
        if self.isMaximized() or self.isFullScreen():
            self._pre_debug_height = None
            return
        self._pre_debug_height = self.height()
        target_h = self.height() + _DEBUG_PANEL_EXTRA_H
        try:
            scr = self.screen()
            if scr is not None:
                target_h = min(target_h, scr.availableGeometry().height())
        except Exception:
            pass
        self.resize(self.width(), target_h)

    def _restore_window_after_debug(self) -> None:
        """Shrink the window back to its pre-Debug-Mode height."""
        if self.isMaximized() or self.isFullScreen():
            self._pre_debug_height = None
            return
        prior = self._pre_debug_height
        if prior is not None:
            self.resize(self.width(), prior)
        self._pre_debug_height = None

    def _drain_debug_log_queue(self) -> None:
        """Move every queued log line into the panel widget. Runs on
        the main thread from ``_tick``."""
        if self._debug_log_handler is None:
            return
        appended = 0
        while True:
            try:
                line = self._debug_log_queue.get_nowait()
            except queue.Empty:
                break
            self._debug_log_lines.append(line)
            self._debug_log_widget.appendPlainText(line)
            appended += 1
            # Defensive: if the queue is being flooded by a runaway
            # logger we don't want to block the GUI for too long. Cap
            # per-tick at a generous batch.
            if appended > 1000:
                break
        if appended:
            # Keep the in-memory list bounded too (the QPlainTextEdit's
            # ``maximumBlockCount`` already trims the visible widget).
            if len(self._debug_log_lines) > _DEBUG_LOG_MAX_LINES:
                drop = len(self._debug_log_lines) - _DEBUG_LOG_MAX_LINES
                self._debug_log_lines = self._debug_log_lines[drop:]
            # Scroll to bottom so newest lines stay visible.
            sb = self._debug_log_widget.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())

    def _on_clear_clicked(self) -> None:
        self._debug_log_lines = []
        self._debug_log_widget.clear()
        # Drain anything that piled up between user clicking Clear and
        # this slot running.
        try:
            while True:
                self._debug_log_queue.get_nowait()
        except queue.Empty:
            pass

    def _on_copy_clicked(self) -> None:
        """Place captured log text + the current overlay image onto the
        clipboard. Most apps paste either text OR image depending on
        what they support; rich-content destinations (Word, email)
        get both at once."""
        log_text = "\n".join(self._debug_log_lines)
        line_count = len(self._debug_log_lines)

        mime = QMimeData()
        mime.setText(log_text)

        image_attached = False
        try:
            if self._cached_pil is not None:
                qim = QImage(ImageQt(self._cached_pil))
                if not qim.isNull():
                    mime.setImageData(qim)
                    image_attached = True
            elif self._overlay_path.is_file():
                qim = QImage(str(self._overlay_path))
                if not qim.isNull():
                    mime.setImageData(qim)
                    image_attached = True
        except Exception:
            image_attached = False

        try:
            cb = QApplication.clipboard()
            cb.setMimeData(mime)
        except Exception as exc:
            self._debug_status_lbl.setText(f"copy failed: {exc}")
            self._debug_status_timer.start(2500)
            return

        if image_attached:
            self._debug_status_lbl.setText(
                f"Copied {line_count} lines + image to clipboard"
            )
        else:
            self._debug_status_lbl.setText(
                f"Copied {line_count} lines to clipboard (no image yet)"
            )
        self._debug_status_timer.start(2500)

    def closeEvent(self, event):
        try:
            self._timer.stop()
        except Exception:
            pass
        # Tear down any attached log handlers so closing the window
        # doesn't leak them onto the global logger registry. Also
        # restore each logger's pre-toggle level — same rationale as
        # the toggle-off branch above (otherwise closing the window
        # with Debug Mode still on leaves DEBUG-level chatter active
        # for the rest of the host process's lifetime).
        try:
            if self._debug_log_handler is not None:
                for name in _DEBUG_LOGGERS:
                    try:
                        logging.getLogger(name).removeHandler(
                            self._debug_log_handler,
                        )
                    except Exception:
                        pass
                self._debug_log_handler = None
            for name, prior_level in self._saved_logger_levels.items():
                try:
                    logging.getLogger(name).setLevel(prior_level)
                except Exception:
                    pass
            self._saved_logger_levels = {}
        except Exception:
            pass
        super().closeEvent(event)
