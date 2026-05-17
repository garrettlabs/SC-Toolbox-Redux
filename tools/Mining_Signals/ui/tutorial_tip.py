"""One-shot tutorial tip popup with 'Do not show again' persistence.

Used by buttons that benefit from a brief upfront explanation the
first time the user clicks them — Set Scanning Region, Set Mining
HUD Region, Start Scan, etc. After the user dismisses the tip with
the checkbox ticked, subsequent clicks of that button skip the
popup entirely.

Singleton across keys: only one ``TutorialTip`` is on screen at a
time. A second call while a tip is up just raises the existing
window — the new tip's ``on_proceed`` callback does NOT fire (the
caller's button click is essentially absorbed by the existing tip,
which the user must dismiss first).

Persistence: dismiss state lives under ``config["tutorial_tip_dismissed"]``
keyed by the caller's ``key`` string. Saved via the caller's
``save_callback`` so the same persistence path that handles the rest
of the app's settings handles these too.

Usage::

    from ui.tutorial_tip import TutorialTip

    def _on_set_region(self):
        TutorialTip.show_once(
            self,
            self._config,
            lambda: _save_config(self._config),
            key="set_scan_region",
            title="Set Scanning Region",
            body_html="<p>Draw a tight box around the signal number...</p>",
            on_proceed=self._open_scan_region_selector,
        )

If the user already ticked "Do not show again" for ``set_scan_region``,
``on_proceed`` runs immediately and no popup appears. Otherwise the
popup shows and ``on_proceed`` fires when the user clicks OK.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from shared.qt.theme import P

TOOL_COLOR = "#33dd88"
_BRACKET_LEN = 14
_DISMISS_KEY = "tutorial_tip_dismissed"


class TutorialTip(QDialog):
    """One-shot tutorial tip with OK + 'Do not show again'."""

    _instance: Optional["TutorialTip"] = None

    # ──────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def show_once(
        cls,
        parent: Optional[QWidget],
        config: dict,
        save_callback: Callable[[], None],
        *,
        key: str,
        title: str,
        body_html: str,
        on_proceed: Optional[Callable[[], None]] = None,
    ) -> None:
        """Display the tip if it hasn't been dismissed for ``key``.

        Behaviour matrix:

        ============================  =================================
        State                         Effect
        ============================  =================================
        Already dismissed (key)       Skips popup; runs ``on_proceed``
                                      immediately (if provided).
        Another tip already showing   Raises that tip; this call is a
                                      no-op (no popup, no proceed).
        Neither                       Shows the popup. ``on_proceed``
                                      fires on OK click. If user
                                      closes via the X without OK,
                                      ``on_proceed`` does NOT fire.
        ============================  =================================
        """
        dismissed_map = config.get(_DISMISS_KEY) or {}
        if dismissed_map.get(key):
            if on_proceed is not None:
                on_proceed()
            return

        existing = cls._instance
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return

        tip = cls(
            parent, config, save_callback,
            key=key, title=title, body_html=body_html,
            on_proceed=on_proceed,
        )
        cls._instance = tip
        tip.show()

    # ──────────────────────────────────────────────────────────────────
    # Dialog construction
    # ──────────────────────────────────────────────────────────────────

    def __init__(
        self,
        parent: Optional[QWidget],
        config: dict,
        save_callback: Callable[[], None],
        *,
        key: str,
        title: str,
        body_html: str,
        on_proceed: Optional[Callable[[], None]],
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._save_callback = save_callback
        self._key = key
        self._on_proceed = on_proceed
        self._drag_pos: QPoint | None = None

        self.setWindowTitle(title)
        self.setWindowFlags(
            Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setModal(True)  # blocks the parent so the action waits
        self.resize(460, 280)
        self.setMinimumSize(380, 220)

        # ``finished`` fires for both accept() and reject() paths,
        # which is the only reliable hook to clear the singleton — Qt
        # bypasses ``closeEvent`` when accept()/reject() are used on
        # a dialog shown via ``show()`` (rather than ``exec()``). This
        # was the cause of the "click OK and nothing happens" bug:
        # closeEvent fired the on_proceed callback, but accept() never
        # triggers closeEvent.
        self.finished.connect(self._on_finished)

        if parent is not None:
            pg = parent.geometry()
            x = pg.x() + (pg.width() - self.width()) // 2
            y = pg.y() + (pg.height() - self.height()) // 2
            self.move(max(0, x), max(0, y))

        self._build(title, body_html)

    def _build(self, title: str, body_html: str) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)

        frame = QWidget(self)
        frame.setStyleSheet("background-color: rgba(11, 14, 20, 235);")
        flay = QVBoxLayout(frame)
        flay.setContentsMargins(0, 0, 0, 0)
        flay.setSpacing(0)

        # Title bar
        title_bar = QWidget(frame)
        title_bar.setFixedHeight(30)
        title_bar.setStyleSheet(f"background-color: {P.bg_header};")
        tb = QHBoxLayout(title_bar)
        tb.setContentsMargins(12, 0, 6, 0)
        tb.setSpacing(8)
        title_lbl = QLabel(title.upper(), title_bar)
        title_lbl.setStyleSheet(
            "font-family: Electrolize, Consolas, monospace; "
            "font-size: 10pt; font-weight: bold; "
            f"color: {TOOL_COLOR}; letter-spacing: 1.5px; "
            "background: transparent;"
        )
        tb.addWidget(title_lbl)
        tb.addStretch(1)

        close_btn = QPushButton("✕", title_bar)  # ✕
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setFixedSize(26, 22)
        close_btn.setStyleSheet(
            "QPushButton {"
            "  background: rgba(255, 60, 60, 0.15); color: #cc6666;"
            "  border: none; border-radius: 3px;"
            "  font-family: Consolas; font-size: 11pt; font-weight: bold;"
            "}"
            "QPushButton:hover { background: rgba(255, 60, 60, 0.30); "
            "color: #ffaaaa; }"
        )
        close_btn.clicked.connect(self.reject)
        tb.addWidget(close_btn)
        flay.addWidget(title_bar)

        # Body
        body = QLabel(body_html, frame)
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        body.setStyleSheet(
            "font-family: Consolas; font-size: 9pt; "
            f"color: {P.fg}; line-height: 1.5; "
            "padding: 14px 16px; background: transparent;"
        )
        body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        flay.addWidget(body, 1)

        # Footer: checkbox (left) + OK (right)
        footer = QWidget(frame)
        footer.setStyleSheet(f"background-color: {P.bg_secondary};")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(14, 8, 12, 10)
        fl.setSpacing(10)

        self._chk_dismiss = QCheckBox("Do not show again", footer)
        self._chk_dismiss.setCursor(Qt.PointingHandCursor)
        self._chk_dismiss.setStyleSheet(
            "QCheckBox {"
            f"  color: {P.fg_dim}; font-family: Consolas; font-size: 9pt;"
            "  spacing: 6px;"
            "}"
            "QCheckBox::indicator {"
            "  width: 14px; height: 14px;"
            f"  border: 1px solid {P.fg_dim}; border-radius: 2px;"
            "  background: transparent;"
            "}"
            "QCheckBox::indicator:checked {"
            f"  background: {TOOL_COLOR}; border-color: {TOOL_COLOR};"
            "}"
        )
        fl.addWidget(self._chk_dismiss)
        fl.addStretch(1)

        ok_btn = QPushButton("OK", footer)
        ok_btn.setCursor(Qt.PointingHandCursor)
        ok_btn.setFixedSize(80, 28)
        ok_btn.setStyleSheet(
            "QPushButton {"
            f"  background: {TOOL_COLOR}; color: #061a0e;"
            "  border: none; border-radius: 3px;"
            "  font-family: Consolas; font-size: 10pt; font-weight: bold;"
            "  letter-spacing: 1px;"
            "}"
            "QPushButton:hover { background: #4ee99a; }"
            "QPushButton:pressed { background: #22aa66; }"
        )
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_ok)
        fl.addWidget(ok_btn)
        flay.addWidget(footer)

        outer.addWidget(frame)

    # ──────────────────────────────────────────────────────────────────
    # OK / Close handling
    # ──────────────────────────────────────────────────────────────────

    def _on_ok(self) -> None:
        """OK clicked: persist the dismiss flag if checked, hide the
        dialog, then fire the proceed callback.

        Order matters: the dialog must be hidden BEFORE on_proceed
        runs, because on_proceed typically opens another window
        (RegionSelector) and we want this tip's modality released
        first. ``self.accept()`` calls ``hide()`` synchronously, so by
        the time ``on_proceed()`` runs the tip is already off-screen.
        """
        if self._chk_dismiss.isChecked():
            dismissed = self._config.setdefault(_DISMISS_KEY, {})
            dismissed[self._key] = True
            try:
                self._save_callback()
            except Exception:
                # Best-effort persistence: if the config save raises,
                # we still want OK to do its thing. The user just sees
                # the tip again next session.
                pass

        # Capture the callback BEFORE accept() — accept() emits the
        # ``finished`` signal which clears the singleton instance, and
        # we want to stay defensive in case any cleanup also nulls
        # ``self._on_proceed`` in some future refactor.
        proceed = self._on_proceed
        self.accept()  # hides + emits finished(QDialog.Accepted)

        if proceed is not None:
            try:
                proceed()
            except Exception:
                # Don't let the action callback's failure rebound into
                # Qt's button-click handler.
                pass

    def _on_finished(self, _result: int) -> None:
        """Singleton cleanup. Fires on every dialog termination —
        OK, X-button, Escape — via Qt's ``finished`` signal. Replaces
        the closeEvent-based cleanup that didn't fire on accept()."""
        TutorialTip._instance = None

    # ──────────────────────────────────────────────────────────────────
    # Cosmetic painting (border + corner brackets, matches TutorialPopup)
    # ──────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()

        edge = QColor(TOOL_COLOR)
        edge.setAlpha(100)
        painter.setPen(QPen(edge, 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        bl = _BRACKET_LEN
        bracket = QColor(TOOL_COLOR)
        bracket.setAlpha(220)
        painter.setPen(QPen(bracket, 2))
        painter.drawLine(0, 0, bl, 0)
        painter.drawLine(0, 0, 0, bl)
        painter.drawLine(w - 1, 0, w - 1 - bl, 0)
        painter.drawLine(w - 1, 0, w - 1, bl)
        painter.drawLine(0, h - 1, bl, h - 1)
        painter.drawLine(0, h - 1, 0, h - 1 - bl)
        painter.drawLine(w - 1, h - 1, w - 1 - bl, h - 1)
        painter.drawLine(w - 1, h - 1, w - 1, h - 1 - bl)
        painter.end()

    # Drag-by-title-bar (the QDialog itself; frameless windows lose
    # the OS-provided drag affordance otherwise).

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)
