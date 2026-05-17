"""Always-on-top GUI launcher for the Panel Finder recorder.

A small frame-on-top window with a Record button. Click it, get a
3-second pre-roll, then 5 seconds of capture, then (optionally) a
notes dialog and a Claude-ready zip bundle. Everything that the CLI
in ``record_panel_finder.py`` does, in one click.

Why GUI instead of just the .bat?
  * Pre-roll countdown is visible in the window (not buried in a
    console behind the game).
  * Settings (duration, pre-roll, bundle on/off) are clickable knobs,
    not CLI flags.
  * The window stays on top of fullscreen-borderless games (same
    SetWindowPos trick the scan_bubble uses), so you can find it
    after alt-tabbing.

Run with:
    python scripts/record_panel_finder_gui.py
or double-click ``record_panel_finder_gui.bat`` in the tool root.
"""
from __future__ import annotations

import ctypes
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Make the sibling CLI module importable when run as a script.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from PIL import Image  # noqa: E402  (sys.path tweak above)
from PySide6.QtCore import Qt, QThread, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QFont  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QLabel, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

import record_panel_finder as recorder  # noqa: E402


# ─────────────────────────── notes dialog ──────────────────────────

class NotesDialog(QDialog):
    """Qt counterpart to the Tkinter dialog in record_panel_finder.py.

    Used when the recorder is invoked from the GUI -- mixing Qt's main
    loop with a Tkinter mainloop in the same process is fragile, so
    the GUI front-end provides its own dialog that fits inside the
    existing QApplication.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Bundle for Claude -- what did you see?")
        self.resize(720, 640)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)

        v.addWidget(_bold_label("What did you see jittering? (1-3 sentences)"))
        v.addWidget(_hint_label(
            "Examples:\n"
            "  the whole panel rectangle shifts ~3px right every other frame\n"
            "  MASS reads 27.43 then 27.48 then 27.43 in alternating frames\n"
            "  the green TOP_LINE bar disappears for one frame and the bubble"
            " flickers off"
        ))
        self.notes = QPlainTextEdit()
        self.notes.setMaximumHeight(120)
        self.notes.setFont(QFont("Consolas", 10))
        v.addWidget(self.notes)

        v.addSpacing(6)
        v.addWidget(_bold_label("Optional: Panel Finder debug log"))
        v.addWidget(_hint_label(
            "In the app: open Panel Finder popout -> tick Debug Mode ->\n"
            "click 'Copy log + image' -> paste here (Ctrl+V)."
        ))
        self.log = QPlainTextEdit()
        self.log.setFont(QFont("Consolas", 9))
        self.log.setLineWrapMode(QPlainTextEdit.NoWrap)
        v.addWidget(self.log, 1)

        bb = QDialogButtonBox()
        save = bb.addButton("Save bundle", QDialogButtonBox.AcceptRole)
        skip = bb.addButton("Skip notes", QDialogButtonBox.RejectRole)
        save.setStyleSheet(
            "QPushButton { background: #3a7; color: white; "
            "padding: 6px 16px; font-weight: bold; }"
        )
        bb.accepted.connect(self.accept)
        # "Skip notes" still saves the bundle -- it just leaves the
        # text fields empty. Treat both buttons as accept so the
        # caller doesn't have to special-case rejection.
        skip.clicked.connect(self.accept)
        v.addWidget(bb)

        self.notes.setFocus()

    def get_values(self) -> Tuple[str, str]:
        return (
            self.notes.toPlainText().strip(),
            self.log.toPlainText().strip(),
        )


def _bold_label(text: str) -> QLabel:
    lbl = QLabel(text)
    f = lbl.font()
    f.setBold(True)
    lbl.setFont(f)
    return lbl


def _hint_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #888; font-size: 9pt;")
    lbl.setWordWrap(True)
    return lbl


# ───────────────────────── recording worker ────────────────────────

class RecorderWorker(QThread):
    """Run pre-roll, capture, and write off the UI thread.

    Signals are the only contract back to the UI -- never touch
    widgets directly from ``run()``.
    """

    status = Signal(str)
    countdown = Signal(int)            # seconds remaining in pre-roll
    frame_captured = Signal(int, float)  # 1-based idx, elapsed_s
    finished_ok = Signal(object)       # session: Path
    failed = Signal(str)

    def __init__(
        self,
        duration: float,
        preroll: float,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._duration = duration
        self._preroll = preroll
        self.frames: List[Tuple[float, Image.Image]] = []
        self.session: Optional[Path] = None

    def run(self) -> None:  # noqa: D401
        try:
            recorder.OUT_ROOT.mkdir(parents=True, exist_ok=True)
            session = recorder.OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
            session.mkdir(parents=True, exist_ok=True)
            self.session = session

            if not recorder.OVERLAY_PATH.exists():
                self.failed.emit(
                    "Overlay PNG not found:\n"
                    f"{recorder.OVERLAY_PATH}\n\n"
                    "Start mining_signals_app.py and make sure the SCAN "
                    "RESULTS panel is visible in-game."
                )
                return

            for remaining in range(int(self._preroll), 0, -1):
                self.countdown.emit(remaining)
                self.msleep(1000)

            self.status.emit(f"Recording {self._duration:.0f}s...")

            def on_frame(idx: int, elapsed: float, img: Image.Image) -> None:
                self.frame_captured.emit(idx, elapsed)

            self.frames = recorder.capture_frames(
                self._duration, on_frame=on_frame,
            )

            if not self.frames:
                self.failed.emit(
                    "No frames captured -- the overlay PNG never updated.\n\n"
                    "Is the OCR pipeline actually scanning right now?\n"
                    "(The SCAN RESULTS panel needs to be visible in-game.)"
                )
                return

            self.status.emit("Writing mosaic + GIF...")
            recorder.write_loose_outputs(session, self.frames, show_diff=True)
            self.finished_ok.emit(session)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Recording failed: {exc!r}")


# ─────────────────────────── main window ───────────────────────────

class RecorderWindow(QWidget):
    """Compact always-on-top window with one Record button."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Panel Finder Recorder")
        self.resize(360, 290)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        # Settings: duration
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Duration:"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 60)
        self.duration_spin.setValue(5)
        self.duration_spin.setSuffix(" sec")
        row1.addWidget(self.duration_spin)
        row1.addStretch()
        v.addLayout(row1)

        # Settings: pre-roll
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Pre-roll: "))
        self.preroll_spin = QSpinBox()
        self.preroll_spin.setRange(0, 30)
        self.preroll_spin.setValue(3)
        self.preroll_spin.setSuffix(" sec")
        row2.addWidget(self.preroll_spin)
        row2.addStretch()
        v.addLayout(row2)

        # Bundle for Claude
        self.bundle_check = QCheckBox("Bundle for Claude (notes + zip)")
        self.bundle_check.setChecked(True)
        self.bundle_check.setToolTip(
            "After capture, prompt for a one-line description and "
            "(optionally) a pasted Panel Finder Debug Mode log, then "
            "zip everything into a single claude_bundle_<ts>.zip."
        )
        v.addWidget(self.bundle_check)

        # The Record button
        self.record_btn = QPushButton("●  RECORD")
        self.record_btn.setMinimumHeight(54)
        self.record_btn.setStyleSheet(
            "QPushButton { background: #c33; color: white; "
            "font-weight: bold; font-size: 16pt; border: none; "
            "border-radius: 4px; }"
            "QPushButton:hover { background: #d44; }"
            "QPushButton:disabled { background: #555; color: #aaa; }"
        )
        self.record_btn.clicked.connect(self._on_record_clicked)
        v.addWidget(self.record_btn)

        # Status
        self.status_lbl = QLabel("Idle")
        self.status_lbl.setStyleSheet(
            "color: #ccc; background: #222; padding: 6px; "
            "font-family: Consolas; font-size: 10pt;"
        )
        self.status_lbl.setMinimumHeight(46)
        self.status_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.status_lbl.setWordWrap(True)
        v.addWidget(self.status_lbl)

        # Open recordings folder
        open_btn = QPushButton("Open recordings folder")
        open_btn.clicked.connect(
            lambda: recorder.open_in_explorer(recorder.OUT_ROOT)
        )
        v.addWidget(open_btn)

        self._worker: Optional[RecorderWorker] = None
        self._frame_count = 0

        # Re-assert HWND_TOPMOST after the window is shown -- Qt's
        # WindowStaysOnTopHint is best-effort and gets undone by some
        # fullscreen-borderless games. Same pattern the scan_bubble uses.
        QTimer.singleShot(150, self._enforce_topmost)

    def _enforce_topmost(self) -> None:
        if not sys.platform.startswith("win"):
            return
        try:
            hwnd = int(self.winId())
            HWND_TOPMOST = -1
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOACTIVATE = 0x0010
            ctypes.windll.user32.SetWindowPos(
                hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        except Exception:
            pass

    # ── Recording lifecycle ────────────────────────────────────────

    def _on_record_clicked(self) -> None:
        self._frame_count = 0
        self.record_btn.setEnabled(False)
        self.record_btn.setText("●  recording...")
        self.status_lbl.setText("Starting...")

        self._worker = RecorderWorker(
            duration=float(self.duration_spin.value()),
            preroll=float(self.preroll_spin.value()),
            parent=self,
        )
        self._worker.status.connect(self._on_status)
        self._worker.countdown.connect(self._on_countdown)
        self._worker.frame_captured.connect(self._on_frame)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_status(self, msg: str) -> None:
        self.status_lbl.setText(msg)

    def _on_countdown(self, remaining: int) -> None:
        self.status_lbl.setText(
            f"Starting in {remaining}...  (switch to the game now)"
        )

    def _on_frame(self, idx: int, elapsed: float) -> None:
        self._frame_count = idx
        self.status_lbl.setText(
            f"Frame #{idx:02d} captured @ {elapsed:.2f}s"
        )

    def _on_finished(self, session: Path) -> None:
        worker = self._worker
        assert worker is not None

        if self.bundle_check.isChecked():
            self.status_lbl.setText(
                f"Captured {self._frame_count} frames -- fill in the notes "
                "dialog to bundle..."
            )
            dlg = NotesDialog(self)
            dlg.exec()
            notes, debug_log = dlg.get_values()
            bundle_path = recorder.build_bundle(
                session, worker.frames, notes, debug_log,
            )
            if bundle_path is not None:
                self.status_lbl.setText(
                    f"Bundle ready: {bundle_path.name}\n"
                    f"Drag it into a chat with Claude."
                )
            else:
                self.status_lbl.setText(
                    f"Captured {self._frame_count} frames "
                    f"(bundle skipped -- no frames)"
                )
        else:
            self.status_lbl.setText(
                f"Done -- {self._frame_count} frames in {session.name}"
            )

        recorder.open_in_explorer(session)
        self._reset_button()

    def _on_failed(self, msg: str) -> None:
        self.status_lbl.setText("Failed -- see message")
        QMessageBox.warning(self, "Recorder", msg)
        self._reset_button()

    def _reset_button(self) -> None:
        self.record_btn.setEnabled(True)
        self.record_btn.setText("●  RECORD")
        self._worker = None


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = RecorderWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
