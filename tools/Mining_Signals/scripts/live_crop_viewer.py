"""Live viewer for the runtime SC-OCR debug crops.

The runtime saves three PNGs per scan:
    debug_value_mass_crop.png
    debug_value_resistance_crop.png
    debug_value_instability_crop.png

This GUI polls them every 400 ms and shows the latest crop alongside
its modification timestamp + size. If you also want, you can log the
predicted value against the actual panel value to spot exactly where
the pipeline misclassifies.

Run with:
    python scripts/live_crop_viewer.py
or double-click LAUNCH_CropViewer.bat in training_data_panels/.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QWidget,
)


THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent

FIELDS = [
    ("MASS",         TOOL / "debug_value_mass_crop.png"),
    ("RESISTANCE",   TOOL / "debug_value_resistance_crop.png"),
    ("INSTABILITY",  TOOL / "debug_value_instability_crop.png"),
]

# Display each tiny ~150x25 crop scaled up so we can see pixel detail
DISPLAY_HEIGHT = 120


class FieldRow(QWidget):
    """One row: label, scaled crop, file metadata, expected/actual inputs."""

    def __init__(self, name: str, path: Path):
        super().__init__()
        self._name = name
        self._path = path
        self._last_mtime = 0.0

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(4)

        # Header line
        head = QHBoxLayout()
        title = QLabel(name)
        f = title.font(); f.setBold(True); f.setPointSize(11); title.setFont(f)
        head.addWidget(title)
        self._meta = QLabel("")
        self._meta.setStyleSheet("color: #888; font-size: 10px;")
        head.addWidget(self._meta, 1, Qt.AlignRight)
        v.addLayout(head)

        # Image
        self._img = QLabel("(no crop yet)")
        self._img.setAlignment(Qt.AlignCenter)
        self._img.setMinimumHeight(DISPLAY_HEIGHT + 8)
        self._img.setStyleSheet(
            "background: #111; color: #666; border: 1px solid #333;"
        )
        v.addWidget(self._img)

        # Expected vs actual side-by-side
        rowx = QHBoxLayout()
        rowx.addWidget(QLabel("Panel:"))
        self._truth = QLineEdit()
        self._truth.setPlaceholderText("type the actual panel value")
        self._truth.setFixedWidth(110)
        rowx.addWidget(self._truth)
        rowx.addWidget(QLabel("Toolbox read:"))
        self._read = QLineEdit()
        self._read.setPlaceholderText("type what toolbox showed")
        self._read.setFixedWidth(110)
        rowx.addWidget(self._read)
        self._diff = QLabel("")
        self._diff.setStyleSheet("font-weight: bold;")
        rowx.addWidget(self._diff, 1, Qt.AlignLeft)
        v.addLayout(rowx)

        self._truth.textChanged.connect(self._update_diff)
        self._read.textChanged.connect(self._update_diff)

        # Visual divider
        div = QFrame(); div.setFrameShape(QFrame.HLine); div.setStyleSheet("color: #333;")
        v.addWidget(div)

    def refresh(self):
        if not self._path.is_file():
            self._meta.setText("(file not found)")
            self._img.setText("(no crop yet)")
            return
        mtime = self._path.stat().st_mtime
        size = self._path.stat().st_size
        ts = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
        delta = max(0, int(datetime.now().timestamp() - mtime))
        self._meta.setText(f"{ts}  ({delta}s ago)  {size}B")

        if mtime == self._last_mtime:
            return  # no change — skip reload
        self._last_mtime = mtime

        try:
            pil = Image.open(self._path).convert("RGB")
        except Exception as exc:
            self._img.setText(f"open failed: {exc}")
            return
        # Scale to display height keeping aspect ratio
        ratio = DISPLAY_HEIGHT / max(1, pil.height)
        new_w = max(40, int(pil.width * ratio))
        # NEAREST keeps pixel grid visible — easier to spot artifacts
        pil = pil.resize((new_w, DISPLAY_HEIGHT), Image.NEAREST)
        self._img.setPixmap(QPixmap.fromImage(ImageQt(pil)))

    def _update_diff(self):
        t = self._truth.text().strip()
        r = self._read.text().strip()
        if not t or not r:
            self._diff.setText("")
            return
        if t == r:
            self._diff.setText("OK")
            self._diff.setStyleSheet("font-weight: bold; color: #2a8;")
        else:
            self._diff.setText(f"MISMATCH: {t!r} vs {r!r}")
            self._diff.setStyleSheet("font-weight: bold; color: #d33;")


class Viewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SC-OCR Live Crop Viewer")
        self.setMinimumSize(640, 540)

        self._rows = [FieldRow(name, path) for name, path in FIELDS]

        v = QVBoxLayout(self)
        v.setSpacing(2)
        head = QLabel(
            "Each row shows the cropped image being fed to the OCR pipeline.\n"
            "If MASS reads correctly but RESIST/INSTAB do not → crops are noisy."
        )
        head.setWordWrap(True)
        head.setStyleSheet("padding: 6px; background: #1d2530; color: #cbd; font-size: 11px;")
        v.addWidget(head)

        for row in self._rows:
            v.addWidget(row)

        # Auto-refresh + manual refresh
        bottom = QHBoxLayout()
        self._status = QLabel("Auto-refreshing every 400 ms…")
        self._status.setStyleSheet("color: #888; font-size: 10px;")
        bottom.addWidget(self._status, 1)
        refresh_btn = QPushButton("Refresh now")
        refresh_btn.clicked.connect(self._tick)
        bottom.addWidget(refresh_btn)
        v.addLayout(bottom)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(400)

    def _tick(self):
        for r in self._rows:
            r.refresh()


def main():
    app = QApplication(sys.argv)
    win = Viewer()
    primary = app.primaryScreen().availableGeometry()
    win.move(primary.left() + 50, primary.top() + 50)
    win.show()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
