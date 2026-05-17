"""Blacklist manager — drag/drop or browse-pick any PNG icons that
should NEVER end up as training glyphs (UI markers, location pins,
icons that look digit-shaped to the segmenter, etc.).

Drops files into ``tools/Mining_Signals/training_data_blacklist/``
where the per-region extractor's ``_is_blacklisted`` check picks
them up via 8×8 perceptual hash comparison. Match threshold is 0.88
(see ``_BLACKLIST_THR`` in extract_labeled_glyphs.py).

How to use:
    1. Snip the offending icon (Win+Shift+S → save as PNG).
    2. Drag the file into this window, or click "Add file…".
    3. The thumbnail appears in the gallery. Re-run extraction
       (`python scripts/train_for_region.py signal --reset`) and the
       icon will be filtered from every staging glyph that matches.

Each entry shows file name + thumbnail. Click an entry to remove it.

Run with:
    python scripts/blacklist_manager.py
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QSize, QMimeData
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPalette, QPixmap, QColor
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)


THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
BLACKLIST_DIR = TOOL / "training_data_blacklist"

THUMB_SIZE = 64

ACCENT = "#33dd88"
RED = "#ff4444"
DIM = "#888888"
BG = "#1e1e1e"
FG = "#e0e0e0"


# ─────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────

def _ensure_dir() -> None:
    BLACKLIST_DIR.mkdir(parents=True, exist_ok=True)


def _list_entries() -> list[Path]:
    _ensure_dir()
    return sorted(BLACKLIST_DIR.glob("*.png"))


def _ingest(src: Path) -> Path | None:
    """Copy ``src`` into the blacklist dir under a unique name. Returns
    the new path on success or None if it wasn't a usable image."""
    _ensure_dir()
    try:
        img = Image.open(src)
        img.load()
    except Exception:
        return None
    # Unique name: timestamp_ms + original stem
    ts = int(time.time() * 1000)
    safe_stem = "".join(
        c for c in src.stem if c.isalnum() or c in ("_", "-")
    )[:40] or "icon"
    dst = BLACKLIST_DIR / f"{ts}_{safe_stem}.png"
    try:
        # Re-save as PNG (handles conversions from jpeg, bmp, …)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        img.save(dst, "PNG")
    except Exception:
        return None
    return dst


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

class ThumbCard(QFrame):
    """One blacklist entry — clickable, shows path + thumbnail."""

    def __init__(self, path: Path, on_remove):
        super().__init__()
        self._path = path
        self._on_remove = on_remove
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            f"background: #2a2a2a; border: 1px solid #333; border-radius: 4px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Thumbnail
        thumb = QLabel(self)
        try:
            pil = Image.open(path).convert("RGBA")
            pil.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.NEAREST)
            qim = ImageQt(pil)
            thumb.setPixmap(QPixmap.fromImage(qim))
        except Exception:
            thumb.setText("(load failed)")
            thumb.setStyleSheet(f"color: {RED};")
        thumb.setAlignment(Qt.AlignCenter)
        thumb.setMinimumSize(THUMB_SIZE, THUMB_SIZE)
        layout.addWidget(thumb)

        # Name (truncated)
        name = path.name
        if len(name) > 22:
            name = name[:19] + "…"
        name_lbl = QLabel(name, self)
        name_lbl.setStyleSheet(
            f"color: {DIM}; font-family: Consolas; font-size: 9pt; "
            f"background: transparent; border: none;"
        )
        name_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(name_lbl)

        # Remove button
        rm = QPushButton("Remove", self)
        rm.setStyleSheet(
            f"background: {RED}; color: white; padding: 3px 8px; "
            f"border: none; border-radius: 3px; font-size: 9pt;"
        )
        rm.clicked.connect(self._on_click)
        layout.addWidget(rm)

    def _on_click(self) -> None:
        try:
            self._path.unlink()
        except Exception as exc:
            QMessageBox.warning(self, "Remove failed", str(exc))
            return
        self._on_remove()


class DropZone(QLabel):
    """Drag-drop target for image files."""

    def __init__(self, on_files):
        super().__init__()
        self._on_files = on_files
        self.setText(
            "  Drop PNG / JPG icons here\n"
            "  (or use the buttons below)"
        )
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(110)
        self.setAcceptDrops(True)
        self.setStyleSheet(
            f"background: #232323; color: {DIM}; border: 2px dashed #444; "
            f"border-radius: 6px; font-family: Consolas; font-size: 11pt;"
        )

    def dragEnterEvent(self, ev: QDragEnterEvent) -> None:
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()
            self.setStyleSheet(
                f"background: #2c3a2c; color: {ACCENT}; "
                f"border: 2px dashed {ACCENT}; border-radius: 6px; "
                f"font-family: Consolas; font-size: 11pt;"
            )

    def dragLeaveEvent(self, ev) -> None:
        self.setStyleSheet(
            f"background: #232323; color: {DIM}; border: 2px dashed #444; "
            f"border-radius: 6px; font-family: Consolas; font-size: 11pt;"
        )

    def dropEvent(self, ev: QDropEvent) -> None:
        urls = ev.mimeData().urls()
        files = [Path(u.toLocalFile()) for u in urls if u.isLocalFile()]
        self.setStyleSheet(
            f"background: #232323; color: {DIM}; border: 2px dashed #444; "
            f"border-radius: 6px; font-family: Consolas; font-size: 11pt;"
        )
        self._on_files(files)


class BlacklistManager(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Glyph Blacklist Manager")
        self.setMinimumSize(640, 540)
        self.setStyleSheet(f"background: {BG}; color: {FG};")

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        title = QLabel("GLYPH BLACKLIST", self)
        from PySide6.QtGui import QFont
        tf = QFont("Consolas")
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        title.setStyleSheet(f"color: {ACCENT}; background: transparent;")
        root.addWidget(title)

        sub = QLabel(
            f"Icons here are filtered from BOTH signal and HUD glyph "
            f"extraction (perceptual-hash match). Folder: "
            f"{BLACKLIST_DIR}",
            self,
        )
        sub.setStyleSheet(
            f"color: {DIM}; background: transparent; font-size: 9pt;"
        )
        sub.setWordWrap(True)
        root.addWidget(sub)

        self._dropzone = DropZone(self._on_files_dropped)
        root.addWidget(self._dropzone)

        button_row = QHBoxLayout()
        add_btn = QPushButton("Add file…", self)
        add_btn.setStyleSheet(
            f"background: {ACCENT}; color: black; padding: 6px 16px; "
            f"border: none; border-radius: 3px; font-weight: bold;"
        )
        add_btn.clicked.connect(self._on_add_clicked)
        button_row.addWidget(add_btn)

        open_btn = QPushButton("Open folder", self)
        open_btn.setStyleSheet(
            f"background: #444; color: {FG}; padding: 6px 16px; "
            f"border: 1px solid #555; border-radius: 3px;"
        )
        open_btn.clicked.connect(self._open_folder)
        button_row.addWidget(open_btn)

        button_row.addStretch(1)

        self._count_lbl = QLabel("0 entries", self)
        self._count_lbl.setStyleSheet(
            f"color: {DIM}; background: transparent; font-family: Consolas;"
        )
        button_row.addWidget(self._count_lbl)

        root.addLayout(button_row)

        # Gallery — scrollable grid of thumbnails
        self._gallery_container = QWidget()
        self._gallery_layout = QGridLayout(self._gallery_container)
        self._gallery_layout.setSpacing(8)
        self._gallery_layout.setContentsMargins(2, 2, 2, 2)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"background: #181818; border: 1px solid #333;"
        )
        scroll.setWidget(self._gallery_container)
        root.addWidget(scroll, 1)

        self._refresh_gallery()

    # ── Slots ──

    def _on_files_dropped(self, files: list[Path]) -> None:
        added, skipped = 0, 0
        for f in files:
            if not f.is_file():
                skipped += 1
                continue
            new = _ingest(f)
            if new is None:
                skipped += 1
            else:
                added += 1
        if added:
            self._refresh_gallery()
        if skipped:
            QMessageBox.warning(
                self, "Some files skipped",
                f"Added {added}; could not read {skipped} (unsupported format?).",
            )

    def _on_add_clicked(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Pick icon image(s) to blacklist",
            "", "Images (*.png *.jpg *.jpeg *.bmp);;All files (*.*)",
        )
        if not files:
            return
        self._on_files_dropped([Path(f) for f in files])

    def _open_folder(self) -> None:
        _ensure_dir()
        try:
            import os
            os.startfile(str(BLACKLIST_DIR))  # Windows
        except AttributeError:
            import subprocess
            subprocess.Popen(["xdg-open", str(BLACKLIST_DIR)])
        except Exception:
            pass

    def _refresh_gallery(self) -> None:
        # Clear
        while self._gallery_layout.count():
            item = self._gallery_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        entries = _list_entries()
        cols = 6
        for i, p in enumerate(entries):
            r, c = divmod(i, cols)
            card = ThumbCard(p, self._refresh_gallery)
            self._gallery_layout.addWidget(card, r, c)

        # Stretch column to keep cards left-aligned
        self._gallery_layout.setColumnStretch(cols, 1)

        self._count_lbl.setText(
            f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'}"
        )


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG))
    palette.setColor(QPalette.WindowText, QColor(FG))
    palette.setColor(QPalette.Base, QColor("#2a2a2a"))
    palette.setColor(QPalette.Text, QColor(FG))
    palette.setColor(QPalette.Button, QColor("#444"))
    palette.setColor(QPalette.ButtonText, QColor(FG))
    app.setPalette(palette)

    mgr = BlacklistManager()
    mgr.show()
    mgr.raise_()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
