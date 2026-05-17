"""Manual template annotation tool for the SC mining HUD.

Loads PNG screen captures from a folder and lets you DRAW bounding
boxes for each fixed HUD element. Saves annotations as JSON sidecar
files (one ``<image>.boxes.json`` per image), then exports them in
batch into NCC templates ready for the OCR pipeline.

ELEMENTS to annotate (drawn in this fixed order):
    1.  SCAN RESULTS        — panel title text
    2.  TOP_LINE            — HUD chrome line under SCAN RESULTS
    3.  RESOURCE            — mineral name row
    4.  MASS_ROW            — entire MASS: <value> row
    5.  RESISTANCE_ROW      — entire RESISTANCE: <value> row
    6.  INSTABILITY_ROW     — entire INSTABILITY: <value> row
    7.  OUTCOME             — difficulty bar (EASY/MEDIUM/HARD/etc.)
    8.  BOT_LINE            — HUD chrome line above COMPOSITION

Workflow:
    1. Run the tool. It opens the configured folder.
    2. Pick an image from the file list (or use Next/Prev buttons).
    3. Click "Draw <ELEMENT>" to enter draw mode for that element.
    4. Click + drag on the image to draw a bounding box.
    5. Box is saved automatically to <image>.boxes.json.
    6. Continue for each element on each image (or skip — partial
       annotations are fine).
    7. When done, click "Export Templates" to bake all boxes into
       averaged templates at ocr/sc_templates/labels.npz (and a
       sidecar regions.json for the row geometry).

Keyboard shortcuts:
    1-8     select element 1..8 for drawing
    Del     delete current element's box on this image
    Left    previous image
    Right   next image
    Ctrl+S  manually save (auto-saves anyway)
    Ctrl+E  export templates
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import (
    QPoint, QPointF, QRect, QRectF, Qt, Signal,
)
from PySide6.QtGui import (
    QAction, QBrush, QColor, QFont, QImage, QKeySequence, QMouseEvent,
    QPainter, QPen, QPixmap, QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QGraphicsItem, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QStatusBar, QVBoxLayout,
    QWidget,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
DEFAULT_SOURCE = TOOL / "training_data_panels" / "user_20260418_154408" / "region1"
TEMPLATES_OUT = TOOL / "ocr" / "sc_templates" / "labels.npz"
REGIONS_OUT = TOOL / "ocr" / "sc_templates" / "regions.json"
PRESETS_DIR = TOOL / "ocr" / "sc_templates" / "annotation_presets"

# Fixed list of elements + display colors.
ELEMENTS: list[tuple[str, tuple[int, int, int]]] = [
    ("scan_results",    (255, 220, 0)),     # yellow
    ("top_line",        (255, 140, 0)),     # orange
    ("resource",        (0, 230, 100)),     # green
    ("mass_row",        (0, 200, 255)),     # cyan
    ("resistance_row",  (200, 100, 255)),   # purple
    ("instability_row", (255, 100, 200)),   # pink
    ("outcome",         (180, 180, 50)),    # olive
    ("bot_line",        (255, 140, 0)),     # orange
]

ELEMENT_NAMES = [n for n, _ in ELEMENTS]


def _color_for(name: str) -> tuple[int, int, int]:
    for n, c in ELEMENTS:
        if n == name:
            return c
    return (255, 255, 255)


# ────────────────────────────────────────────────────────────────────
# QGraphicsScene with click+drag drawing for a single element at a time
# ────────────────────────────────────────────────────────────────────


class AnnotationScene(QGraphicsScene):
    box_drawn = Signal(str, QRectF)   # element_name, scene_rect
    box_deleted = Signal(str)         # element_name

    def __init__(self):
        super().__init__()
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._active_element: Optional[str] = None
        self._dragging = False
        self._drag_start: Optional[QPointF] = None
        self._drag_rect_item: Optional[QGraphicsRectItem] = None
        self._element_rects: dict[str, QGraphicsRectItem] = {}

    def set_image(self, pil_img: Image.Image) -> None:
        self.clear()
        self._element_rects.clear()
        qimg = QImage(
            pil_img.tobytes(), pil_img.width, pil_img.height,
            pil_img.width * 3, QImage.Format_RGB888,
        )
        pix = QPixmap.fromImage(qimg)
        self._pixmap_item = self.addPixmap(pix)
        self.setSceneRect(0, 0, pix.width(), pix.height())

    def set_active_element(self, name: Optional[str]) -> None:
        self._active_element = name

    def add_box(self, name: str, x: int, y: int, w: int, h: int) -> None:
        # Remove any existing box for this element
        if name in self._element_rects:
            self.removeItem(self._element_rects[name])
            del self._element_rects[name]
        color = _color_for(name)
        pen = QPen(QColor(*color), 2)
        rect = QRectF(x, y, w, h)
        item = self.addRect(rect, pen, QBrush(Qt.NoBrush))
        # Add a label near the box
        label = self.addText(name, QFont("Arial", 9))
        label.setDefaultTextColor(QColor(*color))
        label.setPos(x, max(0, y - 16))
        item.setData(0, label)  # so we can remove the label too
        self._element_rects[name] = item

    def remove_box(self, name: str) -> None:
        if name in self._element_rects:
            item = self._element_rects[name]
            label = item.data(0)
            if label is not None:
                self.removeItem(label)
            self.removeItem(item)
            del self._element_rects[name]
            self.box_deleted.emit(name)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._active_element:
            self._dragging = True
            self._drag_start = event.scenePos()
            color = _color_for(self._active_element)
            pen = QPen(QColor(*color), 2, Qt.DashLine)
            self._drag_rect_item = self.addRect(
                QRectF(self._drag_start, self._drag_start),
                pen, QBrush(Qt.NoBrush),
            )
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging and self._drag_rect_item is not None:
            r = QRectF(self._drag_start, event.scenePos()).normalized()
            self._drag_rect_item.setRect(r)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
            r = self._drag_rect_item.rect()
            self.removeItem(self._drag_rect_item)
            self._drag_rect_item = None
            if r.width() >= 4 and r.height() >= 4:
                name = self._active_element
                self.add_box(
                    name, int(r.x()), int(r.y()),
                    int(r.width()), int(r.height()),
                )
                self.box_drawn.emit(name, r)
        else:
            super().mouseReleaseEvent(event)


# ────────────────────────────────────────────────────────────────────
# Main window
# ────────────────────────────────────────────────────────────────────


class AnnotatorWindow(QMainWindow):
    def __init__(self, source_dir: Path):
        super().__init__()
        self.setWindowTitle("SC HUD Template Annotator")
        self.resize(1500, 900)
        self._source_dir = source_dir
        self._current_path: Optional[Path] = None
        self._current_boxes: dict[str, list[int]] = {}
        self._active_element: Optional[str] = None
        # In-memory clipboard for instant copy/paste between slides.
        # Lives in the process; cleared on app exit. Use Save/Load
        # presets for cross-session persistence.
        self._clipboard_boxes: dict[str, list[int]] = {}

        # ── Layout ──
        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)

        # Left: file list
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Captures (annotated → ✓):"))
        self._file_list = QListWidget()
        self._file_list.setMinimumWidth(280)
        # Allow Ctrl-click and Shift-click multi-select for "Paste to SELECTED"
        self._file_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection,
        )
        self._file_list.currentRowChanged.connect(self._on_current_row_changed)
        left_panel.addWidget(self._file_list, 1)
        nav_row = QHBoxLayout()
        prev_btn = QPushButton("◀ Prev")
        prev_btn.clicked.connect(self._prev_image)
        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(self._next_image)
        nav_row.addWidget(prev_btn)
        nav_row.addWidget(next_btn)
        left_panel.addLayout(nav_row)
        h.addLayout(left_panel)

        # Center: image canvas
        self._scene = AnnotationScene()
        self._scene.box_drawn.connect(self._on_box_drawn)
        self._scene.box_deleted.connect(self._on_box_deleted)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.SmoothPixmapTransform)
        self._view.setMinimumWidth(700)
        h.addWidget(self._view, 1)

        # Right: element buttons + actions
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("Click element, then drag on image:"))
        self._element_buttons: dict[str, QPushButton] = {}
        for i, (name, color) in enumerate(ELEMENTS, start=1):
            btn = QPushButton(f"{i}. {name}")
            btn.setCheckable(True)
            btn.setStyleSheet(
                f"QPushButton {{ text-align: left; padding: 6px; }}"
                f"QPushButton:checked {{ background-color: rgb{color}; color: black; "
                "font-weight: bold; }}"
            )
            btn.clicked.connect(
                lambda _checked, n=name: self._select_element(n)
            )
            self._element_buttons[name] = btn
            right_panel.addWidget(btn)
        right_panel.addSpacing(8)

        del_btn = QPushButton("Delete current element's box (Del)")
        del_btn.clicked.connect(self._delete_current_box)
        right_panel.addWidget(del_btn)

        clear_btn = QPushButton("Clear ALL boxes on this image")
        clear_btn.clicked.connect(self._clear_all_boxes)
        right_panel.addWidget(clear_btn)
        right_panel.addStretch(1)

        copy_prev_btn = QPushButton("Copy boxes from Previous image")
        copy_prev_btn.clicked.connect(self._copy_from_prev)
        right_panel.addWidget(copy_prev_btn)

        # ── In-memory clipboard for instant copy/paste between slides ──
        right_panel.addWidget(QLabel("Clipboard (instant copy/paste):"))
        clip_row = QHBoxLayout()
        copy_clip_btn = QPushButton("📋 Copy boxes (Ctrl+C)")
        copy_clip_btn.setToolTip(
            "Copy this image's boxes to clipboard, then switch slides "
            "and paste. Cleared on app exit."
        )
        copy_clip_btn.clicked.connect(self._clipboard_copy)
        clip_row.addWidget(copy_clip_btn)

        paste_clip_btn = QPushButton("📥 Paste boxes (Ctrl+V)")
        paste_clip_btn.setToolTip(
            "Paste clipboard boxes onto the currently displayed image "
            "(overwrites existing boxes)."
        )
        paste_clip_btn.clicked.connect(self._clipboard_paste)
        clip_row.addWidget(paste_clip_btn)

        paste_selected_btn = QPushButton("📥 Paste to SELECTED")
        paste_selected_btn.setToolTip(
            "Paste clipboard boxes onto every image you've Ctrl-clicked "
            "in the file list on the left. Multi-select first."
        )
        paste_selected_btn.clicked.connect(self._clipboard_paste_to_selected)
        clip_row.addWidget(paste_selected_btn)
        right_panel.addLayout(clip_row)

        # Tip about multi-select
        tip = QLabel(
            "<i>Tip: Hold Ctrl/Shift in the file list (left) to "
            "select multiple files, then 'Paste to SELECTED'.</i>"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #888; padding: 2px;")
        right_panel.addWidget(tip)

        # ── Preset row: save/load named annotations ──
        right_panel.addWidget(QLabel("Presets (copy boxes between images):"))
        preset_row = QHBoxLayout()
        save_preset_btn = QPushButton("💾 Save")
        save_preset_btn.setToolTip("Save current image's boxes as a named preset")
        save_preset_btn.clicked.connect(self._save_preset)
        preset_row.addWidget(save_preset_btn)

        load_preset_btn = QPushButton("📂 Load")
        load_preset_btn.setToolTip("Apply a saved preset to this image")
        load_preset_btn.clicked.connect(self._load_preset)
        preset_row.addWidget(load_preset_btn)

        apply_all_btn = QPushButton("⇨ Apply to ALL")
        apply_all_btn.setToolTip(
            "Apply a preset to EVERY image in the folder (for batch annotation)"
        )
        apply_all_btn.clicked.connect(self._apply_preset_to_all)
        preset_row.addWidget(apply_all_btn)
        right_panel.addLayout(preset_row)

        export_btn = QPushButton("⚙ Export Templates (Ctrl+E)")
        export_btn.setStyleSheet(
            "QPushButton { background-color: #2a8; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        export_btn.clicked.connect(self._export_templates)
        right_panel.addWidget(export_btn)
        h.addLayout(right_panel)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)

        # ── Shortcuts ──
        for i, name in enumerate(ELEMENT_NAMES, start=1):
            sc = QShortcut(QKeySequence(str(i)), self)
            sc.activated.connect(lambda n=name: self._select_element(n))
        QShortcut(QKeySequence(Qt.Key_Delete), self,
                  activated=self._delete_current_box)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self._prev_image)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self._next_image)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_current_boxes)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self._export_templates)
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self._clipboard_copy)
        QShortcut(QKeySequence("Ctrl+V"), self, activated=self._clipboard_paste)

        # Populate file list
        self._scan_folder()
        if self._file_list.count() > 0:
            self._file_list.setCurrentRow(0)

    # ──────────────────────────────────────────
    # File handling
    # ──────────────────────────────────────────

    def _scan_folder(self) -> None:
        self._file_list.clear()
        if not self._source_dir.is_dir():
            QMessageBox.critical(
                self, "Folder not found",
                f"Source folder does not exist:\n{self._source_dir}",
            )
            return
        pngs = sorted(self._source_dir.glob("*.png"))
        for p in pngs:
            annotated = self._boxes_path(p).is_file()
            label = ("✓ " if annotated else "   ") + p.name
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, str(p))
            self._file_list.addItem(item)
        self._status.showMessage(
            f"Loaded {len(pngs)} captures from {self._source_dir.name}",
        )

    def _boxes_path(self, png_path: Path) -> Path:
        return png_path.with_suffix(".boxes.json")

    def _on_current_row_changed(self, row: int) -> None:
        """Loads the image for the *currently focused* row.

        Note: with ExtendedSelection mode, multiple items can be
        selected at once (for batch paste). The "currently focused"
        row is the one with the keyboard focus / dotted outline.
        We load THAT image, not the entire selection.
        """
        if row < 0:
            return
        item = self._file_list.item(row)
        if item is None:
            return
        path = Path(item.data(Qt.UserRole))
        self._load_image(path)

    def _load_image(self, path: Path) -> None:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            QMessageBox.warning(self, "Open failed", str(exc))
            return
        self._current_path = path
        self._current_boxes = {}
        self._scene.set_image(img)
        # Load any existing boxes
        bp = self._boxes_path(path)
        if bp.is_file():
            try:
                data = json.loads(bp.read_text())
                for name, box in data.get("boxes", {}).items():
                    if name not in ELEMENT_NAMES:
                        continue
                    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                    self._scene.add_box(name, x, y, w, h)
                    self._current_boxes[name] = [x, y, w, h]
            except Exception as exc:
                log.warning("failed to load %s: %s", bp, exc)
        self._update_status()
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def _next_image(self) -> None:
        row = self._file_list.currentRow()
        if row < self._file_list.count() - 1:
            self._file_list.setCurrentRow(row + 1)

    def _prev_image(self) -> None:
        row = self._file_list.currentRow()
        if row > 0:
            self._file_list.setCurrentRow(row - 1)

    # ──────────────────────────────────────────
    # Box drawing
    # ──────────────────────────────────────────

    def _select_element(self, name: str) -> None:
        self._active_element = name
        for n, btn in self._element_buttons.items():
            btn.setChecked(n == name)
        self._scene.set_active_element(name)
        self._update_status()

    def _on_box_drawn(self, name: str, rect: QRectF) -> None:
        self._current_boxes[name] = [
            int(rect.x()), int(rect.y()),
            int(rect.width()), int(rect.height()),
        ]
        self._save_current_boxes()
        self._update_status()

    def _on_box_deleted(self, name: str) -> None:
        self._current_boxes.pop(name, None)
        self._save_current_boxes()
        self._update_status()

    def _delete_current_box(self) -> None:
        if self._active_element:
            self._scene.remove_box(self._active_element)

    def _clear_all_boxes(self) -> None:
        if self._current_path is None:
            return
        if QMessageBox.question(
            self, "Clear all", "Delete all boxes on this image?",
        ) != QMessageBox.Yes:
            return
        for name in list(self._current_boxes.keys()):
            self._scene.remove_box(name)

    def _copy_from_prev(self) -> None:
        row = self._file_list.currentRow()
        if row <= 0:
            QMessageBox.information(self, "No previous", "This is the first image.")
            return
        prev_path = Path(self._file_list.item(row - 1).data(Qt.UserRole))
        bp = self._boxes_path(prev_path)
        if not bp.is_file():
            QMessageBox.information(
                self, "Previous not annotated",
                f"{prev_path.name} has no boxes file.",
            )
            return
        try:
            data = json.loads(bp.read_text())
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        for name, box in data.get("boxes", {}).items():
            if name not in ELEMENT_NAMES:
                continue
            self._scene.add_box(
                name, box["x"], box["y"], box["w"], box["h"],
            )
            self._current_boxes[name] = [box["x"], box["y"], box["w"], box["h"]]
        self._save_current_boxes()
        self._update_status()

    # ──────────────────────────────────────────
    # In-memory clipboard — copy boxes between any two slides
    # ──────────────────────────────────────────

    def _clipboard_copy(self) -> None:
        """Copy this image's boxes to the in-memory clipboard."""
        if not self._current_boxes:
            self._status.showMessage(
                "Clipboard: nothing to copy (no boxes on this image)",
                3000,
            )
            return
        # Deep copy so subsequent edits don't mutate the clipboard
        self._clipboard_boxes = {
            n: list(b) for n, b in self._current_boxes.items()
        }
        self._status.showMessage(
            f"Clipboard: copied {len(self._clipboard_boxes)} box(es) "
            f"from {self._current_path.name if self._current_path else '?'}",
            3000,
        )

    def _clipboard_paste(self) -> None:
        """Paste clipboard boxes onto the currently displayed image."""
        if not self._clipboard_boxes:
            QMessageBox.information(
                self, "Clipboard empty",
                "Use 'Copy boxes' first to copy boxes from another image.",
            )
            return
        if self._current_path is None:
            return
        # Clear existing boxes, then drop the clipboard ones in.
        for name in list(self._current_boxes.keys()):
            self._scene.remove_box(name)
        for name, b in self._clipboard_boxes.items():
            self._scene.add_box(name, b[0], b[1], b[2], b[3])
            self._current_boxes[name] = list(b)
        self._save_current_boxes()
        self._update_status()
        self._status.showMessage(
            f"Clipboard: pasted {len(self._clipboard_boxes)} box(es) "
            f"onto {self._current_path.name}",
            3000,
        )

    def _clipboard_paste_to_selected(self) -> None:
        """Paste clipboard boxes onto every selected (multi-selected) file."""
        if not self._clipboard_boxes:
            QMessageBox.information(
                self, "Clipboard empty",
                "Use 'Copy boxes' first to copy boxes from another image.",
            )
            return
        selected_items = self._file_list.selectedItems()
        if not selected_items:
            QMessageBox.information(
                self, "No selection",
                "Hold Ctrl or Shift in the file list to select multiple files first.",
            )
            return
        if QMessageBox.question(
            self, "Paste to selected",
            f"Paste clipboard boxes onto {len(selected_items)} "
            "selected image(s)? This OVERWRITES existing boxes on each.",
        ) != QMessageBox.Yes:
            return
        boxes_payload = {
            n: {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}
            for n, b in self._clipboard_boxes.items()
        }
        applied = 0
        for it in selected_items:
            png = Path(it.data(Qt.UserRole))
            try:
                png.with_suffix(".boxes.json").write_text(json.dumps({
                    "image": png.name,
                    "boxes": boxes_payload,
                    "applied_from_clipboard": True,
                }, indent=2))
                applied += 1
            except Exception as exc:
                log.warning(
                    "clipboard paste-to-selected failed for %s: %s",
                    png.name, exc,
                )
        # If the currently displayed image was in the selection, reload it
        if self._current_path is not None:
            for it in selected_items:
                if it.data(Qt.UserRole) == str(self._current_path):
                    self._load_image(self._current_path)
                    break
        self._scan_folder_preserve_selection(selected_items)
        QMessageBox.information(
            self, "Pasted",
            f"Pasted clipboard onto {applied} image(s).",
        )

    def _scan_folder_preserve_selection(self, prev_selected) -> None:
        """Re-scan folder (refresh ✓ marks) without losing selection."""
        prev_paths = {it.data(Qt.UserRole) for it in prev_selected}
        prev_current = (
            str(self._current_path) if self._current_path else None
        )
        self._scan_folder()
        for i in range(self._file_list.count()):
            item = self._file_list.item(i)
            if item.data(Qt.UserRole) in prev_paths:
                item.setSelected(True)
            if item.data(Qt.UserRole) == prev_current:
                self._file_list.setCurrentRow(i)

    # ──────────────────────────────────────────
    # Presets — save/load/apply named annotations
    # ──────────────────────────────────────────

    def _save_preset(self) -> None:
        """Save the current image's boxes as a named preset."""
        if not self._current_boxes:
            QMessageBox.information(
                self, "No boxes",
                "Draw some boxes on this image first.",
            )
            return
        name, ok = QInputDialog.getText(
            self, "Save preset",
            "Preset name (e.g. 'panel_1080p_dark'):",
        )
        if not ok or not name.strip():
            return
        # Sanitize filename
        safe = "".join(c for c in name.strip() if c.isalnum() or c in "_-")
        if not safe:
            QMessageBox.warning(self, "Invalid name", "Preset name has no usable chars.")
            return
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        path = PRESETS_DIR / f"{safe}.json"
        if path.exists():
            if QMessageBox.question(
                self, "Overwrite?",
                f"Preset '{safe}' already exists. Overwrite?",
            ) != QMessageBox.Yes:
                return
        try:
            path.write_text(json.dumps({
                "name": safe,
                "source_image": (
                    self._current_path.name if self._current_path else None
                ),
                "boxes": {
                    n: {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}
                    for n, b in self._current_boxes.items()
                },
            }, indent=2))
            QMessageBox.information(
                self, "Saved",
                f"Preset saved as:\n{path}\n\n"
                "Use 'Load' on another image to apply it.",
            )
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    def _list_presets(self) -> list[Path]:
        if not PRESETS_DIR.is_dir():
            return []
        return sorted(PRESETS_DIR.glob("*.json"))

    def _pick_preset(self, title: str = "Pick preset") -> Optional[Path]:
        presets = self._list_presets()
        if not presets:
            QMessageBox.information(
                self, "No presets",
                "No presets saved yet. Use 'Save' to create one first.",
            )
            return None
        names = [p.stem for p in presets]
        choice, ok = QInputDialog.getItem(
            self, title, "Choose a preset:", names, 0, False,
        )
        if not ok:
            return None
        for p in presets:
            if p.stem == choice:
                return p
        return None

    def _apply_preset_to_current(self, preset_path: Path) -> bool:
        try:
            data = json.loads(preset_path.read_text())
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            return False
        # Clear existing boxes first
        for name in list(self._current_boxes.keys()):
            self._scene.remove_box(name)
        for name, box in data.get("boxes", {}).items():
            if name not in ELEMENT_NAMES:
                continue
            self._scene.add_box(
                name, box["x"], box["y"], box["w"], box["h"],
            )
            self._current_boxes[name] = [
                box["x"], box["y"], box["w"], box["h"],
            ]
        self._save_current_boxes()
        self._update_status()
        return True

    def _load_preset(self) -> None:
        """Apply a saved preset to the currently displayed image."""
        if self._current_path is None:
            return
        preset = self._pick_preset("Load preset to this image")
        if preset is None:
            return
        if self._apply_preset_to_current(preset):
            QMessageBox.information(
                self, "Loaded",
                f"Applied preset '{preset.stem}' to {self._current_path.name}",
            )

    def _apply_preset_to_all(self) -> None:
        """Apply a preset to EVERY image in the folder."""
        preset = self._pick_preset("Apply preset to ALL images")
        if preset is None:
            return
        if QMessageBox.question(
            self, "Apply to all",
            f"Apply preset '{preset.stem}' to all {self._file_list.count()} "
            "images? This will OVERWRITE existing boxes on every image.",
        ) != QMessageBox.Yes:
            return
        try:
            data = json.loads(preset.read_text())
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        boxes = data.get("boxes", {})
        if not boxes:
            QMessageBox.warning(self, "Empty preset", "Preset has no boxes.")
            return
        # Iterate over all image PNGs and write a sidecar boxes file each.
        applied = 0
        for i in range(self._file_list.count()):
            png = Path(self._file_list.item(i).data(Qt.UserRole))
            try:
                png.with_suffix(".boxes.json").write_text(json.dumps({
                    "image": png.name,
                    "boxes": boxes,
                    "applied_from_preset": preset.stem,
                }, indent=2))
                applied += 1
            except Exception as exc:
                log.warning("apply-to-all failed for %s: %s", png.name, exc)
        # Refresh the UI to show checkmarks
        current_path = self._current_path
        self._scan_folder()
        if current_path is not None:
            for i in range(self._file_list.count()):
                if self._file_list.item(i).data(Qt.UserRole) == str(current_path):
                    self._file_list.setCurrentRow(i)
                    break
        QMessageBox.information(
            self, "Applied",
            f"Preset '{preset.stem}' applied to {applied} image(s).\n\n"
            "Review individual images and tweak boxes if needed before "
            "exporting templates.",
        )

    def _save_current_boxes(self) -> None:
        if self._current_path is None:
            return
        bp = self._boxes_path(self._current_path)
        try:
            bp.write_text(json.dumps({
                "image": self._current_path.name,
                "boxes": {
                    name: {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}
                    for name, b in self._current_boxes.items()
                },
            }, indent=2))
        except Exception as exc:
            log.warning("save boxes failed: %s", exc)
        # Update file-list checkmark
        items = self._file_list.selectedItems()
        if items and len(self._current_boxes) > 0:
            items[0].setText("✓ " + self._current_path.name)

    def _update_status(self) -> None:
        n_boxes = len(self._current_boxes)
        active = f" | drawing: {self._active_element}" if self._active_element else ""
        annotated_count = sum(
            1 for i in range(self._file_list.count())
            if self._file_list.item(i).text().startswith("✓")
        )
        total = self._file_list.count()
        self._status.showMessage(
            f"{n_boxes}/{len(ELEMENTS)} boxes on this image | "
            f"{annotated_count}/{total} images annotated{active}",
        )

    # ──────────────────────────────────────────
    # Template export
    # ──────────────────────────────────────────

    def _export_templates(self) -> None:
        # Gather all boxes across all annotated images
        per_element_crops: dict[str, list[np.ndarray]] = {n: [] for n in ELEMENT_NAMES}
        for png in sorted(self._source_dir.glob("*.png")):
            bp = self._boxes_path(png)
            if not bp.is_file():
                continue
            try:
                data = json.loads(bp.read_text())
                img = Image.open(png).convert("L")
            except Exception:
                continue
            for name, box in data.get("boxes", {}).items():
                if name not in per_element_crops:
                    continue
                x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                if w < 4 or h < 4:
                    continue
                crop = img.crop((x, y, x + w, y + h))
                arr = np.array(crop, dtype=np.uint8)
                per_element_crops[name].append(arr)

        # Build templates.npz: average each element's crops at canonical height.
        # For lines (top_line, bot_line) we don't average — they're horizontal
        # markers, just record their geometry.
        text_elements = ("scan_results", "resource",
                         "mass_row", "resistance_row", "instability_row",
                         "outcome")
        line_elements = ("top_line", "bot_line")
        canonical_h = 28

        payload: dict[str, np.ndarray] = {}
        for name in text_elements:
            crops = per_element_crops[name]
            if not crops:
                continue
            normalized = []
            for arr in crops:
                arr = self._canonicalize_polarity(arr)
                arr = self._resize_to_height(arr, canonical_h)
                normalized.append(arr)
            payload[name] = self._average_right_aligned(normalized)
        payload["height"] = np.array(canonical_h, dtype=np.int32)

        # Backwards compatibility: also save under the legacy keys
        # used by label_match.py
        if "mass_row" in payload:
            payload["mass"] = payload["mass_row"]
        if "resistance_row" in payload:
            payload["resistance"] = payload["resistance_row"]
        if "instability_row" in payload:
            payload["instability"] = payload["instability_row"]

        if not any(k for k in payload if k != "height"):
            QMessageBox.warning(
                self, "Nothing to export",
                "No annotated boxes found yet. Annotate some images first.",
            )
            return

        TEMPLATES_OUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez(TEMPLATES_OUT, **payload)

        # Also save geometric region info (for line positions, row pitch, etc.)
        regions = {n: per_element_crops[n] and {
            "count": len(per_element_crops[n]),
            "shape_h": int(np.median([a.shape[0] for a in per_element_crops[n]])),
            "shape_w": int(np.median([a.shape[1] for a in per_element_crops[n]])),
        } for n in ELEMENT_NAMES}
        REGIONS_OUT.write_text(json.dumps(regions, indent=2))

        # Also save preview PNGs for visual inspection
        debug_dir = TEMPLATES_OUT.parent / "labels_debug"
        debug_dir.mkdir(exist_ok=True)
        for k, arr in payload.items():
            if k == "height" or arr.ndim != 2:
                continue
            try:
                Image.fromarray(arr).save(debug_dir / f"{k}_template.png")
            except Exception:
                pass

        QMessageBox.information(
            self, "Export complete",
            f"Templates saved to:\n  {TEMPLATES_OUT}\n\n"
            f"Region metadata:\n  {REGIONS_OUT}\n\n"
            f"Per-element source counts:\n" +
            "\n".join(f"  {n}: {len(per_element_crops[n])} boxes"
                      for n in ELEMENT_NAMES if per_element_crops[n])
            + "\n\nRestart the toolbox to pick up the new templates."
        )

    # Helpers (duplicated from build_label_templates.py — kept here so this
    # script is fully self-contained)
    @staticmethod
    def _otsu(gray: np.ndarray) -> int:
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        total = gray.size
        sum_total = np.sum(np.arange(256) * hist)
        sum_bg, w_bg = 0.0, 0
        max_var, threshold = 0.0, 127
        for t in range(256):
            w_bg += int(hist[t])
            if w_bg == 0:
                continue
            w_fg = total - w_bg
            if w_fg == 0:
                break
            sum_bg += t * int(hist[t])
            m_bg = sum_bg / w_bg
            m_fg = (sum_total - sum_bg) / w_fg
            var = w_bg * w_fg * (m_bg - m_fg) ** 2
            if var > max_var:
                max_var = var
                threshold = t
        return int(threshold)

    @classmethod
    def _canonicalize_polarity(cls, gray: np.ndarray) -> np.ndarray:
        thr = cls._otsu(gray)
        bright = int((gray > thr).sum())
        dark = gray.size - bright
        if dark < bright:
            return (255 - gray).astype(np.uint8)
        return gray.astype(np.uint8)

    @staticmethod
    def _resize_to_height(arr: np.ndarray, target_h: int) -> np.ndarray:
        h, w = arr.shape
        if h == target_h:
            return arr
        scale = target_h / h
        new_w = max(8, int(round(w * scale)))
        pil = Image.fromarray(arr).resize((new_w, target_h), Image.LANCZOS)
        return np.asarray(pil, dtype=np.uint8)

    @staticmethod
    def _average_right_aligned(crops: list[np.ndarray]) -> np.ndarray:
        h = crops[0].shape[0]
        max_w = max(c.shape[1] for c in crops)
        accum = np.zeros((h, max_w), dtype=np.float32)
        counts = np.zeros((h, max_w), dtype=np.float32)
        for c in crops:
            pad = max_w - c.shape[1]
            accum[:, pad:] += c.astype(np.float32)
            counts[:, pad:] += 1.0
        avg = np.where(counts > 0, accum / np.maximum(counts, 1.0), 0.0)
        return avg.astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = AnnotatorWindow(args.source)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
