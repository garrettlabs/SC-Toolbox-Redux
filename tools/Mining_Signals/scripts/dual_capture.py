"""Dual-region screen capture tool with global hotkeys.

Use case: while playing Star Citizen, position two overlay rectangles
on the screen (e.g. one over the SCAN RESULTS panel, one over a
secondary HUD element). Press SPACE to capture region 1, B to capture
region 2. Captures are saved as PNGs under
``training_data_panels/<folder>/region1/`` and ``.../region2/``.

The control window shows:
  - Enable/disable toggle (master switch for hotkeys)
  - Folder name (defaults to user_<timestamp>)
  - Show/hide each overlay rectangle
  - Width/Height spin boxes for each region
  - Live counter of captures saved per region

Each overlay rectangle is a frameless always-on-top window with a
colored border (red for R1, blue for R2). Drag the body to move,
drag the bottom-right corner to resize.

Hotkeys are global (work even when SC has focus) and DON'T suppress
the key — the game still receives SPACE/B normally.

Run with:  python scripts/dual_capture.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import ImageGrab
from PySide6.QtCore import Qt, QPoint, QRect, QSize, Signal, QObject
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)
from pynput import keyboard

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
PANELS_ROOT = TOOL / "training_data_panels"
STATE_FILE = TOOL / ".dual_capture_state.json"


def _load_state() -> dict:
    """Load saved positions / sizes / flags from disk. Returns {} on error."""
    if not STATE_FILE.is_file():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(data: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[dual_capture] state save failed: {exc}", file=sys.stderr)


# ─── Overlay rectangle ──────────────────────────────────────────────
class CaptureOverlay(QWidget):
    """Frameless always-on-top translucent window with colored border.

    Drag body to move; drag bottom-right corner (resize handle) to resize.
    """

    HANDLE = 18  # px size of resize handle

    moved_or_resized = Signal()

    def __init__(self, label: str, color: QColor):
        super().__init__()
        self._label = label
        self._color = color
        self._dragging = False
        self._resizing = False
        self._drag_start_pos: QPoint = QPoint()
        self._drag_start_geom: QRect = QRect()

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        # Do NOT pass through clicks — we need them for drag/resize.
        # Disable while invisible by hiding the widget.
        self.resize(260, 240)

    # Painting ────────────────────────────────────────────────────────
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        # Transparent fill — we just draw the border + label + handle
        pen = QPen(self._color, 3)
        p.setPen(pen)
        p.drawRect(1, 1, self.width() - 3, self.height() - 3)

        # Label in top-left
        p.fillRect(2, 2, 80, 22, self._color)
        p.setPen(QColor("white"))
        p.drawText(8, 18, self._label)

        # Resize handle in bottom-right
        h = self.HANDLE
        p.fillRect(self.width() - h, self.height() - h, h, h, self._color)
        p.setPen(QColor("white"))
        p.drawText(
            self.width() - h + 4, self.height() - 4, "↘",
        )

    # Mouse handling ──────────────────────────────────────────────────
    def _in_handle(self, pos: QPoint) -> bool:
        h = self.HANDLE
        return (
            pos.x() >= self.width() - h
            and pos.y() >= self.height() - h
        )

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        self._drag_start_pos = ev.globalPosition().toPoint()
        self._drag_start_geom = self.geometry()
        if self._in_handle(ev.position().toPoint()):
            self._resizing = True
        else:
            self._dragging = True

    def mouseMoveEvent(self, ev):
        if not (self._dragging or self._resizing):
            return
        delta = ev.globalPosition().toPoint() - self._drag_start_pos
        if self._dragging:
            self.move(self._drag_start_geom.topLeft() + delta)
        elif self._resizing:
            new_w = max(80, self._drag_start_geom.width() + delta.x())
            new_h = max(60, self._drag_start_geom.height() + delta.y())
            self.resize(new_w, new_h)

    def mouseReleaseEvent(self, _ev):
        if self._dragging or self._resizing:
            self._dragging = False
            self._resizing = False
            self.moved_or_resized.emit()

    def capture_bbox(self) -> tuple[int, int, int, int]:
        """Return the screen-space bbox (left, top, right, bottom)
        of the area INSIDE the border (excluding the border line)."""
        g = self.geometry()
        return (g.left() + 3, g.top() + 3, g.right() - 3, g.bottom() - 3)

    def capture_bbox_full(self) -> tuple[int, int, int, int]:
        """Full geometry — used when the overlay is hidden during grab
        so we don't need to inset anything."""
        g = self.geometry()
        return (g.left(), g.top(), g.right(), g.bottom())

    def set_size_only(self, w: int, h: int):
        """Resize without moving (used by spin boxes)."""
        self.resize(w, h)
        self.moved_or_resized.emit()


# ─── Hotkey listener ────────────────────────────────────────────────
class HotkeyBridge(QObject):
    """Bridges pynput's background-thread events to Qt main thread."""

    fire_r1 = Signal()
    fire_r2 = Signal()


# ─── Main control window ────────────────────────────────────────────
class ControlWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dual Capture")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setFixedWidth(380)

        self._enabled = False
        self._r1_count = 0
        self._r2_count = 0
        self._save_root: Path = PANELS_ROOT

        self.r1 = CaptureOverlay("R1  SPACE", QColor(220, 60, 60))
        self.r2 = CaptureOverlay("R2  B",     QColor(60, 130, 220))

        # Load saved state (positions, sizes, folder, flags)
        self._state = _load_state()

        # Apply overlay geometry from saved state, or fall back to
        # sensible defaults centered on the primary screen.
        screen = QApplication.primaryScreen().geometry()
        r1g = self._state.get("r1_geom")
        if r1g and len(r1g) == 4:
            self.r1.setGeometry(*r1g)
        else:
            self.r1.move(screen.width() // 2 - 280, screen.height() // 2 - 200)
        r2g = self._state.get("r2_geom")
        if r2g and len(r2g) == 4:
            self.r2.setGeometry(*r2g)
        else:
            self.r2.move(screen.width() // 2 + 20, screen.height() // 2 - 200)

        self.r1.moved_or_resized.connect(self._sync_r1_inputs)
        self.r2.moved_or_resized.connect(self._sync_r2_inputs)
        # Also persist geometry whenever an overlay is moved/resized
        self.r1.moved_or_resized.connect(self._persist_state)
        self.r2.moved_or_resized.connect(self._persist_state)

        # Hotkey bridge (signal-based, thread-safe Qt invocation)
        self._bridge = HotkeyBridge()
        self._bridge.fire_r1.connect(self._capture_r1)
        self._bridge.fire_r2.connect(self._capture_r2)

        self._listener = keyboard.Listener(on_press=self._on_key)
        self._listener.daemon = True
        self._listener.start()

        self._build_ui()
        self._refresh_save_path()
        # Restore overlay visibility from saved state
        if self._state.get("r1_visible"):
            self.r1.show()
        if self._state.get("r2_visible"):
            self.r2.show()
        # Restore control window geometry
        cwg = self._state.get("ctrl_geom")
        if cwg and len(cwg) == 4:
            self.setGeometry(*cwg)

    # UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Master enable toggle — restores from saved state
        self._enable_box = QCheckBox("ENABLED — hotkeys active")
        self._enable_box.setChecked(bool(self._state.get("enabled", False)))
        self._enabled = self._enable_box.isChecked()
        self._enable_box.toggled.connect(self._on_enable)
        self._enable_box.toggled.connect(lambda v: (
            self._state.__setitem__("enabled", bool(v)),
            self._persist_state(),
        ))
        f = self._enable_box.font()
        f.setPointSize(11)
        f.setBold(True)
        self._enable_box.setFont(f)
        layout.addWidget(self._enable_box)

        # Save folder — restore from saved state, else new timestamp
        folder_box = QGroupBox("Save folder (under training_data_panels/)")
        fb = QVBoxLayout(folder_box)
        saved_folder = (self._state.get("folder") or "").strip()
        default = saved_folder if saved_folder else datetime.now().strftime("user_%Y%m%d_%H%M%S")
        self._folder_input = QLineEdit(default)
        self._folder_input.editingFinished.connect(self._refresh_save_path)
        self._folder_input.editingFinished.connect(self._persist_state)
        fb.addWidget(self._folder_input)
        open_btn = QPushButton("Open folder")
        open_btn.clicked.connect(self._open_folder)
        fb.addWidget(open_btn)
        layout.addWidget(folder_box)

        # Region 1
        layout.addWidget(self._region_group("Region 1 (SPACE)", self.r1, 1))
        # Region 2
        layout.addWidget(self._region_group("Region 2 (B)",     self.r2, 2))

        # Counters
        self._counter_label = QLabel("R1 saved: 0    |    R2 saved: 0")
        f = self._counter_label.font()
        f.setPointSize(10)
        self._counter_label.setFont(f)
        layout.addWidget(self._counter_label)

        # Status line
        self._status = QLabel("Disabled. Toggle ENABLED to start capturing.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    def _region_group(self, title: str, overlay: CaptureOverlay, idx: int) -> QGroupBox:
        box = QGroupBox(title)
        v = QVBoxLayout(box)

        # Show/hide checkbox — restores saved state
        show = QCheckBox("Show overlay")
        saved_visible = bool(self._state.get(f"r{idx}_visible", False))
        show.setChecked(saved_visible)
        def _toggle(vis, o=overlay, i=idx):
            if vis: o.show()
            else: o.hide()
            self._state[f"r{i}_visible"] = bool(vis)
            self._persist_state()
        show.toggled.connect(_toggle)
        v.addWidget(show)

        # Width × Height
        wh_row = QHBoxLayout()
        wh_row.addWidget(QLabel("W:"))
        w_spin = QSpinBox()
        w_spin.setRange(40, 4000)
        w_spin.setValue(overlay.width())
        wh_row.addWidget(w_spin)
        wh_row.addWidget(QLabel("H:"))
        h_spin = QSpinBox()
        h_spin.setRange(40, 4000)
        h_spin.setValue(overlay.height())
        wh_row.addWidget(h_spin)
        v.addLayout(wh_row)

        def on_w(val, o=overlay, hh=h_spin):
            o.set_size_only(val, hh.value())
        def on_h(val, o=overlay, ww=w_spin):
            o.set_size_only(ww.value(), val)
        w_spin.valueChanged.connect(on_w)
        h_spin.valueChanged.connect(on_h)

        # Position label (read-only, updates when overlay moves)
        pos_label = QLabel("X: -   Y: -")
        v.addWidget(pos_label)

        # Stash references on the overlay so _sync_*_inputs can find them
        overlay._w_spin = w_spin
        overlay._h_spin = h_spin
        overlay._pos_label = pos_label

        return box

    # Sync overlay → spin boxes ───────────────────────────────────────
    def _sync_r1_inputs(self): self._sync_inputs(self.r1)
    def _sync_r2_inputs(self): self._sync_inputs(self.r2)
    def _sync_inputs(self, o: CaptureOverlay):
        o._w_spin.blockSignals(True); o._h_spin.blockSignals(True)
        o._w_spin.setValue(o.width()); o._h_spin.setValue(o.height())
        o._w_spin.blockSignals(False); o._h_spin.blockSignals(False)
        g = o.geometry()
        o._pos_label.setText(f"X: {g.left()}   Y: {g.top()}   W: {g.width()}   H: {g.height()}")

    # Save folder ─────────────────────────────────────────────────────
    def _refresh_save_path(self):
        name = self._folder_input.text().strip() or "user_capture"
        self._save_root = PANELS_ROOT / name
        (self._save_root / "region1").mkdir(parents=True, exist_ok=True)
        (self._save_root / "region2").mkdir(parents=True, exist_ok=True)

    def _open_folder(self):
        self._refresh_save_path()
        os.startfile(str(self._save_root))

    # Enable/disable ──────────────────────────────────────────────────
    def _on_enable(self, val: bool):
        self._enabled = val
        if val:
            self._status.setText(
                "ENABLED. SPACE → R1, B → R2 (works even when SC has focus)."
            )
        else:
            self._status.setText("Disabled. Toggle ENABLED to start capturing.")

    # Hotkey path ─────────────────────────────────────────────────────
    def _on_key(self, key):
        """pynput callback — runs in a background thread."""
        if not self._enabled:
            return
        try:
            if key == keyboard.Key.space:
                self._bridge.fire_r1.emit()
            elif hasattr(key, "char") and key.char and key.char.lower() == "b":
                self._bridge.fire_r2.emit()
        except Exception:
            pass

    def _capture_r1(self):
        self._capture(self.r1, 1)

    def _capture_r2(self):
        self._capture(self.r2, 2)

    def _capture(self, overlay: CaptureOverlay, idx: int):
        # Hide the overlay momentarily so its border + label + resize
        # handle don't end up in the captured pixels. Take the FULL
        # geometry as the bbox (no 3-px inset needed now), grab, then
        # re-show. The flicker is ~50 ms and much better than
        # contaminating every capture with our UI chrome.
        bbox = overlay.capture_bbox_full()
        was_visible = overlay.isVisible()
        if was_visible:
            overlay.hide()
            QApplication.processEvents()
            time.sleep(0.05)  # let Windows repaint the region
        try:
            img = ImageGrab.grab(bbox=bbox, all_screens=True)
        except Exception as exc:
            self._status.setText(f"Capture failed: {exc}")
            return
        finally:
            if was_visible:
                overlay.show()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_") + f"{int(time.time()*1000) % 1000:03d}"
        out = self._save_root / f"region{idx}" / f"cap_{ts}.png"
        try:
            img.save(out)
        except Exception as exc:
            self._status.setText(f"Save failed: {exc}")
            return
        if idx == 1:
            self._r1_count += 1
        else:
            self._r2_count += 1
        self._counter_label.setText(
            f"R1 saved: {self._r1_count}    |    R2 saved: {self._r2_count}"
        )

    # State persistence ───────────────────────────────────────────────
    def _persist_state(self):
        """Write current positions + settings to .dual_capture_state.json."""
        g1 = self.r1.geometry()
        g2 = self.r2.geometry()
        cg = self.geometry()
        self._state["r1_geom"] = [g1.x(), g1.y(), g1.width(), g1.height()]
        self._state["r2_geom"] = [g2.x(), g2.y(), g2.width(), g2.height()]
        self._state["ctrl_geom"] = [cg.x(), cg.y(), cg.width(), cg.height()]
        self._state["r1_visible"] = self.r1.isVisible()
        self._state["r2_visible"] = self.r2.isVisible()
        self._state["folder"] = self._folder_input.text().strip()
        _save_state(self._state)

    # Cleanup ─────────────────────────────────────────────────────────
    def closeEvent(self, ev):
        # Save final state before exit
        try:
            self._persist_state()
        except Exception:
            pass
        try:
            self._listener.stop()
        except Exception:
            pass
        self.r1.close()
        self.r2.close()
        ev.accept()

    def moveEvent(self, ev):
        super().moveEvent(ev)
        # Persist on move — Qt fires moveEvent when user drags the window
        if hasattr(self, "_state") and self._state is not None:
            self._persist_state()


def main():
    app = QApplication(sys.argv)
    win = ControlWindow()
    # Force to primary screen top-left — Qt's default position lands
    # off-screen on this setup (multi-monitor / virtual desktop).
    primary = app.primaryScreen().availableGeometry()
    win.move(primary.left() + 50, primary.top() + 50)
    win.show()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
