"""Fullscreen overlay for selecting the OCR capture region.

The user clicks and drags to define a rectangle on screen.  The
selected region is returned via a Qt signal.  Coordinates are
in native screen pixels (matching what ``mss`` captures).
"""

from __future__ import annotations

import ctypes

from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QCursor
from PySide6.QtWidgets import QWidget, QApplication, QMessageBox

from shared.qt.theme import P


def _get_cursor_pos() -> tuple[int, int]:
    """Get the cursor position in native screen pixels via Win32 API.

    This bypasses any Qt coordinate scaling and gives raw pixel
    coordinates that match what ``mss`` captures.
    """
    try:
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return (pt.x, pt.y)
    except Exception:
        pos = QCursor.pos()
        return (pos.x(), pos.y())


class RegionSelector(QWidget):
    """Translucent fullscreen overlay — drag to select a rectangle."""

    region_selected = Signal(dict)  # {"x": int, "y": int, "w": int, "h": int}
    cancelled = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setCursor(Qt.CrossCursor)

        # Cover the entire virtual desktop (all monitors)
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.virtualGeometry()
            self.setGeometry(geom)
        else:
            self.showFullScreen()

        # Store raw native pixel coordinates from Win32 API
        self._origin_native: tuple[int, int] | None = None
        self._current_native: tuple[int, int] | None = None

        # Visual rect in widget coordinates (for painting)
        self._rect: QRect = QRect()
        self._dragging = False

    def paintEvent(self, event) -> None:
        painter = QPainter(self)

        # Semi-transparent dark overlay
        overlay = QColor(0, 0, 0, 140)
        painter.fillRect(self.rect(), overlay)

        if not self._rect.isNull():
            # Clear the selected region (punch a hole)
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(self._rect, Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

            # Draw selection border
            accent = QColor(P.green)
            accent.setAlpha(220)
            painter.setPen(QPen(accent, 2))
            painter.setBrush(QBrush(QColor(51, 221, 136, 30)))
            painter.drawRect(self._rect)

            # Draw size label with native pixel dimensions
            if self._origin_native and self._current_native:
                nx = min(self._origin_native[0], self._current_native[0])
                ny = min(self._origin_native[1], self._current_native[1])
                nw = abs(self._current_native[0] - self._origin_native[0])
                nh = abs(self._current_native[1] - self._origin_native[1])
                label = f"{nw} x {nh}  @ ({nx}, {ny})"
            else:
                label = f"{self._rect.width()} x {self._rect.height()}"
            painter.setPen(QColor(P.fg_bright))
            painter.drawText(
                self._rect.x(), self._rect.y() - 6, label,
            )

        # Instruction text
        painter.setPen(QColor(P.fg_bright))
        painter.drawText(
            self.rect().center().x() - 180,
            40,
            "Click and drag to select the scanner region. Press ESC to cancel.",
        )

        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._origin_native = _get_cursor_pos()
            self._rect = QRect(
                event.position().toPoint(),
                event.position().toPoint(),
            )
            self._dragging = True
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._current_native = _get_cursor_pos()
            origin = self._rect.topLeft()
            # Update visual rect for painting
            self._rect = QRect(
                origin, event.position().toPoint(),
            ).normalized()
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            end_native = _get_cursor_pos()

            # Track sub-floor drops so we can surface a warning AFTER
            # closing the overlay below.  Previously these were silently
            # dropped with no user feedback — the selector just closed,
            # the user assumed their region was saved, and nothing ever
            # reached the handler.  See the comment at the size gate
            # below for why this matters specifically for the signature
            # scanner UI.
            too_small: tuple[int, int] | None = None

            if self._origin_native:
                ox, oy = self._origin_native
                ex, ey = end_native
                x = min(ox, ex)
                y = min(oy, ey)
                w = abs(ex - ox)
                h = abs(ey - oy)

                # Minimum 4x4 native pixels — lowered from 10x10.  The
                # old floor used to silently drop tight drags around
                # small UI elements (notably the signature scanner
                # value digits at 1080p / 100% scaling, where 10
                # native pixels = 10 logical pixels and users were
                # easily drawing a "tight" box that fell below the
                # threshold).  Anything smaller than 4x4 is almost
                # certainly an accidental click, not a real drag.
                if w > 4 and h > 4:
                    self.region_selected.emit({
                        "x": x, "y": y, "w": w, "h": h,
                    })
                else:
                    too_small = (w, h)
            self.close()
            event.accept()
            # Surface sub-floor drops AFTER closing the overlay so the
            # user sees the warning instead of the selector silently
            # disappearing.  Defensive try/except: if the message box
            # itself fails for any reason (rare), at least don't crash
            # the selector — the bumped save-failure logging in
            # ``_save_config`` will still surface the rejection.
            if too_small is not None:
                try:
                    QMessageBox.warning(
                        None,
                        "Region too small",
                        f"The region you drew was "
                        f"{too_small[0]}×{too_small[1]} pixels — too "
                        "small to capture reliably.\n\nPlease draw a "
                        "larger box around the area you want to scan.",
                    )
                except Exception:
                    pass

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.cancelled.emit()
            self.close()
            event.accept()
