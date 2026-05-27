"""Smart region selector — pre-sized bubble overlay with mode-locked aspect.

This module replaces the legacy freehand RegionSelector with a smarter
control: the user is shown a translucent bubble already shaped like the
HUD panel (or signature scanner block) they're trying to frame.  The
bubble can be:

    * dragged around the screen by clicking inside its body,
    * resized via 8 handles (4 corners + 4 edges, aspect locked),
    * scaled by mouse wheel (also aspect locked),
    * confirmed by clicking outside the bubble or pressing Enter,
    * cancelled with ESC.

A wire-diagram is drawn inside the bubble so the user can see exactly
which on-screen elements should land in each "slot" (5 stat rows for
the mining HUD; 2 value boxes for the signature scanner).

Coordinates are returned in native screen pixels, matching what ``mss``
captures.  The legacy ``RegionSelector`` name is preserved as a thin
backward-compat shim so existing callers in ``app.py`` keep working.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from PySide6.QtCore import (
    Qt,
    QRect,
    QRectF,
    QPoint,
    QPointF,
    QSize,
    QTimer,
    Signal,
    Property,
    QPropertyAnimation,
    QEasingCurve,
)
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPen,
    QBrush,
    QCursor,
    QFont,
    QFontMetrics,
    QPainterPath,
)
from PySide6.QtWidgets import (
    QWidget,
    QApplication,
    QMessageBox,
    QGraphicsOpacityEffect,
)

from shared.qt.theme import P


# ─────────────────────────────────────────────────────────────────────
# Native cursor query (Win32 — bypasses Qt DPI scaling so coordinates
# match exactly what mss captures).
# ─────────────────────────────────────────────────────────────────────


def _get_cursor_pos() -> tuple[int, int]:
    """Get cursor position in native screen pixels via Win32 API."""
    try:
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return (pt.x, pt.y)
    except Exception:
        pos = QCursor.pos()
        return (pos.x(), pos.y())


# ─────────────────────────────────────────────────────────────────────
# Mode presets
# ─────────────────────────────────────────────────────────────────────

# Default bubble dimensions (logical px) keyed by mode.  Aspect ratios:
#   HUD       — 448 : 670  (matches the SCAN RESULTS stack)
#   signature — 4   : 1    (the two value digits side-by-side)
_MODE_PRESETS = {
    "hud": {
        "width": 448,
        "height": 670,
        "aspect": 448.0 / 670.0,
        "min_w": 220,
        "max_w": 900,
        "label": "Mining HUD — SCAN RESULTS panel",
    },
    "signature": {
        "width": 400,
        "height": 100,
        "aspect": 4.0 / 1.0,
        "min_w": 200,
        "max_w": 800,
        "label": "Signature scanner values",
    },
}


# ─────────────────────────────────────────────────────────────────────
# Resize handles
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Handle:
    """One of the 8 resize handles around the bubble."""
    name: str             # nw / n / ne / e / se / s / sw / w
    cursor: Qt.CursorShape


_HANDLES: tuple[_Handle, ...] = (
    _Handle("nw", Qt.SizeFDiagCursor),
    _Handle("n",  Qt.SizeVerCursor),
    _Handle("ne", Qt.SizeBDiagCursor),
    _Handle("e",  Qt.SizeHorCursor),
    _Handle("se", Qt.SizeFDiagCursor),
    _Handle("s",  Qt.SizeVerCursor),
    _Handle("sw", Qt.SizeBDiagCursor),
    _Handle("w",  Qt.SizeHorCursor),
)

_HANDLE_SIZE = 14  # px square


# ─────────────────────────────────────────────────────────────────────
# Main widget
# ─────────────────────────────────────────────────────────────────────


class SmartRegionSelector(QWidget):
    """Translucent fullscreen overlay with a pre-sized, draggable, resizable
    bubble.  The bubble's aspect ratio is locked to the target panel
    (mining HUD or signature scanner), and a wire-diagram is drawn inside
    so the user can see exactly where each stat / value will land.

    Emits ``region_selected({x, y, w, h})`` on confirm (Enter, or click
    outside bubble).  Emits ``cancelled()`` on ESC.

    Parameters
    ----------
    mode:
        "hud" or "signature" — controls aspect ratio, default size, and
        the wire-diagram layout drawn inside the bubble.
    initial_region:
        Optional ``{x, y, w, h}`` (native pixels) to pre-position the
        bubble (e.g. the user's previously saved region).  If omitted,
        the bubble is centred on the primary screen.
    validate_callback:
        Optional ``callable(region_dict) -> (ok: bool, msg: str)``.  If
        provided, the bubble shows a live validation indicator (green
        checkmark or red x with message) every ~200 ms while the user
        drags / resizes.  Useful for surfacing issues like "region is
        outside any monitor" before the user commits.
    parent:
        Standard QWidget parent.
    """

    region_selected = Signal(dict)  # {"x": int, "y": int, "w": int, "h": int}
    cancelled = Signal()

    def __init__(
        self,
        mode: str = "hud",
        initial_region: dict | None = None,
        validate_callback=None,
        parent=None,
        game_resolution: dict | None = None,
    ) -> None:
        super().__init__(parent)

        # ── Mode preset ──────────────────────────────────────────────
        if mode not in _MODE_PRESETS:
            mode = "hud"
        self._mode = mode
        preset = _MODE_PRESETS[mode]
        self._aspect = preset["aspect"]
        self._min_w = preset["min_w"]
        self._max_w = preset["max_w"]
        self._mode_label = preset["label"]

        # ── Resolution-aware default sizing ──────────────────────────
        # Annotated HUD panels were captured at ~448x670 from a
        # 2560x1440 native game resolution.  Scale the default
        # bubble size by the ratio of user-screen-height to the
        # 1440 reference so the wire diagram opens roughly the
        # right size on first use regardless of monitor / game
        # resolution.  Caller can override by passing game_resolution
        # explicitly (e.g. loaded from user config).
        #
        # Auto-detect: Qt primary screen height as a proxy for the
        # rendered game height -- accurate on fullscreen / borderless
        # SC, off when the user runs windowed at a different size
        # (they can override via the OCR settings).
        if game_resolution is None:
            primary = QApplication.primaryScreen()
            if primary is not None:
                geom = primary.geometry()
                game_resolution = {"w": int(geom.width()),
                                   "h": int(geom.height())}
            else:
                game_resolution = {"w": 2560, "h": 1440}  # reference fallback
        try:
            self._game_h = max(540, int(game_resolution.get("h", 1440)))
            self._game_w = max(960, int(game_resolution.get("w", 2560)))
        except (TypeError, ValueError):
            self._game_h, self._game_w = 1440, 2560
        # Annotation reference: HUD panel was 670 tall on a 1440 game.
        # On a 1080 game the HUD is ~503 tall.  On 4K it's ~1005 tall.
        _scale_factor = self._game_h / 1440.0
        self._default_w = max(self._min_w,
                              min(self._max_w,
                                  int(preset["width"] * _scale_factor)))
        self._default_h = max(int(self._min_w / self._aspect),
                              int(preset["height"] * _scale_factor))

        self._validate_callback = validate_callback
        self._validate_ok: bool | None = None
        self._validate_msg: str = ""

        # ── Window flags ─────────────────────────────────────────────
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Cover the entire virtual desktop (all monitors)
        screen = QApplication.primaryScreen()
        if screen:
            self._virtual_geom = screen.virtualGeometry()
            self.setGeometry(self._virtual_geom)
        else:
            self._virtual_geom = self.geometry()
            self.showFullScreen()

        # ── Bubble rectangle (widget coords) ─────────────────────────
        # Default to preset size, centred on primary screen.
        primary = QApplication.primaryScreen()
        if primary:
            scr = primary.geometry()
            cx = scr.x() + scr.width() // 2 - self._virtual_geom.x()
            cy = scr.y() + scr.height() // 2 - self._virtual_geom.y()
        else:
            cx = self.width() // 2
            cy = self.height() // 2

        w = self._default_w
        h = self._default_h

        if initial_region:
            # Translate native coords -> widget coords (offset by virtual
            # geometry origin, which is typically (0,0) but can be
            # negative on multi-monitor setups with a left-of-primary
            # display).
            try:
                ix = int(initial_region["x"]) - self._virtual_geom.x()
                iy = int(initial_region["y"]) - self._virtual_geom.y()
                iw = int(initial_region["w"])
                ih = int(initial_region["h"])
                if iw > 20 and ih > 20:
                    w, h = iw, ih
                    self._bubble = QRect(ix, iy, w, h)
                else:
                    self._bubble = QRect(cx - w // 2, cy - h // 2, w, h)
            except (KeyError, TypeError, ValueError):
                self._bubble = QRect(cx - w // 2, cy - h // 2, w, h)
        else:
            self._bubble = QRect(cx - w // 2, cy - h // 2, w, h)

        # ── Interaction state ────────────────────────────────────────
        self._drag_mode: str | None = None   # "move", "resize:<handle>"
        self._drag_anchor_widget: QPoint = QPoint()  # cursor at drag start
        self._drag_anchor_bubble: QRect = QRect()    # bubble at drag start
        self._hover_handle: str | None = None

        # ── Validation debounce timer ────────────────────────────────
        self._validate_timer = QTimer(self)
        self._validate_timer.setSingleShot(True)
        self._validate_timer.setInterval(200)
        self._validate_timer.timeout.connect(self._run_validation)

        # ── Floating instruction banner (top of screen) ──────────────
        self._banner_opacity = 1.0
        self._banner_effect = QGraphicsOpacityEffect(self)
        self._banner_effect.setOpacity(1.0)
        # The banner is drawn directly in paintEvent (not a child widget)
        # so we drive its opacity via a property animation on this widget.
        self._banner_anim = QPropertyAnimation(self, b"bannerOpacity", self)
        self._banner_anim.setDuration(900)
        self._banner_anim.setStartValue(1.0)
        self._banner_anim.setEndValue(0.55)
        self._banner_anim.setEasingCurve(QEasingCurve.InOutQuad)
        # Kick off the fade after a short delay so the user sees the
        # full-opacity banner first.
        QTimer.singleShot(2200, self._banner_anim.start)

        # Run initial validation so the indicator shows immediately.
        QTimer.singleShot(0, self._run_validation)

    # ── Banner opacity property (driven by QPropertyAnimation) ───────

    def _get_banner_opacity(self) -> float:
        return self._banner_opacity

    def _set_banner_opacity(self, val: float) -> None:
        self._banner_opacity = float(val)
        self.update()

    bannerOpacity = Property(float, _get_banner_opacity, _set_banner_opacity)

    # ─────────────────────────────────────────────────────────────────
    # Geometry helpers
    # ─────────────────────────────────────────────────────────────────

    def _handle_rects(self) -> dict[str, QRect]:
        """Return {handle_name: QRect} for the 8 resize handles."""
        s = _HANDLE_SIZE
        b = self._bubble
        cx = b.x() + b.width() // 2
        cy = b.y() + b.height() // 2
        return {
            "nw": QRect(b.x() - s // 2,           b.y() - s // 2,           s, s),
            "n":  QRect(cx - s // 2,              b.y() - s // 2,           s, s),
            "ne": QRect(b.right() - s // 2,       b.y() - s // 2,           s, s),
            "e":  QRect(b.right() - s // 2,       cy - s // 2,              s, s),
            "se": QRect(b.right() - s // 2,       b.bottom() - s // 2,      s, s),
            "s":  QRect(cx - s // 2,              b.bottom() - s // 2,      s, s),
            "sw": QRect(b.x() - s // 2,           b.bottom() - s // 2,      s, s),
            "w":  QRect(b.x() - s // 2,           cy - s // 2,              s, s),
        }

    def _handle_at(self, pt: QPoint) -> str | None:
        """Return the name of the handle under ``pt``, or None."""
        # Generously inflate the hit-test area so handles are easy to grab
        s = _HANDLE_SIZE + 8
        b = self._bubble
        cx = b.x() + b.width() // 2
        cy = b.y() + b.height() // 2
        rects = {
            "nw": QRect(b.x() - s // 2,     b.y() - s // 2,     s, s),
            "n":  QRect(cx - s // 2,        b.y() - s // 2,     s, s),
            "ne": QRect(b.right() - s // 2, b.y() - s // 2,     s, s),
            "e":  QRect(b.right() - s // 2, cy - s // 2,        s, s),
            "se": QRect(b.right() - s // 2, b.bottom() - s // 2, s, s),
            "s":  QRect(cx - s // 2,        b.bottom() - s // 2, s, s),
            "sw": QRect(b.x() - s // 2,     b.bottom() - s // 2, s, s),
            "w":  QRect(b.x() - s // 2,     cy - s // 2,        s, s),
        }
        for name, r in rects.items():
            if r.contains(pt):
                return name
        return None

    def _cursor_for_handle(self, name: str | None) -> Qt.CursorShape:
        if name is None:
            return Qt.ArrowCursor
        for h in _HANDLES:
            if h.name == name:
                return h.cursor
        return Qt.ArrowCursor

    def _clamp_bubble_size(self, w: int, h: int) -> tuple[int, int]:
        """Clamp width to [min_w, max_w] and recompute height from aspect."""
        w = max(self._min_w, min(self._max_w, int(w)))
        h = max(int(round(w / self._aspect)), int(round(self._min_w / self._aspect)))
        return w, h

    def _bubble_to_native(self) -> tuple[int, int, int, int]:
        """Convert the widget-coord bubble to native screen px."""
        b = self._bubble
        # The selector covers the virtual geometry, so widget(0,0) maps
        # to native (virtual_geom.x(), virtual_geom.y()).
        nx = b.x() + self._virtual_geom.x()
        ny = b.y() + self._virtual_geom.y()
        return (nx, ny, b.width(), b.height())

    def _current_region_dict(self) -> dict:
        nx, ny, nw, nh = self._bubble_to_native()
        return {"x": int(nx), "y": int(ny), "w": int(nw), "h": int(nh)}

    # ─────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────

    def _run_validation(self) -> None:
        """Invoke the user-supplied validate_callback (if any)."""
        if self._validate_callback is None:
            # Default validation — at minimum the bubble must be
            # large enough and intersect some screen.
            region = self._current_region_dict()
            ok, msg = self._default_validate(region)
        else:
            try:
                result = self._validate_callback(self._current_region_dict())
                if isinstance(result, tuple) and len(result) == 2:
                    ok, msg = bool(result[0]), str(result[1])
                else:
                    ok, msg = bool(result), ""
            except Exception as exc:
                ok, msg = False, f"validate error: {exc}"
        self._validate_ok = ok
        self._validate_msg = msg
        self.update()

    def _default_validate(self, region: dict) -> tuple[bool, str]:
        if region["w"] < 60 or region["h"] < 60:
            return False, "Region too small"
        # Must intersect some screen
        for scr in QApplication.screens():
            sg = scr.geometry()
            r = QRect(region["x"], region["y"], region["w"], region["h"])
            if sg.intersects(r):
                return True, "OK"
        return False, "Region is outside all monitors"

    def _schedule_validation(self) -> None:
        self._validate_timer.start()

    # ─────────────────────────────────────────────────────────────────
    # Painting
    # ─────────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        # 1. Dim the whole screen
        overlay = QColor(0, 0, 0, 165)
        painter.fillRect(self.rect(), overlay)

        # 2. Punch a hole where the bubble is so the user sees through
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.fillRect(self._bubble, Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # 3. Tint the bubble interior with a faint accent so it's
        #    obvious where the bubble is even on a black background.
        accent_tint = QColor(P.accent)
        accent_tint.setAlpha(18)
        painter.fillRect(self._bubble, accent_tint)

        # 4. Draw the bubble border (validation colour if available)
        if self._validate_ok is False:
            border_col = QColor(P.red)
        elif self._validate_ok is True:
            border_col = QColor(P.green)
        else:
            border_col = QColor(P.accent)
        border_col.setAlpha(230)
        pen = QPen(border_col, 2)
        pen.setStyle(Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self._bubble)

        # 5. Wire-diagram inside the bubble
        self._draw_wireframe(painter)

        # 6. Resize handles
        self._draw_handles(painter, border_col)

        # 7. Size + position label, anchored above bubble
        nx, ny, nw, nh = self._bubble_to_native()
        size_label = f"{nw} x {nh}  @ ({nx}, {ny})  — {self._mode}"
        painter.setPen(QColor(P.fg_bright))
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        fm = QFontMetrics(font)
        label_y = self._bubble.y() - 8
        if label_y < 18:
            label_y = self._bubble.bottom() + 18
        painter.drawText(self._bubble.x(), label_y, size_label)

        # 8. Validation indicator (right side of size label)
        if self._validate_ok is not None:
            ind = "OK" if self._validate_ok else "X"
            ind_col = QColor(P.green) if self._validate_ok else QColor(P.red)
            painter.setPen(ind_col)
            painter.drawText(
                self._bubble.x() + fm.horizontalAdvance(size_label) + 12,
                label_y,
                f"[{ind}] {self._validate_msg}",
            )

        # 9. Floating instruction banner
        self._draw_banner(painter)

        painter.end()

    def _draw_wireframe(self, painter: QPainter) -> None:
        """Draw the mode-specific wire-diagram inside the bubble."""
        if self._mode == "hud":
            self._draw_wire_hud(painter)
        elif self._mode == "signature":
            self._draw_wire_signature(painter)

    def _draw_wire_hud(self, painter: QPainter) -> None:
        """5-row HUD wireframe: SCAN RESULTS title + MASS / INSTABILITY /
        RESISTANCE / [icon strip] / value labels."""
        b = self._bubble
        if b.width() < 40 or b.height() < 80:
            return

        # Faint guide colour
        guide = QColor(P.fg_dim)
        guide.setAlpha(180)
        painter.setPen(QPen(guide, 1, Qt.DashLine))

        # Title band (top ~12%)
        title_h = max(18, int(b.height() * 0.12))
        title_rect = QRect(b.x() + 8, b.y() + 6, b.width() - 16, title_h)
        painter.drawRect(title_rect)
        painter.setPen(QColor(P.fg_dim))
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            title_rect.adjusted(6, 0, 0, 0),
            int(Qt.AlignVCenter | Qt.AlignLeft),
            "SCAN RESULTS",
        )

        # 5 rows below the title
        row_labels = ("MASS", "INSTABILITY", "RESISTANCE", "TYPE", "RESOURCE")
        rows_top = title_rect.bottom() + 8
        rows_h = b.bottom() - rows_top - 8
        if rows_h < 40:
            return
        row_h = rows_h // 5

        painter.setPen(QPen(guide, 1, Qt.DashLine))
        for i, label in enumerate(row_labels):
            ry = rows_top + i * row_h
            row_rect = QRect(b.x() + 8, ry, b.width() - 16, row_h - 4)
            painter.drawRect(row_rect)

            # Label box (left third)
            painter.setPen(QColor(P.fg_dim))
            font2 = QFont()
            font2.setPointSize(8)
            painter.setFont(font2)
            painter.drawText(
                row_rect.adjusted(8, 0, 0, 0),
                int(Qt.AlignVCenter | Qt.AlignLeft),
                label,
            )

            # Value placeholder (right two thirds)
            value_x = row_rect.x() + int(row_rect.width() * 0.45)
            value_rect = QRect(
                value_x, row_rect.y() + 4,
                row_rect.width() - (value_x - row_rect.x()) - 8,
                row_rect.height() - 8,
            )
            value_col = QColor(P.accent)
            value_col.setAlpha(60)
            painter.setPen(QPen(value_col, 1, Qt.DashLine))
            painter.drawRect(value_rect)
            painter.setPen(QColor(P.fg_dim))
            painter.drawText(
                value_rect, int(Qt.AlignCenter), "0.00",
            )
            # Restore guide pen for next outer row rect
            painter.setPen(QPen(guide, 1, Qt.DashLine))

    def _draw_wire_signature(self, painter: QPainter) -> None:
        """2-box signature scanner wireframe (side by side value digits)."""
        b = self._bubble
        if b.width() < 60 or b.height() < 30:
            return

        guide = QColor(P.fg_dim)
        guide.setAlpha(180)
        painter.setPen(QPen(guide, 1, Qt.DashLine))

        pad = 6
        gap = 10
        box_w = (b.width() - 2 * pad - gap) // 2
        box_h = b.height() - 2 * pad
        box_y = b.y() + pad

        left = QRect(b.x() + pad,                       box_y, box_w, box_h)
        right = QRect(b.x() + pad + box_w + gap,        box_y, box_w, box_h)

        painter.drawRect(left)
        painter.drawRect(right)

        # Placeholder digits
        font = QFont()
        font.setPointSize(max(14, box_h // 3))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(P.fg_dim))
        painter.drawText(left,  int(Qt.AlignCenter), "00")
        painter.drawText(right, int(Qt.AlignCenter), "00")

        # Sub-label
        font2 = QFont()
        font2.setPointSize(8)
        painter.setFont(font2)
        painter.drawText(
            QRect(b.x(), b.bottom() + 2, b.width(), 16),
            int(Qt.AlignCenter),
            "value digit boxes",
        )

    def _draw_handles(self, painter: QPainter, col: QColor) -> None:
        """Draw the 8 resize handles as filled squares."""
        painter.setPen(QPen(QColor(0, 0, 0, 200), 1))
        for name, r in self._handle_rects().items():
            fill = QColor(col)
            fill.setAlpha(255 if name == self._hover_handle else 210)
            painter.setBrush(QBrush(fill))
            painter.drawRect(r)
        painter.setBrush(Qt.NoBrush)

    def _draw_banner(self, painter: QPainter) -> None:
        """Floating instruction banner with animated opacity."""
        msg_lines = [
            f"Position the bubble over the {self._mode_label}.",
            "Drag to move  |  Drag handles or scroll to resize  |  "
            "Enter or click outside to confirm  |  ESC to cancel",
        ]

        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        fm = QFontMetrics(font)

        # Compute banner size
        max_w = max(fm.horizontalAdvance(line) for line in msg_lines)
        line_h = fm.height() + 2
        banner_w = max_w + 36
        banner_h = line_h * len(msg_lines) + 16

        # Center horizontally at top of virtual desktop
        bx = (self.width() - banner_w) // 2
        by = 28

        # Background pill
        bg = QColor(P.bg_card)
        bg.setAlphaF(0.92 * self._banner_opacity)
        border = QColor(P.accent)
        border.setAlphaF(0.7 * self._banner_opacity)

        path = QPainterPath()
        path.addRoundedRect(QRectF(bx, by, banner_w, banner_h), 10, 10)
        painter.fillPath(path, bg)
        painter.setPen(QPen(border, 1.5))
        painter.drawPath(path)

        # Text
        text_col = QColor(P.fg_bright)
        text_col.setAlphaF(self._banner_opacity)
        painter.setPen(text_col)
        ty = by + fm.ascent() + 6
        for line in msg_lines:
            painter.drawText(bx + 18, ty, line)
            ty += line_h

    # ─────────────────────────────────────────────────────────────────
    # Mouse interaction
    # ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        pt = event.position().toPoint()

        # Check handles first (they extend slightly outside the bubble)
        handle = self._handle_at(pt)
        if handle is not None:
            self._drag_mode = f"resize:{handle}"
            self._drag_anchor_widget = pt
            self._drag_anchor_bubble = QRect(self._bubble)
            event.accept()
            return

        # Click inside the bubble body → drag-move
        if self._bubble.contains(pt):
            self._drag_mode = "move"
            self._drag_anchor_widget = pt
            self._drag_anchor_bubble = QRect(self._bubble)
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        # Click outside → confirm current region
        self._confirm_and_close()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        pt = event.position().toPoint()

        # Live cursor hint when not dragging
        if self._drag_mode is None:
            handle = self._handle_at(pt)
            if handle != self._hover_handle:
                self._hover_handle = handle
                self.update()
            if handle is not None:
                self.setCursor(self._cursor_for_handle(handle))
            elif self._bubble.contains(pt):
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return

        if self._drag_mode == "move":
            dx = pt.x() - self._drag_anchor_widget.x()
            dy = pt.y() - self._drag_anchor_widget.y()
            new_x = self._drag_anchor_bubble.x() + dx
            new_y = self._drag_anchor_bubble.y() + dy
            self._bubble = QRect(
                new_x, new_y,
                self._drag_anchor_bubble.width(),
                self._drag_anchor_bubble.height(),
            )
            self._schedule_validation()
            self.update()
            event.accept()
            return

        if self._drag_mode.startswith("resize:"):
            handle = self._drag_mode.split(":", 1)[1]
            self._apply_resize(handle, pt)
            self._schedule_validation()
            self.update()
            event.accept()
            return

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        if self._drag_mode is not None:
            self._drag_mode = None
            self.setCursor(Qt.ArrowCursor)
            self._schedule_validation()
            event.accept()

    def _apply_resize(self, handle: str, pt: QPoint) -> None:
        """Apply aspect-locked resize while dragging ``handle`` to ``pt``."""
        b = QRect(self._drag_anchor_bubble)

        # Compute new bounds based on which handle
        left, top, right, bottom = b.left(), b.top(), b.right(), b.bottom()
        if "w" in handle:
            left = pt.x()
        if "e" in handle:
            right = pt.x()
        if "n" in handle:
            top = pt.y()
        if "s" in handle:
            bottom = pt.y()

        new_w = max(self._min_w, right - left + 1)
        new_h = max(int(self._min_w / self._aspect), bottom - top + 1)

        # Aspect-lock: derive the dimension we're NOT directly editing
        # from the one we are.
        if handle in ("e", "w"):
            new_w = min(self._max_w, max(self._min_w, new_w))
            new_h = int(round(new_w / self._aspect))
        elif handle in ("n", "s"):
            target_w = int(round(new_h * self._aspect))
            target_w = min(self._max_w, max(self._min_w, target_w))
            new_w = target_w
            new_h = int(round(new_w / self._aspect))
        else:
            # Corner — let width drive
            new_w = min(self._max_w, max(self._min_w, new_w))
            new_h = int(round(new_w / self._aspect))

        # Reposition: anchor depends on which handle
        anchor = self._drag_anchor_bubble
        if handle == "se":
            x, y = anchor.x(), anchor.y()
        elif handle == "ne":
            x = anchor.x()
            y = anchor.bottom() - new_h + 1
        elif handle == "sw":
            x = anchor.right() - new_w + 1
            y = anchor.y()
        elif handle == "nw":
            x = anchor.right() - new_w + 1
            y = anchor.bottom() - new_h + 1
        elif handle == "e":
            x, y = anchor.x(), anchor.y() + (anchor.height() - new_h) // 2
        elif handle == "w":
            x = anchor.right() - new_w + 1
            y = anchor.y() + (anchor.height() - new_h) // 2
        elif handle == "n":
            x = anchor.x() + (anchor.width() - new_w) // 2
            y = anchor.bottom() - new_h + 1
        elif handle == "s":
            x = anchor.x() + (anchor.width() - new_w) // 2
            y = anchor.y()
        else:
            x, y = anchor.x(), anchor.y()

        self._bubble = QRect(x, y, new_w, new_h)

    # ─────────────────────────────────────────────────────────────────
    # Wheel — scale preserving aspect ratio
    # ─────────────────────────────────────────────────────────────────

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        # 1 wheel notch = ±5% width
        factor = 1.0 + (0.05 if delta > 0 else -0.05)
        old = self._bubble
        new_w = int(round(old.width() * factor))
        new_w, new_h = self._clamp_bubble_size(new_w, int(round(old.height() * factor)))

        # Scale around cursor position so the point under the cursor
        # stays under the cursor.
        cursor_pt = event.position().toPoint()
        if old.contains(cursor_pt):
            # Fraction across the bubble
            fx = (cursor_pt.x() - old.x()) / max(1, old.width())
            fy = (cursor_pt.y() - old.y()) / max(1, old.height())
            new_x = cursor_pt.x() - int(round(fx * new_w))
            new_y = cursor_pt.y() - int(round(fy * new_h))
        else:
            # Scale around bubble centre
            cx = old.x() + old.width() // 2
            cy = old.y() + old.height() // 2
            new_x = cx - new_w // 2
            new_y = cy - new_h // 2

        self._bubble = QRect(new_x, new_y, new_w, new_h)
        self._schedule_validation()
        self.update()
        event.accept()

    # ─────────────────────────────────────────────────────────────────
    # Keyboard
    # ─────────────────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.cancelled.emit()
            self.close()
            event.accept()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._confirm_and_close()
            event.accept()
            return

        # Arrow-key nudge (1 px, or 10 with Shift)
        step = 10 if event.modifiers() & Qt.ShiftModifier else 1
        dx = dy = 0
        if key == Qt.Key_Left:
            dx = -step
        elif key == Qt.Key_Right:
            dx = step
        elif key == Qt.Key_Up:
            dy = -step
        elif key == Qt.Key_Down:
            dy = step

        if dx or dy:
            self._bubble.translate(dx, dy)
            self._schedule_validation()
            self.update()
            event.accept()
            return

        super().keyPressEvent(event)

    # ─────────────────────────────────────────────────────────────────
    # Confirm / close
    # ─────────────────────────────────────────────────────────────────

    def _confirm_and_close(self) -> None:
        region = self._current_region_dict()
        # Final guard: refuse to emit if region is implausible
        if region["w"] < 20 or region["h"] < 20:
            try:
                QMessageBox.warning(
                    None,
                    "Region too small",
                    f"The region is only {region['w']}x{region['h']} px — "
                    "too small to capture reliably.\n\n"
                    "Resize the bubble and try again.",
                )
            except Exception:
                pass
            return
        self.region_selected.emit(region)
        self.close()


# ─────────────────────────────────────────────────────────────────────
# Backward-compatibility shim
# ─────────────────────────────────────────────────────────────────────


class RegionSelector(SmartRegionSelector):
    """Backward-compatible alias for the legacy freehand RegionSelector.

    Existing callers in ``ui/app.py`` do ``RegionSelector(self)`` with a
    single parent arg.  Map that to the new SmartRegionSelector with
    default mode="hud".  Callers that pass keyword args (``mode=``,
    ``initial_region=``, ``validate_callback=``) get the full smart
    behaviour for free.
    """

    def __init__(self, parent=None, **kwargs) -> None:
        # Allow legacy positional-only ``RegionSelector(parent)``
        if "parent" not in kwargs:
            kwargs["parent"] = parent
        super().__init__(**kwargs)


# ─────────────────────────────────────────────────────────────────────
# Standalone test harness
# ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    # Pick mode from argv: python region_selector.py [hud|signature]
    mode = "hud"
    if len(sys.argv) > 1 and sys.argv[1] in _MODE_PRESETS:
        mode = sys.argv[1]

    def _on_region(region: dict) -> None:
        print(f"[region_selected] {region}")
        app.quit()

    def _on_cancel() -> None:
        print("[cancelled]")
        app.quit()

    def _demo_validate(region: dict) -> tuple[bool, str]:
        if region["w"] < 100:
            return False, "too narrow"
        if region["h"] < 50:
            return False, "too short"
        return True, "looks good"

    sel = SmartRegionSelector(mode=mode, validate_callback=_demo_validate)
    sel.region_selected.connect(_on_region)
    sel.cancelled.connect(_on_cancel)
    sel.show()

    sys.exit(app.exec())
