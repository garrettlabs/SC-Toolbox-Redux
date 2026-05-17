"""Per-turret crew assignment popup for MOLE ships.

Opened from the right-click "Configure Turret Crew..." menu on a MOLE
ShipNode in the Mining Ledger canvas. Lets the user pick which crew
member sits on each of the 3 MOLE turrets.

The popup mutates ``ship_node.laser_crew`` in place. The caller's
``on_changed`` callback is invoked when the user closes the popup so
the ledger can persist the changes.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QComboBox,
)

from shared.qt.theme import P

from .theme import ACCENT

_NUM_MOLE_TURRETS = 3

_CLOSE_BTN_STYLE = """
    QPushButton {
        background: rgba(255, 60, 60, 0.15);
        color: #cc6666;
        border: none;
        border-radius: 3px;
        font-family: Consolas;
        font-size: 13pt;
        font-weight: bold;
        padding: 0px;
    }
    QPushButton:hover {
        background-color: rgba(220, 50, 50, 0.85);
        color: #ffffff;
    }
"""


def show_turret_crew_popup(ship_node, on_changed: Callable[[], None]) -> None:
    """Open the per-turret crew assignment popup for a MOLE ship node.

    Args:
        ship_node: ShipNode (must have ``crew`` and ``laser_crew`` attrs)
        on_changed: callback invoked when the popup closes (for save+refresh)
    """
    popup = QWidget(None, Qt.WindowStaysOnTopHint | Qt.Tool | Qt.FramelessWindowHint)
    popup.setAttribute(Qt.WA_DeleteOnClose)

    # Make draggable
    popup._drag_pos = None

    def _mp(event):
        if event.button() == Qt.LeftButton:
            popup._drag_pos = event.globalPosition().toPoint() - popup.frameGeometry().topLeft()

    def _mm(event):
        if popup._drag_pos and event.buttons() & Qt.LeftButton:
            popup.move(event.globalPosition().toPoint() - popup._drag_pos)

    popup.mousePressEvent = _mp
    popup.mouseMoveEvent = _mm

    popup.setFixedWidth(360)
    outer = QVBoxLayout(popup)
    outer.setContentsMargins(0, 0, 0, 0)

    frame = QFrame(popup)
    frame.setObjectName("turret_crew_frame")
    frame.setStyleSheet(
        f"QFrame#turret_crew_frame {{ background: {P.bg_card}; "
        f"border: 1px solid {ACCENT}; border-radius: 4px; }}"
    )
    fl = QVBoxLayout(frame)
    fl.setContentsMargins(12, 12, 12, 12)
    fl.setSpacing(8)

    _ns = "background: transparent; border: none;"

    # Header + close
    hdr = QWidget(frame)
    hl = QHBoxLayout(hdr)
    hl.setContentsMargins(0, 0, 0, 0)
    title = QLabel(f"Configure Turret Crew — {ship_node.ship_name}", hdr)
    title.setStyleSheet(
        f"font-family: Electrolize, Consolas; font-size: 10pt; "
        f"font-weight: bold; color: {ACCENT}; {_ns}"
    )
    hl.addWidget(title)
    hl.addStretch(1)
    close_btn = QPushButton("x", hdr)
    close_btn.setFixedSize(32, 28)
    close_btn.setCursor(Qt.PointingHandCursor)
    close_btn.setStyleSheet(_CLOSE_BTN_STYLE)
    close_btn.clicked.connect(popup.close)
    hl.addWidget(close_btn)
    fl.addWidget(hdr)

    info = QLabel(
        "Pick which crew member sits on each turret. "
        "Empty turrets won't fire during breakability calculations.",
        frame,
    )
    info.setWordWrap(True)
    info.setStyleSheet(
        f"font-family: Consolas; font-size: 8pt; color: {P.fg_dim}; {_ns}"
    )
    fl.addWidget(info)

    # Per-turret rows
    crew_options = ["(none)"] + list(ship_node.crew or [])
    combos: list[QComboBox] = []

    _combo_style = (
        f"QComboBox {{ font-family: Consolas; font-size: 9pt; color: {P.fg}; "
        f"background: {P.bg_card}; border: 1px solid {P.border}; "
        f"border-radius: 3px; padding: 3px 6px; }}"
        f"QComboBox QAbstractItemView {{ background: {P.bg_card}; "
        f"color: {P.fg}; selection-background-color: {ACCENT}; }}"
    )

    for t_idx in range(_NUM_MOLE_TURRETS):
        row = QWidget(frame)
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        lbl = QLabel(f"Turret {t_idx + 1}:", row)
        lbl.setFixedWidth(80)
        lbl.setStyleSheet(
            f"font-family: Consolas; font-size: 9pt; font-weight: bold; "
            f"color: {P.fg}; {_ns}"
        )
        rl.addWidget(lbl)

        combo = QComboBox(row)
        combo.addItems(crew_options)
        combo.setStyleSheet(_combo_style)

        # Pre-select the current assignment if any
        current = ship_node.laser_crew.get(t_idx, [])
        if current and current[0] in ship_node.crew:
            combo.setCurrentText(current[0])
        else:
            combo.setCurrentText("(none)")

        rl.addWidget(combo, 1)
        combos.append(combo)
        fl.addWidget(row)

    # Apply on close: read combos and update ship_node.laser_crew
    def _apply():
        new_map: dict[int, list[str]] = {}
        for t_idx, combo in enumerate(combos):
            val = combo.currentText().strip()
            if val and val != "(none)":
                new_map[t_idx] = [val]
        ship_node.laser_crew = new_map
        on_changed()
        popup.close()

    apply_btn = QPushButton("Apply", frame)
    apply_btn.setCursor(Qt.PointingHandCursor)
    apply_btn.setStyleSheet(
        f"QPushButton {{ font-family: Consolas; font-size: 9pt; font-weight: bold; "
        f"color: {ACCENT}; background: transparent; border: 1px solid {ACCENT}; "
        f"border-radius: 3px; padding: 5px 14px; }}"
        f"QPushButton:hover {{ background: rgba(51, 221, 136, 0.18); }}"
    )
    apply_btn.clicked.connect(_apply)
    fl.addWidget(apply_btn)

    outer.addWidget(frame)
    popup.adjustSize()

    # Center on screen
    try:
        from PySide6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.geometry()
            popup.move(
                geo.center().x() - popup.width() // 2,
                geo.center().y() - popup.height() // 2,
            )
    except Exception:
        pass

    popup.show()
