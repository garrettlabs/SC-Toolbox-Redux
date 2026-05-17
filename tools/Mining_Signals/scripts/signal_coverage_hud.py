"""Per-region training-coverage HUD.

Live dashboard showing how many labeled samples you've captured for
each scan region defined in ``ocr.training_registry``. Defaults to
the ``signal`` region (signature scanner) but the dropdown lets you
flip between any registered kind without restarting.

Strict source isolation: every directory the HUD scans is pulled
from ``training_registry.get_training_sources(kind)`` — no glob is
hard-coded here, so a region can't accidentally read from another
region's corpus. Cross-source pollution is impossible by construction.

Updates every 2 seconds while open, so you can keep capturing in the
background and watch the bars fill in. Color coding:

    red    (below floor)    = model will fail this digit
    yellow (floor)          = marginal
    green  (working)        = usable training set
    bright green (solid)    = matches 3-engine vote quality

Per-region thresholds are defined on the RegionSpec.

Run with:
    python scripts/signal_coverage_hud.py
or double-click LAUNCH_SignalCoverageHUD.bat in training_data_panels/.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget, QComboBox,
)


THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent

# Registry import. The HUD is a thin viewer — every "what counts as
# this region's data" decision lives in training_registry.
sys.path.insert(0, str(TOOL))
from ocr import training_registry  # noqa: E402

POLL_INTERVAL_MS = 2000
BAR_MAX = 200      # progress bar maxes out here for visual consistency

# Theme colors lifted from the Mining Signals palette so this matches
# the rest of the toolbox visually.
ACCENT = "#33dd88"
RED = "#ff4444"
YELLOW = "#ffc107"
DIM = "#888888"
BG = "#1e1e1e"
FG = "#e0e0e0"


# ─────────────────────────────────────────────────────────────
# Capture folder discovery
# ─────────────────────────────────────────────────────────────

def _list_capture_dirs(kind: str) -> list[Path]:
    """Return every directory the registry maps to ``kind``, sorted
    by parent directory name (newest first when names are date-stamped).
    """
    dirs = training_registry.get_training_sources(kind)
    dirs.sort(key=lambda p: p.parent.name, reverse=True)
    return dirs


def _read_label(json_path: Path, label_field: str) -> str:
    """Extract the ground-truth label string from a JSON sidecar.
    Returns '' on any failure or empty label."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return (d.get(label_field) or "").strip()
    except Exception:
        return ""


def _scan_dir(
    region_dir: Path,
    spec: training_registry.RegionSpec,
) -> tuple[Counter, int, int, int]:
    """Tally per-character counts across every labeled JSON in
    ``region_dir`` for the given region spec. Returns
    (counter, total_pngs, labeled_jsons, unlabeled_pngs).

    A capture is "labeled" when its sidecar JSON has a non-empty
    label_field. A PNG without a JSON, or with an empty label, is
    "unlabeled" and contributes nothing to the digit counts."""
    counter: Counter = Counter()
    if not region_dir.is_dir():
        return counter, 0, 0, 0
    # Tripwire: refuse to read from a directory that doesn't belong
    # to this region kind.
    try:
        training_registry.assert_path_belongs_to(spec.kind, region_dir)
    except training_registry.RegistryError:
        return counter, 0, 0, 0
    label_chars = set(spec.label_set)
    # Count PNG captures (raw output of dual_capture). Each one MAY
    # have a paired JSON with a typed value; we tally the label
    # characters only when the JSON exists and has a non-empty value.
    png_paths = list(region_dir.glob(spec.capture_image_glob))
    total = len(png_paths)
    labeled = 0
    unlabeled = 0
    for png in png_paths:
        json_path = png.with_suffix(".json")
        if not json_path.is_file():
            unlabeled += 1
            continue
        v = _read_label(json_path, spec.label_field)
        if not v:
            unlabeled += 1
            continue
        labeled += 1
        for ch in v:
            if ch in label_chars:
                counter[ch] += 1
    return counter, total, labeled, unlabeled


# ─────────────────────────────────────────────────────────────
# UI widgets
# ─────────────────────────────────────────────────────────────

def _tier_color(count: int, spec: training_registry.RegionSpec) -> str:
    if count < spec.floor_per_class:
        return RED
    if count < spec.working_per_class:
        return YELLOW
    if count < spec.solid_per_class:
        return ACCENT
    return "#7fffa0"  # bright green


def _tier_label(count: int, spec: training_registry.RegionSpec) -> str:
    if count < spec.floor_per_class:
        return "below floor"
    if count < spec.working_per_class:
        return "marginal"
    if count < spec.solid_per_class:
        return "working"
    return "solid"


class GlyphRow(QWidget):
    """One per-glyph row: glyph + count + progress bar + tier."""

    def __init__(self, glyph: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        self._glyph = QLabel(glyph, self)
        gf = QFont("Consolas")
        gf.setPointSize(20)
        gf.setBold(True)
        self._glyph.setFont(gf)
        self._glyph.setFixedWidth(36)
        self._glyph.setAlignment(Qt.AlignCenter)
        self._glyph.setStyleSheet(f"color: {FG}; background: transparent;")
        layout.addWidget(self._glyph)

        self._count = QLabel("0", self)
        cf = QFont("Consolas")
        cf.setPointSize(14)
        cf.setBold(True)
        self._count.setFont(cf)
        self._count.setFixedWidth(60)
        self._count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self._count)

        self._bar = QProgressBar(self)
        self._bar.setRange(0, BAR_MAX)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(20)
        self._bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self._bar, 1)

        self._tier = QLabel("below floor", self)
        tf = QFont("Consolas")
        tf.setPointSize(9)
        self._tier.setFont(tf)
        self._tier.setFixedWidth(110)
        self._tier.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self._tier)

    def set_count(self, n: int, spec: training_registry.RegionSpec) -> None:
        color = _tier_color(n, spec)
        self._count.setText(str(n))
        self._count.setStyleSheet(
            f"color: {color}; background: transparent;",
        )
        # Cap the bar visually at BAR_MAX so 9999 doesn't make the
        # tiny digits look identical when normalized.
        self._bar.setValue(min(n, BAR_MAX))
        self._bar.setStyleSheet(
            f"""
            QProgressBar {{
                background: #2a2a2a; border: 1px solid #333;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 2px;
            }}
            """
        )
        self._tier.setText(_tier_label(n, spec))
        self._tier.setStyleSheet(
            f"color: {color}; background: transparent;",
        )


class CoverageHUD(QWidget):
    def __init__(self, initial_kind: str = "signal"):
        super().__init__()
        self.setWindowTitle("Training Coverage HUD")
        self.setMinimumSize(680, 540)
        self.setStyleSheet(f"background: {BG}; color: {FG};")

        # Active region kind (drives every other widget). Set after we
        # build the row pool so set_kind() can populate it.
        self._kind: str = initial_kind

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # ── Header ──
        self._title = QLabel("TRAINING COVERAGE", self)
        tf = QFont("Consolas")
        tf.setPointSize(13)
        tf.setBold(True)
        self._title.setFont(tf)
        self._title.setStyleSheet(f"color: {ACCENT}; background: transparent;")
        root.addWidget(self._title)

        self._description = QLabel("", self)
        self._description.setStyleSheet(
            f"color: {DIM}; background: transparent; font-size: 9pt;"
        )
        self._description.setWordWrap(True)
        root.addWidget(self._description)

        # ── Region kind picker ──
        kind_row = QHBoxLayout()
        kind_row.addWidget(QLabel("Region kind:", self))
        self._kind_combo = QComboBox(self)
        self._kind_combo.setStyleSheet(
            f"background: #2a2a2a; color: {FG}; padding: 3px 6px; border: 1px solid #333;"
        )
        for k in training_registry.list_kinds():
            self._kind_combo.addItem(k)
        self._kind_combo.currentTextChanged.connect(self._on_kind_changed)
        kind_row.addWidget(self._kind_combo, 1)
        root.addLayout(kind_row)

        # ── Capture-folder picker ──
        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Capture session:", self))
        self._folder_combo = QComboBox(self)
        self._folder_combo.setStyleSheet(
            f"background: #2a2a2a; color: {FG}; padding: 3px 6px; border: 1px solid #333;"
        )
        self._folder_combo.currentIndexChanged.connect(self._on_folder_changed)
        picker_row.addWidget(self._folder_combo, 1)
        root.addLayout(picker_row)

        # Separator
        sep = QFrame(self)
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {DIM}; background: {DIM};")
        root.addWidget(sep)

        # ── Glyph rows container (rebuilt per kind) ──
        self._rows_container = QWidget(self)
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        self._rows: dict[str, GlyphRow] = {}
        root.addWidget(self._rows_container)

        sep2 = QFrame(self)
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"color: {DIM}; background: {DIM};")
        root.addWidget(sep2)

        # ── Footer: totals + threshold legend + buttons ──
        self._totals_lbl = QLabel("—", self)
        self._totals_lbl.setStyleSheet(
            f"color: {FG}; background: transparent; font-family: Consolas; font-size: 10pt;"
        )
        root.addWidget(self._totals_lbl)

        self._thresholds_lbl = QLabel("", self)
        self._thresholds_lbl.setStyleSheet(
            f"color: {DIM}; background: transparent; font-family: Consolas; font-size: 9pt;"
        )
        root.addWidget(self._thresholds_lbl)

        button_row = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh now", self)
        self._refresh_btn.setStyleSheet(
            f"background: {ACCENT}; color: black; padding: 6px 16px; "
            f"border: none; border-radius: 3px; font-weight: bold;"
        )
        self._refresh_btn.clicked.connect(self._refresh)
        button_row.addWidget(self._refresh_btn)

        open_btn = QPushButton("Open capture folder", self)
        open_btn.setStyleSheet(
            f"background: #444; color: {FG}; padding: 6px 16px; "
            f"border: 1px solid #555; border-radius: 3px;"
        )
        open_btn.clicked.connect(self._open_folder)
        button_row.addWidget(open_btn)

        button_row.addStretch(1)

        self._mtime_lbl = QLabel("Last refresh: —", self)
        self._mtime_lbl.setStyleSheet(
            f"color: {DIM}; background: transparent; font-family: Consolas; font-size: 9pt;"
        )
        button_row.addWidget(self._mtime_lbl)

        root.addLayout(button_row)

        # ── Initial population ──
        # Set the dropdown without re-firing the change handler before
        # the row pool exists.
        idx = self._kind_combo.findText(self._kind)
        if idx >= 0:
            self._kind_combo.blockSignals(True)
            self._kind_combo.setCurrentIndex(idx)
            self._kind_combo.blockSignals(False)
        self._rebuild_rows_for_kind()
        self._populate_folders()

        # Auto-refresh timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(POLL_INTERVAL_MS)

    # ── Kind handling ──

    def _spec(self) -> training_registry.RegionSpec:
        return training_registry.get(self._kind)

    def _on_kind_changed(self, new_kind: str) -> None:
        if not new_kind or new_kind == self._kind:
            return
        self._kind = new_kind
        self._rebuild_rows_for_kind()
        self._populate_folders()

    def _rebuild_rows_for_kind(self) -> None:
        """Tear down old rows, rebuild for whichever kind is active.
        Glyph set comes from the spec's label_set so HUD shows '.' / '%'
        alongside digits while signal shows just digits + comma."""
        for row in self._rows.values():
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

        spec = self._spec()
        self._title.setText(f"{spec.kind.upper()} TRAINING COVERAGE")
        # Append a how-it-updates note so users don't expect counts to
        # tick up when they snap captures — the counts come from typed
        # labels, not from PNGs.
        self._description.setText(
            spec.description
            + "\n\n"
            "Counts auto-refresh every 2s. They reflect TYPED LABELS "
            "(JSON sidecar files), NOT raw captures. Run the labeler "
            "after taking pictures to make new captures count."
        )
        self._thresholds_lbl.setText(
            f"Floor: {spec.floor_per_class}   |   "
            f"Working: {spec.working_per_class}   |   "
            f"Solid: {spec.solid_per_class}   |   "
            f"Sources: {len(spec.expand_sources())} dir(s) registered"
        )

        for ch in spec.label_set:
            row = GlyphRow(ch, self)
            self._rows[ch] = row
            self._rows_layout.addWidget(row)

    def _populate_folders(self) -> None:
        dirs = _list_capture_dirs(self._kind)
        self._folder_combo.blockSignals(True)
        self._folder_combo.clear()
        if not dirs:
            self._folder_combo.addItem("(no capture sessions found)", None)
        else:
            # "All sessions combined" pseudo-entry first so new users
            # see the totals without picking a session.
            self._folder_combo.addItem("(all sessions — combined)", "__ALL__")
            for d in dirs:
                # Show "<parent>/<basename>" so HUD's training_data_user_panel
                # (no session parent) and signal's "user_*/region2" (with one)
                # both render legibly.
                label = (
                    f"{d.parent.name}/{d.name}"
                    if d.parent != d.parent.parent else d.name
                )
                self._folder_combo.addItem(label, str(d))
        self._folder_combo.blockSignals(False)
        self._refresh()

    def _current_dirs(self) -> list[Path]:
        """Return the list of dirs the current selection covers — one
        directory in the normal case, or all of them when "(all
        sessions — combined)" is picked."""
        sel = self._folder_combo.currentData()
        if not sel:
            return []
        if sel == "__ALL__":
            return _list_capture_dirs(self._kind)
        return [Path(sel)]

    def _on_folder_changed(self, _idx: int) -> None:
        self._refresh()

    def _refresh(self) -> None:
        spec = self._spec()
        dirs = self._current_dirs()
        if not dirs:
            for ch in self._rows:
                self._rows[ch].set_count(0, spec)
            self._totals_lbl.setText("No capture session selected.")
            return

        # Re-scan folder list periodically so brand-new sessions show up
        # without restarting the HUD. Cheap (just iterdir).
        existing_paths = {
            self._folder_combo.itemData(i)
            for i in range(self._folder_combo.count())
            if self._folder_combo.itemData(i) not in (None, "__ALL__")
        }
        live_paths = {str(p) for p in _list_capture_dirs(self._kind)}
        if existing_paths != live_paths and live_paths:
            preserved = self._folder_combo.currentData()
            self._populate_folders()
            for i in range(self._folder_combo.count()):
                if self._folder_combo.itemData(i) == preserved:
                    self._folder_combo.setCurrentIndex(i)
                    break

        # Aggregate counts across every selected dir.
        counter: Counter = Counter()
        total = 0
        labeled = 0
        unlabeled = 0
        for d in dirs:
            c, t, l, u = _scan_dir(d, spec)
            counter += c
            total += t
            labeled += l
            unlabeled += u

        for ch in spec.label_set:
            self._rows[ch].set_count(counter.get(ch, 0), spec)

        # Headline totals
        gross = sum(counter.values())
        per_class_counts = {ch: counter.get(ch, 0) for ch in spec.label_set}
        weakest = min(per_class_counts.values()) if per_class_counts else 0
        weakest_chars = [ch for ch, n in per_class_counts.items() if n == weakest]
        ready_for_train = (
            "✓ READY TO TRAIN"
            if weakest >= spec.working_per_class
            else "⚠ NEED MORE DATA"
        )
        unlabeled_warn = (
            f"  |  ⚠ {unlabeled} unlabeled (run labeler)" if unlabeled else ""
        )
        self._totals_lbl.setText(
            f"{labeled}/{total} captures labeled{unlabeled_warn}  |  "
            f"{gross} glyph instances  |  "
            f"weakest class: {','.join(weakest_chars)} ({weakest})  |  "
            f"{ready_for_train}"
        )
        self._mtime_lbl.setText(
            f"Last refresh: {datetime.now().strftime('%H:%M:%S')}  |  "
            f"Model target: {spec.model_path.name}"
        )
        # Brief visual flash on the refresh button so the user can
        # tell it actually fired (auto or manual). The QSS animation
        # uses a 1-shot timer to revert.
        if hasattr(self, "_refresh_btn"):
            self._refresh_btn.setStyleSheet(
                f"background: #ffd60a; color: black; padding: 6px 16px; "
                f"border: none; border-radius: 3px; font-weight: bold;"
            )
            from PySide6.QtCore import QTimer as _QT
            _QT.singleShot(180, self._restore_refresh_btn_style)

    def _restore_refresh_btn_style(self) -> None:
        if hasattr(self, "_refresh_btn"):
            self._refresh_btn.setStyleSheet(
                f"background: {ACCENT}; color: black; padding: 6px 16px; "
                f"border: none; border-radius: 3px; font-weight: bold;"
            )

    def _open_folder(self) -> None:
        dirs = self._current_dirs()
        if not dirs:
            return
        try:
            os.startfile(str(dirs[0]))  # Windows
        except AttributeError:
            # macOS / Linux — best-effort
            import subprocess
            subprocess.Popen(["xdg-open", str(dirs[0])])
        except Exception:
            pass


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    # Dark palette for the window chrome (combobox dropdown etc.)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG))
    palette.setColor(QPalette.WindowText, QColor(FG))
    palette.setColor(QPalette.Base, QColor("#2a2a2a"))
    palette.setColor(QPalette.Text, QColor(FG))
    palette.setColor(QPalette.Button, QColor("#444"))
    palette.setColor(QPalette.ButtonText, QColor(FG))
    app.setPalette(palette)

    # Allow `python signal_coverage_hud.py hud` to open the HUD-tab.
    initial_kind = "signal"
    if len(sys.argv) > 1 and sys.argv[1] in training_registry.list_kinds():
        initial_kind = sys.argv[1]

    hud = CoverageHUD(initial_kind=initial_kind)
    hud.show()
    hud.raise_()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
