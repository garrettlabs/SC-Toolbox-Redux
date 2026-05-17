"""Glyph review GUI — clean mis-extracted training samples per class.

Walks the active region's staging dir (per training_registry) and
shows a clickable grid of every PNG. Click a glyph to mark it for
removal (red border). Hit 'Move to quarantine' to relocate marked
glyphs to ``<staging_dir>/_quarantine/<class>/`` (recoverable, not
permanent delete).

A region tab at the top lets you flip between scan kinds:
    Signal   → training_data_user_sig/   (digits + comma)
    HUD      → training_data_user_panel/ (digits + . + %)

Class folders + label-set come from training_registry, so adding a
new region kind makes it appear here automatically with no code
changes here.

Goal: cleaning out wrong-class PNGs (e.g. UI icons or 'R' chars
sitting in the 0/ folder) lifts model accuracy way past the noise
ceiling.

Run with:  python scripts/review_glyphs.py
       or: python scripts/review_glyphs.py signal     # open on signal tab
       or: python scripts/review_glyphs.py hud        # open on hud tab
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QSize, QEvent, QPoint, QRect
from PySide6.QtGui import QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QRubberBand, QScrollArea, QSplitter,
    QVBoxLayout, QWidget, QFrame,
)


THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
sys.path.insert(0, str(TOOL))
from ocr import training_registry  # noqa: E402

# Folder-name encoding for non-alphanumeric class labels (filesystem
# safety + matches what extract_labeled_glyphs.py + train_for_region.py
# produce on disk).
_CHAR_TO_DIRNAME = {".": "dot", "%": "pct", ",": "comma", "@": "icon"}
_DIRNAME_TO_CHAR = {v: k for k, v in _CHAR_TO_DIRNAME.items()}


def _char_to_dirname(ch: str) -> str:
    return _CHAR_TO_DIRNAME.get(ch, ch)


def _dirname_to_char(name: str) -> str:
    return _DIRNAME_TO_CHAR.get(name, name)

# Glyph display size — scaled up from 28x28 for clickable visibility
TILE_SIZE = 64
COLS = 14


class GlyphTile(QLabel):
    """Single clickable glyph thumbnail."""

    def __init__(self, path: Path, on_toggle):
        super().__init__()
        self._path = path
        self._on_toggle = on_toggle
        self._marked = False

        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (28, 28), (255, 0, 255))

        # Upscale 2x with NEAREST so pixels are visible
        pil = pil.resize((TILE_SIZE, TILE_SIZE), Image.NEAREST)
        self._pix = QPixmap.fromImage(ImageQt(pil))
        self.setPixmap(self._pix)
        self.setFixedSize(TILE_SIZE + 6, TILE_SIZE + 6)
        self.setAlignment(Qt.AlignCenter)
        self._update_style()

    def _update_style(self):
        if self._marked:
            self.setStyleSheet(
                "border: 3px solid #d22; background: #fee; padding: 0;"
            )
        else:
            self.setStyleSheet(
                "border: 1px solid #ccc; background: #fff; padding: 2px;"
            )

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.set_marked(not self._marked)

    def set_marked(self, marked: bool) -> None:
        """Programmatic mark toggle — used by Mark ALL / Unmark all.
        No-ops if state isn't actually changing so we don't spam
        the on_toggle callback."""
        if bool(marked) == self._marked:
            return
        self._marked = bool(marked)
        self._update_style()
        self._on_toggle(self._path, self._marked)

    @property
    def marked(self) -> bool:
        return self._marked

    @property
    def path(self) -> Path:
        return self._path

    @property
    def marked(self) -> bool:
        return self._marked


class ReviewWindow(QWidget):
    def __init__(self, initial_kind: str = "hud"):
        super().__init__()
        self.setWindowTitle("Glyph Review — click to mark wrong, move to quarantine")
        self.resize(1240, 820)
        self._marked: dict[Path, bool] = {}
        self._tiles: list[GlyphTile] = []
        self._current_class: str = ""

        # ── Rubber-band drag-select state ──
        # When the user mouse-presses on an EMPTY area of the grid
        # (i.e. not on a tile), drag tracks a rectangular selection.
        # On release every tile whose geometry intersects the rect
        # gets marked. Lazy-initialize the rubber-band widget on first
        # drag so we don't allocate it for users who don't use the
        # feature.
        self._rubber_band: QRubberBand | None = None
        self._drag_origin: QPoint | None = None
        self._DRAG_THRESHOLD_PX = 5  # smaller drags treated as clicks

        # Active region kind; everything else (paths, class list,
        # quarantine dir) derives from this through training_registry.
        self._kind: str = initial_kind
        self._build_ui()

        # Open on the first available class for the active kind.
        idx = self._kind_combo.findText(self._kind)
        if idx >= 0:
            self._kind_combo.blockSignals(True)
            self._kind_combo.setCurrentIndex(idx)
            self._kind_combo.blockSignals(False)
        self._reload_class_list()

    # ── Per-kind state derivation ──

    def _spec(self) -> training_registry.RegionSpec:
        return training_registry.get(self._kind)

    def _glyph_root(self) -> Path:
        return self._spec().glyph_staging_dir

    def _quarantine_root(self) -> Path:
        return self._glyph_root() / "_quarantine"

    def _classes_ordered(self) -> list[str]:
        """Filesystem class-folder names in display order, derived
        from the active spec's label_set."""
        return [_char_to_dirname(ch) for ch in self._spec().label_set]

    def _display_for(self, cls_dirname: str) -> str:
        """Pretty label for a class folder (e.g. 'dot' → '.')."""
        return _dirname_to_char(cls_dirname)

    def _build_ui(self):
        # Top: region kind picker
        self._kind_combo = QComboBox()
        self._kind_combo.setFixedWidth(200)
        for k in training_registry.list_kinds():
            self._kind_combo.addItem(k)
        self._kind_combo.currentTextChanged.connect(self._on_kind_changed)
        kind_row = QHBoxLayout()
        kind_lbl = QLabel("Region:")
        f = kind_lbl.font(); f.setBold(True); kind_lbl.setFont(f)
        kind_row.addWidget(kind_lbl)
        kind_row.addWidget(self._kind_combo)
        self._kind_info_lbl = QLabel("")
        self._kind_info_lbl.setStyleSheet("color: #888; font-size: 10px;")
        kind_row.addWidget(self._kind_info_lbl, 1)

        # Left panel: class list
        self._class_list = QListWidget()
        self._class_list.setFixedWidth(140)
        self._class_list.itemClicked.connect(
            lambda it: self._load_class(it.data(Qt.UserRole))
        )

        # Center: scrollable glyph grid
        self._grid_holder = QWidget()
        self._grid_layout = QVBoxLayout(self._grid_holder)
        self._grid_layout.setContentsMargins(8, 8, 8, 8)
        self._grid_layout.setSpacing(6)
        # Install event filter so drag-from-empty-area triggers a
        # rubber-band selection. Mouse events that hit a child tile
        # are consumed by the tile's own ``mousePressEvent`` and never
        # reach this filter — so single-click toggle behavior is
        # preserved untouched. Drag from any GAP between tiles starts
        # a rubber band that finalizes on release.
        self._grid_holder.installEventFilter(self)
        scroll = QScrollArea()
        scroll.setWidget(self._grid_holder)
        scroll.setWidgetResizable(True)

        # Right: status + actions
        right = QVBoxLayout()
        self._status = QLabel("Pick a class on the left.")
        self._status.setWordWrap(True)
        f = self._status.font(); f.setBold(True); self._status.setFont(f)
        right.addWidget(self._status)

        self._marked_label = QLabel("Marked for removal: 0")
        right.addWidget(self._marked_label)

        rm_btn = QPushButton("Move marked to quarantine")
        rm_btn.setStyleSheet("background: #d22; color: white; padding: 8px;")
        rm_btn.clicked.connect(self._quarantine_marked)
        right.addWidget(rm_btn)

        select_all_btn = QPushButton("Mark ALL on this page")
        select_all_btn.clicked.connect(self._mark_all)
        right.addWidget(select_all_btn)

        unmark_all_btn = QPushButton("Unmark all")
        unmark_all_btn.clicked.connect(self._unmark_all)
        right.addWidget(unmark_all_btn)

        recover_btn = QPushButton("Recover quarantined")
        recover_btn.clicked.connect(self._recover_quarantine)
        right.addWidget(recover_btn)

        # ── "Send to..." reclassification grid ──
        # Mark glyphs that are mislabeled (e.g. an '8' sitting in
        # the '5/' folder), pick the correct class here, and they
        # move instead of being quarantined. Saves a labeling pass.
        send_lbl = QLabel("Send marked to class:")
        send_lbl.setStyleSheet("margin-top: 12px;")
        f = send_lbl.font(); f.setBold(True); send_lbl.setFont(f)
        right.addWidget(send_lbl)
        # Build buttons in a grid (4 cols), populated per-kind by
        # _rebuild_send_to_buttons() which is called whenever the
        # active region changes.
        from PySide6.QtWidgets import QGridLayout
        self._send_grid_holder = QWidget()
        self._send_grid = QGridLayout(self._send_grid_holder)
        self._send_grid.setContentsMargins(0, 0, 0, 0)
        self._send_grid.setSpacing(4)
        right.addWidget(self._send_grid_holder)

        right.addStretch(1)
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(220)

        # Layout — region picker on top, three-pane below
        body = QHBoxLayout()
        body.addWidget(self._class_list)
        body.addWidget(scroll, 1)
        body.addWidget(right_widget)
        root = QVBoxLayout(self)
        root.addLayout(kind_row)
        root.addLayout(body, 1)

        # Shortcuts
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._quarantine_marked)
        QShortcut(QKeySequence("Ctrl+A"),      self, activated=self._mark_all)
        QShortcut(QKeySequence("Ctrl+Z"),      self, activated=self._unmark_all)

    # ── Region change ──

    def _on_kind_changed(self, new_kind: str) -> None:
        if not new_kind or new_kind == self._kind:
            return
        if new_kind not in training_registry.list_kinds():
            return
        self._kind = new_kind
        self._reload_class_list()

    def _reload_class_list(self) -> None:
        """Rebuild the left class list from the active kind's spec.
        Auto-loads the first non-empty class so the user never sees
        a blank grid after switching tabs."""
        self._class_list.clear()
        first_nonempty: str | None = None
        for c in self._classes_ordered():
            d = self._glyph_root() / c
            n = len(list(d.glob("*.png"))) if d.is_dir() else 0
            display = self._display_for(c)
            item = QListWidgetItem(f"{display}    ({n})")
            item.setData(Qt.UserRole, c)
            self._class_list.addItem(item)
            if n > 0 and first_nonempty is None:
                first_nonempty = c

        spec = self._spec()
        self._kind_info_lbl.setText(
            f"staging: {spec.glyph_staging_dir.name}  →  "
            f"model: {spec.model_path.name}"
        )
        target = first_nonempty or self._classes_ordered()[0]
        self._rebuild_send_to_buttons()
        self._load_class(target)
        # Highlight the active row
        for i in range(self._class_list.count()):
            if self._class_list.item(i).data(Qt.UserRole) == target:
                self._class_list.setCurrentRow(i)
                break

    def _rebuild_send_to_buttons(self) -> None:
        """Render one button per class in the active region's
        label_set into the send-to grid. Clicking a button moves all
        currently-marked glyphs from the active class to that target
        class folder (renames + moves the file)."""
        # Wipe existing
        while self._send_grid.count():
            item = self._send_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        cols = 4
        for i, cls_dirname in enumerate(self._classes_ordered()):
            display = self._display_for(cls_dirname)
            btn = QPushButton(display)
            btn.setFixedHeight(28)
            btn.setStyleSheet(
                "background: #2e7; color: white; font-weight: bold; "
                "padding: 2px 6px; border-radius: 3px;"
            )
            # Capture cls_dirname per-button via default arg trick.
            btn.clicked.connect(
                lambda _checked=False, dst=cls_dirname: self._send_marked_to(dst)
            )
            r, c = divmod(i, cols)
            self._send_grid.addWidget(btn, r, c)

    # Class loading ────────────────────────────────────────────────
    def _load_class(self, cls: str):
        self._current_class = cls
        self._marked = {}
        # Clear grid
        for t in self._tiles:
            t.setParent(None); t.deleteLater()
        self._tiles = []
        # Clear layout
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        d = self._glyph_root() / cls
        if not d.is_dir():
            self._status.setText(f"No folder: {d}")
            return

        files = sorted(d.glob("*.png"))
        display = self._display_for(cls)
        self._status.setText(
            f"[{self._kind}] Class {display!r}: {len(files)} glyphs.\n"
            "Click any that are NOT a clear {display!r}.".format(display=display)
        )
        self._marked_label.setText("Marked for removal: 0")

        # Build rows of TILE grid
        row_w = None
        cur_row_layout = None
        for i, f in enumerate(files):
            if i % COLS == 0:
                row_w = QWidget()
                cur_row_layout = QHBoxLayout(row_w)
                cur_row_layout.setContentsMargins(0, 0, 0, 0)
                cur_row_layout.setSpacing(4)
                self._grid_layout.addWidget(row_w)
            tile = GlyphTile(f, on_toggle=self._on_toggle)
            cur_row_layout.addWidget(tile)
            self._tiles.append(tile)
        # Pad last row
        if cur_row_layout is not None:
            cur_row_layout.addStretch(1)

    def _on_toggle(self, path: Path, marked: bool):
        if marked:
            self._marked[path] = True
        else:
            self._marked.pop(path, None)
        self._marked_label.setText(f"Marked for removal: {len(self._marked)}")

    def _mark_all(self):
        for t in self._tiles:
            t.set_marked(True)

    def _unmark_all(self):
        for t in self._tiles:
            t.set_marked(False)

    # ── Rubber-band drag selection ──
    # Drag from any empty grid area to draw a selection rectangle.
    # On mouse-up, every tile whose geometry intersects the
    # rectangle gets MARKED (additive — already-marked tiles stay
    # marked, no toggling). Use the explicit "Unmark all" button or
    # individual clicks to undo.
    #
    # Implementation: ``installEventFilter`` on ``self._grid_holder``
    # in ``_build_ui`` routes mouse events through this method. Tile
    # widgets handle their own ``mousePressEvent`` (single-click
    # toggle) so events on tiles never reach this filter — drags
    # MUST start in the gaps between tiles, but once started they
    # can sweep over tiles freely (mouse capture stays with the
    # grid_holder until release).

    def eventFilter(self, obj, event):
        if obj is self._grid_holder:
            t = event.type()
            if (
                t == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                # Begin drag tracking. Lazy-create the QRubberBand
                # the first time it's needed.
                self._drag_origin = event.position().toPoint()
                if self._rubber_band is None:
                    self._rubber_band = QRubberBand(
                        QRubberBand.Shape.Rectangle, self._grid_holder,
                    )
                self._rubber_band.setGeometry(
                    QRect(self._drag_origin, QSize()),
                )
                self._rubber_band.show()
                return True
            elif (
                t == QEvent.Type.MouseMove
                and self._drag_origin is not None
                and self._rubber_band is not None
            ):
                cur = event.position().toPoint()
                rect = QRect(self._drag_origin, cur).normalized()
                self._rubber_band.setGeometry(rect)
                return True
            elif (
                t == QEvent.Type.MouseButtonRelease
                and self._drag_origin is not None
                and self._rubber_band is not None
            ):
                rect = self._rubber_band.geometry()
                self._rubber_band.hide()
                self._drag_origin = None
                # Treat as a real drag only when the rectangle is
                # large enough to be intentional. A click-and-release
                # on empty space (rect basically zero) is ignored —
                # if the user wanted to mark a tile they'd click on
                # the tile directly.
                if (
                    rect.width() >= self._DRAG_THRESHOLD_PX
                    or rect.height() >= self._DRAG_THRESHOLD_PX
                ):
                    self._select_in_rect(rect)
                return True
        return super().eventFilter(obj, event)

    def _select_in_rect(self, rect: QRect) -> None:
        """Mark every tile whose geometry intersects ``rect`` (in
        ``self._grid_holder`` coordinates). Additive: doesn't unmark
        anything. The status line + marked-count label update via
        ``GlyphTile.set_marked`` → ``_on_toggle`` propagation."""
        n_newly_marked = 0
        for tile in self._tiles:
            if tile.marked:
                continue  # already marked, additive semantics
            # Tile's geometry is in its row's coords; map up to
            # grid_holder coords.
            top_left = tile.mapTo(self._grid_holder, QPoint(0, 0))
            tile_rect = QRect(top_left, tile.size())
            if rect.intersects(tile_rect):
                tile.set_marked(True)
                n_newly_marked += 1
        if n_newly_marked > 0:
            self._status.setText(
                f"Drag-selected {n_newly_marked} additional tile(s) "
                f"(total marked: {len(self._marked)})."
            )

    # ── Augmentation cascade helpers ──
    # When a sample is augmented (geometric transforms: rotation,
    # translation, brightness jitter), the augmentation files live
    # alongside the original in the same class directory and follow
    # a strict naming convention: ``aug_<original_stem>_NNN.png``.
    # Quarantining or relabeling the original WITHOUT moving its
    # augmentations leaves stale augmented copies of a (now wrong-
    # class) shape in the training pool — defeating the cleanup. So
    # any move of an original ALSO moves its augmentations to the
    # same destination, atomically.
    #
    # Reverse direction: if the user marks an `aug_*` file directly
    # (uncommon but possible), only that one file moves — we don't
    # touch its sibling augmentations or the original, since those
    # might be valid.

    def _augmentation_siblings(
        self, original_path: Path,
    ) -> list[Path]:
        """Find ``aug_<stem>_*.png`` siblings of an original file in
        the same directory. Empty list if the file is itself an
        augmentation (filename starts with ``aug_``) or if no siblings
        exist."""
        if original_path.name.startswith("aug_"):
            return []
        parent = original_path.parent
        stem = original_path.stem
        return list(parent.glob(f"aug_{stem}_*.png"))

    def _move_with_aug_cascade(
        self, src: Path, dst_dir: Path,
    ) -> tuple[int, int]:
        """Move ``src`` into ``dst_dir`` and cascade-move any
        augmentation siblings to the same destination. Returns
        ``(originals_moved, augmentations_moved)``.

        Avoids clobber: if a destination file already exists with
        the same name, appends ``_dupN`` until a free name is found
        (matches the existing send-to logic).
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        n_orig = 0
        n_aug = 0
        # Collect aug siblings BEFORE moving the original (parent /
        # stem are derived from src.path which becomes invalid after
        # shutil.move).
        aug_siblings = self._augmentation_siblings(src)
        try:
            target = dst_dir / src.name
            if target.exists():
                n = 1
                while True:
                    candidate = dst_dir / f"{src.stem}_dup{n}{src.suffix}"
                    if not candidate.exists():
                        target = candidate
                        break
                    n += 1
            shutil.move(str(src), str(target))
            n_orig = 1
        except Exception as exc:
            print(f"  move failed {src.name}: {exc}")
            return (0, 0)
        for aug_path in aug_siblings:
            try:
                aug_target = dst_dir / aug_path.name
                if aug_target.exists():
                    n = 1
                    while True:
                        candidate = (
                            dst_dir / f"{aug_path.stem}_dup{n}{aug_path.suffix}"
                        )
                        if not candidate.exists():
                            aug_target = candidate
                            break
                        n += 1
                shutil.move(str(aug_path), str(aug_target))
                n_aug += 1
            except Exception as exc:
                print(f"  aug-cascade move failed {aug_path.name}: {exc}")
        return (n_orig, n_aug)

    def _send_marked_to(self, dst_class_dirname: str) -> None:
        """Move every currently-marked glyph from ``self._current_class``
        into ``dst_class_dirname``. Augmentation siblings (``aug_<stem>_*``)
        cascade-move to the same destination so the training pool
        stays consistent with the relabeling."""
        if not self._marked:
            self._status.setText("Nothing marked.")
            return
        if dst_class_dirname == self._current_class:
            self._status.setText(
                f"Already in '{self._display_for(dst_class_dirname)}' — no-op."
            )
            return
        dst_dir = self._glyph_root() / dst_class_dirname
        moved_orig = 0
        moved_aug = 0
        for path in list(self._marked.keys()):
            n_o, n_a = self._move_with_aug_cascade(path, dst_dir)
            moved_orig += n_o
            moved_aug += n_a
        msg = (
            f"Moved {moved_orig} glyph(s)"
            + (f" + {moved_aug} aug sibling(s)" if moved_aug else "")
            + f" → '{self._display_for(dst_class_dirname)}/'."
        )
        self._status.setText(msg)
        self._refresh_class_counts()
        self._load_class(self._current_class)

    def _quarantine_marked(self):
        """Move every currently-marked glyph to the quarantine subdir
        for the current class. Augmentation siblings cascade-move so
        the training pool doesn't keep stale aug variants of a now-
        quarantined original."""
        if not self._marked:
            self._status.setText("Nothing marked.")
            return
        dst_dir = self._quarantine_root() / self._current_class
        moved_orig = 0
        moved_aug = 0
        for path in list(self._marked.keys()):
            n_o, n_a = self._move_with_aug_cascade(path, dst_dir)
            moved_orig += n_o
            moved_aug += n_a
        msg = (
            f"Moved {moved_orig} glyphs"
            + (f" + {moved_aug} aug sibling(s)" if moved_aug else "")
            + " to quarantine.\nReloading class..."
        )
        self._status.setText(msg)
        self._refresh_class_counts()
        self._load_class(self._current_class)

    def _recover_quarantine(self):
        """Move ALL quarantined glyphs back into their original folders."""
        qroot = self._quarantine_root()
        if not qroot.is_dir():
            self._status.setText("No quarantine folder yet.")
            return
        recovered = 0
        for cls_dir in qroot.iterdir():
            if not cls_dir.is_dir():
                continue
            target = self._glyph_root() / cls_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for f in cls_dir.glob("*.png"):
                try:
                    shutil.move(str(f), str(target / f.name))
                    recovered += 1
                except Exception:
                    pass
        self._status.setText(f"Recovered {recovered} glyphs from quarantine.")
        self._refresh_class_counts()
        self._load_class(self._current_class)

    def _refresh_class_counts(self):
        for i in range(self._class_list.count()):
            item = self._class_list.item(i)
            cls = item.data(Qt.UserRole)
            d = self._glyph_root() / cls
            n = len(list(d.glob("*.png"))) if d.is_dir() else 0
            label = self._display_for(cls)
            item.setText(f"{label}    ({n})")


def main():
    app = QApplication(sys.argv)
    # Optional CLI arg: open straight on a region tab.
    initial_kind = "hud"
    if len(sys.argv) > 1 and sys.argv[1] in training_registry.list_kinds():
        initial_kind = sys.argv[1]
    win = ReviewWindow(initial_kind=initial_kind)
    # Force to primary screen (same trick as labeler)
    primary = app.primaryScreen().availableGeometry()
    win.resize(min(1240, primary.width() - 100), min(820, primary.height() - 100))
    win.move(primary.left() + 50, primary.top() + 50)
    win.show()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
