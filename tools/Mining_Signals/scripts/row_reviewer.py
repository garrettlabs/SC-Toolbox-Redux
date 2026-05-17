"""Row Reviewer - verify labelled captures before feeding them to
the RGB CRNN (or any other downstream model).

Walks every capture that has a sibling ``<capture>.glyphs.json``
sidecar (produced by the new Glyph Forge), shows the source image
with the verified bbox positions overlaid, and lets the user
Approve / Reject each row. Persists the verdict to the sidecar's
``review_status`` field (``"approved"`` / ``"rejected"`` /
``"pending"``).

Downstream consumers (training scripts, kerning recalibrator) can
then filter by ``review_status == "approved"`` to use only reviewed
data.

Keyboard shortcuts:
    Ctrl+Enter / PgDn = Approve & next
    Ctrl+Backspace / R = Reject & next
    PgUp              = Previous
    S                 = Skip (no verdict, advance)

Run with:
    python scripts/row_reviewer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QPixmap, QColor, QFont, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget, QProgressBar,
)


THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent

# Theme mirrors Glyph Forge so the two windows feel like part of the same workflow.
ACCENT = "#33dd88"
RED = "#ff4444"
DIM = "#888888"
BG = "#1e1e1e"
FG = "#e0e0e0"
WARN = "#ffc107"
BLUE = "#5fa8d3"

# Bbox overlay colours
COLOR_DIGIT = (51, 221, 136, 220)        # green digit border
COLOR_DIGIT_DISAGREE = (255, 68, 68, 220)  # red when voter disagreed with expected
COLOR_REJECT_BG = (255, 68, 68, 60)       # red wash for rejected status
COLOR_APPROVE_BG = (51, 221, 136, 40)     # green wash for approved status


# ─────────────────────────────────────────────────────────────
# Sidecar walker
# ─────────────────────────────────────────────────────────────

PANEL_ROOT_CANDIDATES = [
    TOOL / "training_data_panels",
]


def _find_all_sidecars() -> list[Path]:
    """Walk all panel-root candidates and collect every
    ``<capture>.glyphs.json`` sidecar found. Returns sorted list
    so the review queue is deterministic across runs.
    """
    out: list[Path] = []
    for root in PANEL_ROOT_CANDIDATES:
        if not root.is_dir():
            continue
        for sc in sorted(root.rglob("*.glyphs.json")):
            out.append(sc)
    return out


def _load_sidecar(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_sidecar(path: Path, doc: dict) -> bool:
    try:
        path.write_text(
            json.dumps(doc, indent=2),
            encoding="utf-8",
        )
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# Overlay renderer
# ─────────────────────────────────────────────────────────────

def _render_overlay(
    source_png: Path,
    tiles: list[dict],
    label: str,
    review_status: str,
    scale: int = 3,
) -> Optional[Image.Image]:
    """Load the source PNG, upscale by ``scale``, and draw bbox
    overlays for each tile. Colour: GREEN if voter consensus
    agreed with expected, RED otherwise (disagreements get a red
    border so you can spot suspect labels at a glance).

    The whole image is washed with a faint GREEN if already
    approved, RED if already rejected — so re-reviewing existing
    verdicts is unambiguous.
    """
    if not source_png.is_file():
        return None
    try:
        img = Image.open(source_png).convert("RGB")
    except Exception:
        return None
    img = img.resize(
        (img.width * scale, img.height * scale), Image.NEAREST,
    )
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Apply review-status wash (subtle, doesn't obscure content).
    if review_status == "approved":
        draw.rectangle(
            [0, 0, img.width, img.height], fill=COLOR_APPROVE_BG,
        )
    elif review_status == "rejected":
        draw.rectangle(
            [0, 0, img.width, img.height], fill=COLOR_REJECT_BG,
        )

    # Try a readable font; fall back to default if unavailable.
    try:
        font = ImageFont.truetype("consola.ttf", 10 * scale // 2)
    except Exception:
        font = ImageFont.load_default()

    # The sidecar stores x coords in source-image space. We need
    # to know the y-range of the digit row. The sidecar doesn't
    # carry y, so we use the FULL image height as the bbox y
    # bounds (the digits live across the entire row band in the
    # source PNG). This is approximate but good enough for a
    # visual review.
    H = img.height
    y1 = int(H * 0.20)
    y2 = int(H * 0.85)

    for t in tiles:
        try:
            x1 = int(t["x1"]) * scale
            x2 = int(t["x2"]) * scale
        except Exception:
            continue
        saved_class = t.get("saved_class")
        expected = t.get("expected", "?")
        skipped = bool(t.get("skipped", saved_class is None))
        if skipped:
            colour = (136, 136, 136, 220)
            label_text = "skip"
        else:
            # Check voter consensus from the pipeline snapshot.
            consensus = (t.get("pipeline") or {}).get("consensus")
            voter_disagrees = False
            if consensus is not None and len(consensus) >= 1:
                if str(consensus[0]) != saved_class:
                    voter_disagrees = True
            colour = COLOR_DIGIT_DISAGREE if voter_disagrees else COLOR_DIGIT
            label_text = f"{saved_class}"

        # Draw bbox.
        draw.rectangle(
            [x1, y1, x2, y2], outline=colour, width=2,
        )
        # Class label centered above the bbox.
        tx = (x1 + x2) // 2 - 4 * scale // 2
        ty = max(0, y1 - 14 * scale // 2)
        draw.text((tx, ty), label_text, fill=colour, font=font)

    out = Image.alpha_composite(img.convert("RGBA"), overlay)
    return out.convert("RGB")


# ─────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────

class RowReviewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Row Reviewer - approve/reject Glyph Forge sidecars")
        self.resize(1100, 580)
        self.setStyleSheet(f"background: {BG}; color: {FG};")

        self._sidecars = _find_all_sidecars()
        # Resume at first unreviewed (status missing OR "pending").
        self._idx = self._first_unreviewed()
        self._build_ui()
        self._show_current()

    def _first_unreviewed(self) -> int:
        for i, sc in enumerate(self._sidecars):
            doc = _load_sidecar(sc)
            if doc is None:
                continue
            status = str(doc.get("review_status", "")).strip().lower()
            if status not in ("approved", "rejected"):
                return i
        return 0

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # Header
        title = QLabel("ROW REVIEWER", self)
        tf = QFont("Consolas")
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        title.setStyleSheet(f"color: {ACCENT}; background: transparent;")
        root.addWidget(title)

        sub = QLabel(
            "Approve verified rows or reject suspect ones. Persists "
            "review_status to each sidecar JSON. Downstream training "
            "scripts can filter by approved-only.",
            self,
        )
        sub.setStyleSheet(
            f"color: {DIM}; font-size: 9pt; background: transparent;"
        )
        sub.setWordWrap(True)
        root.addWidget(sub)

        # Overview image (full strip + bbox overlays)
        self._img_lbl = QLabel("(loading)", self)
        self._img_lbl.setMinimumHeight(180)
        self._img_lbl.setAlignment(Qt.AlignCenter)
        self._img_lbl.setStyleSheet(
            f"background: #181818; border: 1px solid #333; padding: 8px;"
        )
        root.addWidget(self._img_lbl)

        # Status / metadata line
        self._status_lbl = QLabel("", self)
        self._status_lbl.setStyleSheet(
            f"color: {FG}; font-family: Consolas; font-size: 10pt; "
            f"background: transparent;"
        )
        root.addWidget(self._status_lbl)

        # Per-tile detail line (one row per tile)
        self._detail_lbl = QLabel("", self)
        self._detail_lbl.setStyleSheet(
            f"color: {DIM}; font-family: Consolas; font-size: 9pt; "
            f"background: transparent;"
        )
        self._detail_lbl.setWordWrap(True)
        root.addWidget(self._detail_lbl)

        # Buttons
        btn_row = QHBoxLayout()

        prev_btn = QPushButton("< Prev (PgUp)", self)
        prev_btn.clicked.connect(lambda: self._step(-1))
        btn_row.addWidget(prev_btn)

        skip_btn = QPushButton("Skip (S)", self)
        skip_btn.clicked.connect(lambda: self._step(+1))
        btn_row.addWidget(skip_btn)

        reject_btn = QPushButton("Reject (Ctrl+Backspace)", self)
        reject_btn.setStyleSheet(
            f"background: {RED}; color: white; padding: 6px 14px; "
            f"font-weight: bold; border: none; border-radius: 3px;"
        )
        reject_btn.clicked.connect(self._reject)
        btn_row.addWidget(reject_btn)

        approve_btn = QPushButton("Approve & Next (Ctrl+Enter)", self)
        approve_btn.setStyleSheet(
            f"background: {ACCENT}; color: black; padding: 6px 16px; "
            f"font-weight: bold; border: none; border-radius: 3px;"
        )
        approve_btn.clicked.connect(self._approve)
        btn_row.addWidget(approve_btn)

        btn_row.addStretch(1)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, max(1, len(self._sidecars)))
        self._progress.setFixedWidth(260)
        btn_row.addWidget(self._progress)

        root.addLayout(btn_row)

        # Bottom summary line (counts approved / rejected / pending)
        self._summary_lbl = QLabel("", self)
        self._summary_lbl.setStyleSheet(
            f"color: {DIM}; font-family: Consolas; font-size: 9pt; "
            f"background: transparent;"
        )
        root.addWidget(self._summary_lbl)

        # Shortcuts
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._approve)
        QShortcut(QKeySequence("Ctrl+S"),      self, activated=self._approve)
        QShortcut(QKeySequence("PgDown"),      self, activated=self._approve)
        QShortcut(QKeySequence("Ctrl+Backspace"), self, activated=self._reject)
        QShortcut(QKeySequence("R"),           self, activated=self._reject)
        QShortcut(QKeySequence("S"),           self, activated=lambda: self._step(+1))
        QShortcut(QKeySequence("PgUp"),        self, activated=lambda: self._step(-1))

    def _step(self, delta: int) -> None:
        new = self._idx + delta
        if 0 <= new < len(self._sidecars):
            self._idx = new
            self._show_current()

    def _show_current(self) -> None:
        if not self._sidecars:
            self._status_lbl.setText("No sidecar JSON files found.")
            self._detail_lbl.setText("")
            self._summary_lbl.setText("")
            return

        sc = self._sidecars[self._idx]
        doc = _load_sidecar(sc) or {}
        tiles = doc.get("tiles") or []
        label_value = doc.get("label", "")
        review_status = str(doc.get("review_status", "pending")).lower()
        # Source PNG sits next to the sidecar with the .png extension.
        png_path = sc.with_suffix("").with_suffix(".png")
        if not png_path.is_file():
            # Sidecar stem ends with ".glyphs", strip that to get
            # the source name.
            stem = sc.stem.replace(".glyphs", "")
            png_path = sc.parent / f"{stem}.png"

        img = _render_overlay(png_path, tiles, label_value, review_status)
        if img is not None:
            qim = ImageQt(img)
            self._img_lbl.setPixmap(QPixmap.fromImage(qim))
        else:
            self._img_lbl.setText(f"(image load failed: {png_path.name})")

        # Status line
        n_total = len(self._sidecars)
        n_tiles = len(tiles)
        n_saved = sum(1 for t in tiles if not t.get("skipped", False))
        n_skipped = sum(1 for t in tiles if t.get("skipped", False))
        n_disagree = 0
        for t in tiles:
            if t.get("skipped"):
                continue
            cons = (t.get("pipeline") or {}).get("consensus")
            if cons and len(cons) >= 1:
                if str(cons[0]) != t.get("saved_class"):
                    n_disagree += 1

        status_color = {
            "approved": ACCENT,
            "rejected": RED,
            "pending": FG,
        }.get(review_status, FG)
        self._progress.setValue(self._idx + 1)
        self._progress.setFormat(f"{self._idx + 1} / {n_total}")
        self._status_lbl.setText(
            f"[{self._idx + 1}/{n_total}]  {png_path.name}  "
            f"label={label_value!r}  tiles={n_tiles} "
            f"(saved={n_saved}, skipped={n_skipped}, "
            f"voter_disagree={n_disagree})  "
            f"status={review_status}"
        )
        self._status_lbl.setStyleSheet(
            f"color: {status_color}; font-family: Consolas; font-size: 10pt; "
            f"background: transparent;"
        )

        # Per-tile detail
        detail_parts: list[str] = []
        for i, t in enumerate(tiles):
            if t.get("skipped"):
                detail_parts.append(
                    f"[{i}] x={t.get('x1')}..{t.get('x2')} SKIP"
                )
                continue
            cons = (t.get("pipeline") or {}).get("consensus")
            cs = (
                f"voter={cons[0]}@{float(cons[1]):.2f}"
                if cons else "voter=None"
            )
            expected = t.get("expected", "?")
            saved = t.get("saved_class", "?")
            match = "ok" if saved == expected else "EDITED"
            detail_parts.append(
                f"[{i}] x={t.get('x1')}..{t.get('x2')} "
                f"saved={saved!r} exp={expected!r} ({match}) {cs}"
            )
        self._detail_lbl.setText("  ".join(detail_parts))

        # Summary across all sidecars
        n_approved = 0
        n_rejected = 0
        n_pending = 0
        for s in self._sidecars:
            d = _load_sidecar(s) or {}
            st = str(d.get("review_status", "pending")).lower()
            if st == "approved":
                n_approved += 1
            elif st == "rejected":
                n_rejected += 1
            else:
                n_pending += 1
        self._summary_lbl.setText(
            f"Approved: {n_approved}   Rejected: {n_rejected}   "
            f"Pending: {n_pending}"
        )

    def _approve(self) -> None:
        self._set_status("approved")

    def _reject(self) -> None:
        self._set_status("rejected")

    def _set_status(self, status: str) -> None:
        if not self._sidecars:
            return
        sc = self._sidecars[self._idx]
        doc = _load_sidecar(sc) or {}
        doc["review_status"] = status
        _save_sidecar(sc, doc)
        # Advance to next unreviewed (or just next).
        new = self._idx + 1
        if new < len(self._sidecars):
            self._idx = new
        self._show_current()


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

    print("row_reviewer: starting", flush=True)
    reviewer = RowReviewer()
    reviewer.show()
    reviewer.raise_()
    print(f"row_reviewer: window shown, {len(reviewer._sidecars)} sidecars",
          flush=True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
