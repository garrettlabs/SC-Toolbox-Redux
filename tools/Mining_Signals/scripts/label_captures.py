"""Label user-captured panels for OCR training.

Walks all PNGs under ``training_data_panels/<folder>/region1`` and
``region2`` and lets you enter ground-truth values for each. Labels
are saved as JSON sidecar files next to the image:
    cap_*.png  →  cap_*.json

The schema differs per region:
  region1 (SCAN RESULTS panel):
    mineral, mass, resistance, instability, difficulty, scu,
    composition (multi-line "pct  name  count" rows)
  region2 (small numeric readout):
    value, kind (free text label of what it is)

Already-labeled images are skipped on relaunch; use the "Edit" button
to re-open one. The progress bar shows total labeled / total images
across all folders.

Keyboard:
  Ctrl+Enter / Ctrl+S  →  save and advance
  PgDown / PgUp        →  skip without saving / go back
  Esc                  →  quit

Run with:  python scripts/label_captures.py
       or: python scripts/label_captures.py --folder user_20260418_081525
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL.ImageQt import ImageQt
from PIL import Image

# Path setup
_SCRIPT_DIR = Path(__file__).resolve().parent
_TOOL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_TOOL_DIR))

# pytesseract is imported LAZILY on first OCR call — importing it at
# module load has been observed to hang on some Windows Python installs
# (issue related to subprocess init in the WindowsApps python stub).
_TESS_OK: Optional[bool] = None  # None = untested, True/False = result
_pytesseract = None

# Debug log — writes every OCR run so we can see what the labeler is doing
_LOG_PATH = _TOOL_DIR / "labeler.log"
def _log(msg: str) -> None:
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            from datetime import datetime as _dt
            f.write(f"[{_dt.now().strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


_TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def _ensure_tesseract() -> bool:
    """Check if tesseract.exe is available. We call it directly via
    subprocess — pytesseract import hangs on this Python install."""
    global _TESS_OK
    if _TESS_OK is not None:
        return _TESS_OK
    import os as _os
    _TESS_OK = _os.path.isfile(_TESSERACT_EXE)
    return _TESS_OK


def _run_tesseract(img: Image.Image, config_args: list[str]) -> str:
    """Call tesseract.exe directly, piping the image via temp file."""
    import tempfile
    import os as _os
    with tempfile.NamedTemporaryFile(
        suffix=".png", delete=False, dir=tempfile.gettempdir(),
    ) as tf:
        tmp_path = tf.name
    try:
        img.save(tmp_path)
        result = subprocess.run(
            [_TESSERACT_EXE, tmp_path, "-"] + config_args,
            capture_output=True, timeout=15, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.stdout or ""
    except Exception as exc:
        print(f"  tesseract error: {exc}")
        return ""
    finally:
        try:
            _os.unlink(tmp_path)
        except Exception:
            pass

# Glyph extraction (for training-data generation on save)
try:
    from scripts.extract_labeled_glyphs import (
        extract_one as _extract_glyphs,
        PANEL_GLYPH_ROOT, SIG_GLYPH_ROOT,
    )
    _EXTRACTOR_OK = True
except Exception as _exc:
    _EXTRACTOR_OK = False
    print(f"[labeler] glyph extractor unavailable ({_exc})")
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QProgressBar, QPushButton, QPlainTextEdit,
    QScrollArea, QSizePolicy, QSpinBox, QSplitter, QStatusBar,
    QVBoxLayout, QWidget,
)


# Mineral list loaded from the Mining Signals chart cache, with (Raw)
# variants auto-generated for each. Names match the in-game panel
# capitalization so labels auto-select via the dropdown's findText.
def _load_mineral_list() -> list[str]:
    """Pull every mineral name from .mining_chart_cache.json + generate
    Raw variants. Returns sorted uppercase list with 'INERT MATERIALS'
    first.
    """
    cache = _TOOL_DIR / ".mining_chart_cache.json"
    names: set[str] = set()
    try:
        import json as _json
        with open(cache, "r", encoding="utf-8") as f:
            data = _json.load(f)
        me = data.get("mining_data", {}).get("mineableElements", {})
        for v in me.values():
            n = (v.get("name") or "").strip()
            if not n:
                continue
            # Uppercase — panel shows ALL CAPS
            upper = n.upper()
            names.add(upper)
            # Also add the "base" form (without (RAW)/(ORE) suffix) and
            # an explicit (RAW) variant for every mineral.
            import re as _re
            base = _re.sub(r"\s*\((?:RAW|ORE)\)\s*$", "", upper).strip()
            if base:
                names.add(base)
                names.add(f"{base} (RAW)")
                names.add(f"{base} (ORE)")
                names.add(f"RAW {base}")
    except Exception as exc:
        print(f"[labeler] mineral-list load failed ({exc}) — using fallback")
    # Always include these canonical entries
    names.add("INERT MATERIALS")
    names.add("UNKNOWN MATERIAL")
    # Inert Materials first, then alphabetical
    out = sorted(names)
    if "INERT MATERIALS" in out:
        out.remove("INERT MATERIALS")
        out.insert(0, "INERT MATERIALS")
    return out


KNOWN_MATERIALS = _load_mineral_list()

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
PANELS_ROOT = TOOL / "training_data_panels"


def _scan_user_images(folder: Optional[str] = None) -> list[Path]:
    """Return list of all user-capture PNGs under PANELS_ROOT."""
    out: list[Path] = []
    if folder:
        roots = [PANELS_ROOT / folder]
    else:
        roots = [p for p in PANELS_ROOT.iterdir() if p.is_dir() and p.name.startswith("user_")]
    for root in roots:
        for region_dir in ("region1", "region2"):
            d = root / region_dir
            if d.is_dir():
                out.extend(sorted(d.glob("cap_*.png")))
    return out


def _label_path(img_path: Path) -> Path:
    return img_path.with_suffix(".json")


def _load_label(img_path: Path) -> dict:
    p = _label_path(img_path)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_label(img_path: Path, data: dict) -> None:
    _label_path(img_path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _region_of(img_path: Path) -> str:
    return img_path.parent.name  # "region1" or "region2"


# ─── OCR pre-fill ───────────────────────────────────────────────────
# Regexes are lenient about:
#   - Tesseract reading '.' as '-' (INSTABILITY: 2-22 → 2.22)
#   - Missing colons or noise punctuation (MASS. → MASS:)
#   - Extra non-digit noise chars in values
_RE_MASS  = re.compile(r"M[A4]SS\s*[:.,]?\s*([\d,]+(?:[\.\-]\d+)?)", re.IGNORECASE)
_RE_RES   = re.compile(r"RES(?:IS|1S)?T?AN?CE?\s*[:.,]?\s*([\d,]+(?:[\.\-]\d+)?)\s*%?", re.IGNORECASE)
_RE_INST  = re.compile(r"IN?STAB(?:ILITY|L?ITY)?\s*[:.,]?\s*([\d,]+(?:[\.\-]\d+)?)", re.IGNORECASE)
_RE_SCU   = re.compile(r"([\d,]+[\.\-]\d+)\s*SCU", re.IGNORECASE)
_RE_DIFF  = re.compile(r"\b(EASY|MEDIUM|HARD|VERY\s*HARD|IMPOSSIBLE)\b", re.IGNORECASE)
# Composition row: "  4.11% RAW ICE  514"  or  "82.26%  INERT MATERIALS  0"
# Case-insensitive so Tesseract's mixed-case output still matches.
_RE_COMP  = re.compile(
    r"([\d]+\.[\d]+)\s*%\s+([A-Za-z][A-Za-z0-9 \(\)\-]*?[A-Za-z\)0-9])\s+(\d+)",
)
_RE_NUM_ONLY = re.compile(r"[\d,]+")


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Preprocess for Tesseract using the ORANGE-signature channel.

    SC HUD text is always warm orange/cream (R > G > B) regardless of
    background brightness. Using (R - B) isolates the warm channel:
      - Orange text: R high, B low → R-B large (positive)
      - Neutral nebula: R ≈ B → R-B small
      - Dark space: R ≈ B ≈ 0 → R-B small
    After this transform, orange text is BRIGHT on a dark field, which
    is the same for both dark-space and light-nebula backgrounds.
    """
    import numpy as np
    target_h = 800
    if img.height < target_h:
        scale = target_h / img.height
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    R = arr[..., 0]
    G = arr[..., 1]
    B = arr[..., 2]
    # Orange signature: R dominates B, R slightly > G
    warm = np.clip(R - B, 0, 255).astype(np.uint8)
    # Also try red dominance over green (catches cream/yellow text)
    warm2 = np.clip(R - G, 0, 255).astype(np.uint8)
    warm = np.maximum(warm, warm2 * 2)
    # Stretch contrast
    lo, hi = int(np.percentile(warm, 50)), int(np.percentile(warm, 99))
    if hi > lo:
        warm = np.clip((warm.astype(np.int32) - lo) * 255 // max(1, hi - lo),
                       0, 255).astype(np.uint8)
    # Threshold: bright warm = text (now black on white for tesseract)
    thr = 60
    binary = np.where(warm > thr, 0, 255).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def _ocr_region1(img: Image.Image) -> dict:
    """Run Tesseract on a SCAN RESULTS panel and parse fields."""
    if not _ensure_tesseract():
        return {}
    # Upscale for small composition text + try THREE preprocessings:
    #   1. Grayscale upscale (best for dark-bg clean text)
    #   2. Warm-channel isolate (best for light-bg / colored-text cases)
    #   3. Aggressive threshold (catches what the others miss)
    # Combining outputs lets the regex parser pick up fields that any
    # one preprocessing finds.
    if img.height < 2000:
        scale = 2000 / img.height
        img2 = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )
    else:
        img2 = img
    text_gray = _run_tesseract(img2.convert("L"), ["--psm", "6"]) or ""
    text_warm = _run_tesseract(_preprocess_for_ocr(img), ["--psm", "6"]) or ""
    text = text_gray + "\n" + text_warm
    if not text.strip():
        return {}

    def _norm(s: str) -> str:
        # Tesseract mis-reads decimals as dashes; normalize back.
        return s.replace(",", "").replace("-", ".")

    out: dict = {}
    if (m := _RE_MASS.search(text)):
        out["mass"] = _norm(m.group(1)).split(".")[0]  # mass is int
    if (m := _RE_RES.search(text)):
        # Resistance is a percentage — keep the % suffix so the %
        # glyph ends up in training data.
        out["resistance"] = _norm(m.group(1)).split(".")[0] + "%"
    if (m := _RE_INST.search(text)):
        out["instability"] = _norm(m.group(1))
    if (m := _RE_SCU.search(text)):
        out["scu"] = _norm(m.group(1))
    if (m := _RE_DIFF.search(text)):
        out["difficulty"] = m.group(1).upper().replace("  ", " ")

    # Mineral name: the first non-empty line above the MASS line that
    # looks like an uppercase identifier (and isn't "SCAN RESULTS")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    mass_idx = next(
        (i for i, ln in enumerate(lines) if "MASS" in ln.upper()), None,
    )
    if mass_idx is not None and mass_idx > 0:
        for cand in reversed(lines[:mass_idx]):
            up = cand.upper()
            if "SCAN" in up and "RESULT" in up:
                continue
            # Prefer lines that are mostly uppercase letters
            letters = [c for c in cand if c.isalpha()]
            if len(letters) >= 3 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
                # Strip trailing punctuation/numbers
                cleaned = re.sub(r"[^A-Z\s\(\)\-]", "", up).strip()
                if cleaned:
                    out["mineral"] = cleaned
                    break

    # Composition rows — parse line-by-line to handle noisy OCR output.
    # Each composition line has: <pct>% <name> <count>
    # Where pct may use comma or dot, name may have OCR noise, count
    # is the last integer on the line.
    import re as _re
    comp: list[dict] = []
    seen: set[tuple] = set()
    # Percentage anywhere on the line (decimal or comma, followed by %)
    pct_pat = _re.compile(r"([\d]+[.,][\d]+)\s*%")
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = pct_pat.search(line)
        if not m:
            continue
        pct = m.group(1).replace(",", ".")
        # Skip garbage like "0.00%" or pct out of sane range (1-100)
        try:
            pct_val = float(pct)
            if pct_val < 0.01 or pct_val > 100:
                continue
        except ValueError:
            continue
        # Extract all numbers from the line — first is pct, last int is count
        nums = _re.findall(r"\d+", line)
        count = ""
        if len(nums) >= 2:
            # Find the last integer-like number AFTER the pct position
            pct_end = m.end()
            tail = line[pct_end:]
            tail_ints = _re.findall(r"\b(\d+)\b", tail)
            if tail_ints:
                count = tail_ints[-1]
        # Name: uppercase letter run(s) on the line (dedupe whitespace)
        name_hits = _re.findall(r"[A-Za-z]+(?:\s+[A-Za-z]+)*", line)
        # Prefer the longest letter-sequence that's clearly text
        name_hits = [n.strip() for n in name_hits if len(n.strip()) >= 2]
        name = ""
        if name_hits:
            name = max(name_hits, key=len).upper()
            # Strip stray single letters at edges (OCR artifacts)
            name = _re.sub(r"^\W+|\W+$", "", name).strip()
        # Dedupe by rounded pct (same physical row can OCR slightly
        # differently on different passes).
        try:
            pct_round = round(float(pct), 0)
        except ValueError:
            pct_round = pct
        if pct_round in seen:
            continue
        seen.add(pct_round)
        # Store pct with trailing % so the % character is accumulated
        # by the glyph extractor during training-data generation.
        pct_with_pct = pct if pct.endswith("%") else pct + "%"
        comp.append({"pct": pct_with_pct, "name": name, "count": count})
    if comp:
        # Cap at 6 rows (typical SC panel has ≤4)
        out["composition"] = comp[:6]

    return out


def _ocr_region2(img: Image.Image) -> dict:
    """Read a signature scanner crop — auto-fill the value field.

    Uses the live runtime's 3-engine vote (Tesseract A + Tesseract B
    + Paddle, with consensus voting) when available so the labeler's
    auto-fill matches the scanner's own accuracy. Falls back to a
    single Tesseract pass when ``ocr.screen_reader`` can't be imported
    (e.g. tools running outside the project)."""
    # Preferred path: borrow the runtime extractor. It already does
    # multi-engine vote, returns a clean int (or None on failure).
    try:
        from ocr.screen_reader import extract_number
        val = extract_number(img)
        if val is not None:
            # Re-format with thousands separator so it matches what
            # the user typically types in the value field.
            return {"value": f"{val:,}"}
    except Exception:
        pass

    # Fallback: single Tesseract pass with digit/comma/period whitelist.
    if not _ensure_tesseract():
        return {}
    if img.height < 80:
        scale = 80 / img.height
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )
    text = _run_tesseract(
        img,
        ["--psm", "7", "-c", "tessedit_char_whitelist=0123456789,."],
    ).strip()
    if not text:
        return {}
    if (m := _RE_NUM_ONLY.search(text)):
        return {"value": m.group(0)}
    return {}


def _ocr_prefill(img_path: Path) -> dict:
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return {}
    if _region_of(img_path) == "region1":
        return _ocr_region1(img)
    return _ocr_region2(img)


# ─── Region-specific field forms ───────────────────────────────────
class CompositionRow(QWidget):
    """One composition row: [%] [material dropdown] [count] [×]."""

    def __init__(self, parent_form: "Region1Form"):
        super().__init__()
        self._parent_form = parent_form
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self.pct = QLineEdit()
        self.pct.setPlaceholderText("%")
        self.pct.setFixedWidth(60)
        self.name = QComboBox()
        self.name.setEditable(True)
        self.name.addItems([""] + KNOWN_MATERIALS)
        self.name.setMinimumWidth(180)
        self.count = QLineEdit()
        self.count.setPlaceholderText("count")
        self.count.setFixedWidth(70)
        self.del_btn = QPushButton("×")
        self.del_btn.setFixedWidth(24)
        self.del_btn.setStyleSheet("color:#a33;")
        self.del_btn.clicked.connect(self._delete_self)
        h.addWidget(self.pct)
        h.addWidget(self.name, 1)
        h.addWidget(self.count)
        h.addWidget(self.del_btn)

    def _delete_self(self):
        self._parent_form._remove_row(self)

    def to_dict(self) -> dict:
        return {
            "pct":   self.pct.text().strip(),
            "name":  self.name.currentText().strip(),
            "count": self.count.text().strip(),
        }

    def from_dict(self, d: dict):
        self.pct.setText(d.get("pct", ""))
        nm = d.get("name", "")
        idx = self.name.findText(nm)
        if idx >= 0:
            self.name.setCurrentIndex(idx)
        else:
            self.name.setEditText(nm)
        self.count.setText(d.get("count", ""))


class Region1Form(QWidget):
    """SCAN RESULTS panel: mineral, mass, resistance, instability, difficulty, scu, composition."""

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self.mineral = QComboBox()
        self.mineral.setEditable(True)
        self.mineral.addItems([""] + KNOWN_MATERIALS)
        self.mineral.setMinimumWidth(220)
        self.mineral.currentTextChanged.connect(self._on_mineral_changed)
        self.mass = QLineEdit();         self.mass.setPlaceholderText("e.g. 3577")
        self.resistance = QLineEdit();   self.resistance.setPlaceholderText("e.g. 0%  (include %)")
        self.instability = QLineEdit();  self.instability.setPlaceholderText("e.g. 0.00")
        self.difficulty = QComboBox()
        self.difficulty.addItems(["", "EASY", "MEDIUM", "HARD", "VERY HARD", "IMPOSSIBLE"])
        self.scu = QLineEdit();          self.scu.setPlaceholderText("e.g. 25.04")
        form.addRow("Mineral:",     self.mineral)
        form.addRow("Mass:",        self.mass)
        form.addRow("Resistance:",  self.resistance)
        form.addRow("Instability:", self.instability)
        form.addRow("Difficulty:",  self.difficulty)
        form.addRow("SCU:",         self.scu)
        outer.addLayout(form)

        # Composition section
        comp_label = QLabel("Composition rows:")
        f = comp_label.font(); f.setBold(True); comp_label.setFont(f)
        outer.addWidget(comp_label)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setSpacing(2)
        outer.addLayout(self._rows_layout)
        self._rows: list[CompositionRow] = []

        # Action buttons
        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add row")
        add_btn.clicked.connect(lambda: self._add_row())
        autofill_btn = QPushButton("Auto-fill 3 rows (mineral / mineral / INERT)")
        autofill_btn.clicked.connect(self._autofill_default)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(autofill_btn)
        outer.addLayout(btn_row)

        outer.addStretch(1)

    # Row management ────────────────────────────────────────────────
    def _add_row(self, data: Optional[dict] = None) -> CompositionRow:
        row = CompositionRow(self)
        if data:
            row.from_dict(data)
        else:
            # Default the material to current mineral name
            mineral = self.mineral.currentText().strip()
            if mineral:
                row.name.setEditText(mineral)
        self._rows.append(row)
        self._rows_layout.addWidget(row)
        return row

    def _remove_row(self, row: CompositionRow):
        if row in self._rows:
            self._rows.remove(row)
            row.setParent(None)
            row.deleteLater()

    def _clear_rows(self):
        for r in list(self._rows):
            self._remove_row(r)

    def _autofill_default(self):
        """Pre-populate rows: default 3 (mineral / mineral / INERT),
        but if OCR finds MORE than 3 composition rows in the panel,
        expand to match. Always ends with INERT MATERIALS.
        """
        # Ask the Labeler to run OCR and tell us how many rows it found
        ocr_row_count = 3
        if hasattr(self, "_on_autofill_query_count") and self._on_autofill_query_count:
            try:
                ocr_row_count = int(self._on_autofill_query_count() or 3)
            except Exception:
                ocr_row_count = 3
        # At least 3, cap at 8
        n_rows = max(3, min(8, ocr_row_count))

        self._clear_rows()
        mineral = self.mineral.currentText().strip()
        # Rows 1..n-1 use the mineral name; last row is INERT MATERIALS
        first_row = None
        for i in range(n_rows - 1):
            r = self._add_row(); r.name.setEditText(mineral)
            if first_row is None:
                first_row = r
        last = self._add_row(); last.name.setEditText("INERT MATERIALS")
        if first_row:
            first_row.pct.setFocus()
        # Signal parent to fill in OCR numeric values
        if hasattr(self, "_on_autofill_cb") and self._on_autofill_cb:
            self._on_autofill_cb()

    def _on_mineral_changed(self, text: str):
        """When the user updates the main mineral, propagate to any
        composition rows that still have the old default placeholder
        or whose name matches the previous mineral. Skip rows whose
        material was explicitly set to something different."""
        # Only auto-update rows whose name is empty
        for r in self._rows:
            if not r.name.currentText().strip():
                r.name.setEditText(text)

    # Data ──────────────────────────────────────────────────────────
    def to_data(self) -> dict:
        comp = [r.to_dict() for r in self._rows]
        # Keep only rows with at least one field filled
        comp = [c for c in comp if any(c.values())]
        return {
            "schema": "region1",
            "mineral":     self.mineral.currentText().strip(),
            "mass":        self.mass.text().strip(),
            "resistance":  self.resistance.text().strip(),
            "instability": self.instability.text().strip(),
            "difficulty":  self.difficulty.currentText(),
            "scu":         self.scu.text().strip(),
            "composition": comp,
        }

    def from_data(self, data: dict):
        self.mineral.setEditText(data.get("mineral", ""))
        self.mass.setText(data.get("mass", ""))
        self.resistance.setText(data.get("resistance", ""))
        self.instability.setText(data.get("instability", ""))
        idx = self.difficulty.findText(data.get("difficulty", ""))
        self.difficulty.setCurrentIndex(idx if idx >= 0 else 0)
        self.scu.setText(data.get("scu", ""))
        # Composition: handle both new (list of dicts) and old (string) formats
        self._clear_rows()
        comp = data.get("composition", [])
        if isinstance(comp, str):
            # Migrate old free-form text to rows
            for line in comp.splitlines():
                parts = line.split()
                if not parts:
                    continue
                pct = parts[0] if parts[0].replace(".", "").replace("%", "").isdigit() else ""
                count = parts[-1] if parts[-1].isdigit() else ""
                name = " ".join(parts[1:-1]) if pct and count else " ".join(parts)
                self._add_row({"pct": pct, "name": name, "count": count})
        else:
            for c in comp:
                self._add_row(c)

    def clear(self):
        self.mineral.setEditText("")
        self.mass.clear(); self.resistance.clear()
        self.instability.clear(); self.difficulty.setCurrentIndex(0)
        self.scu.clear()
        self._clear_rows()

    def first_focus(self):
        self.mineral.setFocus()


class Region2Form(QWidget):
    """Small numeric readout: value + free-form 'kind' label."""

    def __init__(self):
        super().__init__()
        layout = QFormLayout(self)
        self.value = QLineEdit()
        self.value.setPlaceholderText("e.g. 17020  or  17,020")
        self.kind = QLineEdit()
        self.kind.setPlaceholderText("what is this number? (payout, distance, quantity...)")
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("optional notes")
        layout.addRow("Value:", self.value)
        layout.addRow("Kind:",  self.kind)
        layout.addRow("Notes:", self.notes)

    def to_data(self) -> dict:
        return {
            "schema": "region2",
            "value": self.value.text().strip(),
            "kind":  self.kind.text().strip(),
            "notes": self.notes.text().strip(),
        }

    def from_data(self, data: dict):
        self.value.setText(data.get("value", ""))
        self.kind.setText(data.get("kind", ""))
        self.notes.setText(data.get("notes", ""))

    def clear(self):
        self.value.clear(); self.kind.clear(); self.notes.clear()

    def first_focus(self):
        self.value.setFocus()


# ─── Main window ───────────────────────────────────────────────────
# Target counts for classifier reliability (per class)
TRAIN_TARGETS = {
    "minimum": 20,    # barely usable
    "reliable": 100,  # solid accuracy
    "production": 500,  # diminishing returns beyond this
}

# Glyph classes in display order
ALL_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "%"]
FOLDER_MAP = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    ".": "dot", "%": "pct",
}


class TrainingProgressHUD(QWidget):
    """Per-class progress bars showing how many glyphs have been collected
    and how many more are needed to hit 'reliable' (100 per class).
    """

    def __init__(self, root_dir: Path, target: int = TRAIN_TARGETS["reliable"]):
        super().__init__()
        self._root = root_dir
        self._target = target
        self._bars: dict[str, QProgressBar] = {}
        self._counts: dict[str, QLabel] = {}

        layout = QFormLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        for ch in ALL_CLASSES:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            bar = QProgressBar()
            bar.setRange(0, target)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(14)
            cnt_lbl = QLabel("0 / 100")
            cnt_lbl.setFixedWidth(70)
            cnt_lbl.setStyleSheet("font-size: 10px; color: #666;")
            h.addWidget(bar, 1)
            h.addWidget(cnt_lbl)
            self._bars[ch] = bar
            self._counts[ch] = cnt_lbl
            layout.addRow(f"  {ch}  ", row)

        self.setStyleSheet("""
            QProgressBar { border: 1px solid #bbb; background: #eee; }
            QProgressBar::chunk { background-color: #3a7; }
        """)
        self._total_label = QLabel()
        f = self._total_label.font(); f.setBold(True); self._total_label.setFont(f)
        layout.addRow(" ", self._total_label)
        self.refresh()

    def refresh(self, pending: int = 0):
        """Re-read disk and update bars.
        Shows total count AND unique-slide count per class (files from
        the same source panel all count — only the SOURCE name is
        deduped for the unique count).
        """
        import re as _re
        total = 0
        unique_total = 0
        missing_classes = []
        classes_hit = 0
        for ch in ALL_CLASSES:
            folder = FOLDER_MAP[ch]
            d = self._root / folder
            n_total = 0
            unique_srcs: set[str] = set()
            if d.is_dir():
                for f in d.glob("*.png"):
                    n_total += 1
                    # Strip trailing "_N.png" and a "_pct"/"_cnt"/"_scu"
                    # suffix to get the source panel name.
                    stem = f.stem
                    stem = _re.sub(r"_(pct|cnt|scu)_\d+$", "", stem)
                    stem = _re.sub(r"_\d+$", "", stem)
                    unique_srcs.add(stem)
            n_uniq = len(unique_srcs)
            total += n_total
            unique_total += n_uniq
            if n_total == 0:
                missing_classes.append(ch)
            elif n_total >= self._target:
                classes_hit += 1
            bar = self._bars[ch]
            bar.setValue(min(n_total, self._target))
            # Color by level
            if n_total >= self._target:
                color = "#2a9"
            elif n_total >= TRAIN_TARGETS["minimum"]:
                color = "#e80"
            elif n_total > 0:
                color = "#c33"
            else:
                color = "#888"
            bar.setStyleSheet(f"""
                QProgressBar {{ border: 1px solid #bbb; background: #eee; }}
                QProgressBar::chunk {{ background-color: {color}; }}
            """)
            # Show "total (unique)" so user sees same-slide duplicates counted
            if n_total >= self._target:
                self._counts[ch].setText(f"{n_total} ({n_uniq}) ✓")
                self._counts[ch].setStyleSheet("font-size: 10px; color: #2a9;")
            else:
                self._counts[ch].setText(f"{n_total} ({n_uniq}) / {self._target}")
                self._counts[ch].setStyleSheet("font-size: 10px; color: #666;")

        miss_txt = f"missing: {', '.join(missing_classes)}" if missing_classes else "all classes present"
        pending_txt = f"   |   ⏳ {pending} extracting" if pending else ""
        self._total_label.setText(
            f"Total: {total} ({unique_total} unique-slide) glyphs   |   "
            f"{classes_hit}/{len(ALL_CLASSES)} reliable   |   {miss_txt}{pending_txt}"
        )


class Labeler(QWidget):
    def __init__(self, images: list[Path]):
        super().__init__()
        self.setWindowTitle("Capture Labeler")
        self.resize(1100, 780)
        # Master image list (every region). Visible/active list is
        # filtered through ``_region_filter`` below so the user can
        # focus on signature captures without tearing through 200+
        # HUD ones first.
        self._all_images: list[Path] = images
        self._region_filter: str = "all"  # "all" | "region1" | "region2"
        self._images = self._apply_filter()
        self._idx = self._first_unlabeled()

        self.r1_form = Region1Form()
        self.r2_form = Region2Form()
        # Wire up the autofill callback so the "Auto-fill 3 rows" button
        # also pulls numeric values from OCR, and so it can ask OCR
        # how many composition rows the panel actually has.
        self.r1_form._on_autofill_cb = lambda: self._fill_comp_numbers_from_ocr()
        self.r1_form._on_autofill_query_count = lambda: self._ocr_row_count()

        # Training-data counters (auto-populated from disk on launch)
        self._saves_since_train = 0
        self._train_proc: Optional[subprocess.Popen] = None
        self._train_started_at: Optional[datetime] = None

        self._build_ui()
        # Defer initial load/OCR until after the window is shown.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(50, self._refresh_glyph_counts)
        QTimer.singleShot(100, self._show_current)
        # Periodic poller — updates training status + HUD every 2s
        # so the UI reflects background extraction/training progress.
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._periodic_poll)
        self._poll_timer.start(2000)
        # Track extract subprocesses
        self._extract_procs: list[subprocess.Popen] = []

    # UI ────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Left: image preview + status + nav
        self._image_label = QLabel("(no image)")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(QSize(420, 420))
        self._image_label.setStyleSheet("background:#111; color:#888;")
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._path_label = QLabel("")
        self._path_label.setStyleSheet("color:#666; font-size:10px;")
        self._path_label.setWordWrap(True)

        self._progress = QProgressBar()
        self._progress.setRange(0, max(1, len(self._images)))

        # OCR pre-fill controls — assume available; lazy-check on first use
        ocr_row = QHBoxLayout()
        self._ocr_auto = QCheckBox("Auto-fill from OCR")
        self._ocr_auto.setChecked(True)
        rerun_btn = QPushButton("Re-run OCR")
        rerun_btn.clicked.connect(self._rerun_ocr)
        ocr_row.addWidget(self._ocr_auto)
        ocr_row.addWidget(rerun_btn)
        ocr_row.addStretch(1)

        # Region filter — lets the user focus on JUST the signature
        # scanner captures (region2) or JUST the HUD panel ones
        # (region1) instead of marching through both in capture order.
        # Each option shows the unlabeled-count so you know whether
        # there's anything left to do in that bucket before you switch.
        from PySide6.QtWidgets import QComboBox  # local import keeps top of file clean
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Show:"))
        self._region_filter_combo = QComboBox()
        self._region_filter_combo.addItem("All regions", "all")
        self._region_filter_combo.addItem("Region 1 only — HUD", "region1")
        self._region_filter_combo.addItem("Region 2 only — Signature", "region2")
        self._region_filter_combo.currentIndexChanged.connect(
            self._on_region_filter_changed
        )
        filter_row.addWidget(self._region_filter_combo, 1)
        self._filter_status_lbl = QLabel("")
        self._filter_status_lbl.setStyleSheet("color:#666; font-size:10px;")
        filter_row.addWidget(self._filter_status_lbl)

        nav_row = QHBoxLayout()
        prev_btn = QPushButton("◀ Prev (PgUp)")
        next_btn = QPushButton("Skip ▶ (PgDn)")
        save_btn = QPushButton("Save + Next (Ctrl+Enter)")
        save_btn.setStyleSheet("background:#2a6; color:white; padding:6px;")
        skip_unlabeled_btn = QPushButton("→ Next unlabeled")
        prev_btn.clicked.connect(lambda: self._step(-1))
        next_btn.clicked.connect(lambda: self._step(+1))
        save_btn.clicked.connect(self._save_and_advance)
        skip_unlabeled_btn.clicked.connect(self._jump_to_next_unlabeled)
        nav_row.addWidget(prev_btn); nav_row.addWidget(next_btn)
        nav_row.addWidget(skip_unlabeled_btn); nav_row.addWidget(save_btn)

        delete_btn = QPushButton("🗑 Delete this capture")
        delete_btn.setStyleSheet("color:#a33;")
        delete_btn.clicked.connect(self._delete_current)

        # Training panel
        train_box = QGroupBox("Training (live)")
        tb = QVBoxLayout(train_box)
        self._train_auto = QCheckBox("Auto-train digit classifier every")
        self._train_auto.setChecked(False)  # off by default — use manual button
        self._train_auto.setEnabled(_EXTRACTOR_OK)
        tb_row = QHBoxLayout()
        tb_row.addWidget(self._train_auto)
        self._train_every = QSpinBox()
        self._train_every.setRange(1, 500)
        self._train_every.setValue(50)
        self._train_every.setSuffix(" labels")
        tb_row.addWidget(self._train_every)
        tb.addLayout(tb_row)
        self._glyph_label = QLabel("Glyphs: panel=0 / signature=0")
        tb.addWidget(self._glyph_label)
        self._train_status = QLabel("Training: idle")
        self._train_status.setStyleSheet("color:#26a;")
        tb.addWidget(self._train_status)
        train_now_btn = QPushButton("Train now (KNN, <1s)")
        train_now_btn.clicked.connect(self._start_training)
        train_now_btn.setEnabled(_EXTRACTOR_OK)
        tb.addWidget(train_now_btn)

        # Training progress HUD — per-class bars showing how close
        # each digit is to "reliable" (100 samples).
        hud_label = QLabel("Training progress (target: 100 per class)")
        f = hud_label.font(); f.setBold(True); hud_label.setFont(f)
        tb.addWidget(hud_label)
        self._train_hud = TrainingProgressHUD(PANEL_GLYPH_ROOT)
        tb.addWidget(self._train_hud)

        left = QVBoxLayout()
        left.addWidget(self._image_label, 1)
        left.addWidget(self._path_label)
        left.addWidget(self._progress)
        left.addLayout(ocr_row)
        left.addLayout(filter_row)
        left.addLayout(nav_row)
        left.addWidget(delete_btn)
        left.addWidget(train_box)
        left_widget = QWidget(); left_widget.setLayout(left)

        # Right: form (swapped depending on region)
        self._form_box = QGroupBox("Labels")
        fb = QVBoxLayout(self._form_box)
        self._region_label = QLabel("region: -")
        self._region_label.setStyleSheet("font-weight:bold; color:#26a;")
        fb.addWidget(self._region_label)
        fb.addWidget(self.r1_form)
        fb.addWidget(self.r2_form)

        # Layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self._form_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        root = QVBoxLayout(self)
        root.addWidget(splitter)

        # Shortcuts
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._save_and_advance)
        QShortcut(QKeySequence("Ctrl+S"),      self, activated=self._save_and_advance)
        QShortcut(QKeySequence("PgDown"),      self, activated=lambda: self._step(+1))
        QShortcut(QKeySequence("PgUp"),        self, activated=lambda: self._step(-1))
        QShortcut(QKeySequence("Esc"),         self, activated=self.close)

    # State ─────────────────────────────────────────────────────────
    def _first_unlabeled(self) -> int:
        for i, p in enumerate(self._images):
            if not _label_path(p).is_file():
                return i
        return 0

    # ── Region filter ──────────────────────────────────────────────
    def _apply_filter(self) -> list[Path]:
        """Return the subset of ``self._all_images`` that matches the
        active region filter. The visible navigation list (
        ``self._images``) is always this filtered view."""
        if self._region_filter == "all":
            return list(self._all_images)
        return [p for p in self._all_images if _region_of(p) == self._region_filter]

    def _on_region_filter_changed(self, _idx: int) -> None:
        """User picked a different filter — preserve the currently
        viewed image if it's still in the filtered set, otherwise jump
        to the first unlabeled in the new view."""
        new_filter = self._region_filter_combo.currentData()
        if new_filter == self._region_filter:
            return
        prev_path = (
            self._images[self._idx]
            if self._images and 0 <= self._idx < len(self._images) else None
        )
        self._region_filter = new_filter
        self._images = self._apply_filter()

        # Try to keep the user on the same image if visible.
        new_idx = -1
        if prev_path is not None:
            try:
                new_idx = self._images.index(prev_path)
            except ValueError:
                new_idx = -1
        if new_idx < 0:
            new_idx = self._first_unlabeled()
        self._idx = max(0, new_idx)

        # Resize the progress bar to the new visible count.
        self._progress.setRange(0, max(1, len(self._images)))
        self._update_filter_status()
        self._show_current()
        self._refresh_glyph_counts()

    def _update_filter_status(self) -> None:
        """Show 'Nx in view (M unlabeled)' next to the dropdown so the
        user always sees what they're looking at."""
        n = len(self._images)
        unlabeled = sum(
            1 for q in self._images if not _label_path(q).is_file()
        )
        # Also show the OTHER bucket's unlabeled-count so the user
        # knows whether switching would actually help.
        if self._region_filter == "all":
            r1 = sum(1 for p in self._all_images
                     if _region_of(p) == "region1"
                     and not _label_path(p).is_file())
            r2 = sum(1 for p in self._all_images
                     if _region_of(p) == "region2"
                     and not _label_path(p).is_file())
            self._filter_status_lbl.setText(
                f"{n} in view  |  {unlabeled} unlabeled  "
                f"(R1: {r1} unlabeled, R2: {r2} unlabeled)"
            )
        else:
            other = "region2" if self._region_filter == "region1" else "region1"
            other_short = "R2" if other == "region2" else "R1"
            other_unlab = sum(1 for p in self._all_images
                              if _region_of(p) == other
                              and not _label_path(p).is_file())
            self._filter_status_lbl.setText(
                f"{n} in view  |  {unlabeled} unlabeled  "
                f"({other_short}: {other_unlab} unlabeled)"
            )

    def _step(self, delta: int):
        new = self._idx + delta
        if 0 <= new < len(self._images):
            self._idx = new
            self._show_current()
            # Also refresh HUD on nav so user sees latest counts
            self._refresh_glyph_counts()

    def _jump_to_next_unlabeled(self):
        for i in range(self._idx + 1, len(self._images)):
            if not _label_path(self._images[i]).is_file():
                self._idx = i
                self._show_current()
                return
        # Wrap from start
        for i in range(0, self._idx):
            if not _label_path(self._images[i]).is_file():
                self._idx = i
                self._show_current()
                return
        self._status("all images labeled — nothing to jump to")

    def _show_current(self):
        if not self._images:
            self._image_label.setText("(no images found)")
            return
        p = self._images[self._idx]
        # Image
        try:
            img = Image.open(p).convert("RGB")
        except Exception as exc:
            self._image_label.setText(f"failed to open: {exc}")
            return
        # Scale to fit while keeping aspect
        max_w = self._image_label.width()
        max_h = self._image_label.height()
        ratio = min(max_w / img.width, max_h / img.height, 4.0)
        if ratio > 1:
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.NEAREST)
        elif ratio < 1:
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        qim = ImageQt(img)
        self._image_label.setPixmap(QPixmap.fromImage(qim))

        # Path + progress
        rel = p.relative_to(PANELS_ROOT)
        labeled = _label_path(p).is_file()
        tag = "  ✓ labeled" if labeled else "  ⋯ unlabeled"
        self._path_label.setText(f"[{self._idx+1}/{len(self._images)}]  {rel}{tag}")
        n_labeled = sum(1 for q in self._images if _label_path(q).is_file())
        self._progress.setValue(n_labeled)
        self._progress.setFormat(f"{n_labeled} / {len(self._images)} labeled")
        # Also keep the filter status line fresh — counts shift every
        # time a label is saved.
        self._update_filter_status()

        # Region-specific form
        region = _region_of(p)
        self._region_label.setText(f"region: {region}")
        existing = _load_label(p)
        _log(f"loading {p.name} region={region} existing_keys={list(existing.keys())}")
        if "composition" in existing:
            _log(f"  saved comp type={type(existing['composition']).__name__} "
                 f"len={len(existing['composition'])}")
        # Run OCR and MERGE into existing — fill empty fields but
        # never overwrite what was already saved.
        ocr_filled = False
        if self._ocr_auto.isChecked() and _ensure_tesseract():
            _log("  running OCR...")
            ocr = _ocr_prefill(p)
            _log(f"  OCR returned keys={list(ocr.keys()) if ocr else []}")
            if "composition" in (ocr or {}):
                _log(f"  OCR comp rows={len(ocr['composition'])}: {ocr['composition']}")
            if ocr:
                merged = dict(existing)  # start from saved data
                for k, v in ocr.items():
                    if k == "composition":
                        cur_comp = existing.get("composition") or []
                        ocr_comp = v or []
                        if isinstance(cur_comp, str):
                            cur_comp = []  # migrate old format
                        if not cur_comp:
                            merged["composition"] = ocr_comp
                        else:
                            # Fill empty fields in saved rows from OCR.
                            # DON'T extend with extra OCR rows — saved
                            # row count is authoritative.
                            new = []
                            for i, row in enumerate(cur_comp):
                                o = ocr_comp[i] if i < len(ocr_comp) else {}
                                new.append({
                                    "pct":   row.get("pct")   or o.get("pct", ""),
                                    "name":  row.get("name")  or o.get("name", ""),
                                    "count": row.get("count") or o.get("count", ""),
                                })
                            merged["composition"] = new
                    elif not existing.get(k):
                        merged[k] = v
                merged["schema"] = region
                existing = merged
                ocr_filled = True
        _log(f"  final existing.composition = {existing.get('composition', 'NONE')}")
        self.r1_form.clear(); self.r2_form.clear()
        if region == "region1":
            self.r1_form.setVisible(True); self.r2_form.setVisible(False)
            self.r1_form.from_data(existing)
            _log(f"  after from_data, rows={len(self.r1_form._rows)}")
            for i, row in enumerate(self.r1_form._rows):
                _log(f"    row {i}: pct={row.pct.text()!r} name={row.name.currentText()!r} count={row.count.text()!r}")
            self.r1_form.first_focus()
        else:
            self.r1_form.setVisible(False); self.r2_form.setVisible(True)
            self.r2_form.from_data(existing)
            self.r2_form.first_focus()
        if ocr_filled:
            cur = self._path_label.text()
            self._path_label.setText(cur + "  ⚙ OCR merged (verify before saving)")

    # Actions ───────────────────────────────────────────────────────
    def _save_and_advance(self):
        if not self._images:
            return
        p = self._images[self._idx]
        region = _region_of(p)
        data = self.r1_form.to_data() if region == "region1" else self.r2_form.to_data()
        try:
            _save_label(p, data)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        # Spawn glyph extraction for THIS image in a subprocess so
        # it doesn't block the labeler. Extractor writes directly
        # to training_data_user_panel/<class>/ and the periodic
        # poller will refresh the HUD when new glyphs appear.
        python_exe = sys.executable
        if python_exe.lower().endswith("pythonw.exe"):
            python_exe = python_exe[:-len("pythonw.exe")] + "python.exe"
        try:
            proc = subprocess.Popen(
                [python_exe, "-u", "scripts/extract_labeled_glyphs.py",
                 "--path", str(p)],
                cwd=str(_TOOL_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._extract_procs.append(proc)
        except Exception as exc:
            print(f"[labeler] extract spawn failed: {exc}")
        self._status(f"saved → {_label_path(p).name}")
        self._refresh_glyph_counts()
        self._saves_since_train += 1
        # Auto-train trigger
        if (self._train_auto.isChecked()
                and not self._train_active()
                and self._saves_since_train >= self._train_every.value()):
            self._start_training(auto=True)
        # Poll training status (cheap)
        self._poll_training()
        # Advance to NEXT sequential image (not next unlabeled) —
        # user wants to review/correct every slide.
        self._step(+1)

    # Training ─────────────────────────────────────────────────────
    def _refresh_glyph_counts(self):
        if not _EXTRACTOR_OK:
            self._glyph_label.setText("Glyph extraction unavailable")
            return
        def count(root: Path) -> int:
            if not root.is_dir():
                return 0
            return sum(
                len(list(cls_dir.glob("*.png")))
                for cls_dir in root.iterdir() if cls_dir.is_dir()
            )
        panel_n = count(PANEL_GLYPH_ROOT)
        sig_n   = count(SIG_GLYPH_ROOT)
        next_at = self._train_every.value() - self._saves_since_train
        hint = ""
        if self._train_auto.isChecked():
            hint = f"  |  auto-train in {max(0, next_at)} labels"
        self._glyph_label.setText(f"Glyphs: panel={panel_n} / signature={sig_n}{hint}")
        # Refresh per-class HUD with current pending-extraction count
        if hasattr(self, "_train_hud"):
            pending = len([p for p in getattr(self, "_extract_procs", []) if p.poll() is None])
            self._train_hud.refresh(pending=pending)

    def _train_active(self) -> bool:
        return self._train_proc is not None and self._train_proc.poll() is None

    def _start_training(self, auto: bool = False):
        if self._train_active():
            self._train_status.setText("Training: already running")
            return
        # Write glyphs to training_data/ (where ocr/train_model.py reads)
        # We symlink-or-copy our training_data_user_panel into training_data
        # before launching the trainer, or just point the trainer at our path.
        # Simplest: the trainer supports a fixed path, so we TEMPORARILY
        # copy our user_panel glyphs into training_data/ via symlink. To
        # keep it simple and portable, we instead pass through the env var
        # the trainer now respects (patched alongside this).
        env = dict(**__import__("os").environ)
        env["OCR_TRAIN_SOURCE"] = str(PANEL_GLYPH_ROOT)
        self._train_status.setText("Training: starting...")
        # Prefer Python 3.13 (with torch) for CNN training; fall back
        # to current Python (3.14, KNN-only) if 3.13 not present.
        import os as _os
        py313 = _os.path.join(
            _os.environ.get("LOCALAPPDATA", ""),
            "Programs", "Python", "Python313", "python.exe",
        )
        if py313 and _os.path.isfile(py313):
            python_exe = py313
            trainer_module = "ocr.train_torch"  # CNN trainer
        else:
            python_exe = sys.executable
            if python_exe.lower().endswith("pythonw.exe"):
                python_exe = python_exe[:-len("pythonw.exe")] + "python.exe"
            trainer_module = "ocr.train_sklearn"  # KNN fallback
        self._train_log_path = _TOOL_DIR / "training.log"
        try:
            self._train_log_fh = open(self._train_log_path, "w", encoding="utf-8")
            self._train_proc = subprocess.Popen(
                [python_exe, "-u", "-m", trainer_module],
                cwd=str(_TOOL_DIR),
                env=env,
                stdout=self._train_log_fh,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:
            self._train_status.setText(f"Training: failed to start — {exc}")
            self._train_proc = None
            return
        self._train_started_at = datetime.now()
        self._saves_since_train = 0
        tag = " (auto)" if auto else ""
        self._train_status.setText(
            f"Training{tag}: running... PID {self._train_proc.pid}"
        )
        self._refresh_glyph_counts()

    def _poll_training(self):
        if self._train_proc is None:
            return
        ret = self._train_proc.poll()
        if ret is None:
            return  # still running
        # Finished — close log file and read it
        try:
            if hasattr(self, "_train_log_fh") and self._train_log_fh:
                self._train_log_fh.close()
                self._train_log_fh = None
        except Exception:
            pass
        out = ""
        try:
            out = self._train_log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            out = ""
        self._train_proc = None
        dt = (datetime.now() - self._train_started_at).total_seconds() if self._train_started_at else 0
        # Find best val acc in output
        m = re.search(r"Best validation accuracy:\s*([\d.]+)%", out)
        acc = m.group(1) + "%" if m else "?"
        if ret == 0:
            self._train_status.setText(
                f"Training: done in {dt:.0f}s — best val_acc={acc} "
                f"(model: ocr/models/model_cnn_finetuned.onnx)"
            )
        else:
            tail = "\n".join(out.splitlines()[-6:]) or "(no output)"
            self._train_status.setText(
                f"Training: failed (exit {ret}).  {tail}"
            )

    def _periodic_poll(self):
        """Runs every 2s. Refreshes HUD (picks up newly-extracted
        glyphs from subprocesses) and polls training status."""
        # Reap finished extraction procs
        self._extract_procs = [p for p in self._extract_procs if p.poll() is None]
        # Refresh glyph counts (picks up anything subprocess produced)
        self._refresh_glyph_counts()
        # Poll training
        self._poll_training()

    def _ocr_row_count(self) -> int:
        """Return how many composition rows OCR finds on the current
        image. Used by Auto-fill 3 rows to decide whether to expand
        beyond the default 3."""
        if not self._images or not _ensure_tesseract():
            return 3
        p = self._images[self._idx]
        try:
            ocr = _ocr_prefill(p)
        except Exception:
            return 3
        comp = ocr.get("composition", []) if ocr else []
        return max(3, len(comp))

    def _fill_comp_numbers_from_ocr(self):
        """Run OCR on current image and fill pct/count for the
        composition rows that have them empty. Doesn't touch names."""
        if not self._images:
            return
        if not _ensure_tesseract():
            print("[labeler] tesseract not available")
            self._status("⚠ Tesseract not available")
            return
        p = self._images[self._idx]
        if _region_of(p) != "region1":
            return
        try:
            ocr = _ocr_prefill(p)
        except Exception as exc:
            print(f"[labeler] OCR error: {exc}")
            self._status(f"⚠ OCR error: {exc}")
            return
        ocr_comp = ocr.get("composition") or []
        print(f"[labeler] OCR found {len(ocr_comp)} composition rows")
        if not ocr_comp:
            self._status("OCR found no composition rows — try a different image")
            return
        filled = 0
        for i, row in enumerate(self.r1_form._rows):
            if i >= len(ocr_comp):
                break
            o = ocr_comp[i]
            if not row.pct.text().strip() and o.get("pct"):
                row.pct.setText(o["pct"])
                filled += 1
            if not row.count.text().strip() and o.get("count"):
                row.count.setText(o["count"])
                filled += 1
        self._status(f"OCR filled {filled} composition numbers")

    def _rerun_ocr(self):
        """Run OCR and MERGE results with current form values — only
        fill empty fields, keep what the user already typed."""
        if not self._images or not _ensure_tesseract():
            return
        p = self._images[self._idx]
        ocr = _ocr_prefill(p)
        if not ocr:
            self._status("OCR returned nothing")
            return
        region = _region_of(p)
        if region == "region1":
            # Merge with current form values
            current = self.r1_form.to_data()
            merged = dict(current)
            for k, v in ocr.items():
                if k == "composition":
                    # For composition: if current has rows, fill empty
                    # fields in each row from the OCR result; else use
                    # OCR wholesale.
                    cur_comp = current.get("composition") or []
                    ocr_comp = v or []
                    if not cur_comp:
                        merged["composition"] = ocr_comp
                    else:
                        new = []
                        for i, row in enumerate(cur_comp):
                            o = ocr_comp[i] if i < len(ocr_comp) else {}
                            new.append({
                                "pct":   row.get("pct")   or o.get("pct", ""),
                                "name":  row.get("name")  or o.get("name", ""),
                                "count": row.get("count") or o.get("count", ""),
                            })
                        # Extend with extra OCR rows if any
                        new.extend(ocr_comp[len(cur_comp):])
                        merged["composition"] = new
                else:
                    # Scalar: only fill if currently empty
                    if not current.get(k):
                        merged[k] = v
            merged["schema"] = "region1"
            self.r1_form.clear(); self.r1_form.from_data(merged)
        else:
            current = self.r2_form.to_data()
            merged = dict(current)
            for k, v in ocr.items():
                if not current.get(k):
                    merged[k] = v
            merged["schema"] = "region2"
            self.r2_form.clear(); self.r2_form.from_data(merged)
        self._status(f"OCR merged — filled empty fields")

    def _delete_current(self):
        if not self._images:
            return
        p = self._images[self._idx]
        ok = QMessageBox.question(
            self, "Delete capture?",
            f"Delete this image and its label?\n{p.name}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            return
        try:
            p.unlink(missing_ok=True)
            _label_path(p).unlink(missing_ok=True)
        except Exception as exc:
            QMessageBox.warning(self, "Delete failed", str(exc))
            return
        del self._images[self._idx]
        if self._idx >= len(self._images):
            self._idx = max(0, len(self._images) - 1)
        self._show_current()

    def _status(self, msg: str):
        self._path_label.setText(self._path_label.text().split("\n")[0] + f"\n  {msg}")

    # Resize handler — re-fit image (debounced via timer so we don't
    # block the initial show() which fires a resize synchronously).
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # Skip — image rescale on resize is a nice-to-have; if needed
        # it can be wired via QTimer to avoid blocking the first show.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", help="Specific user_* folder to label (default: all)")
    args = p.parse_args()

    images = _scan_user_images(args.folder)
    if not images:
        print("No user capture images found under training_data_panels/")
        sys.exit(0)

    # High-DPI — enable before QApplication is created
    import os as _os
    _os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    print("[labeler] creating QApplication...", flush=True)
    app = QApplication(sys.argv)
    print(f"[labeler] creating Labeler ({len(images)} images)...", flush=True)
    win = Labeler(images)
    # Force window to top-left of primary screen — Qt defaults to a
    # position that may be on a secondary/virtual monitor.
    primary = app.primaryScreen().availableGeometry()
    win.resize(min(1100, primary.width() - 100), min(780, primary.height() - 100))
    win.move(primary.left() + 50, primary.top() + 50)
    print(f"[labeler] forcing pos to {primary.left()+50},{primary.top()+50}", flush=True)
    win.show()
    win.raise_()
    win.activateWindow()
    print(f"[labeler] after show, visible={win.isVisible()} geom={win.geometry()}", flush=True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
