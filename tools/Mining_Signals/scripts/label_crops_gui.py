"""Tkinter GUI labeler for mining-value crops.

Shows one crop at a time (upscaled 6×), with a text field for you
to type the value you see. Tab/Enter moves to next, keyboard
shortcuts for common actions.

Sources it pulls crops from (in priority order):
  1. training_data_crnn/pending/        (live scans, high value)
  2. training_data_crnn/yt_*.png entries where manifest label is
     non-numeric (Paddle-mislabeled, many have real values if you
     squint)

Saves labeled crops into the main corpus:
  * Correct label → moved/renamed into training_data_crnn/ with
    filename encoding `yt_user_<ts>_<i>_<safelabel>.png`
  * Manifest is updated in place (append mode — no race because
    only this labeler writes)
  * Crops you mark as ``garbage`` go into training_data_crnn/rejected/
    (preserves them in case you change your mind)

Keyboard shortcuts:
  Enter       Accept the typed label and advance
  Ctrl+Enter  Accept + mark as "perfect" (higher confidence)
  Esc / S     Skip this crop (revisit later)
  Del / G     Mark as garbage (move to rejected/)
  Ctrl+Z      Undo last action (only previous entry)
  Ctrl+Q      Quit (state is saved incrementally)

Usage:
    python scripts/label_crops_gui.py
    python scripts/label_crops_gui.py --mode polluted   # work on yt_*
    python scripts/label_crops_gui.py --mode pending    # default

Label format hints shown in the UI:
  - Mass: plain integer or decimal, e.g. '4878', '12040', '958'
  - Resistance: integer with %, e.g. '8%', '0%', '32%'
  - Instability: decimal, e.g. '4.65', '27.64', '0.76', '12.10'
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from tkinter import (
    Tk, Label, Entry, Frame, StringVar, BOTH, TOP, BOTTOM, LEFT, RIGHT, X, Y,
    PhotoImage, messagebox,
)

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("Pillow required: pip install Pillow")

REPO = Path(__file__).resolve().parent.parent
CROPS_DIR = REPO / "training_data_crnn"
PENDING_DIR = CROPS_DIR / "pending"
REJECTED_DIR = CROPS_DIR / "rejected"
MANIFEST = CROPS_DIR / "manifest.json"
PROGRESS_FILE = CROPS_DIR / ".labeler_progress.json"

_NUMERIC_RE = re.compile(r"^[0-9][0-9.,]*%?$")


def _phash(path: Path) -> str:
    """Tiny perceptual hash — resize to 8×8 grayscale, threshold at
    mean, return 64-bit hex. Near-identical crops collide on this."""
    try:
        img = Image.open(path).convert("L").resize((8, 8), Image.BILINEAR)
    except Exception:
        return str(path)  # degenerate bucket
    import numpy as _np
    arr = _np.asarray(img, dtype=_np.uint8)
    mean = float(arr.mean())
    bits = (arr > mean).astype(_np.uint8).flatten()
    # Pack 64 bits into a hex string
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:016x}"


def _dedupe_by_similarity(paths: list[Path]) -> tuple[list[Path], dict[str, list[Path]]]:
    """Group near-identical crops by perceptual hash. Returns
    (representatives, groups) — one path per group plus the full
    mapping so labels can be propagated to all duplicates on save."""
    groups: dict[str, list[Path]] = {}
    for p in paths:
        h = _phash(p)
        groups.setdefault(h, []).append(p)
    reps = [g[0] for g in groups.values()]
    return reps, groups


def _list_pending() -> list[Path]:
    if not PENDING_DIR.is_dir():
        return []
    return sorted(PENDING_DIR.glob("*.png"))


def _list_polluted_yt() -> list[Path]:
    """YouTube crops whose label is non-numeric — but the image might
    still contain a numeric value worth relabeling.

    Reads from ``manifest_unfiltered_backup.json`` when present
    (because the live manifest has been stripped to numeric-only).
    """
    backup = CROPS_DIR / "manifest_unfiltered_backup.json"
    source = backup if backup.is_file() else MANIFEST
    if not source.is_file():
        return []
    with open(source) as f:
        man = json.load(f)
    out: list[Path] = []
    for e in man.get("files", []):
        lab = e.get("label", "") or ""
        if _NUMERIC_RE.match(lab):
            continue
        path = CROPS_DIR / e["path"]
        if path.is_file():
            out.append(path)
    return out


def _load_progress() -> dict:
    if PROGRESS_FILE.is_file():
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"done": [], "skipped": []}


def _save_progress(state: dict) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _append_manifest_entry(path: str, label: str) -> None:
    """Append one entry to manifest.json (sole writer — no race)."""
    manifest: dict = {"files": []}
    if MANIFEST.is_file():
        try:
            with open(MANIFEST) as f:
                manifest = json.load(f)
        except Exception:
            pass
    files = manifest.setdefault("files", [])
    files.append({"path": path, "label": label, "source": "user_labeled"})
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f)


def _safe_filename(label: str, idx: int) -> str:
    safe = label.replace(".", "dot").replace("%", "pct").replace(",", "c")
    return f"user_{int(time.time())}_{idx:04d}_{safe}.png"


class LabelerApp:
    def __init__(self, root: Tk, crops: list[Path], source_name: str):
        self.root = root
        self.crops = crops
        self.source_name = source_name
        self.progress = _load_progress()
        self._undo_stack: list[dict] = []

        # Dedupe near-identical crops via perceptual hash. training_collector
        # saves every ~5s per field, so staying on one rock for a minute
        # produces ~12 pixel-adjacent crops. Show the user one representative
        # per group; on save, the label propagates to every duplicate.
        reps, self.groups = _dedupe_by_similarity(crops)
        # Keep stable sort order
        reps = sorted(reps, key=lambda p: p.name)
        original_count = len(crops)
        dupes_collapsed = original_count - len(reps)

        # Filter out already-done crops (resume support)
        done_set = set(self.progress.get("done", []))
        self.queue = [c for c in reps if str(c) not in done_set]
        self.pos = 0
        self._dedupe_stats = (original_count, len(reps), dupes_collapsed)

        # UI
        root.title(f"Crop Labeler — {source_name}")
        root.geometry("800x520")
        # root-level bindings (fire when focus isn't on the Entry)
        root.bind("<Escape>", self._on_skip)
        root.bind("<Delete>", self._on_garbage)
        root.bind("<Control-z>", self._on_undo)
        root.bind("<Control-q>", lambda e: root.destroy())

        self.img_label = Label(root, bg="#111")
        self.img_label.pack(side=TOP, pady=10)

        self.meta_var = StringVar()
        Label(root, textvariable=self.meta_var, font=("Consolas", 11)).pack()

        self.hint_var = StringVar()
        Label(root, textvariable=self.hint_var, fg="#444", font=("Consolas", 9)).pack()

        self.entry_var = StringVar()
        self.entry = Entry(root, textvariable=self.entry_var, font=("Consolas", 18), width=24)
        self.entry.pack(pady=10)
        self.entry.focus_set()

        # Entry-level bindings — Tk's Entry widget does NOT propagate
        # <Return> to root, so we bind directly on the Entry. Also
        # handle numpad Enter via <KP_Enter>.
        def _accept(e, perfect: bool = False):
            self._on_enter(e, perfect=perfect)
            return "break"  # stop Tk's default Entry handling

        self.entry.bind("<Return>",              lambda e: _accept(e))
        self.entry.bind("<KP_Enter>",            lambda e: _accept(e))
        self.entry.bind("<Control-Return>",      lambda e: _accept(e, perfect=True))
        self.entry.bind("<Control-KP_Enter>",    lambda e: _accept(e, perfect=True))
        # Allow Esc/Delete from inside the entry too
        self.entry.bind("<Escape>",              lambda e: (self._on_skip(e), "break")[1])

        legend = ("Enter = accept    Ctrl+Enter = perfect    "
                  "Esc/S = skip    Del/G = garbage    Ctrl+Z = undo    Ctrl+Q = quit")
        Label(root, text=legend, fg="#666", font=("Consolas", 8)).pack(side=BOTTOM, pady=5)

        self.count_var = StringVar()
        Label(root, textvariable=self.count_var, fg="#888", font=("Consolas", 9)).pack(side=BOTTOM)

        self._render()

    def _render(self) -> None:
        if self.pos >= len(self.queue):
            messagebox.showinfo("Done", f"No more crops to label in {self.source_name}.")
            self.root.destroy()
            return

        crop_path = self.queue[self.pos]
        try:
            img = Image.open(crop_path)
        except Exception:
            self.pos += 1
            self._render()
            return

        # Upscale to ~400 px tall for inspection
        target_h = 200
        w, h = img.size
        scale = max(2, target_h // max(1, h))
        big = img.resize((w * scale, h * scale), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(big)
        self.img_label.config(image=self._tk_img)

        # Metadata: field hint from filename
        name_lower = crop_path.name.lower()
        hint = "mining value"
        if "mass" in name_lower:
            hint = "MASS (integer, e.g. '4878', '958')"
        elif "resist" in name_lower:
            hint = "RESISTANCE (e.g. '8%', '0%')"
        elif "instab" in name_lower:
            hint = "INSTABILITY (decimal, e.g. '4.65', '27.64')"
        self.meta_var.set(f"{crop_path.name}")
        self.hint_var.set(f"Field: {hint}")

        # Pre-fill with any Paddle guess from filename suffix
        guess = ""
        m = re.search(r"_([0-9][0-9dotpctc]*)\.png$", crop_path.name)
        if m:
            g = m.group(1).replace("dot", ".").replace("pct", "%").replace("c", ",")
            if _NUMERIC_RE.match(g):
                guess = g
        self.entry_var.set(guess)
        self.entry.select_range(0, "end")
        self.entry.focus_set()

        # Show dedupe info so user knows one click = many saves
        group_size = 1
        for g in self.groups.values():
            if crop_path in g:
                group_size = len(g)
                break
        orig, deduped, collapsed = self._dedupe_stats
        self.count_var.set(
            f"{self.pos + 1} / {len(self.queue)}   "
            f"(1 save → {group_size} crops)   "
            f"[{collapsed} duplicates collapsed from {orig} total]"
        )

    def _on_enter(self, event, perfect: bool = False) -> None:
        label = self.entry_var.get().strip()
        if not label:
            return
        if not _NUMERIC_RE.match(label):
            if not messagebox.askyesno(
                "Non-numeric label",
                f"'{label}' isn't in numeric format. Save anyway?"
            ):
                return

        crop_path = self.queue[self.pos]
        # Expand the representative to its full dedupe group so the
        # label propagates to all near-identical crops.
        group_members = []
        for g in self.groups.values():
            if crop_path in g:
                group_members = g
                break
        if not group_members:
            group_members = [crop_path]

        saved_paths = []
        for idx, member in enumerate(group_members):
            new_name = _safe_filename(label, self.pos * 100 + idx)
            dst = CROPS_DIR / new_name
            try:
                if member.parent == CROPS_DIR:
                    shutil.copy(str(member), str(dst))
                else:
                    shutil.move(str(member), str(dst))
                _append_manifest_entry(new_name, label)
                saved_paths.append(str(member))
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc))
                return

        done = self.progress.setdefault("done", [])
        for s in saved_paths:
            done.append(s)
        _save_progress(self.progress)
        self._undo_stack.append({
            "action": "saved", "src": str(crop_path),
            "group_size": len(group_members), "label": label,
        })
        self.pos += 1
        self._render()

    def _group_for(self, crop_path: Path) -> list[Path]:
        for g in self.groups.values():
            if crop_path in g:
                return g
        return [crop_path]

    def _on_skip(self, event=None) -> None:
        crop_path = self.queue[self.pos]
        # Skipping only marks the representative — keeps all duplicates
        # queued for when user returns (resume).
        self.progress.setdefault("skipped", []).append(str(crop_path))
        _save_progress(self.progress)
        self._undo_stack.append({"action": "skipped", "src": str(crop_path)})
        self.pos += 1
        self._render()

    def _on_garbage(self, event=None) -> None:
        crop_path = self.queue[self.pos]
        REJECTED_DIR.mkdir(exist_ok=True)
        # Propagate garbage decision to all duplicates
        for member in self._group_for(crop_path):
            dst = REJECTED_DIR / member.name
            try:
                if member.parent == CROPS_DIR:
                    shutil.copy(str(member), str(dst))
                else:
                    shutil.move(str(member), str(dst))
            except Exception:
                pass
            self.progress.setdefault("done", []).append(str(member))
        _save_progress(self.progress)
        self._undo_stack.append({"action": "garbage", "src": str(crop_path)})
        self.pos += 1
        self._render()

    def _on_undo(self, event=None) -> None:
        if not self._undo_stack or self.pos == 0:
            return
        last = self._undo_stack.pop()
        # Move back one position
        self.pos = max(0, self.pos - 1)
        # Remove from progress done list if applicable
        done = self.progress.get("done", [])
        if last["src"] in done:
            done.remove(last["src"])
        skipped = self.progress.get("skipped", [])
        if last["src"] in skipped:
            skipped.remove(last["src"])
        _save_progress(self.progress)
        self._render()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pending", "polluted"), default="pending",
                    help="pending = live scan buffer; polluted = yt_* "
                    "crops with non-numeric labels (much bigger pool)")
    args = ap.parse_args()

    if args.mode == "pending":
        crops = _list_pending()
        source = "pending (live scans)"
    else:
        crops = _list_polluted_yt()
        source = "polluted YT crops"

    if not crops:
        print(f"No crops to label in {source}.")
        return

    print(f"Found {len(crops)} crops in {source}.")
    root = Tk()
    app = LabelerApp(root, crops, source)
    root.mainloop()


if __name__ == "__main__":
    main()
