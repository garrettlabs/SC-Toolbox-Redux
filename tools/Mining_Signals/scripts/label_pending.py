"""Label pending value crops and fold them into the CRNN training set.

After mining a rock, the running app auto-saves every value crop
(mass/resistance/instability) to ``training_data_crnn/pending/`` via
``ocr.sc_ocr.api::scan_hud_onnx``. This script lets you promote
those crops into labeled training data.

Usage (two modes):

1. Bulk: you know what recent crops are.
     python scripts/label_pending.py mass:367 resistance:0% instability:0.45
     → moves the LATEST pending crop for each field into the training
       set with the specified label. Appends to manifest.json.

2. Pattern: match files containing a substring.
     python scripts/label_pending.py --match "_735" 367 resistance:0%
     → labels any pending files whose name contains the substring.

3. List: show pending crops so you can decide what to label.
     python scripts/label_pending.py --list

After labeling, run:
     python -m ocr.train_crnn --epochs 15 --n 6000 --real-aug 250

The newly-labeled samples are included automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent

CRNN_DIR = TOOL / "training_data_crnn"
PENDING_DIR = CRNN_DIR / "pending"
MANIFEST_PATH = CRNN_DIR / "manifest.json"


def _load_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        return {"files": []}
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"files": []}


def _save_manifest(m: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)


def _list_pending() -> list[Path]:
    if not PENDING_DIR.is_dir():
        return []
    return sorted(
        [p for p in PENDING_DIR.glob("*.png")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _move_latest(field: str, label: str) -> bool:
    """Move the most recent pending crop for `field` → labeled dataset."""
    candidates = [p for p in _list_pending() if p.name.startswith(field + "_")]
    if not candidates:
        print(f"  no pending crops for field={field}", file=sys.stderr)
        return False
    src = candidates[0]
    return _promote(src, label, source="label_pending_latest")


def _move_by_match(pattern: str, label: str) -> int:
    """Move all pending crops whose filename contains `pattern`."""
    n = 0
    for src in _list_pending():
        if pattern in src.name:
            if _promote(src, label, source="label_pending_match"):
                n += 1
    return n


def _promote(src: Path, label: str, source: str) -> bool:
    _allowed = "0123456789.-% ()ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    if not label or any(c not in _allowed for c in label):
        print(f"  bad label {label!r}; must be digits/letters/.-% ()", file=sys.stderr)
        return False
    safe = (label.replace(".", "dot").replace("%", "pct")
                 .replace(" ", "_").replace("(", "-").replace(")", "-"))
    stamp = int(time.time() * 1000) % 1_000_000_000
    dst_name = f"labeled_{stamp}_{safe}.png"
    dst = CRNN_DIR / dst_name
    try:
        shutil.move(str(src), str(dst))
    except OSError as exc:
        print(f"  move failed: {exc}", file=sys.stderr)
        return False
    m = _load_manifest()
    m.setdefault("files", []).append({
        "path": dst_name,
        "label": label,
        "source": source,
    })
    _save_manifest(m)
    print(f"  {src.name} -> {dst_name} (label={label!r})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true",
                        help="List pending crops by recency and exit")
    parser.add_argument("--match", type=str,
                        help="Instead of matching by field-latest, match by filename substring")
    parser.add_argument("assignments", nargs="*",
                        help="<field>:<label> pairs, e.g. mass:367 resistance:0%% instability:0.45")
    args = parser.parse_args()

    if args.list:
        pending = _list_pending()
        print(f"Pending crops: {len(pending)} in {PENDING_DIR}")
        for p in pending[:30]:
            age = time.time() - p.stat().st_mtime
            print(f"  {p.name} ({age:.0f}s old)")
        return

    if not args.assignments:
        parser.error("provide at least one <field>:<label> pair, or --list")

    if args.match:
        # All assignments must be plain labels in this mode
        for raw in args.assignments:
            label = raw.split(":")[-1] if ":" in raw else raw
            n = _move_by_match(args.match, label)
            print(f"  matched {n} files for label {label!r}")
        return

    for raw in args.assignments:
        if ":" not in raw:
            print(f"  skip {raw!r}: expected <field>:<label>", file=sys.stderr)
            continue
        field, label = raw.split(":", 1)
        field = field.strip().lower()
        if field not in ("mass", "resistance", "instability"):
            print(f"  skip {raw!r}: field must be mass/resistance/instability", file=sys.stderr)
            continue
        _move_latest(field, label)


if __name__ == "__main__":
    main()
