"""Mirror the canonical glyph training dataset with polarity-inverted
copies, for training the sibling inverted-polarity classifier
(``model_cnn_inv.onnx``).

The OCR pipeline runs two ONNX classifiers in parallel:

  * **Primary**  — ``model_cnn.onnx``       (canonical: bright text on dark)
  * **Secondary** — ``model_cnn_inv.onnx``   (inverted: dark text on light)

Both consume the SAME crops from the primary segmentation path; the
secondary path inverts each crop before classification so the two
voters see polarity-decorrelated views of the same source pixels.
For that vote to be meaningful, the inverted model must be trained
on inverted data — that's what this script produces.

What it does:
  1. Reads every PNG under ``training_data/{0..9}/``.
  2. For each, writes ``255 - img`` to the same filename under
     ``training_data_inv/{0..9}/``.
  3. Idempotent: re-running overwrites previous outputs.

Run:
    python scripts/make_inverted_dataset.py
    python scripts/make_inverted_dataset.py --clear
    python scripts/make_inverted_dataset.py --source training_data_user_sig

Then train the inverted model:
    python -m ocr.train_model --inverted
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

THIS = Path(__file__).resolve()
TOOL_DIR = THIS.parent.parent

DEFAULT_SRC = TOOL_DIR / "training_data_user_panel"
DEFAULT_DST_SUFFIX = "_inv"

# Class folders to mirror.  The torch_digit pipeline (train_torch.py +
# export_torch_to_onnx.py — the canonical pipeline that produces
# model_cnn.onnx) uses 12 classes: 0..9 plus 'dot' and 'pct'.  We
# mirror exactly the same folder layout into the _inv sibling so the
# inverted training run sees an identical class structure.
# `_quarantine` is intentionally skipped (those are bad samples).
CLASS_FOLDERS = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                 "dot", "pct")


def _wipe_destination(dst: Path) -> int:
    """Recursively remove dst's per-class subdirs, return file count removed."""
    n = 0
    for cls in CLASS_FOLDERS:
        d = dst / cls
        if not d.is_dir():
            continue
        for f in d.glob("*.png"):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    return n


def _invert_class_dir(src_dir: Path, dst_dir: Path) -> int:
    """Mirror every PNG in src_dir to dst_dir with 255 - pixels."""
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src_png in src_dir.glob("*.png"):
        try:
            with Image.open(src_png) as img:
                gray = np.asarray(img.convert("L"), dtype=np.uint8)
            inv = (255 - gray).astype(np.uint8)
            out = dst_dir / src_png.name
            Image.fromarray(inv, mode="L").save(out)
            n += 1
        except (OSError, ValueError):
            pass
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path, default=DEFAULT_SRC,
        help=(
            f"Canonical-polarity dataset to mirror. "
            f"Default: {DEFAULT_SRC.relative_to(TOOL_DIR)}"
        ),
    )
    p.add_argument(
        "--dest", type=Path, default=None,
        help=(
            "Destination directory. Default: <source>_inv (sibling of source)."
        ),
    )
    p.add_argument(
        "--clear", action="store_true",
        help="Wipe the destination's per-class PNGs before generating.",
    )
    args = p.parse_args()

    src = args.source.resolve() if args.source.is_absolute() \
        else (TOOL_DIR / args.source).resolve()
    if not src.is_dir():
        print(f"[!] source dir does not exist: {src}", file=sys.stderr)
        return 2

    dst = args.dest if args.dest is not None \
        else src.parent / (src.name + DEFAULT_DST_SUFFIX)
    dst = dst.resolve() if dst.is_absolute() else (TOOL_DIR / dst).resolve()

    if src == dst:
        print("[!] source and dest must differ", file=sys.stderr)
        return 2

    print(f"=== Inverted-dataset mirror ===")
    print(f"    source: {src}")
    print(f"    dest:   {dst}")
    print()

    if args.clear:
        n = _wipe_destination(dst)
        print(f"[clear] removed {n} previous PNGs from {dst.name}/")

    dst.mkdir(parents=True, exist_ok=True)

    per_class: dict[str, int] = {}
    total = 0
    for cls in CLASS_FOLDERS:
        n = _invert_class_dir(src / cls, dst / cls)
        per_class[cls] = n
        total += n

    print()
    print("Per-class inverted PNGs written:")
    for cls in CLASS_FOLDERS:
        print(f"  {cls!r}: {per_class[cls]}")
    print()
    print(f"[done] {total} PNGs mirrored")
    print()
    print("Next steps to retrain + deploy the inverted classifier:")
    print("    python -m ocr.train_torch --inverted --epochs 60")
    print("    python scripts/export_torch_to_onnx.py --inverted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
