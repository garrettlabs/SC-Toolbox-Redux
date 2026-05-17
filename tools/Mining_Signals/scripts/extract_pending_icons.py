"""Extract pending-review icon crops from labeled region2 captures.

Walks ``training_data_panels/<run>/region2/`` looking for
``*.boxes.json`` sidecars that carry an ``icon`` bbox alongside the
``value`` digit bbox. For each one, opens the matching capture PNG,
crops the icon region (with ~2 px padding, clamped to image bounds),
and writes TWO normalized 28×28 PNGs into the staging pool:

    training_data_pending_review_signal/icon/
        pending_<source_id>_gray.png   — grayscale 28×28
        pending_<source_id>_rgb.png    — RGB 28×28

``<source_id>`` is the original capture filename without extension
(e.g. ``cap_20260418_155446_555``). The grayscale variant matches
what the current signal CNN expects; the RGB variant is queued for
future RGB-CNN training.

After running, sort the staging pool with
``python scripts/review_glyphs.py pending_signal_icon`` — click any
mis-cropped or non-icon images, hit "Move to quarantine". Then run
``python scripts/promote_reviewed_icons.py`` to move surviving
crops into ``training_data_user_sig/icon/`` so they replace the
synthetic ``aug_bad_crop_*.png`` corpus that currently trains the
``@`` class.

Idempotent: skips a (run, source_id) pair if any
``pending_<source_id>_*.png`` already exists in the pool.
Defensive: skips sidecars whose ``icon`` bbox is malformed (zero
area, falls outside the image, can't be cropped) with a warning
rather than crashing.

Usage:
    python scripts/extract_pending_icons.py
    python scripts/extract_pending_icons.py --run user_20260418_154408
    python scripts/extract_pending_icons.py --pad 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
sys.path.insert(0, str(TOOL))

from ocr import training_registry  # noqa: E402

# Source root for labeled region2 captures (one subdir per run).
PANELS_ROOT = TOOL / "training_data_panels"

# Pull the staging dir from the registry so we never drift if the
# spec moves.
_SPEC = training_registry.get("pending_signal_icon")
STAGING_ROOT = _SPEC.glyph_staging_dir
ICON_CLASS_DIR = STAGING_ROOT / "icon"

# 28×28 — same input shape as the existing signal CNN. Use BICUBIC
# (better for sub-pixel preservation than NEAREST when the source
# crop is e.g. 37×37).
TARGET_SIZE = (28, 28)


def _resolve_runs(runs_arg: str | None) -> list[Path]:
    """Return concrete ``training_data_panels/<run>/region2`` dirs.

    If the user passed ``--run NAME`` we honor it (and only that
    run); otherwise we walk every run under ``training_data_panels``
    that has a ``region2/`` child.
    """
    if not PANELS_ROOT.is_dir():
        return []
    if runs_arg:
        candidate = PANELS_ROOT / runs_arg / "region2"
        return [candidate] if candidate.is_dir() else []
    out: list[Path] = []
    for run_dir in sorted(PANELS_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        region2 = run_dir / "region2"
        if region2.is_dir():
            out.append(region2)
    return out


def _validate_bbox(
    bbox: dict, img_w: int, img_h: int,
) -> tuple[int, int, int, int] | None:
    """Sanity-check an icon bbox dict ``{x, y, w, h}`` and return
    ``(x1, y1, x2, y2)`` integer pixel coords on success, or ``None``
    if the bbox is malformed / out-of-bounds / zero-area.

    Reasons a bbox is invalid: missing keys, non-numeric values,
    zero-or-negative dimensions, fully outside the image, or so
    far off-image that there's no overlap to crop.
    """
    try:
        x = int(bbox["x"])
        y = int(bbox["y"])
        w = int(bbox["w"])
        h = int(bbox["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _padded(
    coords: tuple[int, int, int, int],
    pad: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Expand a crop by ``pad`` px on all sides, clamped to the image."""
    x1, y1, x2, y2 = coords
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(img_w, x2 + pad),
        min(img_h, y2 + pad),
    )


def _already_extracted(source_id: str) -> bool:
    """Return True if any ``pending_<source_id>_*.png`` exists already.

    Idempotent re-run guard: re-extracting the same capture is a
    no-op so the user can run this freely without duplicating the
    review queue.
    """
    if not ICON_CLASS_DIR.is_dir():
        return False
    return any(ICON_CLASS_DIR.glob(f"pending_{source_id}_*.png"))


def _extract_one(
    region2_dir: Path,
    sidecar: Path,
    pad: int,
) -> tuple[bool, str]:
    """Extract gray + RGB icon crops for one sidecar.

    Returns ``(extracted, status)`` where ``status`` is one of
    ``"written"``, ``"skipped:already_extracted"``,
    ``"skipped:no_icon_bbox"``, ``"skipped:no_image"``,
    ``"skipped:bad_bbox"``, or ``"skipped:error"``.
    """
    source_id = sidecar.name.removesuffix(".boxes.json")
    image_path = region2_dir / f"{source_id}.png"

    if _already_extracted(source_id):
        return False, "skipped:already_extracted"

    if not image_path.is_file():
        return False, "skipped:no_image"

    try:
        with sidecar.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [warn] cannot read {sidecar.name}: {exc}")
        return False, "skipped:error"

    boxes = payload.get("boxes") or {}
    icon_box = boxes.get("icon")
    if not isinstance(icon_box, dict):
        return False, "skipped:no_icon_bbox"

    try:
        with Image.open(image_path) as im:
            im.load()
            img_w, img_h = im.size
            coords = _validate_bbox(icon_box, img_w, img_h)
            if coords is None:
                print(
                    f"  [warn] {sidecar.name}: icon bbox is invalid or "
                    f"out-of-bounds (bbox={icon_box}, image={img_w}x{img_h})"
                )
                return False, "skipped:bad_bbox"
            crop_box = _padded(coords, pad, img_w, img_h)
            # Save TWO normalized variants. Convert the RGB variant
            # first (so we keep color), then derive grayscale from
            # the same source — this is faster than re-cropping and
            # keeps both views perfectly aligned.
            rgb_crop = im.convert("RGB").crop(crop_box)
    except (OSError, ValueError) as exc:
        print(f"  [warn] {sidecar.name}: cannot crop icon: {exc}")
        return False, "skipped:error"

    try:
        ICON_CLASS_DIR.mkdir(parents=True, exist_ok=True)
        rgb_28 = rgb_crop.resize(TARGET_SIZE, Image.BICUBIC)
        gray_28 = rgb_28.convert("L")
        gray_path = ICON_CLASS_DIR / f"pending_{source_id}_gray.png"
        rgb_path = ICON_CLASS_DIR / f"pending_{source_id}_rgb.png"
        gray_28.save(gray_path)
        rgb_28.save(rgb_path)
    except OSError as exc:
        print(f"  [warn] {sidecar.name}: cannot write crops: {exc}")
        return False, "skipped:error"

    return True, "written"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run", default=None,
        help="Restrict extraction to a single run dir under "
             "training_data_panels/ (e.g. 'user_20260418_154408'). "
             "Default: walk every run.",
    )
    p.add_argument(
        "--pad", type=int, default=2,
        help="Padding pixels added on each side of the icon bbox "
             "before cropping; clamped to image bounds. Default 2.",
    )
    args = p.parse_args()

    print("=== Extract pending signal-icon crops ===")
    print(f"  panels root:   {PANELS_ROOT}")
    print(f"  staging dir:   {ICON_CLASS_DIR}")
    print(f"  bbox padding:  {args.pad} px")
    print(f"  target size:   {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"  run filter:    {args.run or '(all runs)'}")
    print()

    # Always ensure the staging dir exists so review_glyphs.py picks
    # up the region tab even when there are zero crops yet.
    ICON_CLASS_DIR.mkdir(parents=True, exist_ok=True)

    region_dirs = _resolve_runs(args.run)
    if not region_dirs:
        if args.run:
            print(f"  no region2 dir for run '{args.run}'.")
        else:
            print(f"  no region2 dirs found under {PANELS_ROOT}.")
        print()
        print("Staging dir created (empty). Re-run after labeling some "
              "region2 captures.")
        return 0

    n_sidecars = 0
    n_with_icon = 0
    n_written = 0
    n_already = 0
    n_no_image = 0
    n_bad_bbox = 0
    n_errors = 0

    for region2_dir in region_dirs:
        run_name = region2_dir.parent.name
        sidecars = sorted(region2_dir.glob("*.boxes.json"))
        print(f"[run] {run_name}: {len(sidecars)} sidecar(s)")
        for sidecar in sidecars:
            n_sidecars += 1
            extracted, status = _extract_one(
                region2_dir, sidecar, pad=args.pad,
            )
            if status != "skipped:no_icon_bbox":
                n_with_icon += 1
            if extracted:
                n_written += 1
            elif status == "skipped:already_extracted":
                n_already += 1
            elif status == "skipped:no_image":
                n_no_image += 1
            elif status == "skipped:bad_bbox":
                n_bad_bbox += 1
            elif status == "skipped:error":
                n_errors += 1

    print()
    print("=== Summary ===")
    print(f"  sidecars seen:                {n_sidecars}")
    print(f"  with icon bbox:               {n_with_icon}")
    print(f"  crop pairs written:           {n_written} "
          f"({n_written * 2} files: gray + rgb)")
    print(f"  skipped (already extracted):  {n_already}")
    print(f"  skipped (no image on disk):   {n_no_image}")
    print(f"  skipped (bad/oob bbox):       {n_bad_bbox}")
    print(f"  skipped (errors):             {n_errors}")
    print()
    if n_written == 0 and n_already == 0:
        print("Staging dir is empty. Once region2 captures get labeled "
              "with icon bboxes, re-run this to populate the reviewer.")
    else:
        print("Next: python scripts/review_glyphs.py pending_signal_icon")
        print("      (click trash on bad crops, hit 'Move to quarantine')")
        print("Then: python scripts/promote_reviewed_icons.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
