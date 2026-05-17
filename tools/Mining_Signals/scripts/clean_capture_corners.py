"""Strip the contaminated corners from captures made by an older
version of dual_capture.py that baked the overlay's label + resize
handle into saved PNGs.

Contamination positions (from CaptureOverlay geometry):
  - Top-left label: 80 wide × 22 tall (from x=2, y=2) + 3-px border
  - Bottom-right handle: 18 wide × 18 tall + 3-px border

Strategy:
  - Crop the image with a fixed inset on top (24 px) to remove the
    whole label row, plus 3 px on every other side to remove the
    border. Bottom-right handle gets removed when we additionally
    crop 18 px from the bottom and 18 px from the right INSIDE the
    cropped bottom-right corner… except that would destroy the
    rest of the image.

Better approach: just overwrite the label/handle pixels with
background black — the detector/splitter ignores non-orange,
non-white regions anyway. Safer than cropping (which shifts all
panels downward by 24 px, breaking the spatial consistency the
model relies on).

Usage:
  python scripts/clean_capture_corners.py
  python scripts/clean_capture_corners.py --folder user_20260418_081525
  python scripts/clean_capture_corners.py --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
PANELS_ROOT = TOOL / "training_data_panels"

# Overlay contamination regions (in the captured image's own coordinates)
# Label: sat at (2, 2) in overlay; since capture_bbox stripped 3-px border,
# in saved image it sits at (-1, -1) roughly. Use generous padding.
LABEL_BOX = (0, 0, 85, 25)       # top-left
HANDLE_PAD_R = 22
HANDLE_PAD_B = 22


def clean_one(path: Path, dry_run: bool = False) -> bool:
    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        print(f"  SKIP {path.name}: {exc}")
        return False
    W, H = img.size
    draw = ImageDraw.Draw(img)
    # Black out top-left label
    draw.rectangle(LABEL_BOX, fill=(0, 0, 0))
    # Black out bottom-right handle
    draw.rectangle(
        (W - HANDLE_PAD_R, H - HANDLE_PAD_B, W, H),
        fill=(0, 0, 0),
    )
    if dry_run:
        return True
    img.save(path)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", help="user_* folder name (default: all)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.folder:
        roots = [PANELS_ROOT / args.folder]
    else:
        roots = [
            p for p in PANELS_ROOT.iterdir()
            if p.is_dir() and p.name.startswith("user_")
        ]

    total = cleaned = 0
    for root in roots:
        for region in ("region1", "region2"):
            d = root / region
            if not d.is_dir():
                continue
            for img_path in sorted(d.glob("cap_*.png")):
                total += 1
                ok = clean_one(img_path, dry_run=args.dry_run)
                if ok:
                    cleaned += 1

    verb = "would clean" if args.dry_run else "cleaned"
    print(f"\n{verb} {cleaned}/{total} images")


if __name__ == "__main__":
    main()
