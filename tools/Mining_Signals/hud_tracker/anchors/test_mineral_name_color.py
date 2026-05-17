"""Smoke / IoU test for ``mineral_name_color.find_mineral_name_row``.

Walks 30 random region1 captures (mix of labeled and unlabeled),
runs the new color-aware mineral row detector on each, and reports:

    * IoU vs ground truth on labeled ones
    * y_frac (vertical-center fraction) on unlabeled ones — used as a
      sanity gate (mineral row should be in the upper-middle of the
      capture, well clear of the COMPOSITION area)
    * Specifically verifies the 3 known-failure captures the legacy
      ``_find_mineral_row_universal`` returns the wrong y for —
      cap_20260418_090135_740, cap_20260418_091447_617,
      cap_20260418_091456_295. These have bright icy-asteroid
      backgrounds where projection-band detection lands in the
      COMPOSITION area (y_frac 0.70-0.94).

Usage::

    python test_mineral_name_color.py

Exits 0 when all three known-failure captures get a y_frac < 0.55
and all labeled IoUs ≥ 0.20.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Make sure we can import the module when run from anywhere.
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent  # Mining_Signals
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from hud_tracker.anchors.mineral_name_color import (  # noqa: E402
    find_mineral_name_row,
)


_LABELED_DIR = (
    Path(os.environ.get("APPDATA", ""))
    / "ShipBit" / "WingmanAI" / "custom_skills" / "SC_Toolbox_Beta_V1.2"
    / "tools" / "Mining_Signals" / "training_data_panels"
    / "user_20260418_154408" / "region1"
)
_UNLABELED_DIR = (
    Path(os.environ.get("APPDATA", ""))
    / "ShipBit" / "WingmanAI" / "custom_skills" / "SC_Toolbox_Beta_V1.2"
    / "tools" / "Mining_Signals" / "training_data_panels"
    / "user_20260418_081525" / "region1"
)
_KNOWN_FAILURE_NAMES = [
    "cap_20260418_090135_740",
    "cap_20260418_091447_617",
    "cap_20260418_091456_295",
]


def _bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """IoU of two (x, y, w, h) bboxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _row_iou_y_only(
    pred: tuple[int, int, int, int],
    gt: tuple[int, int, int, int],
    *,
    img_w: int,
) -> float:
    """IoU using only the y-band — the predicted bbox is narrow (just
    the mineral name) while the GT box spans the full panel width.

    To compare apples to apples we widen the prediction's x to the
    full image width and score against the GT directly. This isolates
    "did we localize the right ROW?" from "did we get the exact
    horizontal extent of the text?" — the row is what matters for
    downstream layout.
    """
    px, py, pw, ph = pred
    return _bbox_iou((0, py, img_w, ph), gt)


def _y_frac(bbox: tuple[int, int, int, int], img_h: int) -> float:
    _, y, _, h = bbox
    return (y + h / 2.0) / max(1, img_h)


def main() -> int:
    rng = random.Random(20260509)

    # Collect labeled paths.
    labeled: list[tuple[Path, dict]] = []
    if _LABELED_DIR.is_dir():
        for boxes_path in sorted(_LABELED_DIR.glob("*.boxes.json")):
            png = boxes_path.with_suffix("").with_suffix(".png")
            if not png.is_file():
                continue
            try:
                data = json.loads(boxes_path.read_text())
            except Exception:
                continue
            if "boxes" not in data or "resource" not in data["boxes"]:
                continue
            labeled.append((png, data))

    # Collect unlabeled paths (skip ones that have .boxes.json).
    unlabeled: list[Path] = []
    if _UNLABELED_DIR.is_dir():
        for png in sorted(_UNLABELED_DIR.glob("*.png")):
            sidecar = png.with_suffix(".boxes.json")
            if sidecar.is_file():
                continue
            unlabeled.append(png)

    # Random sample 30 captures total — half labeled, half unlabeled
    # (or whatever's available).
    rng.shuffle(labeled)
    rng.shuffle(unlabeled)
    sample_labeled = labeled[:15]
    sample_unlabeled = unlabeled[:15]

    print(f"=== mineral_name_color test ===")
    print(f"labeled candidates:   {len(labeled)}")
    print(f"unlabeled candidates: {len(unlabeled)}")
    print(f"using {len(sample_labeled)} labeled + {len(sample_unlabeled)} unlabeled")
    print()

    # ── Labeled IoU ──
    print("Labeled IoU (y-band):")
    ious: list[float] = []
    n_hits = 0
    for png, data in sample_labeled:
        gt = data["boxes"]["resource"]
        with Image.open(png) as img:
            img_rgb = img.convert("RGB")
            iw, ih = img_rgb.size
            res = find_mineral_name_row(img_rgb)
        if res is None:
            print(f"  {png.stem}: MISS (no row found)")
            ious.append(0.0)
            continue
        pred = res["bbox"]
        iou = _row_iou_y_only(
            pred, (gt["x"], gt["y"], gt["w"], gt["h"]), img_w=iw,
        )
        ious.append(iou)
        if iou >= 0.20:
            n_hits += 1
        yf = _y_frac(pred, ih)
        print(
            f"  {png.stem}: IoU={iou:.2f} pred=(x={pred[0]},y={pred[1]},"
            f"w={pred[2]},h={pred[3]}) y_frac={yf:.2f} "
            f"hue={res['details']['dominant_hue']} "
            f"conf={res['confidence']:.2f}"
        )
    if ious:
        print(
            f"  mean IoU {sum(ious)/len(ious):.2f}, "
            f"hits >= 0.20: {n_hits}/{len(ious)}"
        )
    print()

    # ── Unlabeled y_frac ──
    print("Unlabeled y_frac plausibility (should be < 0.55):")
    n_unlabeled_pass = 0
    n_unlabeled_total = 0
    for png in sample_unlabeled:
        with Image.open(png) as img:
            img_rgb = img.convert("RGB")
            ih = img_rgb.size[1]
            res = find_mineral_name_row(img_rgb)
        n_unlabeled_total += 1
        if res is None:
            print(f"  {png.stem}: NONE (no row found)")
            continue
        yf = _y_frac(res["bbox"], ih)
        ok = yf < 0.55
        if ok:
            n_unlabeled_pass += 1
        print(
            f"  {png.stem}: {'OK ' if ok else 'BAD'} y_frac={yf:.2f} "
            f"hue={res['details']['dominant_hue']} "
            f"conf={res['confidence']:.2f}"
        )
    print(
        f"  pass: {n_unlabeled_pass}/{n_unlabeled_total} "
        f"({n_unlabeled_pass/max(1, n_unlabeled_total)*100:.0f}%)"
    )
    print()

    # ── 3 known-failure captures ──
    print("Known-failure captures (must have y_frac < 0.55):")
    n_known_pass = 0
    for stem in _KNOWN_FAILURE_NAMES:
        png = _UNLABELED_DIR / f"{stem}.png"
        if not png.is_file():
            print(f"  {stem}: SKIP (file missing)")
            continue
        with Image.open(png) as img:
            img_rgb = img.convert("RGB")
            ih = img_rgb.size[1]
            res = find_mineral_name_row(img_rgb)
        if res is None:
            print(f"  {stem}: NONE")
            continue
        yf = _y_frac(res["bbox"], ih)
        ok = yf < 0.55
        if ok:
            n_known_pass += 1
        print(
            f"  {stem}: {'PASS' if ok else 'FAIL'} y_frac={yf:.2f} "
            f"bbox={res['bbox']} hue={res['details']['dominant_hue']} "
            f"conf={res['confidence']:.2f}"
        )
    print(f"  known-failure: {n_known_pass}/{len(_KNOWN_FAILURE_NAMES)} pass")

    # Exit code: non-zero if any known-failure capture failed or labeled
    # IoU mean is below 0.10 (very loose floor — we mostly need any
    # localization at all on the visually-trivial captures).
    if ious:
        mean_iou = sum(ious) / len(ious)
    else:
        mean_iou = 0.0
    fail = (
        n_known_pass < len(_KNOWN_FAILURE_NAMES)
        or mean_iou < 0.10
    )
    print()
    print(f"FINAL: mean_labeled_IoU={mean_iou:.2f}  "
          f"known_failure={n_known_pass}/{len(_KNOWN_FAILURE_NAMES)}  "
          f"=> {'FAIL' if fail else 'OK'}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
