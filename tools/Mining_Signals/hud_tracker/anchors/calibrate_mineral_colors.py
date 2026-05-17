"""Calibrate the mineral-name palette for ``mineral_name_color``.

Reads ``.boxes.json`` ground-truth files alongside their PNG, samples
*bright* pixels (V high, S high) inside the labeled ``resource``
boxes (the mineral name row), clusters the sampled hues into the
warm / cyan / purple palette buckets, computes robust percentile
bounds, and writes
``hud_tracker/mineral_color_calibration.json``.

Usage
-----
Run with no args to default to the user_20260418_154408 set::

    python calibrate_mineral_colors.py

Or point at any folder of labeled captures::

    python calibrate_mineral_colors.py --source <folder> --output <json>

The auto-annotator already labels the ``resource`` (mineral name)
box, so the captures we use are the same set used for hud_color
calibration.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("calibrate_mineral_colors")


_DEFAULT_SOURCE = (
    Path(os.environ.get("APPDATA", ""))
    / "ShipBit" / "WingmanAI" / "custom_skills" / "SC_Toolbox_Beta_V1.2"
    / "tools" / "Mining_Signals" / "training_data_panels"
    / "user_20260418_154408" / "region1"
)


def _iter_labeled_captures(source: Path) -> Iterable[tuple[Path, dict]]:
    """Yield (png_path, boxes_json_dict) for every capture in ``source``
    that has both a PNG and a ``.boxes.json`` sibling."""
    for boxes_path in sorted(source.glob("*.boxes.json")):
        png_path = boxes_path.with_suffix("").with_suffix(".png")
        if not png_path.is_file():
            log.warning("orphan boxes file with no PNG: %s", boxes_path)
            continue
        try:
            data = json.loads(boxes_path.read_text())
        except Exception as exc:
            log.warning("failed to read %s: %s", boxes_path, exc)
            continue
        if "boxes" not in data:
            continue
        yield png_path, data


def _sample_resource_pixels(
    img: Image.Image,
    boxes: dict,
    *,
    val_floor: int = 130,
    sat_floor: int = 90,
) -> np.ndarray:
    """Return (N, 3) HSV pixels (uint8) that are bright + saturated
    inside the ``resource`` (mineral name) bbox."""
    rb = boxes.get("resource")
    if not rb:
        return np.empty((0, 3), dtype=np.uint8)
    x = max(0, int(rb["x"]))
    y = max(0, int(rb["y"]))
    w = max(0, int(rb["w"]))
    h = max(0, int(rb["h"]))
    if w == 0 or h == 0:
        return np.empty((0, 3), dtype=np.uint8)
    iw, ih = img.size
    x2 = min(iw, x + w)
    y2 = min(ih, y + h)
    if x >= x2 or y >= y2:
        return np.empty((0, 3), dtype=np.uint8)
    crop = img.crop((x, y, x2, y2)).convert("HSV")
    arr = np.asarray(crop)  # (h, w, 3) uint8
    H, W, _ = arr.shape
    pixels = arr.reshape(-1, 3)
    s = pixels[:, 1]
    v = pixels[:, 2]
    keep = (s >= sat_floor) & (v >= val_floor)
    return pixels[keep]


def _cluster_hue_bands(hues: np.ndarray) -> tuple[dict, dict, dict]:
    """Split a set of PIL hues (0..255) into warm / cyan / purple.

    Centers (PIL units, ≈ degrees * 255/360):
      warm   ≈   0..50   (deg) →   0..35   PIL
      cyan   ≈  85..110         →  60..78
      purple ≈ 200..240         → 141..170

    For each bucket, return ``{"h_min": p5, "h_max": p95,
    "_n_samples": n}`` derived from the captured pixels that fall
    within a permissive bucket window. If a bucket is empty, fall
    back to its default window edges with n_samples=0.
    """
    # Permissive bucket windows (so we don't lose pixels near the edges).
    windows = {
        "warm":   (0, 50),
        "cyan":   (50, 95),
        "purple": (130, 180),
    }
    defaults = {
        "warm":   (0, 35),
        "cyan":   (60, 78),
        "purple": (141, 170),
    }
    out: dict[str, dict] = {}
    for name, (lo, hi) in windows.items():
        in_band = hues[(hues >= lo) & (hues <= hi)]
        if in_band.size < 50:
            d_lo, d_hi = defaults[name]
            out[name] = {
                "h_min": int(d_lo),
                "h_max": int(d_hi),
                "_n_samples": int(in_band.size),
                "_fallback": True,
            }
            continue
        p5 = int(round(float(np.percentile(in_band, 5))))
        p95 = int(round(float(np.percentile(in_band, 95))))
        # Pad ±2 to soak up edge pixels we'd otherwise reject.
        p5 = max(0, p5 - 2)
        p95 = min(255, p95 + 2)
        out[name] = {
            "h_min": int(p5),
            "h_max": int(p95),
            "_n_samples": int(in_band.size),
        }
    return out["warm"], out["cyan"], out["purple"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--source", type=Path, default=_DEFAULT_SOURCE,
        help="folder of labeled region1 captures (PNG + .boxes.json)",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent.parent
        / "mineral_color_calibration.json",
        help="output calibration JSON path",
    )
    args = p.parse_args()

    source: Path = args.source
    if not source.is_dir():
        log.error("source %s is not a directory", source)
        return 2

    log.info("calibrating from %s", source)
    all_pixels: list[np.ndarray] = []
    per_capture: list[dict] = []
    n_caps = 0

    for png_path, data in _iter_labeled_captures(source):
        boxes = data.get("boxes", {})
        if "resource" not in boxes:
            continue
        try:
            with Image.open(png_path) as img:
                img_rgb = img.convert("RGB")
                pixels = _sample_resource_pixels(img_rgb, boxes)
        except Exception as exc:
            log.warning("could not sample %s: %s", png_path, exc)
            continue
        if pixels.size == 0:
            continue
        all_pixels.append(pixels)
        per_capture.append({
            "image": png_path.name,
            "n_samples": int(pixels.shape[0]),
        })
        n_caps += 1

    if not all_pixels:
        log.error("no usable captures found")
        return 1

    pix = np.concatenate(all_pixels, axis=0)
    hues = pix[:, 0]
    log.info(
        "sampled %d bright pixels from %d captures (hue range %d..%d)",
        pix.shape[0], n_caps,
        int(hues.min()), int(hues.max()),
    )

    warm_band, cyan_band, purple_band = _cluster_hue_bands(hues)
    sat_floor = int(round(float(np.percentile(pix[:, 1], 5))))
    val_floor = int(round(float(np.percentile(pix[:, 2], 5))))

    out = {
        "version": 1,
        "source": str(source),
        "n_captures": int(n_caps),
        "n_total_samples": int(pix.shape[0]),
        "warm_band": warm_band,
        "cyan_band": cyan_band,
        "purple_band": purple_band,
        "sat_min": int(max(60, sat_floor - 5)),
        "val_min": int(max(80, val_floor - 5)),
        # Geometry / morph defaults — copied from the module's
        # DEFAULT_CALIBRATION; tuneable separately if needed.
        "min_width_px":  50,
        "min_height_px": 10,
        "max_height_px": 35,
        "min_aspect":     2.0,
        "morph_horiz_close_px": 11,
        "position_y_min_frac": 0.05,
        "position_y_max_frac": 0.55,
        "panel_y_min_frac": 0.10,
        "panel_y_max_frac": 0.55,
        "_per_capture_summary": per_capture[:8],
        "_doc": (
            "Mineral name palette calibration. PIL HSV scale: "
            "H, S, V each in 0-255. "
            "Hue bands cover warm (orange/yellow), cyan/teal "
            "(BERYL-style) and purple (rare mineral types)."
        ),
    }

    args.output.write_text(json.dumps(out, indent=2))
    log.info("wrote %s", args.output)
    log.info("warm  band: H ∈ [%d, %d] (%d samples)",
             warm_band["h_min"], warm_band["h_max"], warm_band["_n_samples"])
    log.info("cyan  band: H ∈ [%d, %d] (%d samples)",
             cyan_band["h_min"], cyan_band["h_max"], cyan_band["_n_samples"])
    log.info("purple band: H ∈ [%d, %d] (%d samples)",
             purple_band["h_min"], purple_band["h_max"], purple_band["_n_samples"])
    log.info("sat_min=%d, val_min=%d", out["sat_min"], out["val_min"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
