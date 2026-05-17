"""Exercise the HUD OCR pipeline on captured value crops and report
which of the recently-added features fired on each read.

Reads the debug_value_*_crop.png files that the live pipeline drops
to disk after every scan and runs them back through
``_ocr_value_crop`` directly — same entry point the live scan path
uses, just with file-loaded input instead of an in-memory crop.

For each crop we log:
  * the final returned (text, mean_conf)
  * whether the HUD-RGB CRNN gate accepted (and which threshold)
  * whether beam-lexicon rerank changed the winner (#1)
  * whether the lexicon-confirmed gate relaxed (#2)
  * whether anchor-check rejected (#4)
  * the cascade path taken when CRNN failed

The lexicon is seeded with values the pipeline would have learned
from prior captures (10810, 13451, 12.09, etc. from the recent logs)
so we can see the #1/#2 paths exercise without needing a live
frozen_panel.freeze() call.

Note: ``_ocr_value_crop`` is the per-row entry point. The full
``scan_hud_onnx`` path adds the panel-finder + label-row anchoring
on top — we test those via the existing pytest suite, not here,
because they need a full-frame input.

Usage:
    python scripts/run_hud_pipeline_check.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from PIL import Image

# Make ``ocr.sc_ocr`` importable when running from the tree root.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Configure logging so we see the INFO lines the pipeline emits
# (HUD-RGB CRNN gate accept, beam-lexicon rerank, anchor reject etc.)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet down anything below ocr.sc_ocr so the output is readable.
for noisy in ("PIL", "onnxruntime", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from ocr.sc_ocr import api as sc_api  # noqa: E402
from ocr.sc_ocr import hud_lexicon  # noqa: E402


CROPS = [
    ("mass",        _ROOT / "debug_value_mass_crop.png"),
    ("resistance",  _ROOT / "debug_value_resistance_crop.png"),
    ("instability", _ROOT / "debug_value_instability_crop.png"),
]


def seed_lexicon() -> None:
    """Seed the lexicon with values from the recent session logs so
    the lexicon-confirmed gate and beam-rerank can fire even on a
    cold-start install (before frozen_panel has had a chance to
    populate it organically).
    """
    seeds = {
        "mass":        [10810, 13451, 27265, 3384],
        "resistance":  [0, 1, 50, 100],
        "instability": [11.01, 12.09, 1.26, 1.43, 12.79, 12.6],
    }
    for field, values in seeds.items():
        for v in values:
            hud_lexicon.observe(field, v)
    print(
        f"[seed] mass={hud_lexicon.size('mass')} "
        f"resistance={hud_lexicon.size('resistance')} "
        f"instability={hud_lexicon.size('instability')}",
    )


def run_one(field: str, path: Path) -> None:
    if not path.is_file():
        print(f"[skip] {field}: {path.name} not present")
        return
    img = Image.open(path)
    print(f"\n-- {field.upper()} -- {path.name} ({img.size[0]}x{img.size[1]} {img.mode})")
    try:
        text, confs = sc_api._ocr_value_crop(img, field=field)
    except Exception as exc:
        print(f"  PIPELINE RAISED: {exc!r}")
        return
    if not text:
        print(f"  → (empty read)")
        return
    mean = sum(confs) / len(confs) if confs else 0.0
    print(f"  → text={text!r} mean_conf={mean:.2f} (per-char {[f'{c:.2f}' for c in confs]})")
    # Surface the lexicon state — did this read land in-lexicon?
    try:
        from . import priors  # noqa: F401
    except Exception:
        pass
    digits = "".join(c for c in text if c.isdigit() or c == ".")
    if digits:
        try:
            val = float(digits)
            hit = hud_lexicon.is_known(field, val)
            print(f"  lex_hit({val})={hit}")
        except ValueError:
            pass


def main() -> int:
    print("=" * 78)
    print("HUD pipeline end-to-end test on captured value crops")
    print("=" * 78)
    seed_lexicon()
    for field, path in CROPS:
        run_one(field, path)
    print("\n" + "=" * 78)
    print(
        f"final lexicon: mass={hud_lexicon.size('mass')} "
        f"resistance={hud_lexicon.size('resistance')} "
        f"instability={hud_lexicon.size('instability')}",
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
