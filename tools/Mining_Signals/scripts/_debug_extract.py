"""Debug harness for one image + its label."""
import json
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOL))

from scripts.extract_labeled_glyphs import (
    extract_region1_glyphs, PANEL_GLYPH_ROOT,
)

# Find first labeled region1 image
region1 = TOOL / "training_data_panels" / "user_20260418_081525" / "region1"
for img in sorted(region1.glob("cap_*.png")):
    jp = img.with_suffix(".json")
    if not jp.is_file():
        continue
    label = json.loads(jp.read_text(encoding="utf-8"))
    print(f"\n=== {img.name} ===")
    print(f"label: {label}")
    counts = extract_region1_glyphs(img, label, PANEL_GLYPH_ROOT)
    print(f"counts: {counts}")
    break
