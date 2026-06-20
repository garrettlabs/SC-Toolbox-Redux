import importlib
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MINING_SIGNALS_ROOT = PROJECT_ROOT / "tools" / "Mining_Signals"

sys.path.insert(0, os.path.normpath(str(PROJECT_ROOT)))
sys.path.insert(0, os.path.normpath(str(MINING_SIGNALS_ROOT)))

from core.skill_registry import discover_skills  # noqa: E402
from redux_mining_launcher import REDUX_MINING_SKILL_IDS, filter_redux_skills  # noqa: E402


NON_MINING_SENTINEL_IDS = {"dps", "trade", "cargo", "market", "craft_db"}


def test_redux_source_discovery_filters_to_mining_only_apps():
    """Source discovery plus Redux filtering exposes only the mining apps."""
    discovered = discover_skills(str(PROJECT_ROOT))
    discovered_ids = {skill.id for skill in discovered}

    assert discovered_ids >= set(REDUX_MINING_SKILL_IDS), (
        "discovery seam failed: Mining Redux source skills were not discovered"
    )

    surviving_non_mining_ids = discovered_ids & NON_MINING_SENTINEL_IDS
    assert not surviving_non_mining_ids, (
        "strict source-boundary seam failed: archived non-mining skill IDs remain "
        f"discoverable after cleanup: {sorted(surviving_non_mining_ids)}"
    )

    redux_skills = filter_redux_skills(discovered)

    assert list(redux_skills) == list(REDUX_MINING_SKILL_IDS), (
        "allowlist seam failed: Redux filtering did not preserve the mining-only order"
    )


@pytest.mark.parametrize(
    ("module_name", "expected_attrs", "seam_label"),
    [
        (
            "skills.Mining_Loadout.mining_loadout_app",
            ("main",),
            "Mining Loadout app import seam failed",
        ),
        (
            "tools.Mining_Signals.mining_signals_app",
            ("main",),
            "Mining Signals app import seam failed",
        ),
        (
            "tools.Mining_Signals.ocr.onnx_hud_reader",
            ("scan_hud_onnx",),
            "Mining Signals OCR HUD entrypoint import seam failed",
        ),
        (
            "tools.Mining_Signals.ocr.sc_ocr.scan_results_match",
            ("find_scan_results_anchor", "reset_anchor_tracker", "reset_cache"),
            "Mining Signals OCR scan-results gate import seam failed",
        ),
    ],
)
def test_redux_source_imports_mining_apps_and_ocr_entrypoints(
    module_name, expected_attrs, seam_label
):
    """Import Redux mining source seams without launching GUI processes or event loops."""
    module = importlib.import_module(module_name)

    missing_attrs = [name for name in expected_attrs if not hasattr(module, name)]
    assert not missing_attrs, f"{seam_label}: missing {missing_attrs!r}"
