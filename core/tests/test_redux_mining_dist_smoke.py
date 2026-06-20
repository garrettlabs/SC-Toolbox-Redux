import os
import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIST_ROOT = PROJECT_ROOT / "dist" / "SC_Toolbox_Redux_Mining"

NON_MINING_DIST_ROOTS = (
    Path("skills/Cargo_loader"),
    Path("skills/DPS_Calculator"),
    Path("skills/Trade_Hub"),
    Path("skills/Mission_Database"),
    Path("skills/Craft_Database"),
    Path("skills/Market_Finder"),
    Path("tools/Battle_Buddy"),
    Path("tools/PlayTime_Calculator"),
)

DIST_IMPORT_SCRIPT = r"""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

project_root = Path(os.environ["SC_TOOLBOX_SOURCE_ROOT"]).resolve()
dist_root = Path(os.environ["SC_TOOLBOX_DIST_ROOT"]).resolve()
mining_signals_root = dist_root / "tools" / "Mining_Signals"

# Keep the staged root and Mining Signals local package root first, then remove
# source-checkout paths so imports cannot silently pass by falling back to source.
sys.path[:] = [
    str(dist_root),
    str(mining_signals_root),
    *[
        entry
        for entry in sys.path
        if entry
        and not Path(entry).resolve().is_relative_to(project_root)
        and Path(entry).resolve() != dist_root
        and Path(entry).resolve() != mining_signals_root
    ],
]
os.chdir(dist_root)

modules = {
    "redux_mining_launcher": ("REDUX_MINING_SKILL_IDS", "filter_redux_skills"),
    "shared.app_bootstrap": ("bootstrap_skill",),
    "skills.Mining_Loadout.mining_loadout_app": ("main",),
    "tools.Mining_Signals.mining_signals_app": ("main",),
    "tools.Mining_Signals.ocr.onnx_hud_reader": ("scan_hud_onnx",),
    "tools.Mining_Signals.ocr.sc_ocr.scan_results_match": (
        "find_scan_results_anchor",
        "reset_anchor_tracker",
        "reset_cache",
    ),
}

loaded = {}
for module_name, attrs in modules.items():
    module = importlib.import_module(module_name)
    module_file = Path(getattr(module, "__file__", "")).resolve()
    if not module_file.is_relative_to(dist_root):
        raise AssertionError(
            f"source-path leakage: {module_name} imported from {module_file}, "
            f"expected a path under {dist_root}"
        )
    missing = [attr for attr in attrs if not hasattr(module, attr)]
    if missing:
        raise AssertionError(f"{module_name} missing expected attrs: {missing!r}")
    loaded[module_name] = str(module_file.relative_to(dist_root))

for forbidden in (
    "skills.DPS_Calculator.dps_calculator_app",
    "skills.Trade_Hub.trade_hub_app",
    "tools.Battle_Buddy.battle_buddy_app",
):
    try:
        spec = importlib.util.find_spec(forbidden)
    except ModuleNotFoundError:
        spec = None
    if spec is not None:
        raise AssertionError(f"non-mining module unexpectedly importable from staged dist: {forbidden}")

print(json.dumps(loaded, sort_keys=True))
"""


def test_redux_mining_dist_tree_exists_before_dist_smoke():
    assert DIST_ROOT.exists(), (
        "missing staged dist tree: run build\\build_redux_mining.bat before the dist smoke"
    )
    assert (DIST_ROOT / "redux_mining_launcher.py").exists(), (
        "missing staged launcher: build output is incomplete"
    )


def test_redux_mining_dist_excludes_obvious_non_mining_roots():
    missing_contract = [rel.as_posix() for rel in NON_MINING_DIST_ROOTS if (DIST_ROOT / rel).exists()]
    assert not missing_contract, (
        "non-mining root leaked into Redux mining dist: " + ", ".join(missing_contract)
    )


def test_redux_mining_dist_imports_from_staged_root_without_source_checkout_fallback():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(DIST_ROOT), str(DIST_ROOT / "tools" / "Mining_Signals")]
    )
    env["SC_TOOLBOX_SOURCE_ROOT"] = str(PROJECT_ROOT)
    env["SC_TOOLBOX_DIST_ROOT"] = str(DIST_ROOT)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    completed = subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(DIST_IMPORT_SCRIPT)],
        cwd=str(DIST_ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, (
        "dist import isolation failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert "redux_mining_launcher.py" in completed.stdout
    assert "tools" in completed.stdout
