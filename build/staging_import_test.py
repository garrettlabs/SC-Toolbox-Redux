"""Staging import smoke test.

For every skill in ``staging/skills/*`` and every tool in ``staging/tools/*``,
discover its entry script (via ``skill.json`` or the built-in fallback list
that mirrors ``core/skill_registry.py``) and import it in a fresh subprocess
of the staging Python.

The entry scripts are guarded by ``if __name__ == "__main__"``, so importing
them executes their top-level statements (sys.path setup, bootstrap_skill,
all transitive imports) without launching the GUI. Any missing pip package,
stripped runtime module, or broken import surfaces as a failed subprocess.

Each skill runs in its own subprocess so state cannot leak between tests.

Usage:
    staging\\python\\python.exe build\\staging_import_test.py <staging_root>

Exit codes:
    0 — all entry points imported cleanly
    1 — at least one failed (per-skill traceback printed)
    2 — bad invocation
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Mirrors core/skill_registry._BUILTIN_SKILLS — used when a skill folder
# has no skill.json. Adding a new skill that ships a skill.json is fully
# auto-discovered; only legacy folders without skill.json need an entry here.
_BUILTIN_SKILLS = [
    {"folder": "DPS_Calculator", "script": "dps_calc_app.py"},
    {"folder": "Cargo_loader", "script": "cargo_app.py"},
    {"folder": "Mission_Database", "script": "mission_db_app.py"},
    {"folder": "Mining_Loadout", "script": "mining_loadout_app.py"},
    {"folder": "Market_Finder", "script": "uex_item_browser.py"},
    {"folder": "Trade_Hub", "script": "trade_hub_app.py"},
    {"folder": "Craft_Database", "script": "craft_db_app.py"},
    {"folder": "Battle_Buddy", "script": "hud_app.py"},
    {"folder": "Mouse_Blocker", "script": "mouse_blocker_app.py"},
]


def _discover_entries(staging_root: Path) -> list[tuple[str, Path]]:
    """Return [(skill_id, entry_script_abs_path)] for every staging skill/tool."""
    found: list[tuple[str, Path]] = []
    for parent in ("skills", "tools"):
        parent_dir = staging_root / parent
        if not parent_dir.is_dir():
            continue
        for entry in sorted(parent_dir.iterdir()):
            if not entry.is_dir():
                continue

            script_name: str | None = None
            sid = entry.name

            sj = entry / "skill.json"
            if sj.is_file():
                try:
                    data = json.loads(sj.read_text(encoding="utf-8"))
                    script_name = data.get("script")
                    sid = data.get("id", entry.name)
                except Exception as exc:
                    print(f"  [WARN] {entry.name}: invalid skill.json — {exc}")

            if script_name is None:
                for builtin in _BUILTIN_SKILLS:
                    if builtin["folder"] == entry.name:
                        script_name = builtin["script"]
                        break

            if script_name is None:
                # Folder we don't know how to enter — not a hard fail.
                # If you add a new skill, ship a skill.json (preferred) or
                # add it to _BUILTIN_SKILLS above.
                print(f"  [WARN] {entry.name}: no skill.json + not in built-in list — skipped")
                continue

            script_path = entry / script_name
            if not script_path.is_file():
                print(f"  [WARN] {entry.name}: entry script {script_name} missing on disk")
                continue
            found.append((sid, script_path))
    return found


def _smoke_test(staging_python: Path, skill_id: str, script_path: Path) -> tuple[bool, str]:
    """Spawn staging Python on the entry script in import-only mode."""
    # Run the entry as if Python imported it (not as __main__), so the
    # if __name__ == '__main__' guard skips main() but every top-level
    # statement, including the sys.path bootstrap and all transitive
    # imports, executes exactly as it would at launch.
    bootstrap = (
        "import os, sys, importlib.util\n"
        # Headless Qt — every skill is a PySide6 GUI; this lets imports
        # finish without trying to open a real window.
        "os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')\n"
        "os.environ.setdefault('QT_MEDIA_BACKEND', 'windows')\n"
        f"_path = r'''{script_path}'''\n"
        "spec = importlib.util.spec_from_file_location('_smoke', _path)\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        # Pretend it's not __main__ so the guard skips main()
        "mod.__name__ = '_smoke'\n"
        "sys.modules['_smoke'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "print('IMPORT_OK')\n"
    )
    try:
        result = subprocess.run(
            [str(staging_python), "-c", bootstrap],
            capture_output=True,
            timeout=180,
            cwd=str(script_path.parent),
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT — import did not complete within 180s"
    except Exception as exc:
        return False, f"failed to spawn staging Python: {exc}"

    out = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    if result.returncode == 0 and "IMPORT_OK" in out:
        return True, ""
    return False, out or f"(no output, rc={result.returncode})"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: staging_import_test.py <staging_root>", file=sys.stderr)
        return 2

    staging_root = Path(sys.argv[1]).resolve()
    staging_python = staging_root / "python" / "python.exe"
    if not staging_root.is_dir():
        print(f"  [!] staging root not a directory: {staging_root}", file=sys.stderr)
        return 2
    if not staging_python.is_file():
        print(f"  [!] staging Python missing at: {staging_python}", file=sys.stderr)
        return 2

    print(f"  [*] Smoke-testing imports under {staging_python}")
    entries = _discover_entries(staging_root)
    if not entries:
        print("  [!] No skill/tool entry scripts discovered — staging looks empty")
        return 1

    failures: list[tuple[str, Path, str]] = []
    for skill_id, script_path in entries:
        rel = script_path.relative_to(staging_root)
        sys.stdout.write(f"  [*] Import {skill_id:<24} ({rel}) ... ")
        sys.stdout.flush()
        ok, err = _smoke_test(staging_python, skill_id, script_path)
        if ok:
            sys.stdout.write("OK\n")
        else:
            sys.stdout.write("FAIL\n")
            failures.append((skill_id, rel, err))
        sys.stdout.flush()

    if failures:
        print()
        print(f"  [!] {len(failures)} of {len(entries)} skill/tool import(s) failed:")
        print()
        for sid, rel, err in failures:
            print(f"  ===== {sid} ({rel}) =====")
            print(err.rstrip())
            print()
        return 1

    print(f"  [OK] All {len(entries)} skill/tool entry points import cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
