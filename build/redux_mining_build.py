"""Build and validate the Redux mining-only distributable tree.

The default build is intentionally lightweight: it creates a source-based
one-folder runtime that can be inspected quickly and launched with Python.  An
optional PyInstaller mode is available for executable experiments, but the
contract remains the same: include the mining runtime roots and exclude unrelated
SC Toolbox modules.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

DIST_NAME = "SC_Toolbox_Redux_Mining"

REQUIRED_PATHS: tuple[str, ...] = (
    "redux_mining_launcher.py",
    "redux_mining_launcher_settings.json",
    "skill_launcher.py",
    "core",
    "shared",
    "ui",
    "skills/Mining_Loadout",
    "tools/Mining_Signals",
    "assets/sc_toolbox.ico",
    "assets/sc_toolbox_logo.png",
    "assets/screenshots/launcher.png",
    "assets/screenshots/mining_loadout.png",
    "assets/screenshots/mining_signals.png",
    "requirements.txt",
)

EXCLUDED_PATHS: tuple[str, ...] = (
    "skills/Cargo_loader",
    "skills/DPS_Calculator",
    "skills/Trade_Hub",
    "skills/Mission_Database",
    "skills/Craft_Database",
    "skills/Market_Finder",
    "tools/Battle_Buddy",
    "tools/PlayTime_Calculator",
)

SOURCE_ROOTS: tuple[str, ...] = (
    "redux_mining_launcher.py",
    "redux_mining_launcher_settings.json",
    "skill_launcher.py",
    "requirements.txt",
    "core",
    "shared",
    "ui",
    "skills/Mining_Loadout",
    "tools/Mining_Signals",
)

ASSET_FILES: tuple[str, ...] = (
    "assets/sc_toolbox.ico",
    "assets/sc_toolbox_logo.png",
    "assets/screenshots/launcher.png",
    "assets/screenshots/mining_loadout.png",
    "assets/screenshots/mining_signals.png",
)

IGNORE_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".api_cache",
    "logs",
}
IGNORE_SUFFIXES = (".pyc", ".pyo", ".log")


class BuildContractError(RuntimeError):
    """Raised when the Redux mining distributable content contract fails."""


def _display(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _ignore_runtime_noise(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in IGNORE_NAMES or name.endswith(IGNORE_SUFFIXES):
            ignored.add(name)
    return ignored


def _copy_path(src: Path, dst: Path) -> None:
    if not src.exists():
        raise BuildContractError(f"required source path is missing: {src.as_posix()}")
    if src.is_dir():
        shutil.copytree(src, dst, ignore=_ignore_runtime_noise, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def create_source_dist(project_root: Path, output_dir: Path, clean: bool = True) -> Path:
    """Create the inspectable Redux mining source distributable."""
    project_root = project_root.resolve()
    dist_root = output_dir.resolve()
    print(f"[redux-build] project root: {_display(project_root)}")
    print(f"[redux-build] output directory: {_display(dist_root)}")

    if clean and dist_root.exists():
        print(f"[redux-build] cleaning existing output: {_display(dist_root)}")
        shutil.rmtree(dist_root)
    dist_root.mkdir(parents=True, exist_ok=True)

    for rel in SOURCE_ROOTS:
        print(f"[redux-build] include: {rel}")
        _copy_path(project_root / rel, dist_root / rel)
    for rel in ASSET_FILES:
        print(f"[redux-build] include: {rel}")
        _copy_path(project_root / rel, dist_root / rel)

    validate_dist(dist_root, verbose=True)
    print(f"[redux-build] complete: {_display(dist_root)}")
    return dist_root


def validate_dist(dist_root: Path, verbose: bool = True) -> None:
    """Assert the Redux mining distributable include/exclude contract."""
    dist_root = dist_root.resolve()
    missing = [rel for rel in REQUIRED_PATHS if not (dist_root / rel).exists()]
    present_excluded = [rel for rel in EXCLUDED_PATHS if (dist_root / rel).exists()]

    if verbose:
        for rel in REQUIRED_PATHS:
            status = "OK" if (dist_root / rel).exists() else "MISSING"
            print(f"[redux-build] require {status}: {rel}")
        for rel in EXCLUDED_PATHS:
            status = "ABSENT" if not (dist_root / rel).exists() else "PRESENT"
            print(f"[redux-build] exclude {status}: {rel}")

    if missing or present_excluded:
        parts: list[str] = []
        if missing:
            parts.append("missing required paths: " + ", ".join(missing))
        if present_excluded:
            parts.append("excluded non-mining paths present: " + ", ".join(present_excluded))
        raise BuildContractError("; ".join(parts))


def run_pyinstaller(project_root: Path, output_parent: Path) -> int:
    """Run the optional PyInstaller spec, failing clearly when unavailable."""
    pyinstaller = shutil.which("pyinstaller")
    if not pyinstaller:
        print(
            "[redux-build] prerequisite failure: PyInstaller is not installed or not on PATH.\n"
            "[redux-build] Install it with: python -m pip install pyinstaller\n"
            "[redux-build] Or run without --pyinstaller for the fast source distributable.",
            file=sys.stderr,
        )
        return 2

    spec = project_root / "build" / "redux_mining_launcher.spec"
    cmd = [pyinstaller, "--noconfirm", "--clean", "--distpath", str(output_parent), str(spec)]
    print(f"[redux-build] invoked command: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(project_root), check=False)
    if completed.returncode == 0:
        validate_dist(output_parent / DIST_NAME, verbose=True)
    return completed.returncode


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or validate the Redux mining-only distributable.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--output", default=None, type=Path, help="Output dist root; defaults to dist/SC_Toolbox_Redux_Mining")
    parser.add_argument("--validate-only", action="store_true", help="Only validate an existing staged dist tree")
    parser.add_argument("--no-clean", action="store_true", help="Do not remove the existing output before staging")
    parser.add_argument("--pyinstaller", action="store_true", help="Run PyInstaller using build/redux_mining_launcher.spec")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    project_root: Path = args.project_root.resolve()
    output = (args.output or (project_root / "dist" / DIST_NAME)).resolve()

    print(f"[redux-build] selected Python: {sys.executable}")
    print(f"[redux-build] mode: {'validate-only' if args.validate_only else 'pyinstaller' if args.pyinstaller else 'source-dist'}")
    print(f"[redux-build] output directory: {_display(output)}")

    try:
        if args.validate_only:
            validate_dist(output, verbose=True)
            return 0
        if args.pyinstaller:
            return run_pyinstaller(project_root, output.parent)
        create_source_dist(project_root, output, clean=not args.no_clean)
        return 0
    except BuildContractError as exc:
        print(f"[redux-build] contract failure: {exc}", file=sys.stderr)
        return 3
    except (OSError, shutil.Error, subprocess.SubprocessError) as exc:
        print(f"[redux-build] build failure: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
