# -*- mode: python ; coding: utf-8 -*-
"""Optional PyInstaller spec for the Redux mining-only launcher.

The primary S03 build path is the fast source distributable created by
build/redux_mining_build.py.  This spec is intentionally narrow and mirrors the
same mining-only include/exclude contract for teams that explicitly choose
--pyinstaller.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent.parent
DIST_NAME = "SC_Toolbox_Redux_Mining"


def tree(rel: str, prefix: str | None = None):
    target = prefix or rel
    return Tree(str(ROOT / rel), prefix=target, excludes=["__pycache__", "*.pyc", "*.pyo", "*.log", ".api_cache"])


datas = [
    (str(ROOT / "redux_mining_launcher_settings.json"), "."),
    (str(ROOT / "requirements.txt"), "."),
    (str(ROOT / "assets" / "sc_toolbox.ico"), "assets"),
    (str(ROOT / "assets" / "sc_toolbox_logo.png"), "assets"),
    (str(ROOT / "assets" / "screenshots" / "launcher.png"), "assets/screenshots"),
    (str(ROOT / "assets" / "screenshots" / "mining_loadout.png"), "assets/screenshots"),
    (str(ROOT / "assets" / "screenshots" / "mining_signals.png"), "assets/screenshots"),
    tree("core"),
    tree("shared"),
    tree("ui"),
    tree("skills/Mining_Loadout"),
    tree("tools/Mining_Signals"),
]

hiddenimports = []
hiddenimports += collect_submodules("core")
hiddenimports += collect_submodules("shared")
hiddenimports += collect_submodules("ui")


a = Analysis(
    [str(ROOT / "redux_mining_launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="redux_mining_launcher",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=str(ROOT / "assets" / "sc_toolbox.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=DIST_NAME,
)
