import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from build.redux_mining_build import (  # noqa: E402
    BuildContractError,
    EXCLUDED_PATHS,
    REQUIRED_PATHS,
    create_source_dist,
    main,
    validate_dist,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_build_command_is_dedicated_redux_surface():
    command = PROJECT_ROOT / "build" / "build_redux_mining.bat"
    helper = PROJECT_ROOT / "build" / "redux_mining_build.py"
    spec = PROJECT_ROOT / "build" / "redux_mining_launcher.spec"

    assert command.exists(), "Redux mining build batch command is missing"
    assert helper.exists(), "Redux mining build helper is missing"
    assert spec.exists(), "Redux mining PyInstaller spec is missing"

    text = command.read_text(encoding="utf-8")
    forbidden = (
        "build_installer.bat",
        "SC_Toolbox_Installer.iss",
        "iscc",
        "installer_profile",
        "bootstrapper",
        "get-pip.py",
        "tesseract-setup.exe",
    )
    for token in forbidden:
        assert token.lower() not in text.lower(), f"Redux build command should not invoke {token}"

    assert "redux_mining_build.py" in text
    assert "SC_Toolbox_Redux_Mining" in text


def test_readme_documents_fast_redux_run_build_and_installer_boundary():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    normalized = readme.replace("/", "\\")
    lowered = normalized.lower()

    assert "RUN_REDUX_MINING.bat" in readme
    assert "build\\build_redux_mining.bat" in normalized
    assert "dist\\SC_Toolbox_Redux_Mining" in normalized
    assert "tools\\Mining_Signals\\tests\\test_ocr_redux_regressions.py" in normalized
    assert "core\\tests\\test_redux_mining_smoke.py" in normalized
    assert "core\\tests\\test_redux_mining_dist_smoke.py" in normalized
    assert "core\\tests\\test_redux_mining_build.py" in normalized
    assert "tools\\Mining_Signals\\tests\\test_local_search.py" in normalized
    assert "Mining Loadout" in readme
    assert "Mining Signals" in readme
    assert "PyInstaller" in readme
    assert "build\\build_installer.bat" in normalized
    assert "Inno Setup" in readme
    assert "bootstrapper" in lowered
    assert "installer profile UI" in readme

    boundary_phrases = (
        "fast redux iteration should not require",
        "separate from the legacy full installer",
        "without the legacy installer flow",
    )
    assert any(phrase in lowered for phrase in boundary_phrases), (
        "README must explain that fast Redux iteration does not use the legacy full installer path"
    )


def test_source_build_stages_required_mining_roots_and_excludes_non_mining(tmp_path):
    out = tmp_path / "SC_Toolbox_Redux_Mining"

    create_source_dist(PROJECT_ROOT, out)

    for rel in REQUIRED_PATHS:
        assert (out / rel).exists(), f"missing required Redux runtime path: {rel}"
    for rel in EXCLUDED_PATHS:
        assert not (out / rel).exists(), f"non-mining path leaked into Redux dist: {rel}"

    assert (out / "skills" / "Mining_Loadout" / "mining_loadout_app.py").exists()
    assert (out / "tools" / "Mining_Signals" / "mining_signals_app.py").exists()
    assert not (out / "skills" / "Mining_Loadout" / ".api_cache").exists()


def test_validate_dist_reports_missing_required_roots(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()

    with pytest.raises(BuildContractError) as excinfo:
        validate_dist(dist, verbose=False)

    message = str(excinfo.value)
    assert "missing required paths" in message
    assert "redux_mining_launcher.py" in message


def test_validate_dist_reports_excluded_non_mining_roots(tmp_path):
    dist = tmp_path / "dist"
    for rel in REQUIRED_PATHS:
        path = dist / rel
        if Path(rel).suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture", encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)
    leaked = dist / "skills" / "DPS_Calculator"
    leaked.mkdir(parents=True)

    with pytest.raises(BuildContractError) as excinfo:
        validate_dist(dist, verbose=False)

    message = str(excinfo.value)
    assert "excluded non-mining paths present" in message
    assert "skills/DPS_Calculator" in message


def test_pyinstaller_mode_fails_with_clear_guidance_when_prerequisite_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("build.redux_mining_build.shutil.which", lambda name: None if name == "pyinstaller" else name)

    rc = main(["--project-root", str(PROJECT_ROOT), "--output", str(tmp_path / "SC_Toolbox_Redux_Mining"), "--pyinstaller"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "PyInstaller is not installed" in captured.err
    assert "python -m pip install pyinstaller" in captured.err
