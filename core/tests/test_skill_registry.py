import os, sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import shared.path_setup
shared.path_setup.ensure_path(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import json
import pytest

from core.skill_registry import (
    _try_load_skill_json,
    discover_skills,
    resolve_skill_path,
    resolve_script_path,
)
from redux_mining_launcher import REDUX_MINING_SKILL_IDS, filter_redux_skills
from skill_launcher import filter_skills_for_launcher
from shared.config_models import SkillConfig


def _make_skill_json(directory, data):
    """Write a skill.json into *directory* and return the path."""
    path = os.path.join(directory, "skill.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def _valid_skill_data(**overrides):
    base = {
        "id": "test_skill",
        "name": "Test Skill",
        "icon": "T",
        "color": "#123456",
        "folder": "Test_Skill",
        "script": "test_app.py",
        "hotkey": "<shift>+9",
        "settings_key": "hotkey_test",
    }
    base.update(overrides)
    return base


# ── _try_load_skill_json ────────────────────────────────────────────────────


def test_try_load_skill_json_valid(tmp_path):
    data = _valid_skill_data()
    _make_skill_json(tmp_path, data)
    cfg = _try_load_skill_json(str(tmp_path))
    assert cfg is not None
    assert isinstance(cfg, SkillConfig)
    assert cfg.id == "test_skill"
    assert cfg.script == "test_app.py"
    assert cfg.name == "Test Skill"


def test_try_load_skill_json_missing_file(tmp_path):
    assert _try_load_skill_json(str(tmp_path)) is None


def test_try_load_skill_json_invalid_json(tmp_path):
    bad = tmp_path / "skill.json"
    bad.write_text("{not valid json!!!", encoding="utf-8")
    assert _try_load_skill_json(str(tmp_path)) is None


def test_try_load_skill_json_missing_id(tmp_path):
    data = _valid_skill_data()
    del data["id"]  # from_dict defaults missing id to "", which is falsy
    _make_skill_json(tmp_path, data)
    assert _try_load_skill_json(str(tmp_path)) is None


# ── launcher filtering seam ─────────────────────────────────────────────────


def test_filter_skills_for_launcher_default_preserves_all_skills():
    skills = [
        SkillConfig(id="dps", name="DPS", icon="D", color="#111", folder="DPS", script="dps.py"),
        SkillConfig(id="mining", name="Mining", icon="M", color="#222", folder="Mining", script="mining.py"),
    ]

    assert filter_skills_for_launcher(skills) == skills


def test_filter_skills_for_launcher_returns_allowed_subset_in_discovery_order():
    skills = [
        SkillConfig(id="dps", name="DPS", icon="D", color="#111", folder="DPS", script="dps.py"),
        SkillConfig(id="mining", name="Mining", icon="M", color="#222", folder="Mining", script="mining.py"),
        SkillConfig(id="cargo", name="Cargo", icon="C", color="#333", folder="Cargo", script="cargo.py"),
        SkillConfig(id="mining_signals", name="Mining Signals", icon="S", color="#444", folder="Signals", script="signals.py"),
    ]

    filtered = filter_skills_for_launcher(skills, {"mining_signals", "mining"})

    assert [skill.id for skill in filtered] == ["mining", "mining_signals"]


def test_filter_skills_for_launcher_rejects_unknown_required_ids():
    skills = [
        SkillConfig(id="mining", name="Mining", icon="M", color="#222", folder="Mining", script="mining.py"),
    ]

    with pytest.raises(ValueError, match="missing_required"):
        filter_skills_for_launcher(skills, {"mining", "missing_required"})


def test_redux_mining_filter_matches_repository_skill_boundary():
    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    skills = discover_skills(repo_root)

    all_by_id = {skill.id: skill for skill in skills}
    assert "dps" in all_by_id

    filtered = filter_redux_skills(skills)

    assert list(filtered) == ["mining", "mining_signals"]
    assert set(filtered) == set(REDUX_MINING_SKILL_IDS)
    assert "dps" not in filtered

    expected_scripts = {
        "mining": os.path.join("skills", "Mining_Loadout", "mining_loadout_app.py"),
        "mining_signals": os.path.join("tools", "Mining_Signals", "mining_signals_app.py"),
    }
    for skill_id, expected_relative_path in expected_scripts.items():
        script_path = resolve_script_path(filtered[skill_id], repo_root)
        assert script_path is not None
        assert os.path.normcase(os.path.relpath(script_path, repo_root)) == os.path.normcase(expected_relative_path)


def test_redux_source_run_targets_entrypoint_and_settings_file():
    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    wrapper_path = os.path.join(repo_root, "RUN_REDUX_MINING.bat")

    from redux_mining_launcher import REDUX_MINING_SETTINGS_FILE

    assert os.path.isfile(os.path.join(repo_root, "redux_mining_launcher.py"))
    assert os.path.isfile(wrapper_path)
    assert os.path.normcase(os.path.relpath(REDUX_MINING_SETTINGS_FILE, repo_root)) == os.path.normcase(
        "redux_mining_launcher_settings.json"
    )

    wrapper = open(wrapper_path, encoding="utf-8").read().lower()
    assert "redux_mining_launcher.py" in wrapper
    assert "selected python" in wrapper
    assert "command:" in wrapper


def test_redux_source_run_wrapper_avoids_legacy_full_installer_surface():
    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    wrapper = open(os.path.join(repo_root, "RUN_REDUX_MINING.bat"), encoding="utf-8").read().lower()

    assert "skill_launcher.py" not in wrapper
    assert "build_installer" not in wrapper
    assert "install_and_launch" not in wrapper
    assert "iscc" not in wrapper
    assert "inno" not in wrapper
    assert "%1" not in wrapper
    assert "%*" not in wrapper


# ── discover_skills ─────────────────────────────────────────────────────────


def test_discover_skills_with_json(tmp_path):
    base = tmp_path / "project"
    skills_dir = base / "skills"

    # Create two skill folders with valid skill.json
    for name, sid in [("Alpha_Skill", "alpha"), ("Beta_Skill", "beta")]:
        d = skills_dir / name
        d.mkdir(parents=True)
        _make_skill_json(str(d), _valid_skill_data(id=sid, name=name, folder=name, script="app.py"))

    result = discover_skills(str(base))
    ids = [s.id for s in result]
    assert "alpha" in ids
    assert "beta" in ids
    assert len(result) == 2


def test_discover_skills_fallback_builtin(tmp_path):
    base = tmp_path / "project"
    skills_dir = base / "skills"

    # Create a folder matching a built-in skill but without skill.json
    (skills_dir / "Trade_Hub").mkdir(parents=True)

    result = discover_skills(str(base))
    ids = [s.id for s in result]
    assert "trade" in ids
    # Should use the built-in metadata
    trade = [s for s in result if s.id == "trade"][0]
    assert trade.script == "trade_hub_app.py"


def test_discover_skills_empty(tmp_path):
    base = tmp_path / "project"
    (base / "skills").mkdir(parents=True)
    result = discover_skills(str(base))
    assert result == []


# ── resolve_skill_path ──────────────────────────────────────────────────────


def test_resolve_skill_path_found(tmp_path):
    base = tmp_path / "project"
    skill_dir = base / "skills" / "My_Skill"
    skill_dir.mkdir(parents=True)

    skill = SkillConfig(
        id="my", name="My Skill", icon="M", color="#000",
        folder="My_Skill", script="app.py",
    )
    result = resolve_skill_path(skill, str(base))
    assert result is not None
    assert os.path.basename(result) == "My_Skill"


def test_resolve_skill_path_not_found(tmp_path):
    base = tmp_path / "project"
    (base / "skills").mkdir(parents=True)

    skill = SkillConfig(
        id="nope", name="Nope", icon="N", color="#000",
        folder="Nonexistent", script="app.py",
    )
    assert resolve_skill_path(skill, str(base)) is None


# ── resolve_script_path ─────────────────────────────────────────────────────


def test_resolve_script_path_found(tmp_path):
    base = tmp_path / "project"
    skill_dir = base / "skills" / "My_Skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "app.py").write_text("# entry", encoding="utf-8")

    skill = SkillConfig(
        id="my", name="My Skill", icon="M", color="#000",
        folder="My_Skill", script="app.py",
    )
    result = resolve_script_path(skill, str(base))
    assert result is not None
    assert result.endswith("app.py")
    assert os.path.isfile(result)


def test_resolve_script_path_not_found(tmp_path):
    base = tmp_path / "project"
    skill_dir = base / "skills" / "My_Skill"
    skill_dir.mkdir(parents=True)
    # Folder exists but script file does not

    skill = SkillConfig(
        id="my", name="My Skill", icon="M", color="#000",
        folder="My_Skill", script="missing.py",
    )
    assert resolve_script_path(skill, str(base)) is None


# ── launcher visibility install_state boundary (unittest-discoverable) ─────────

import tempfile
import unittest

from core.launcher_visibility import discover_launcher_skills
from core.skill_registry import load_launcher_entry_ids_from_install_state


class LauncherVisibilityInstallStateTests(unittest.TestCase):
    def _make_runtime_root(self):
        temp = tempfile.TemporaryDirectory()
        root = temp.name
        os.makedirs(os.path.join(root, "skills", "Mining_Loadout"), exist_ok=True)
        mining_signals_dir = os.path.join(root, "tools", "Mining_Signals")
        os.makedirs(mining_signals_dir, exist_ok=True)
        _make_skill_json(
            mining_signals_dir,
            {
                "id": "mining_signals",
                "name": "Mining Signals",
                "icon": "M",
                "color": "#ffaa22",
                "folder": "Mining_Signals",
                "script": "main.py",
            },
        )
        return temp, root

    def test_valid_mining_install_state_filters_visible_launcher_entries(self):
        temp, root = self._make_runtime_root()
        self.addCleanup(temp.cleanup)

        visible = discover_launcher_skills(
            root,
            install_state={"launcher_entry_ids": ["mining-signals"]},
        )

        self.assertEqual([skill.id for skill in visible], ["mining_signals"])
        self.assertEqual([skill.launcher_entry_id for skill in visible], ["mining-signals"])

    def test_missing_install_state_preserves_legacy_full_launcher(self):
        temp, root = self._make_runtime_root()
        self.addCleanup(temp.cleanup)

        visible = discover_launcher_skills(root)
        visible_ids = {skill.launcher_entry_id for skill in visible}

        self.assertIn("mining-loadout", visible_ids)
        self.assertIn("mining-signals", visible_ids)

    def test_malformed_launcher_entry_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "launcher_entry_ids must contain only non-empty strings"):
            load_launcher_entry_ids_from_install_state({"launcher_entry_ids": ["mining-signals", 42]})

    def test_unknown_launcher_entry_id_rejected(self):
        temp, root = self._make_runtime_root()
        self.addCleanup(temp.cleanup)

        with self.assertRaisesRegex(ValueError, "unknown launcher_entry_ids"):
            discover_launcher_skills(root, install_state={"launcher_entry_ids": ["not-a-runtime-entry"]})
