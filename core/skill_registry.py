"""
Dynamic skill discovery — replaces the static SKILLS list.

Scans the ``skills/`` directory for ``skill.json`` metadata files.
Falls back to a built-in default list so existing skill folders
(which may not yet have a skill.json) continue to work.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping

from shared.config_models import SkillConfig
from shared.i18n import N_

log = logging.getLogger(__name__)

_DEFAULT_LAUNCHER_ENTRY_IDS: dict[str, str] = {
    'mining': 'mining-loadout',
    'mining_signals': 'mining-signals',
}


def _apply_launcher_entry_default(config: SkillConfig) -> SkillConfig:
    if not config.launcher_entry_id:
        config.launcher_entry_id = _DEFAULT_LAUNCHER_ENTRY_IDS.get(config.id, "")
    return config


def load_launcher_entry_ids_from_install_state(install_state) -> set[str] | None:
    """Load an optional launcher-entry allowlist from installer state.

    ``None`` and missing ``launcher_entry_ids`` preserve legacy full-launcher
    discovery.  A supplied allowlist is intentionally strict so corrupted local
    installer metadata cannot silently broaden launcher visibility.
    """
    if install_state is None:
        return None

    if isinstance(install_state, (str, os.PathLike)):
        try:
            with open(install_state, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"install_state is not valid JSON: {exc}") from exc
    else:
        state = install_state

    if not isinstance(state, Mapping):
        raise ValueError("install_state must be a JSON object")

    raw_ids = state.get("launcher_entry_ids")
    if raw_ids is None:
        return None
    if not isinstance(raw_ids, list):
        raise ValueError("install_state launcher_entry_ids must be a list")

    launcher_entry_ids: set[str] = set()
    for raw_id in raw_ids:
        if not isinstance(raw_id, str) or not raw_id:
            raise ValueError("install_state launcher_entry_ids must contain only non-empty strings")
        launcher_entry_ids.add(raw_id)
    return launcher_entry_ids

# Built-in defaults — used when a skill folder has no skill.json
_BUILTIN_SKILLS: list[dict] = [
    {
        'id': 'mining',
        'name': 'Mining Loadout',
        'icon': '⛏',
        'color': '#ffaa22',
        'folder': 'Mining_Loadout',
        'script': 'mining_loadout_app.py',
        'hotkey': '<shift>+4',
        'settings_key': 'hotkey_mining',
        'launcher_entry_id': 'mining-loadout',
    },
]

_BUILTIN_INDEX: dict[str, dict] = {s["id"]: s for s in _BUILTIN_SKILLS}


def _try_load_skill_json(skill_dir: str) -> SkillConfig | None:
    """Load a ``skill.json`` from *skill_dir*, or return None."""
    path = os.path.join(skill_dir, "skill.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cfg = SkillConfig.from_dict(data)
        if not cfg.id or not cfg.script:
            log.warning("skill_registry: invalid skill.json in %s (missing id/script)", skill_dir)
            return None
        return _apply_launcher_entry_default(cfg)
    except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
        log.warning("skill_registry: failed to load %s: %s", path, exc)
        return None


def discover_skills(base_dir: str) -> list[SkillConfig]:
    """Scan for skills and return an ordered list of SkillConfig objects.

    Discovery order:
    1. Scan ``<base_dir>/skills/`` for directories containing ``skill.json``
    2. For known built-in skills whose folders exist but lack ``skill.json``,
       use the built-in default metadata
    3. Result is sorted: discovered skills first (alphabetical), then
       built-in skills in their canonical order

    Parameters
    ----------
    base_dir:
        The SC_Toolbox root directory (contains ``skills/``).
    """
    skills_root = os.path.join(base_dir, "skills")
    tools_root = os.path.join(base_dir, "tools")
    parent_skills = os.path.dirname(base_dir)  # custom_skills/ level

    found: dict[str, SkillConfig] = {}

    # Phase 1: scan for skill.json files (in both skills/ and tools/)
    for scan_root in (skills_root, tools_root):
        if os.path.isdir(scan_root):
            try:
                for entry in sorted(os.listdir(scan_root)):
                    entry_path = os.path.join(scan_root, entry)
                    if not os.path.isdir(entry_path):
                        continue
                    cfg = _try_load_skill_json(entry_path)
                    if cfg:
                        # Override folder to match actual directory name
                        cfg.folder = entry
                        found[cfg.id] = cfg
                        log.debug("skill_registry: discovered %s from skill.json", cfg.id)
            except OSError as exc:
                log.warning("skill_registry: error scanning %s: %s", scan_root, exc)

    # Phase 2: fill in built-in skills that weren't discovered via skill.json
    result: list[SkillConfig] = []
    for builtin in _BUILTIN_SKILLS:
        sid = builtin["id"]
        if sid in found:
            result.append(found.pop(sid))
            continue

        # Check if the folder exists (under skills/, tools/, or parent custom_skills/)
        local = os.path.join(skills_root, builtin["folder"])
        tools = os.path.join(tools_root, builtin["folder"])
        parent = os.path.join(parent_skills, builtin["folder"])
        if os.path.isdir(local) or os.path.isdir(tools) or os.path.isdir(parent):
            result.append(SkillConfig.from_dict(builtin))
            log.debug("skill_registry: using built-in metadata for %s", sid)

    # Phase 3: append any extra discovered skills not in the built-in list
    for cfg in sorted(found.values(), key=lambda c: c.name):
        result.append(cfg)

    log.info("skill_registry: %d skill(s) registered", len(result))
    return result


def resolve_skill_path(skill: SkillConfig, base_dir: str) -> str | None:
    """Return the absolute directory path for a skill, or None if not found."""
    skills_root = os.path.join(base_dir, "skills")
    tools_root = os.path.join(base_dir, "tools")
    parent_skills = os.path.dirname(base_dir)

    local = os.path.join(skills_root, skill.folder)
    if os.path.isdir(local):
        return local
    tools = os.path.join(tools_root, skill.folder)
    if os.path.isdir(tools):
        return tools
    parent = os.path.join(parent_skills, skill.folder)
    if os.path.isdir(parent):
        return parent
    return None


def resolve_script_path(skill: SkillConfig, base_dir: str) -> str | None:
    """Return the absolute path to the skill's entry script, or None."""
    folder = resolve_skill_path(skill, base_dir)
    if not folder:
        return None
    script = os.path.join(folder, skill.script)
    return script if os.path.isfile(script) else None
