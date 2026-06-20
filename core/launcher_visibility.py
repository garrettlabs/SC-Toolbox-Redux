"""Runtime launcher skill visibility helpers.

This module is the runtime seam used by installer smoke checks and installed
launchers.  Installer profile membership stays in ``build.installer_profiles``;
this code only applies the launcher-entry allowlist recorded in local installer
state to the discovered runtime skills.
"""
from __future__ import annotations

from pathlib import Path

from core.skill_registry import discover_skills, load_launcher_entry_ids_from_install_state
from skill_launcher import filter_skills_for_launcher
from shared.config_models import SkillConfig


def discover_launcher_skills(base_dir: str | Path, *, install_state=None) -> list[SkillConfig]:
    """Discover skills visible to the launcher for an optional install state.

    ``install_state`` may be ``None``, a mapping, or a path to JSON metadata.
    Missing launcher-entry state preserves the legacy full launcher.  When an
    allowlist is present, every requested launcher-entry ID must correspond to a
    discovered skill; unknown IDs raise ``ValueError`` rather than silently
    broadening or narrowing the installed launcher.
    """
    skills = discover_skills(str(base_dir))
    launcher_entry_ids = load_launcher_entry_ids_from_install_state(install_state)
    if launcher_entry_ids is None:
        return filter_skills_for_launcher(skills)

    entry_to_skill_id: dict[str, str] = {}
    for skill in skills:
        launcher_entry_id = getattr(skill, "launcher_entry_id", "")
        if launcher_entry_id:
            entry_to_skill_id[launcher_entry_id] = skill.id

    unknown = sorted(launcher_entry_ids.difference(entry_to_skill_id))
    if unknown:
        raise ValueError(f"install_state references unknown launcher_entry_ids: {unknown}")

    allowed_skill_ids = [
        entry_to_skill_id[entry_id]
        for entry_id in launcher_entry_ids
    ]
    return filter_skills_for_launcher(skills, allowed_skill_ids)
