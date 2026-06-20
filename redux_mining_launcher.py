"""Redux mining-only entrypoint for SC Toolbox.

This module keeps the existing launcher CLI contract while constraining the
visible/launchable surface to the mining tools Redux needs.
"""

from __future__ import annotations

from collections.abc import Iterable
import os

from shared.config_models import SkillConfig

REDUX_MINING_SKILL_IDS = ("mining", "mining_signals")

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REDUX_MINING_SETTINGS_FILE = os.path.join(
    _ROOT_DIR,
    "redux_mining_launcher_settings.json",
)


def filter_redux_skills(skills: Iterable[SkillConfig]) -> dict[str, SkillConfig]:
    """Return Redux mining launcher skills keyed by their hard-allowlisted IDs."""
    from skill_launcher import filter_skills_for_launcher

    return {
        skill.id: skill
        for skill in filter_skills_for_launcher(skills, REDUX_MINING_SKILL_IDS)
    }


def main() -> None:
    """Run the existing launcher in Redux mining-only mode."""
    from skill_launcher import main as launcher_main

    launcher_main(
        allowed_skill_ids=REDUX_MINING_SKILL_IDS,
        settings_file=REDUX_MINING_SETTINGS_FILE,
        logging_name="redux_mining_launcher",
    )


if __name__ == "__main__":
    main()
