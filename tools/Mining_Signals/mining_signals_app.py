# Mining Signals — powered by Star Citizen Mining Signals spreadsheet
#
# This is the sole entry point.  The sys.path adjustment here is the
# only place it exists — all internal modules use relative imports.

import os
import sys

# Force Qt Multimedia to use the Windows-native backend BEFORE any
# Qt modules load. Qt6's default FFmpeg backend fails on our bundled
# mp3 with "# channels not specified" (Qt reads this env var once at
# plugin load, so the assignment must precede every Qt import).
if os.name == "nt" and not os.environ.get("QT_MEDIA_BACKEND"):
    os.environ["QT_MEDIA_BACKEND"] = "windows"

# Bootstrap project root and skill directory
_SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SKILL_DIR, '..', '..'))
for _path in (_PROJECT_ROOT, _SKILL_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from shared.app_bootstrap import bootstrap_skill  # noqa: E402
bootstrap_skill(__file__)

from ui.app import main  # noqa: E402


if __name__ == "__main__":
    main()
