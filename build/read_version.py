"""Print the pyproject.toml version string. Used by build_installer.bat
to feed `vpk pack -v` without fighting cmd's quote/paren handling."""
import re
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("usage: read_version.py <path-to-pyproject.toml>", file=sys.stderr)
    sys.exit(2)

text = Path(sys.argv[1]).read_text(encoding="utf-8")
m = re.search(r'^\s*version\s*=\s*"([^"]+)"', text, re.M)
print(m.group(1) if m else "")
