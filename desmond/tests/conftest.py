"""Make desmond/ (for `src.*`) and the repository root (advisor's fdtd2d, ct_to_speed) importable."""

import sys
from pathlib import Path

DESMOND = Path(__file__).resolve().parents[1]
REPO_ROOT = DESMOND.parent
for path in (DESMOND, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: GPU/model runs taking minutes (deselect with -m 'not slow')")
