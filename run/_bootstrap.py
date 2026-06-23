from __future__ import annotations

import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def activate_project_root() -> Path:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(PROJECT_ROOT)
    return PROJECT_ROOT
