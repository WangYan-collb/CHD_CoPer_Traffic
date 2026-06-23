from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from _bootstrap import activate_project_root

activate_project_root()


REQUIRED_MODULES = [
    "numpy",
    "torch",
    "pandas",
    "yaml",
    "matplotlib",
    "traci",
    "sumolib",
]


def main() -> None:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print("Recommended Python version: 3.9.17")
    missing = []
    for module_name in REQUIRED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
            print(f"[missing] {module_name}")
        else:
            print(f"[ok] {module_name}")
    if missing:
        print("Install dependencies with:")
        print("  .venv/bin/python -m pip install -r requirements.txt")
        return
    print("Python dependency check passed.")


if __name__ == "__main__":
    main()
