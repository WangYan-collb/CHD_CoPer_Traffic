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
    print("Recommended Windows interpreter: Python 3.9.17 x64")
    missing = []
    for module_name in REQUIRED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
            print(f"[missing] {module_name}")
        else:
            print(f"[ok] {module_name}")
    if missing:
        print("Install dependencies with:")
        print(r"  .venv\Scripts\python.exe -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121")
        print(r"  .venv\Scripts\python.exe -m pip install -r requirements-windows-gpu.txt")
        return
    import torch

    print(f"torch version: {torch.__version__}")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device count: {torch.cuda.device_count()}")
        print(f"cuda device 0: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Check NVIDIA driver and the CUDA PyTorch wheel.")
    print("Python dependency check passed.")


if __name__ == "__main__":
    main()
