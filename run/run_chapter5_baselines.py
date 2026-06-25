from __future__ import annotations

import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from _bootstrap import activate_project_root

activate_project_root()

from src.cli.run_suite import main


SUITE_PATH = "configs/meta_rl/comparison_suite.yaml"
SMOKE = False


def build_args() -> list[str]:
    args = ["--suite", SUITE_PATH]
    if SMOKE:
        args.append("--smoke")
    return args


if __name__ == "__main__":
    raise SystemExit(main(build_args()))
