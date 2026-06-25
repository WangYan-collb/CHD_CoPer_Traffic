from __future__ import annotations

import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from _bootstrap import activate_project_root

activate_project_root()

from src.cli.test import main


CONFIG_PATH = "configs/rl/trans_beta_ppo.yaml"
CHECKPOINT: str | None = None
SMOKE = False


def build_args() -> list[str]:
    args = ["--config", CONFIG_PATH]
    if CHECKPOINT:
        args.extend(["--checkpoint", CHECKPOINT])
    if SMOKE:
        args.append("--smoke")
    return args


if __name__ == "__main__":
    raise SystemExit(main(build_args()))
