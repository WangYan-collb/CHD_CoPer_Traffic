from __future__ import annotations

import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from _bootstrap import activate_project_root

activate_project_root()

from src.cli.meta_train import main


CONFIG_PATH = "configs/meta_rl/reptile_trans_beta_ppo.yaml"
SMOKE = False
EPISODES: int | None = None


def build_args() -> list[str]:
    args = ["--config", CONFIG_PATH]
    if SMOKE:
        args.append("--smoke")
    if EPISODES is not None:
        args.extend(["--episodes", str(EPISODES)])
    return args


if __name__ == "__main__":
    raise SystemExit(main(build_args()))
