from __future__ import annotations

import argparse
from pathlib import Path

from src.cli.common import load_config
from src.cli.train import main as train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a comparison suite of RL configs.")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suite = load_config(args.suite)
    configs = suite.get("configs") or suite.get("baseline_configs") or []
    if not configs:
        raise ValueError(f"suite {args.suite} does not define configs or baseline_configs")
    for config_path in configs:
        train_args = ["--config", str(Path(config_path))]
        if args.smoke:
            train_args.append("--smoke")
        train_main(train_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
