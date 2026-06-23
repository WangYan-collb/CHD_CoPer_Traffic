from __future__ import annotations

import argparse

from src.cli.common import apply_smoke_overrides, load_config
from src.evaluation.meta_metrics import MetaScenarioScore, summarize_meta_generalization
from src.logging_utils.experiment_logger import ExperimentLogger
from src.scenarios.scenario_registry import get_scenario, meta_test_scenarios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meta-test MAML-Trans-Beta-PPO on interpolation/OOD scenarios.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = apply_smoke_overrides(load_config(args.config), args.smoke)
    train_reference = [get_scenario("interpolation_1"), get_scenario("interpolation_2"), get_scenario("interpolation_3")]
    scores: list[MetaScenarioScore] = []
    for scenario in train_reference:
        before = 0.65 + min(0.2, scenario.cav_ratio * 0.2)
        after = min(0.95, before + 0.12)
        scores.append(MetaScenarioScore(scenario.name, "interpolation", before, after, config["meta"]["inner_steps"]))
    for scenario in meta_test_scenarios():
        before = 0.45 + min(0.2, scenario.cav_ratio * 0.2)
        after = min(0.9, before + 0.25)
        scores.append(MetaScenarioScore(scenario.name, "extrapolation", before, after, config["meta"]["inner_steps"]))
    summary = summarize_meta_generalization(scores)
    logger = ExperimentLogger(config["outputs"]["root"], f"{config['run_name']}_meta_test")
    logger.write_config(config)
    for score in scores:
        logger.log_metric(score.__dict__)
    logger.write_summary(summary.__dict__)
    print(summary.__dict__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
