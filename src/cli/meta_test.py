from __future__ import annotations

import argparse
import copy

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
    import numpy as np

    from src.algorithms.registry import create_agent
    from src.envs.sumo.env import SumoMovingBottleneckEnv

    config = apply_smoke_overrides(load_config(args.config), args.smoke)
    env_cfg = config["environment"]
    algorithm_config = dict(config["algorithm"])
    algorithm_name = algorithm_config.pop("name", "trans_beta_ppo")
    agent = create_agent(
        algorithm_name,
        state_dim=env_cfg["state_dim"],
        action_dim=env_cfg["action_dim"],
        sequence_length=env_cfg["sequence_length"],
        config=algorithm_config,
    )
    if args.checkpoint:
        agent.load(args.checkpoint)
    base_state = copy.deepcopy(agent.policy.state_dict())

    def make_env(scenario):
        return SumoMovingBottleneckEnv(
            scenario=scenario,
            sequence_length=env_cfg["sequence_length"],
            state_dim=env_cfg["state_dim"],
            smoke=config["smoke"],
            aggregation_time_s=int(env_cfg.get("aggregation_time_s", env_cfg.get("control_cycle_s", 120))),
            control_cycle_s=int(env_cfg.get("control_cycle_s", env_cfg.get("aggregation_time_s", 120))),
            simulation_time_s=int(env_cfg.get("simulation_time_s", 3600)),
            congestion_prediction_enabled=bool(env_cfg.get("congestion_prediction_enabled", True)),
            net_file=env_cfg.get("net_file", "data/sumo/base_network/test1.net.xml"),
            additional_file=env_cfg.get("additional_file", "data/sumo/base_network/E2_info.xml"),
            use_gui=bool(env_cfg.get("use_gui", False)),
        )

    def evaluate_episode(scenario) -> float:
        env = make_env(scenario)
        rewards: list[float] = []
        try:
            state, _ = env.reset()
            done = False
            while not done:
                action, _ = agent.select_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                rewards.append(reward)
                done = terminated or truncated
        finally:
            env.close()
        return float(np.mean(rewards)) if rewards else 0.0

    def adapt_episode(scenario) -> float:
        env = make_env(scenario)
        rewards: list[float] = []
        try:
            state, _ = env.reset()
            done = False
            while not done:
                action, info = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.store_transition(
                    state,
                    np.asarray(action, dtype=np.float32),
                    info["log_prob"],
                    reward,
                    terminated or truncated,
                    info["value"],
                )
                rewards.append(reward)
                state = next_state
                done = terminated or truncated
            agent.update()
        finally:
            env.close()
        return float(np.mean(rewards)) if rewards else 0.0

    train_reference = [
        get_scenario("interpolation_1"),
        get_scenario("interpolation_2"),
        get_scenario("interpolation_3"),
    ]
    scores: list[MetaScenarioScore] = []
    for split, scenarios in (("interpolation", train_reference), ("extrapolation", meta_test_scenarios())):
        for scenario in scenarios:
            agent.policy.load_state_dict(base_state)
            before = evaluate_episode(scenario)
            for _ in range(int(config["meta"]["inner_steps"])):
                adapt_episode(scenario)
            after = evaluate_episode(scenario)
            scores.append(
                MetaScenarioScore(
                    scenario.name,
                    split,
                    before,
                    after,
                    config["meta"]["inner_steps"],
                )
            )
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
