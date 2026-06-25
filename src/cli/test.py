from __future__ import annotations

import argparse

from src.cli.common import apply_smoke_overrides, load_config
from src.scenarios.scenario_registry import get_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test a Trans-Beta-PPO checkpoint.")
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
    scenario = get_scenario(config.get("scenario", "base"))
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
    env = SumoMovingBottleneckEnv(
        scenario,
        env_cfg["sequence_length"],
        env_cfg["state_dim"],
        smoke=config["smoke"],
        aggregation_time_s=int(env_cfg.get("aggregation_time_s", env_cfg.get("control_cycle_s", 120))),
        control_cycle_s=int(env_cfg.get("control_cycle_s", env_cfg.get("aggregation_time_s", 120))),
        simulation_time_s=int(env_cfg.get("simulation_time_s", 3600)),
        congestion_prediction_enabled=bool(env_cfg.get("congestion_prediction_enabled", True)),
        control_activation_score=float(env_cfg.get("control_activation_score", 0.45)),
        net_file=env_cfg.get("net_file", "data/sumo/base_network/test1.net.xml"),
        additional_file=env_cfg.get("additional_file", "data/sumo/base_network/E2_info.xml"),
        use_gui=bool(env_cfg.get("use_gui", False)),
    )
    state, _ = env.reset()
    rewards: list[float] = []
    done = False
    while not done:
        action, _ = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    print({"scenario": scenario.name, "mean_reward": float(np.mean(rewards)) if rewards else 0.0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
