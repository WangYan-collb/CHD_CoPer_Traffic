from __future__ import annotations

import argparse
import random

from src.cli.common import apply_smoke_overrides, load_config
from src.logging_utils.experiment_logger import ExperimentLogger
from src.scenarios.scenario_registry import get_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Trans-Beta-PPO on a SUMO moving bottleneck scenario.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import numpy as np

    from src.algorithms.registry import create_agent
    from src.envs.sumo.env import SumoMovingBottleneckEnv

    config = apply_smoke_overrides(load_config(args.config), args.smoke)
    if args.episodes is not None:
        config.setdefault("training", {})["episodes"] = args.episodes
    random.seed(config.get("seed", 0))
    np.random.seed(config.get("seed", 0))

    env_cfg = config["environment"]
    scenario = get_scenario(config.get("scenario", "base"))
    logger = ExperimentLogger(config["outputs"]["root"], f"{config['run_name']}_{scenario.name}")
    logger.write_config(config)

    algorithm_config = dict(config["algorithm"])
    algorithm_name = algorithm_config.pop("name", "trans_beta_ppo")
    agent = create_agent(
        algorithm_name,
        state_dim=env_cfg["state_dim"],
        action_dim=env_cfg["action_dim"],
        sequence_length=env_cfg["sequence_length"],
        config=algorithm_config,
    )
    env = SumoMovingBottleneckEnv(
        scenario=scenario,
        sequence_length=env_cfg["sequence_length"],
        state_dim=env_cfg["state_dim"],
        smoke=config["smoke"],
        aggregation_time_s=int(env_cfg.get("aggregation_time_s", 30)),
        net_file=env_cfg.get("net_file", "data/sumo/base_network/test1.net.xml"),
        additional_file=env_cfg.get("additional_file", "data/sumo/base_network/E2_info.xml"),
        use_gui=bool(env_cfg.get("use_gui", False)),
    )

    episode_count = int(config["training"]["episodes"])
    for episode in range(episode_count):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            action, info = agent.select_action(state)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            if algorithm_name == "td3":
                agent.store_transition(state, action, reward, next_state, terminated or truncated)
            else:
                agent.store_transition(state, action, info["log_prob"], reward, terminated or truncated, info["value"])
            total_reward += reward
            logger.log_action({
                "episode": episode,
                "step": step,
                "a0": action[0],
                "a1": action[1],
                "a2": action[2],
                "speed_limit_kmh": step_info.get("speed_limit_kmh"),
                "longitudinal_gap_m": step_info.get("longitudinal_gap_m"),
                "selected_cav_count": step_info.get("selected_cav_count"),
            })
            state = next_state
            done = terminated or truncated
            step += 1
        stats = agent.update()
        if isinstance(stats, dict):
            loss_value = stats.get("loss", stats.get("critic_loss"))
        else:
            loss_value = None if stats is None else stats.loss
        logger.log_metric({
            "episode": episode,
            "reward": total_reward,
            "loss": loss_value,
            "density": step_info.get("density"),
            "speed_mps": step_info.get("speed_mps"),
            "queue_m": step_info.get("queue_m"),
        })
    checkpoint = logger.checkpoint_dir / f"{algorithm_name}.pth"
    agent.save(checkpoint)
    logger.write_summary({"episodes": episode_count, "checkpoint": str(checkpoint)})
    env.close()
    print(f"Run directory: {logger.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
