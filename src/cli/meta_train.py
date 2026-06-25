from __future__ import annotations

import argparse
import random

from src.cli.common import apply_smoke_overrides, load_config
from src.logging_utils.experiment_logger import ExperimentLogger
from src.scenarios.scenario_registry import meta_train_scenarios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meta-train MAML-Trans-Beta-PPO.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import numpy as np

    from src.algorithms.meta_trans_beta_ppo.maml import MAMLTransBetaPPO
    from src.algorithms.meta_trans_beta_ppo.reptile import ReptileTransBetaPPO
    from src.algorithms.registry import create_agent
    from src.envs.sumo.env import SumoMovingBottleneckEnv

    config = apply_smoke_overrides(load_config(args.config), args.smoke)
    if args.episodes is not None:
        config.setdefault("meta", {})["episodes"] = args.episodes
    random.seed(int(config.get("seed", 0)))
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
    meta_cfg = config["meta"]
    meta_algorithm = str(meta_cfg.get("algorithm", "maml")).lower()
    if meta_algorithm == "maml":
        meta_agent = MAMLTransBetaPPO(
            agent=agent,
            inner_steps=int(meta_cfg["inner_steps"]),
            meta_lr=float(meta_cfg["meta_lr"]),
        )
    elif meta_algorithm == "reptile":
        meta_agent = ReptileTransBetaPPO(
            agent=agent,
            inner_steps=int(meta_cfg["inner_steps"]),
            meta_lr=float(meta_cfg["meta_lr"]),
        )
    else:
        raise ValueError("meta.algorithm must be 'maml' or 'reptile'")
    logger = ExperimentLogger(config["outputs"]["root"], config["run_name"])
    logger.write_config(config)
    result = None
    scenario_pool = meta_train_scenarios()
    batch_size = min(int(meta_cfg["meta_batch_size"]), len(scenario_pool))

    def rollout_episode(rollout_agent, scenario):
        env = SumoMovingBottleneckEnv(
            scenario=scenario,
            sequence_length=env_cfg["sequence_length"],
            state_dim=env_cfg["state_dim"],
            smoke=config["smoke"],
            aggregation_time_s=int(env_cfg.get("aggregation_time_s", env_cfg.get("control_cycle_s", 120))),
            control_cycle_s=int(env_cfg.get("control_cycle_s", env_cfg.get("aggregation_time_s", 120))),
            simulation_time_s=int(env_cfg.get("simulation_time_s", 3600)),
            congestion_prediction_enabled=bool(env_cfg.get("congestion_prediction_enabled", True)),
            control_activation_score=float(env_cfg.get("control_activation_score", 0.45)),
            topology_state_enabled=bool(env_cfg.get("topology_state_enabled", False)),
            topology_reward_enabled=bool(env_cfg.get("topology_reward_enabled", False)),
            topology_reward_weight=float(env_cfg.get("topology_reward_weight", 0.10)),
            net_file=env_cfg.get("net_file", "data/sumo/base_network/test1.net.xml"),
            additional_file=env_cfg.get("additional_file", "data/sumo/base_network/E2_info.xml"),
            use_gui=bool(env_cfg.get("use_gui", False)),
        )
        state, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action, info = rollout_agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            rollout_agent.store_transition(
                state,
                np.asarray(action, dtype=np.float32),
                info["log_prob"],
                reward,
                terminated or truncated,
                info["value"],
            )
            total += reward
            state = next_state
            done = terminated or truncated
        env.close()
        return total

    for episode in range(int(meta_cfg["episodes"])):
        if batch_size == len(scenario_pool):
            scenarios = list(scenario_pool)
        else:
            scenarios = random.sample(scenario_pool, batch_size)
        result = meta_agent.meta_update(scenarios, rollout_episode)
        logger.log_metric({
            "episode": episode,
            "meta_algorithm": meta_algorithm,
            "mean_reward": result.mean_reward,
            "mean_improvement": result.mean_improvement,
            "task_count": result.task_count,
            "scenarios": "|".join(scenario.name for scenario in scenarios),
        })
    checkpoint = logger.checkpoint_dir / f"{config['run_name']}.pth"
    agent.save(checkpoint)
    logger.write_summary({
        "meta_algorithm": meta_algorithm,
        "train_scenarios": [scenario.name for scenario in scenario_pool],
        "checkpoint": str(checkpoint),
        "last_result": None if result is None else result.__dict__,
    })
    print(f"Run directory: {logger.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
