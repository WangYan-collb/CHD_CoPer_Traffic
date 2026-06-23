from __future__ import annotations

import argparse

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
    from src.algorithms.registry import create_agent
    from src.envs.sumo.env import SumoMovingBottleneckEnv

    config = apply_smoke_overrides(load_config(args.config), args.smoke)
    if args.episodes is not None:
        config.setdefault("meta", {})["episodes"] = args.episodes
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
    maml = MAMLTransBetaPPO(
        agent=agent,
        inner_steps=int(config["meta"]["inner_steps"]),
        meta_lr=float(config["meta"]["meta_lr"]),
    )
    logger = ExperimentLogger(config["outputs"]["root"], config["run_name"])
    logger.write_config(config)
    result = None
    scenarios = meta_train_scenarios()[: int(config["meta"]["meta_batch_size"])]

    def rollout_episode(rollout_agent, scenario):
        env = SumoMovingBottleneckEnv(
            scenario=scenario,
            sequence_length=env_cfg["sequence_length"],
            state_dim=env_cfg["state_dim"],
            smoke=config["smoke"],
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

    for episode in range(int(config["meta"]["episodes"])):
        result = maml.meta_update(scenarios, rollout_episode)
        logger.log_metric({
            "episode": episode,
            "mean_reward": result.mean_reward,
            "mean_improvement": result.mean_improvement,
            "task_count": result.task_count,
        })
    checkpoint = logger.checkpoint_dir / "maml_trans_beta_ppo.pth"
    agent.save(checkpoint)
    logger.write_summary({"checkpoint": str(checkpoint), "last_result": None if result is None else result.__dict__})
    print(f"Run directory: {logger.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
