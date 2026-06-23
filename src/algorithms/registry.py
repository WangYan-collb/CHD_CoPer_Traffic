from __future__ import annotations

from typing import Any


def algorithm_names() -> list[str]:
    return [
        "beta_ppo",
        "classic_vsl",
        "continuous_ppo",
        "dr_ppo",
        "td3",
        "traditional_drl_vsl",
        "trans_beta_ppo",
        "vanilla_ppo",
    ]


def create_agent(name: str, *, state_dim: int, action_dim: int, sequence_length: int, config: dict[str, Any]):
    if name == "trans_beta_ppo":
        from src.algorithms.trans_beta_ppo.agent import TransBetaPPOAgent

        return TransBetaPPOAgent(state_dim, action_dim, sequence_length, **config)
    if name == "beta_ppo":
        from src.algorithms.beta_ppo.agent import BetaPPOAgent

        return BetaPPOAgent(state_dim, action_dim, sequence_length, **config)
    if name == "continuous_ppo":
        from src.algorithms.continuous_ppo.agent import ContinuousPPOAgent

        return ContinuousPPOAgent(state_dim, action_dim, sequence_length, **config)
    if name == "vanilla_ppo":
        from src.algorithms.continuous_ppo.agent import ContinuousPPOAgent

        return ContinuousPPOAgent(state_dim, action_dim, sequence_length, **config)
    if name == "traditional_drl_vsl":
        from src.algorithms.traditional_drl_vsl.agent import TraditionalDRLVSLAgent

        return TraditionalDRLVSLAgent(state_dim, action_dim, sequence_length, **config)
    if name == "dr_ppo":
        from src.algorithms.dr_ppo.agent import DRPPOAgent

        return DRPPOAgent(state_dim, action_dim, sequence_length, **config)
    if name == "classic_vsl":
        from src.algorithms.rule_based.classic_vsl import ClassicVSLAgent

        return ClassicVSLAgent(state_dim, action_dim, sequence_length, **config)
    if name == "td3":
        from src.algorithms.td3.agent import TD3Agent

        return TD3Agent(state_dim, action_dim, sequence_length, **config)
    raise KeyError(f"unknown algorithm '{name}', available: {', '.join(algorithm_names())}")
