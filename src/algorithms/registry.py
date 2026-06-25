from __future__ import annotations

from typing import Any


def algorithm_names() -> list[str]:
    return [
        "beta_ppo",
        "classic_vsl",
        "continuous_ppo",
        "dr_ppo",
        "feedback_vsl",
        "mpc_vsl",
        "sac",
        "context_trans_beta_ppo",
        "td3",
        "traditional_drl_vsl",
        "trans_beta_ppo",
        "vanilla_ppo",
    ]


def create_agent(name: str, *, state_dim: int, action_dim: int, sequence_length: int, config: dict[str, Any]):
    if name == "trans_beta_ppo":
        from src.algorithms.trans_beta_ppo.agent import TransBetaPPOAgent

        return TransBetaPPOAgent(state_dim, action_dim, sequence_length, **config)
    if name == "context_trans_beta_ppo":
        from src.algorithms.trans_beta_ppo.context_agent import ContextTransBetaPPOAgent

        return ContextTransBetaPPOAgent(state_dim, action_dim, sequence_length, **config)
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
    if name == "feedback_vsl":
        from src.algorithms.rule_based.feedback_vsl import FeedbackVSLAgent

        return FeedbackVSLAgent(state_dim, action_dim, sequence_length, **config)
    if name == "mpc_vsl":
        from src.algorithms.rule_based.mpc_vsl import MPCVSLAgent

        return MPCVSLAgent(state_dim, action_dim, sequence_length, **config)
    if name == "td3":
        from src.algorithms.td3.agent import TD3Agent

        return TD3Agent(state_dim, action_dim, sequence_length, **config)
    if name == "sac":
        from src.algorithms.sac.agent import SACAgent

        return SACAgent(state_dim, action_dim, sequence_length, **config)
    raise KeyError(f"unknown algorithm '{name}', available: {', '.join(algorithm_names())}")
