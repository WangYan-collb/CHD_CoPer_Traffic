from __future__ import annotations

from src.algorithms.continuous_ppo.agent import ContinuousPPOAgent


class TraditionalDRLVSLAgent(ContinuousPPOAgent):
    """Plain MLP continuous PPO baseline used as traditional DRL-VSL."""
