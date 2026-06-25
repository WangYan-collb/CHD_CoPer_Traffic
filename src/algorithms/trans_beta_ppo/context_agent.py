from __future__ import annotations

from src.algorithms.trans_beta_ppo.agent import TransBetaPPOAgent
from src.models.context_beta_actor_critic import ContextBetaActorCritic


class ContextTransBetaPPOAgent(TransBetaPPOAgent):
    """Trans-Beta-PPO with online traffic-context conditioning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sequence_length: int,
        context_dim: int = 32,
        **kwargs,
    ):
        self.context_dim = context_dim
        super().__init__(state_dim, action_dim, sequence_length, **kwargs)

    def _build_policy(self):
        return ContextBetaActorCritic(
            self.state_dim,
            self.action_dim,
            self.sequence_length,
            context_dim=self.context_dim,
        ).to(self.device)
