from __future__ import annotations

from src.algorithms.trans_beta_ppo.agent import TransBetaPPOAgent
from src.models.mlp_beta_actor_critic import MLPBetaActorCritic


class BetaPPOAgent(TransBetaPPOAgent):
    """Beta-PPO ablation: same PPO update, MLP encoder instead of Transformer."""

    def _build_policy(self):
        return MLPBetaActorCritic(self.state_dim, self.action_dim, self.sequence_length).to(self.device)
