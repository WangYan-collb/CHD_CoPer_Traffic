from __future__ import annotations

from src.algorithms.trans_beta_ppo.agent import TransBetaPPOAgent
from src.models.gaussian_actor_critic import GaussianActorCritic


class ContinuousPPOAgent(TransBetaPPOAgent):
    """Continuous PPO baseline: squashed Gaussian actor with PPO updates."""

    def _build_policy(self):
        return GaussianActorCritic(self.state_dim, self.action_dim, self.sequence_length).to(self.device)
