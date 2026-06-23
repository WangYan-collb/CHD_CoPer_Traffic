from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Beta


class MLPBetaActorCritic(nn.Module):
    """Beta actor-critic baseline without Transformer temporal encoding."""

    def __init__(self, state_dim: int, action_dim: int, sequence_length: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        flat_dim = state_dim * sequence_length
        self.backbone = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.alpha_head = nn.Linear(hidden_dim, action_dim)
        self.beta_head = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.softplus = nn.Softplus()

    def _flat(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError("states must have shape (batch, sequence, state_dim)")
        return states.float().reshape(states.shape[0], -1)

    def distribution_params(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(self._flat(states))
        return self.softplus(self.alpha_head(features)) + 1.0, self.softplus(self.beta_head(features)) + 1.0

    def distribution(self, states: torch.Tensor) -> Beta:
        alpha, beta = self.distribution_params(states)
        return Beta(alpha, beta)

    def value(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(self._flat(states)).squeeze(-1)

    def act(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(states)
        action = torch.clamp(dist.rsample(), 1e-6, 1.0 - 1e-6)
        return action, dist.log_prob(action).sum(dim=-1), dist.entropy().sum(dim=-1), self.value(states)

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(states)
        action = torch.clamp(actions.float(), 1e-6, 1.0 - 1e-6)
        return dist.log_prob(action).sum(dim=-1), dist.entropy().sum(dim=-1), self.value(states)
