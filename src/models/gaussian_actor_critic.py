from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class GaussianActorCritic(nn.Module):
    """Continuous PPO baseline using a squashed Gaussian policy."""

    def __init__(self, state_dim: int, action_dim: int, sequence_length: int, hidden_dim: int = 128):
        super().__init__()
        flat_dim = state_dim * sequence_length
        self.actor = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def _flat(self, states: torch.Tensor) -> torch.Tensor:
        return states.float().reshape(states.shape[0], -1)

    def distribution(self, states: torch.Tensor) -> Normal:
        features = self.actor(self._flat(states))
        mean = torch.sigmoid(self.mean_head(features))
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(self._flat(states)).squeeze(-1)

    def act(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(states)
        raw_action = dist.rsample()
        action = torch.clamp(raw_action, 1e-6, 1.0 - 1e-6)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        return action, log_prob, dist.entropy().sum(dim=-1), self.value(states)

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(states)
        clipped = torch.clamp(actions.float(), 1e-6, 1.0 - 1e-6)
        return dist.log_prob(clipped).sum(dim=-1), dist.entropy().sum(dim=-1), self.value(states)
