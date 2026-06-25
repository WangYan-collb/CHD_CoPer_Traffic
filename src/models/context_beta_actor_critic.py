from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Beta

from src.models.transformer_encoder import TrafficTransformerEncoder


class ContextBetaActorCritic(nn.Module):
    """Beta actor-critic with online traffic-context conditioning.

    The transformer encodes short-term temporal traffic state. The context branch
    summarizes the same recent window into a compact task embedding, giving the
    policy a lightweight way to infer demand/CAV/bottleneck shifts without
    adding an external replay-based meta learner.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sequence_length: int,
        embed_dim: int = 64,
        context_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.encoder = TrafficTransformerEncoder(
            state_dim=state_dim,
            sequence_length=sequence_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
            nn.GELU(),
        )
        feature_dim = embed_dim + context_dim
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.alpha_head = nn.Linear(hidden_dim, action_dim)
        self.beta_head = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.softplus = nn.Softplus()

    def _context(self, states: torch.Tensor) -> torch.Tensor:
        states = states.float()
        mean = states.mean(dim=1)
        std = states.std(dim=1, unbiased=False)
        return self.context_encoder(torch.cat([mean, std], dim=-1))

    def _features(self, states: torch.Tensor) -> torch.Tensor:
        temporal = self.encoder(states.float())
        context = self._context(states)
        return torch.cat([temporal, context], dim=-1)

    def distribution_params(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actor_features = self.actor(self._features(states))
        alpha = self.softplus(self.alpha_head(actor_features)) + 1.0
        beta = self.softplus(self.beta_head(actor_features)) + 1.0
        return alpha, beta

    def distribution(self, states: torch.Tensor) -> Beta:
        alpha, beta = self.distribution_params(states)
        return Beta(alpha, beta)

    def value(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(self._features(states)).squeeze(-1)

    def act(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(states)
        action = dist.rsample()
        action = torch.clamp(action, 1e-6, 1.0 - 1e-6)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(states)
        return action, log_prob, entropy, value

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(states)
        clipped_actions = torch.clamp(actions.float(), 1e-6, 1.0 - 1e-6)
        log_prob = dist.log_prob(clipped_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(states)
        return log_prob, entropy, value
