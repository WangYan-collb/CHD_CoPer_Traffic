from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn


class TD3Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, sequence_length: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states.float().reshape(states.shape[0], -1))


class TD3Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, sequence_length: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = state_dim * sequence_length + action_dim
        self.q1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([states.float().reshape(states.shape[0], -1), actions.float()], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


@dataclass(frozen=True)
class TD3Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class TD3Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sequence_length: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        batch_size: int = 64,
        replay_size: int = 100000,
        device: str | None = None,
        **_: object,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = TD3Actor(state_dim, action_dim, sequence_length).to(self.device)
        self.actor_target = TD3Actor(state_dim, action_dim, sequence_length).to(self.device)
        self.critic = TD3Critic(state_dim, action_dim, sequence_length).to(self.device)
        self.critic_target = TD3Critic(state_dim, action_dim, sequence_length).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.replay: deque[TD3Transition] = deque(maxlen=replay_size)
        self.total_updates = 0

    def select_action(self, state_sequence: np.ndarray | torch.Tensor) -> tuple[np.ndarray, dict[str, float]]:
        state = torch.as_tensor(state_sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).squeeze(0).cpu().numpy()
        noise = np.random.normal(0.0, self.exploration_noise, size=action.shape)
        action = np.clip(action + noise, 0.0, 1.0)
        return action.astype(np.float32), {"value": 0.0, "log_prob": 0.0}

    def store_transition(
        self,
        state_sequence: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state_sequence: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.append(TD3Transition(state_sequence, action, float(reward), next_state_sequence, bool(done)))

    def update(self):
        if len(self.replay) < self.batch_size:
            return None
        batch = random.sample(self.replay, self.batch_size)
        states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.stack([item.action for item in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            noise = torch.clamp(torch.randn_like(actions) * self.policy_noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(self.actor_target(next_states) + noise, 0.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1.0 - dones) * torch.minimum(target_q1, target_q2)
        q1, q2 = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.total_updates += 1
        if self.total_updates % 2 == 0:
            actor_loss = -self.critic(states, self.actor(states))[0].mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)
        return {"critic_loss": float(critic_loss.detach().cpu().item())}

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(source_param.data, alpha=self.tau)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)
