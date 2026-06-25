from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class SACActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, sequence_length: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = state_dim * sequence_length
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = states.float().reshape(states.shape[0], -1)
        features = self.backbone(x)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), -5.0, 2.0)
        return mean, log_std

    def sample(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(states)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        tanh_action = torch.tanh(z)
        action = (tanh_action + 1.0) * 0.5
        log_prob = normal.log_prob(z) - torch.log(1.0 - tanh_action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def deterministic(self, states: torch.Tensor) -> torch.Tensor:
        mean, _ = self(states)
        return (torch.tanh(mean) + 1.0) * 0.5


class SACCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, sequence_length: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = state_dim * sequence_length + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([states.float().reshape(states.shape[0], -1), actions.float()], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


@dataclass(frozen=True)
class SACTransition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sequence_length: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
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
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = SACActor(state_dim, action_dim, sequence_length).to(self.device)
        self.critic = SACCritic(state_dim, action_dim, sequence_length).to(self.device)
        self.critic_target = SACCritic(state_dim, action_dim, sequence_length).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.replay: deque[SACTransition] = deque(maxlen=replay_size)

    def select_action(self, state_sequence: np.ndarray | torch.Tensor) -> tuple[np.ndarray, dict[str, float]]:
        state = torch.as_tensor(state_sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.actor.sample(state)
            q1, q2 = self.critic(state, action)
        value = torch.minimum(q1, q2)
        return action.squeeze(0).cpu().numpy().astype(np.float32), {
            "value": float(value.item()),
            "log_prob": float(log_prob.item()),
        }

    def store_transition(
        self,
        state_sequence: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state_sequence: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.append(SACTransition(state_sequence, action, float(reward), next_state_sequence, bool(done)))

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
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1.0 - dones) * (
                torch.minimum(target_q1, target_q2) - self.alpha * next_log_probs
            )

        q1, q2 = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_probs = self.actor.sample(states)
        new_q1, new_q2 = self.critic(states, new_actions)
        actor_loss = (self.alpha * log_probs - torch.minimum(new_q1, new_q2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self._soft_update(self.critic_target, self.critic)

        return {
            "loss": float((actor_loss + critic_loss).detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
        }

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(source_param.data, alpha=self.tau)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint.get("critic_target", checkpoint["critic"]))
