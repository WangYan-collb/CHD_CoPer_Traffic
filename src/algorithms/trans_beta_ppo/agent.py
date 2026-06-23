from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.algorithms.trans_beta_ppo.buffer import RolloutBuffer
from src.models.beta_actor_critic import BetaActorCritic


@dataclass(frozen=True)
class PPOUpdateStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float


class TransBetaPPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sequence_length: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.1,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        update_epochs: int = 4,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = self._build_policy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def _build_policy(self):
        return BetaActorCritic(self.state_dim, self.action_dim, self.sequence_length).to(self.device)

    def _state_tensor(self, state_sequence: np.ndarray | torch.Tensor) -> torch.Tensor:
        state = torch.as_tensor(state_sequence, dtype=torch.float32, device=self.device)
        if state.ndim != 2:
            raise ValueError("state_sequence must have shape (sequence_length, state_dim)")
        if state.shape != (self.sequence_length, self.state_dim):
            raise ValueError(
                f"expected state shape {(self.sequence_length, self.state_dim)}, got {tuple(state.shape)}"
            )
        return state.unsqueeze(0)

    def select_action(self, state_sequence: np.ndarray | torch.Tensor) -> tuple[np.ndarray, dict[str, float]]:
        with torch.no_grad():
            state = self._state_tensor(state_sequence)
            action, log_prob, entropy, value = self.policy.act(state)
        info = {
            "log_prob": float(log_prob.item()),
            "entropy": float(entropy.item()),
            "value": float(value.item()),
        }
        return action.squeeze(0).detach().cpu().numpy(), info

    def store_transition(
        self,
        state_sequence: np.ndarray | torch.Tensor,
        action: np.ndarray | torch.Tensor,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        state = self._state_tensor(state_sequence).squeeze(0).cpu()
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        self.buffer.add(
            state=state,
            action=action_tensor,
            log_prob=torch.tensor(log_prob, dtype=torch.float32),
            reward=reward,
            done=done,
            value=torch.tensor(value, dtype=torch.float32),
        )

    def _returns_and_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = torch.stack(self.buffer.values).float()
        advantages = torch.zeros(len(rewards), dtype=torch.float32)
        gae = 0.0
        next_value = 0.0
        for step in reversed(range(len(rewards))):
            mask = 0.0 if dones[step] else 1.0
            delta = rewards[step] + self.gamma * next_value * mask - values[step].item()
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[step] = gae
            next_value = values[step].item()
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return returns.to(self.device), advantages.to(self.device)

    def update(self) -> PPOUpdateStats | None:
        if len(self.buffer) == 0:
            return None

        states = torch.stack(self.buffer.states).to(self.device)
        actions = torch.stack(self.buffer.actions).to(self.device)
        old_log_probs = torch.stack(self.buffer.log_probs).to(self.device)
        returns, advantages = self._returns_and_advantages()

        last_stats: PPOUpdateStats | None = None
        for _ in range(self.update_epochs):
            log_probs, entropy, values = self.policy.evaluate_actions(states, actions)
            ratios = torch.exp(log_probs - old_log_probs)
            unclipped = ratios * advantages
            clipped = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(values, returns)
            entropy_loss = entropy.mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            last_stats = PPOUpdateStats(
                loss=float(loss.detach().cpu().item()),
                policy_loss=float(policy_loss.detach().cpu().item()),
                value_loss=float(value_loss.detach().cpu().item()),
                entropy=float(entropy_loss.detach().cpu().item()),
            )

        self.buffer.clear()
        return last_stats

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.policy.state_dict()}, path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model"])
