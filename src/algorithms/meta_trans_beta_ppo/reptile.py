from __future__ import annotations

from dataclasses import dataclass
import copy

from src.algorithms.trans_beta_ppo.agent import TransBetaPPOAgent
from src.scenarios.scenario_config import ScenarioConfig


@dataclass(frozen=True)
class ReptileUpdateResult:
    task_count: int
    inner_steps: int
    mean_reward: float
    mean_improvement: float


class ReptileTransBetaPPO:
    """First-order meta-RL wrapper for Trans-Beta-PPO.

    Reptile is a practical fit for SUMO experiments because it avoids expensive
    second-order gradients. Each task adapts a copied initialization with PPO,
    then the meta initialization moves toward the average adapted policy.
    """

    def __init__(
        self,
        agent: TransBetaPPOAgent,
        inner_steps: int = 3,
        meta_lr: float = 0.05,
    ):
        self.agent = agent
        self.inner_steps = inner_steps
        self.meta_lr = meta_lr

    def adapt_task(self, scenario: ScenarioConfig, rollout_fn) -> list[float]:
        rewards: list[float] = []
        for _ in range(self.inner_steps):
            reward = float(rollout_fn(self.agent, scenario))
            self.agent.update()
            rewards.append(reward)
        return rewards

    def meta_update(self, scenarios: list[ScenarioConfig], rollout_fn) -> ReptileUpdateResult:
        all_rewards: list[float] = []
        improvements: list[float] = []
        base_state = copy.deepcopy(self.agent.policy.state_dict())
        adapted_states = []
        for scenario in scenarios:
            self.agent.policy.load_state_dict(base_state)
            task_rewards = self.adapt_task(scenario, rollout_fn)
            all_rewards.extend(task_rewards)
            if task_rewards:
                improvements.append(task_rewards[-1] - task_rewards[0])
            adapted_states.append(copy.deepcopy(self.agent.policy.state_dict()))
        self.agent.policy.load_state_dict(self._move_toward_adapted_mean(base_state, adapted_states))
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        mean_improvement = sum(improvements) / len(improvements) if improvements else 0.0
        return ReptileUpdateResult(
            task_count=len(scenarios),
            inner_steps=self.inner_steps,
            mean_reward=mean_reward,
            mean_improvement=mean_improvement,
        )

    def _move_toward_adapted_mean(self, base_state, adapted_states):
        if not adapted_states:
            return base_state
        meta_state = {}
        for key, base_value in base_state.items():
            if not base_value.is_floating_point():
                meta_state[key] = base_value
                continue
            mean_delta = sum(adapted[key] - base_value for adapted in adapted_states) / len(adapted_states)
            meta_state[key] = base_value + self.meta_lr * mean_delta
        return meta_state
