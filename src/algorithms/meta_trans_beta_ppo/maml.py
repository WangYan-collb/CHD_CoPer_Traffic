from __future__ import annotations

from dataclasses import dataclass
import copy

from src.algorithms.trans_beta_ppo.agent import TransBetaPPOAgent
from src.scenarios.scenario_config import ScenarioConfig


@dataclass(frozen=True)
class MetaUpdateResult:
    task_count: int
    inner_steps: int
    mean_reward: float
    mean_improvement: float


class MAMLTransBetaPPO:
    """First-order MAML orchestration wrapper.

    The heavy PPO gradient work stays inside `TransBetaPPOAgent`; this class owns
    task sampling and bookkeeping so SUMO workers can plug in rollout functions.
    """

    def __init__(
        self,
        agent: TransBetaPPOAgent,
        inner_steps: int = 3,
        meta_lr: float = 5e-4,
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

    def meta_update(self, scenarios: list[ScenarioConfig], rollout_fn) -> MetaUpdateResult:
        all_rewards: list[float] = []
        improvements: list[float] = []
        base_state = copy.deepcopy(self.agent.policy.state_dict())
        adapted_states = []
        for scenario in scenarios:
            self.agent.policy.load_state_dict(base_state)
            all_rewards.extend(self.adapt_task(scenario, rollout_fn))
            if len(all_rewards) >= self.inner_steps:
                task_rewards = all_rewards[-self.inner_steps :]
                improvements.append(task_rewards[-1] - task_rewards[0])
            adapted_states.append(copy.deepcopy(self.agent.policy.state_dict()))
        self.agent.policy.load_state_dict(self._meta_interpolate(base_state, adapted_states))
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        mean_improvement = sum(improvements) / len(improvements) if improvements else 0.0
        return MetaUpdateResult(
            task_count=len(scenarios),
            inner_steps=self.inner_steps,
            mean_reward=mean_reward,
            mean_improvement=mean_improvement,
        )

    def _meta_interpolate(self, base_state, adapted_states):
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
