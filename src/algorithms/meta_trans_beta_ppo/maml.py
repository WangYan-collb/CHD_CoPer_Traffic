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
        for scenario in scenarios:
            self.agent.policy.load_state_dict(base_state)
            all_rewards.extend(self.adapt_task(scenario, rollout_fn))
            if len(all_rewards) >= self.inner_steps:
                task_rewards = all_rewards[-self.inner_steps :]
                improvements.append(task_rewards[-1] - task_rewards[0])
        self.agent.policy.load_state_dict(base_state)
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        mean_improvement = sum(improvements) / len(improvements) if improvements else 0.0
        return MetaUpdateResult(
            task_count=len(scenarios),
            inner_steps=self.inner_steps,
            mean_reward=mean_reward,
            mean_improvement=mean_improvement,
        )
