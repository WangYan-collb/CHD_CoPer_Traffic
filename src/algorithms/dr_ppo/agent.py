from __future__ import annotations

from src.algorithms.continuous_ppo.agent import ContinuousPPOAgent


class DRPPOAgent(ContinuousPPOAgent):
    """Robust PPO baseline.

    The model is standard continuous PPO; robustness is configured through
    scenario/domain-randomization parameters consumed by the training pipeline.
    """

    def __init__(self, *args, robustness_noise_std: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.robustness_noise_std = robustness_noise_std
