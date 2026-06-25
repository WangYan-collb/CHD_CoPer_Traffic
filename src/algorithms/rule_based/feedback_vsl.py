from __future__ import annotations

from pathlib import Path

import numpy as np


class FeedbackVSLAgent:
    """Density-feedback VSL controller.

    This is a traditional closed-loop baseline: if the measured density exceeds
    the critical density or queues grow, the controller lowers the CAV speed
    command. It is intentionally model-free and non-learning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sequence_length: int,
        critical_density: float = 27.0,
        kp_density: float = 0.018,
        kp_queue: float = 0.006,
        base_speed_action: float = 0.78,
        min_speed_action: float = 0.12,
        max_speed_action: float = 0.90,
        **_: object,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.critical_density = critical_density
        self.kp_density = kp_density
        self.kp_queue = kp_queue
        self.base_speed_action = base_speed_action
        self.min_speed_action = min_speed_action
        self.max_speed_action = max_speed_action

    def select_action(self, state_sequence: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        latest = np.asarray(state_sequence, dtype=np.float32)[-1]
        density = float(latest[0]) * 100.0 if latest.size > 0 else 0.0
        queue_m = float(latest[2]) * 100.0 if latest.size > 2 else 0.0
        density_error = max(0.0, density - self.critical_density)
        speed_action = self.base_speed_action - self.kp_density * density_error - self.kp_queue * queue_m
        action = np.array([speed_action, 0.35, 0.75, 0.5], dtype=np.float32)
        return _fit_action_dim(action, self.action_dim), {"log_prob": 0.0, "value": 0.0}

    def update(self):
        return None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("feedback_vsl has no learned parameters\n", encoding="utf-8")


def _fit_action_dim(action: np.ndarray, action_dim: int) -> np.ndarray:
    if action_dim <= action.size:
        fitted = action[:action_dim]
    else:
        fitted = np.pad(action, (0, action_dim - action.size), constant_values=0.5)
    return np.clip(fitted, 0.0, 1.0).astype(np.float32)
