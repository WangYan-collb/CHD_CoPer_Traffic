from __future__ import annotations

from pathlib import Path

import numpy as np


class ClassicVSLAgent:
    """Non-learning rule-based VSL baseline for thesis comparisons.

    The action format matches learning agents: speed, start-position, gap are
    normalized to [0, 1]. The speed action decreases as density increases.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sequence_length: int,
        critical_density: float = 27.0,
        **_: object,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.critical_density = critical_density

    def select_action(self, state_sequence: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        latest = np.asarray(state_sequence, dtype=np.float32)[-1]
        normalized_density = float(latest[0]) if latest.size else 0.0
        density = normalized_density * 100.0
        if density >= self.critical_density * 1.2:
            speed_action = 0.15
        elif density >= self.critical_density:
            speed_action = 0.35
        else:
            speed_action = 0.75
        action = np.array([speed_action, 0.5, 0.5], dtype=np.float32)
        if self.action_dim != 3:
            action = np.resize(action, self.action_dim).astype(np.float32)
        return np.clip(action, 0.0, 1.0), {"log_prob": 0.0, "value": 0.0}

    def update(self):
        return None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("classic_vsl has no learned parameters\n", encoding="utf-8")
