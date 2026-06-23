from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationSummary:
    mean_reward: float
    mean_speed: float = 0.0
    mean_queue: float = 0.0
    tet: float = 0.0
    tit: float = 0.0
