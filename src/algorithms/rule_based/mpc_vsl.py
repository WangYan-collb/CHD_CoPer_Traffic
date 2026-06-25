from __future__ import annotations

from pathlib import Path

import numpy as np


class MPCVSLAgent:
    """Lightweight model-predictive VSL baseline.

    The controller evaluates a discrete set of normalized speed actions with a
    simple macroscopic prediction objective. It represents the "traditional
    optimization control" family without depending on an external MPC solver.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sequence_length: int,
        critical_density: float = 27.0,
        free_flow_speed_mps: float = 30.0,
        horizon_steps: int = 4,
        density_relaxation: float = 0.12,
        queue_weight: float = 0.35,
        speed_weight: float = 0.25,
        smoothness_weight: float = 0.10,
        **_: object,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.critical_density = critical_density
        self.free_flow_speed_mps = free_flow_speed_mps
        self.horizon_steps = horizon_steps
        self.density_relaxation = density_relaxation
        self.queue_weight = queue_weight
        self.speed_weight = speed_weight
        self.smoothness_weight = smoothness_weight
        self.last_speed_action = 0.75

    def select_action(self, state_sequence: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        latest = np.asarray(state_sequence, dtype=np.float32)[-1]
        density = float(latest[0]) * 100.0 if latest.size > 0 else 0.0
        speed_mps = float(latest[1]) * self.free_flow_speed_mps if latest.size > 1 else self.free_flow_speed_mps
        queue_m = float(latest[2]) * 100.0 if latest.size > 2 else 0.0

        candidates = np.linspace(0.10, 0.90, 9)
        best_action = float(candidates[0])
        best_cost = float("inf")
        for candidate in candidates:
            cost = self._rollout_cost(candidate, density, speed_mps, queue_m)
            if cost < best_cost:
                best_cost = cost
                best_action = float(candidate)
        self.last_speed_action = best_action
        action = np.array([best_action, 0.30, 0.80, 0.5], dtype=np.float32)
        return _fit_action_dim(action, self.action_dim), {"log_prob": 0.0, "value": -best_cost}

    def _rollout_cost(self, speed_action: float, density: float, speed_mps: float, queue_m: float) -> float:
        predicted_density = density
        predicted_speed = speed_mps
        predicted_queue = queue_m
        cost = 0.0
        target_speed = 40.0 / 3.6 + speed_action * ((100.0 - 40.0) / 3.6)
        for _ in range(self.horizon_steps):
            density_error = predicted_density - self.critical_density
            predicted_density -= self.density_relaxation * density_error * (1.0 - speed_action)
            predicted_speed = 0.7 * predicted_speed + 0.3 * target_speed
            predicted_queue = max(0.0, predicted_queue + max(0.0, density_error) * 0.6 - (target_speed / 30.0) * 4.0)
            cost += abs(predicted_density - self.critical_density)
            cost += self.queue_weight * predicted_queue
            cost += self.speed_weight * max(0.0, self.free_flow_speed_mps - predicted_speed)
        cost += self.smoothness_weight * abs(speed_action - self.last_speed_action) * 100.0
        return float(cost)

    def update(self):
        return None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("mpc_vsl has no learned parameters\n", encoding="utf-8")


def _fit_action_dim(action: np.ndarray, action_dim: int) -> np.ndarray:
    if action_dim <= action.size:
        fitted = action[:action_dim]
    else:
        fitted = np.pad(action, (0, action_dim - action.size), constant_values=0.5)
    return np.clip(fitted, 0.0, 1.0).astype(np.float32)
