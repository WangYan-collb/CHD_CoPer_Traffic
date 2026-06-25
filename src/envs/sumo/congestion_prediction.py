from __future__ import annotations

from src.envs.sumo.metrics import CongestionPrediction, TrafficMetrics


class CongestionPredictor:
    """Chapter-2-style early congestion predictor.

    The thesis uses multi-dimensional state perception: density/occupancy,
    flow decay, queue gradient and speed oscillation. This implementation keeps
    those signals explicit and lightweight enough for per-simulation-second use.
    """

    def __init__(
        self,
        critical_density: float = 27.0,
        low_speed_mps: float = 15.0,
        queue_warning_m: float = 50.0,
        flow_decay_threshold: float = 0.15,
        queue_growth_threshold_m: float = 15.0,
        trigger_score: float = 0.55,
    ):
        self.critical_density = critical_density
        self.low_speed_mps = low_speed_mps
        self.queue_warning_m = queue_warning_m
        self.flow_decay_threshold = flow_decay_threshold
        self.queue_growth_threshold_m = queue_growth_threshold_m
        self.trigger_score = trigger_score

    def predict(self, samples: list[TrafficMetrics]) -> CongestionPrediction:
        if not samples:
            return CongestionPrediction(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        latest = samples[-1]
        density_score = _clip01(latest.density / self.critical_density)
        speed_score = _clip01((self.low_speed_mps - latest.speed_mps) / max(self.low_speed_mps, 1.0))
        queue_score = _clip01(latest.queue_m / self.queue_warning_m)
        flow_decay_score = self._flow_decay_score(samples)
        queue_growth_score = self._queue_growth_score(samples)
        score = (
            0.30 * density_score
            + 0.20 * speed_score
            + 0.20 * queue_score
            + 0.15 * flow_decay_score
            + 0.15 * queue_growth_score
        )
        return CongestionPrediction(
            is_congested=score >= self.trigger_score,
            score=round(score, 4),
            density_score=round(density_score, 4),
            speed_score=round(speed_score, 4),
            queue_score=round(queue_score, 4),
            flow_decay_score=round(flow_decay_score, 4),
            queue_growth_score=round(queue_growth_score, 4),
        )

    def _flow_decay_score(self, samples: list[TrafficMetrics]) -> float:
        if len(samples) < 4:
            return 0.0
        midpoint = len(samples) // 2
        early = sum(item.throughput for item in samples[:midpoint]) / max(midpoint, 1)
        late_count = len(samples) - midpoint
        late = sum(item.throughput for item in samples[midpoint:]) / max(late_count, 1)
        if early <= 1e-6:
            return 0.0
        decay = max(0.0, (early - late) / early)
        return _clip01(decay / self.flow_decay_threshold)

    def _queue_growth_score(self, samples: list[TrafficMetrics]) -> float:
        if len(samples) < 2:
            return 0.0
        growth = max(0.0, samples[-1].queue_m - samples[0].queue_m)
        return _clip01(growth / self.queue_growth_threshold_m)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
