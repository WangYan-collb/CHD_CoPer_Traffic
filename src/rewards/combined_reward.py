from __future__ import annotations

from dataclasses import dataclass

from src.rewards.density_reward import density_efficiency_reward
from src.rewards.efficiency_reward import queue_penalty, speed_efficiency_reward
from src.rewards.safety_reward import risk_penalty, ttc_safety_reward
from src.rewards.topology_reward import moving_bottleneck_topology_reward


@dataclass(frozen=True)
class RewardWeights:
    density: float = 0.45
    speed: float = 0.25
    queue: float = 0.10
    safety: float = 0.15
    smoothness: float = 0.05
    topology: float = 0.00


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    density: float
    speed: float
    queue_penalty: float
    safety: float
    smoothness_penalty: float
    topology: float = 0.0


def combined_reward(
    density: float,
    critical_density: float,
    speed_mps: float,
    free_flow_speed_mps: float,
    queue_m: float,
    ttc_s: float,
    tet_s: float,
    tit_s: float,
    action_delta: float,
    chain_coverage: float = 0.0,
    control_coverage_ratio: float = 0.0,
    fallback_used: bool = False,
    selected_cav_count: int = 0,
    target_vehicle_count: int = 0,
    speed_limit_delta_kmh: float = 0.0,
    actual_gap_mean_m: float = 0.0,
    target_gap_m: float = 0.0,
    congestion_score_delta: float = 0.0,
    queue_delta_m: float = 0.0,
    weights: RewardWeights | None = None,
) -> tuple[float, RewardBreakdown]:
    weights = weights or RewardWeights()
    density_term = density_efficiency_reward(density, critical_density)
    speed_term = speed_efficiency_reward(speed_mps, free_flow_speed_mps)
    queue_term = queue_penalty(queue_m)
    safety_term = 0.5 * ttc_safety_reward(ttc_s) + 0.5 * risk_penalty(tet_s, tit_s)
    smoothness_term = -abs(action_delta)
    topology_term = moving_bottleneck_topology_reward(
        chain_coverage=chain_coverage,
        control_coverage_ratio=control_coverage_ratio,
        fallback_used=fallback_used,
        selected_cav_count=selected_cav_count,
        target_vehicle_count=target_vehicle_count,
        speed_limit_delta_kmh=speed_limit_delta_kmh,
        actual_gap_mean_m=actual_gap_mean_m,
        target_gap_m=target_gap_m,
        congestion_score_delta=congestion_score_delta,
        queue_delta_m=queue_delta_m,
    )
    total = (
        weights.density * density_term
        + weights.speed * speed_term
        + weights.queue * queue_term
        + weights.safety * safety_term
        + weights.smoothness * smoothness_term
        + weights.topology * topology_term
    )
    breakdown = RewardBreakdown(
        total=float(total),
        density=float(density_term),
        speed=float(speed_term),
        queue_penalty=float(queue_term),
        safety=float(safety_term),
        smoothness_penalty=float(smoothness_term),
        topology=float(topology_term),
    )
    return breakdown.total, breakdown
