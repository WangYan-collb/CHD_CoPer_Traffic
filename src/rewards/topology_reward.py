from __future__ import annotations


def moving_bottleneck_topology_reward(
    chain_coverage: float,
    control_coverage_ratio: float,
    fallback_used: bool,
    selected_cav_count: int,
    target_vehicle_count: int,
    speed_limit_delta_kmh: float,
    actual_gap_mean_m: float = 0.0,
    target_gap_m: float = 0.0,
    congestion_score_delta: float = 0.0,
    queue_delta_m: float = 0.0,
) -> float:
    """Reward executable and stable CAV moving-bottleneck configurations."""

    chain_term = _clip01(chain_coverage)
    coverage_term = _clip01(control_coverage_ratio)
    selection_term = _clip01(selected_cav_count / max(target_vehicle_count, 1))
    fallback_penalty = 0.25 if fallback_used else 0.0
    no_control_penalty = 0.35 if selected_cav_count <= 0 else 0.0
    speed_jump_penalty = _clip01(abs(speed_limit_delta_kmh) / 40.0) * 0.20
    gap_error = abs(actual_gap_mean_m - target_gap_m) if actual_gap_mean_m > 0 and target_gap_m > 0 else 0.0
    gap_tracking_penalty = _clip01(gap_error / max(target_gap_m, 1.0)) * 0.15
    congestion_worsening_penalty = _clip01(congestion_score_delta / 0.30) * 0.20
    queue_growth_penalty = _clip01(queue_delta_m / 80.0) * 0.15
    return float(
        0.35 * chain_term
        + 0.35 * coverage_term
        + 0.20 * selection_term
        - fallback_penalty
        - no_control_penalty
        - speed_jump_penalty
        - gap_tracking_penalty
        - congestion_worsening_penalty
        - queue_growth_penalty
    )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
