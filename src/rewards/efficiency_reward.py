from __future__ import annotations


def speed_efficiency_reward(speed_mps: float, free_flow_speed_mps: float) -> float:
    if free_flow_speed_mps <= 0:
        raise ValueError("free_flow_speed_mps must be positive")
    return max(0.0, min(1.0, speed_mps / free_flow_speed_mps))


def queue_penalty(queue_m: float, scale_m: float = 100.0) -> float:
    if scale_m <= 0:
        raise ValueError("scale_m must be positive")
    return -max(0.0, queue_m) / scale_m
