from __future__ import annotations


def map_speed_action(
    action: float,
    min_speed_kmh: float = 40.0,
    max_speed_kmh: float = 100.0,
    step_kmh: float = 5.0,
) -> float:
    clipped = max(0.0, min(1.0, float(action)))
    speed = min_speed_kmh + clipped * (max_speed_kmh - min_speed_kmh)
    if step_kmh > 0:
        speed = round(speed / step_kmh) * step_kmh
    return max(min_speed_kmh, min(max_speed_kmh, speed))
