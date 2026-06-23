from __future__ import annotations

import math


def density_efficiency_reward(density: float, critical_density: float, decay: float = 0.1) -> float:
    """Piecewise density reward: peak near critical density, zero when over-critical."""

    if critical_density <= 0:
        raise ValueError("critical_density must be positive")
    if density > critical_density:
        return 0.0
    if abs(density - critical_density) < 1e-9:
        return 1.0
    return float(1.0 / (1.0 + math.exp(-decay * (critical_density - density))))
