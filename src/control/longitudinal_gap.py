from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class LongitudinalGapMapper:
    """Map normalized policy actions to physically realistic highway CAV gaps."""

    vehicle_length_m: float = 4.5
    standstill_gap_m: float = 2.0
    min_headway_s: float = 1.0
    max_headway_s: float = 1.8
    absolute_min_gap_m: float = 12.0
    absolute_max_gap_m: float = 80.0
    min_range_width_m: float = 5.0

    def safe_gap_range(self, speed_mps: float) -> tuple[float, float]:
        if speed_mps < 0:
            raise ValueError("speed_mps must be non-negative")

        base = self.vehicle_length_m + self.standstill_gap_m
        low = _clamp(
            base + speed_mps * self.min_headway_s,
            self.absolute_min_gap_m,
            self.absolute_max_gap_m,
        )
        high = _clamp(
            base + speed_mps * self.max_headway_s,
            low + self.min_range_width_m,
            self.absolute_max_gap_m,
        )
        return round(low, 3), round(high, 3)

    def map_action(self, action: float, speed_mps: float) -> float:
        low, high = self.safe_gap_range(speed_mps)
        clipped = _clamp(float(action), 0.0, 1.0)
        return round(low + clipped * (high - low), 3)
