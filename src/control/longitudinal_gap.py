from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class LongitudinalGapMapper:
    """Map normalized policy actions to physically realistic highway CAV gaps.

    The range follows an IDM/time-headway idea:

    s_safe = vehicle_length + standstill_gap + speed * time_headway.

    Parameters use the project's CAV vehicle type by default, so the learned
    search gap stays tied to actual vehicle length, minimum gap and normal
    highway car-following behavior.
    """

    vehicle_length_m: float = 3.5
    standstill_gap_m: float = 1.4
    comfortable_accel_mps2: float = 3.0
    comfortable_decel_mps2: float = 5.5
    min_headway_s: float = 1.1
    max_headway_s: float = 2.0
    absolute_min_gap_m: float = 12.0
    absolute_max_gap_m: float = 95.0
    min_range_width_m: float = 5.0

    def safe_gap_range(self, speed_mps: float) -> tuple[float, float]:
        if speed_mps < 0:
            raise ValueError("speed_mps must be non-negative")

        base = self.vehicle_length_m + self.standstill_gap_m
        braking_margin = speed_mps / max(2.0 * (self.comfortable_accel_mps2 * self.comfortable_decel_mps2) ** 0.5, 1.0)
        low = _clamp(
            base + speed_mps * self.min_headway_s + 0.25 * braking_margin,
            self.absolute_min_gap_m,
            self.absolute_max_gap_m,
        )
        high = _clamp(
            base + speed_mps * self.max_headway_s + braking_margin,
            low + self.min_range_width_m,
            self.absolute_max_gap_m,
        )
        return round(low, 3), round(high, 3)

    def map_action(self, action: float, speed_mps: float) -> float:
        low, high = self.safe_gap_range(speed_mps)
        clipped = _clamp(float(action), 0.0, 1.0)
        return round(low + clipped * (high - low), 3)
