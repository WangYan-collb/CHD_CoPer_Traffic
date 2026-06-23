from __future__ import annotations

from dataclasses import dataclass

from src.control.conflict_resolution import ControlledVehicle, filter_spacing_conflicts
from src.control.longitudinal_gap import LongitudinalGapMapper
from src.control.speed_limit_mapper import map_speed_action


@dataclass(frozen=True)
class MovingBottleneckCommand:
    speed_limit_kmh: float
    longitudinal_gap_m: float
    start_position_m: float
    selected_vehicles: list[ControlledVehicle]


class MovingBottleneckController:
    def __init__(self, gap_mapper: LongitudinalGapMapper | None = None):
        self.gap_mapper = gap_mapper or LongitudinalGapMapper()

    def build_command(
        self,
        actions: list[float],
        speed_mps: float,
        candidates_by_lane: dict[str, list[ControlledVehicle]],
        start_position_bounds: tuple[float, float] = (300.0, 1600.0),
        search_tolerance_m: float = 35.0,
    ) -> MovingBottleneckCommand:
        if len(actions) < 3:
            raise ValueError("moving bottleneck action must contain speed, start position, and gap")
        speed_limit_kmh = map_speed_action(actions[0])
        low, high = start_position_bounds
        start_position = low + max(0.0, min(1.0, actions[1])) * (high - low)
        gap = self.gap_mapper.map_action(actions[2], speed_mps=speed_mps)
        selected: list[ControlledVehicle] = []
        for lane_index, (lane_id, candidates) in enumerate(candidates_by_lane.items()):
            if not candidates:
                continue
            target_position = start_position + lane_index * gap
            local_candidates = [
                candidate
                for candidate in candidates
                if abs(candidate.position_m - target_position) <= search_tolerance_m
            ]
            search_pool = local_candidates or candidates
            closest = min(search_pool, key=lambda item: abs(item.position_m - target_position))
            selected.append(closest)
        selected = filter_spacing_conflicts(selected, min_gap_m=max(12.0, gap * 0.5))
        return MovingBottleneckCommand(
            speed_limit_kmh=speed_limit_kmh,
            longitudinal_gap_m=gap,
            start_position_m=round(start_position, 3),
            selected_vehicles=selected,
        )
