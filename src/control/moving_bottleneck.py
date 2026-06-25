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
    end_position_m: float
    selected_vehicles: list[ControlledVehicle]
    fallback_used: bool = False


class MovingBottleneckController:
    def __init__(self, gap_mapper: LongitudinalGapMapper | None = None):
        self.gap_mapper = gap_mapper or LongitudinalGapMapper()

    def build_command(
        self,
        actions: list[float],
        speed_mps: float,
        candidates_by_lane: dict[str, list[ControlledVehicle]],
        control_position_bounds: tuple[float, float] = (300.0, 7200.0),
        min_control_length_m: float = 200.0,
        search_tolerance_m: float = 35.0,
    ) -> MovingBottleneckCommand:
        if len(actions) < 3:
            raise ValueError("moving bottleneck action must contain speed, start position, and gap")
        speed_limit_kmh = map_speed_action(actions[0])
        low, high = control_position_bounds
        start_action = max(0.0, min(1.0, actions[1]))
        if len(actions) >= 4:
            end_action = max(0.0, min(1.0, actions[2]))
            gap_action = actions[3]
        else:
            end_action = min(1.0, start_action + 0.25)
            gap_action = actions[2]
        start_position = low + start_action * (high - low)
        end_position = low + end_action * (high - low)
        if end_position < start_position:
            start_position, end_position = end_position, start_position
        end_position = max(end_position, start_position + min_control_length_m)
        end_position = min(end_position, high)
        gap = self.gap_mapper.map_action(gap_action, speed_mps=speed_mps)
        selected: list[ControlledVehicle] = []
        for lane_index, (lane_id, candidates) in enumerate(candidates_by_lane.items()):
            if not candidates:
                continue
            candidates = [
                candidate for candidate in candidates if start_position <= candidate.position_m <= end_position
            ]
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
            end_position_m=round(end_position, 3),
            selected_vehicles=selected,
        )
