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
    target_vehicle_count: int = 0
    chain_coverage: float = 0.0
    construction_mode: str = "none"
    fallback_used: bool = False


@dataclass(frozen=True)
class SpatialControlMapping:
    """Map normalized start/end actions into a constrained upstream VSL region."""

    bottleneck_position_m: float = 7500.0
    upstream_control_length_m: float = 2500.0
    recovery_length_m: float = 300.0
    start_fraction: float = 0.40
    end_fraction: float = 0.35
    min_control_length_m: float = 1500.0

    @property
    def upstream_start_m(self) -> float:
        return self.bottleneck_position_m - self.upstream_control_length_m

    @property
    def downstream_limit_m(self) -> float:
        return self.bottleneck_position_m - self.recovery_length_m

    @property
    def effective_length_m(self) -> float:
        return max(1.0, self.downstream_limit_m - self.upstream_start_m)

    def map_actions(self, start_action: float, end_action: float) -> tuple[float, float]:
        start_fraction = _clip01(self.start_fraction)
        end_fraction = _clip01(self.end_fraction)
        start = self.upstream_start_m + _clip01(start_action) * start_fraction * self.effective_length_m
        max_start = self.downstream_limit_m - self.min_control_length_m
        start = min(start, max_start)
        raw_end_low = self.downstream_limit_m - end_fraction * self.effective_length_m
        raw_end = raw_end_low + _clip01(end_action) * end_fraction * self.effective_length_m
        end = max(raw_end, start + self.min_control_length_m)
        end = min(end, self.downstream_limit_m)
        if end <= start:
            end = min(self.downstream_limit_m, start + max(1.0, self.min_control_length_m))
        return round(start, 3), round(end, 3)


class MovingBottleneckController:
    def __init__(
        self,
        gap_mapper: LongitudinalGapMapper | None = None,
        spatial_mapping: SpatialControlMapping | None = None,
    ):
        self.gap_mapper = gap_mapper or LongitudinalGapMapper()
        self.spatial_mapping = spatial_mapping or SpatialControlMapping()

    def build_command(
        self,
        actions: list[float],
        speed_mps: float,
        candidates_by_lane: dict[str, list[ControlledVehicle]],
        search_tolerance_m: float = 35.0,
        expansion_m: float = 300.0,
        support_vehicles_per_lane: int = 2,
        min_chain_coverage: float = 0.6,
    ) -> MovingBottleneckCommand:
        if len(actions) < 3:
            raise ValueError("moving bottleneck action must contain speed, start position, and gap")
        speed_limit_kmh = map_speed_action(actions[0])
        start_action = max(0.0, min(1.0, actions[1]))
        if len(actions) >= 4:
            end_action = max(0.0, min(1.0, actions[2]))
            gap_action = actions[3]
        else:
            end_action = min(1.0, start_action + 0.25)
            gap_action = actions[2]
        start_position, end_position = self.spatial_mapping.map_actions(start_action, end_action)
        gap = self.gap_mapper.map_action(gap_action, speed_mps=speed_mps)
        selected: list[ControlledVehicle] = []
        primary_selected: list[ControlledVehicle] = []
        sorted_lanes = sorted(candidates_by_lane.items())
        target_vehicle_count = len([lane_id for lane_id, candidates in sorted_lanes if candidates])
        for lane_index, (lane_id, candidates) in enumerate(sorted_lanes):
            if not candidates:
                continue
            in_area = [
                candidate for candidate in candidates if start_position <= candidate.position_m <= end_position
            ]
            expanded = [
                candidate
                for candidate in candidates
                if start_position - expansion_m <= candidate.position_m <= end_position + expansion_m
            ]
            target_position = start_position + lane_index * gap
            local_candidates = [
                candidate
                for candidate in in_area
                if abs(candidate.position_m - target_position) <= search_tolerance_m
            ]
            search_pool = local_candidates or in_area or expanded or candidates
            closest = min(search_pool, key=lambda item: abs(item.position_m - target_position))
            primary_selected.append(closest)
            selected.append(closest)

            support_pool = [
                candidate
                for candidate in in_area
                if candidate.vehicle_id != closest.vehicle_id
            ]
            support_pool = sorted(
                support_pool,
                key=lambda item: (abs(item.position_m - target_position), item.position_m),
            )
            selected.extend(support_pool[:support_vehicles_per_lane])

        selected = _unique_vehicles(selected)
        selected = filter_spacing_conflicts(selected, min_gap_m=max(12.0, gap * 0.45))
        primary_count = len(_unique_vehicles(primary_selected))
        chain_coverage = primary_count / target_vehicle_count if target_vehicle_count else 0.0
        if not selected:
            mode = "none"
        elif chain_coverage >= min_chain_coverage and len(selected) > primary_count:
            mode = "chain_with_support"
        elif chain_coverage >= min_chain_coverage:
            mode = "chain"
        elif selected:
            mode = "area_fallback"
        else:
            mode = "none"
        return MovingBottleneckCommand(
            speed_limit_kmh=speed_limit_kmh,
            longitudinal_gap_m=gap,
            start_position_m=round(start_position, 3),
            end_position_m=round(end_position, 3),
            selected_vehicles=selected,
            target_vehicle_count=target_vehicle_count,
            chain_coverage=round(chain_coverage, 4),
            construction_mode=mode,
            fallback_used=chain_coverage < min_chain_coverage,
        )


def _unique_vehicles(vehicles: list[ControlledVehicle]) -> list[ControlledVehicle]:
    seen: set[str] = set()
    unique: list[ControlledVehicle] = []
    for vehicle in vehicles:
        if vehicle.vehicle_id in seen:
            continue
        seen.add(vehicle.vehicle_id)
        unique.append(vehicle)
    return unique


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
