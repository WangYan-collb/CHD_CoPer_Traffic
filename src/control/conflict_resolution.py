from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlledVehicle:
    vehicle_id: str
    lane_id: str
    position_m: float


def has_spacing_conflict(vehicles: list[ControlledVehicle], min_gap_m: float) -> bool:
    ordered = sorted(vehicles, key=lambda vehicle: vehicle.position_m)
    return any(
        abs(right.position_m - left.position_m) < min_gap_m
        for left, right in zip(ordered, ordered[1:])
    )


def filter_spacing_conflicts(
    vehicles: list[ControlledVehicle], min_gap_m: float
) -> list[ControlledVehicle]:
    """Keep vehicles separated in the same lane.

    Vehicles in different lanes may be intentionally staggered for a moving
    bottleneck, so conflicts are evaluated lane by lane instead of globally.
    """

    accepted: list[ControlledVehicle] = []
    for vehicle in vehicles:
        if all(
            vehicle.lane_id != kept.lane_id or abs(vehicle.position_m - kept.position_m) >= min_gap_m
            for kept in accepted
        ):
            accepted.append(vehicle)
    return accepted
