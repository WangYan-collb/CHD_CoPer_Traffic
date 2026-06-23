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
    accepted: list[ControlledVehicle] = []
    for vehicle in sorted(vehicles, key=lambda item: item.position_m):
        if all(abs(vehicle.position_m - kept.position_m) >= min_gap_m for kept in accepted):
            accepted.append(vehicle)
    return accepted
