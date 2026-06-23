from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoadNetworkDesign:
    """Highway merge geometry used by the SUMO scenario generator.

    The base geometry follows the thesis M25 merge experiment: 7.5 km upstream
    control area, a merge/bottleneck area, 500 m single-entry ramp and 2 km
    downstream recovery area. Chapter 4/5 scenarios vary lane count and
    bottleneck length while keeping the same edge naming used by legacy code.
    """

    upstream_length_m: float = 7500.0
    bottleneck_length_m: float = 150.0
    ramp_length_m: float = 500.0
    downstream_length_m: float = 2000.0
    lane_width_m: float = 3.5
    speed_mps: float = 33.33
    upstream_edge: str = "E1"
    ramp_edge: str = "E2"
    ramp_merge_edge: str = "E3"
    merge_edge: str = "E4"
    downstream_edge: str = "E6"

    @property
    def main_route_edges(self) -> str:
        return f"{self.upstream_edge} {self.merge_edge} {self.downstream_edge}"

    @property
    def ramp_route_edges(self) -> str:
        return f"{self.ramp_edge} {self.ramp_merge_edge} {self.merge_edge} {self.downstream_edge}"
