from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoadNetworkDesign:
    upstream_length_m: float = 1900.0
    bottleneck_length_m: float = 150.0
    ramp_length_m: float = 260.0
    downstream_length_m: float = 642.0
    upstream_edge: str = "E1"
    ramp_edge: str = "E2"
    merge_edge: str = "E4"
    downstream_edge: str = "E6"

    @property
    def main_route_edges(self) -> str:
        return f"{self.upstream_edge} {self.merge_edge} {self.downstream_edge}"

    @property
    def ramp_route_edges(self) -> str:
        return f"{self.ramp_edge} E3 {self.merge_edge} {self.downstream_edge}"
