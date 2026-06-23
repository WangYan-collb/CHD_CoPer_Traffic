from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrafficMetrics:
    density: float
    speed_mps: float
    queue_m: float
    ttc_s: float
    tet_s: float
    tit_s: float
    throughput: float = 0.0
