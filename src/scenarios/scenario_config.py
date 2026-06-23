from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    lane_count: int
    bottleneck_length_m: int
    cav_ratio: float
    main_flow: int
    ramp_flow: int
    description: str
    is_meta_train: bool
    seed: int
    route_hours: int = 1
    main_flow_std: float = 154.24640151702852
    ramp_flow_std: float = 182.98711898460425
    extra: dict[str, object] = field(default_factory=dict)
