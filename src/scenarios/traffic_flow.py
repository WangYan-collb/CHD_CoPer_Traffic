from __future__ import annotations

from dataclasses import dataclass
import random

from src.scenarios.scenario_config import ScenarioConfig


@dataclass(frozen=True)
class FlowSegment:
    begin_s: int
    end_s: int
    main_flow_vph: int
    ramp_flow_vph: int


@dataclass(frozen=True)
class VehicleFlowSplit:
    total_main_vph: int
    total_ramp_vph: int
    cav_main_vph: int
    hdv_main_vph: tuple[int, int, int]
    hdv_ramp_vph: tuple[int, int, int]


class NormalTrafficFlowSampler:
    """Third-chapter style segmented normal traffic demand sampler."""

    def __init__(self, segment_seconds: int = 600, hdv_mix: tuple[float, float, float] = (0.4, 0.4, 0.2)):
        if sum(hdv_mix) <= 0:
            raise ValueError("hdv_mix must contain positive proportions")
        self.segment_seconds = segment_seconds
        total = sum(hdv_mix)
        self.hdv_mix = tuple(item / total for item in hdv_mix)

    def sample_segments(self, scenario: ScenarioConfig) -> list[FlowSegment]:
        rng = random.Random(scenario.seed)
        segment_count = int(3600 * scenario.route_hours / self.segment_seconds)
        profile = self._profile(segment_count)
        segments: list[FlowSegment] = []
        for index, factor in enumerate(profile):
            main_mean = scenario.main_flow * factor
            ramp_mean = scenario.ramp_flow * factor
            main_flow = max(1, int(rng.gauss(main_mean, scenario.main_flow_std)))
            ramp_flow = max(1, int(rng.gauss(ramp_mean, scenario.ramp_flow_std)))
            begin = index * self.segment_seconds
            segments.append(
                FlowSegment(
                    begin_s=begin,
                    end_s=begin + self.segment_seconds,
                    main_flow_vph=main_flow,
                    ramp_flow_vph=ramp_flow,
                )
            )
        return segments

    def split_vehicle_flows(self, segment: FlowSegment, cav_ratio: float) -> VehicleFlowSplit:
        cav_ratio = max(0.0, min(1.0, cav_ratio))
        cav_main = int(segment.main_flow_vph * cav_ratio)
        hdv_main_total = int(segment.main_flow_vph * (1.0 - cav_ratio))
        hdv_ramp_total = int(segment.ramp_flow_vph * (1.0 - cav_ratio))
        hdv_main = tuple(int(hdv_main_total * ratio) for ratio in self.hdv_mix)
        hdv_ramp = tuple(int(hdv_ramp_total * ratio) for ratio in self.hdv_mix)
        return VehicleFlowSplit(
            total_main_vph=segment.main_flow_vph,
            total_ramp_vph=segment.ramp_flow_vph,
            cav_main_vph=cav_main,
            hdv_main_vph=hdv_main,
            hdv_ramp_vph=hdv_ramp,
        )

    def _profile(self, count: int) -> list[float]:
        if count == 6:
            return [0.92, 1.02, 1.10, 1.06, 1.00, 0.90]
        return [1.0 for _ in range(count)]
