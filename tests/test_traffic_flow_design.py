from src.scenarios.scenario_registry import get_scenario
from src.scenarios.traffic_flow import NormalTrafficFlowSampler
from src.scenarios.vehicle_params import default_vehicle_types


def test_normal_sampler_generates_ten_minute_flow_segments():
    scenario = get_scenario("base")
    sampler = NormalTrafficFlowSampler(segment_seconds=600)

    segments = sampler.sample_segments(scenario)

    assert len(segments) == 6
    assert all(segment.end_s - segment.begin_s == 600 for segment in segments)
    assert all(segment.main_flow_vph > 0 for segment in segments)
    assert all(segment.ramp_flow_vph > 0 for segment in segments)


def test_flow_split_preserves_cav_ratio_and_hdv_mix():
    scenario = get_scenario("base")
    sampler = NormalTrafficFlowSampler(segment_seconds=600)
    split = sampler.split_vehicle_flows(sampler.sample_segments(scenario)[0], scenario.cav_ratio)

    assert split.cav_main_vph > 0
    assert len(split.hdv_main_vph) == 3
    assert len(split.hdv_ramp_vph) == 3
    assert sum(split.hdv_main_vph) + split.cav_main_vph <= split.total_main_vph


def test_default_vehicle_types_include_cav_hdv_and_heavy_vehicle():
    vehicle_types = default_vehicle_types()
    ids = {vehicle.id for vehicle in vehicle_types}

    assert {"CAV", "HDV_0", "HDV_1", "HDV_2"}.issubset(ids)
