from src.control.longitudinal_gap import LongitudinalGapMapper


def test_gap_mapping_uses_highway_time_headway_at_80_kmh():
    mapper = LongitudinalGapMapper()
    low, high = mapper.safe_gap_range(speed_mps=80 / 3.6)

    assert 28.0 <= low <= 30.0
    assert 46.0 <= high <= 48.0
    assert mapper.map_action(0.0, speed_mps=80 / 3.6) == low
    assert mapper.map_action(1.0, speed_mps=80 / 3.6) == high


def test_gap_mapping_clamps_extreme_actions_and_keeps_minimum_width():
    mapper = LongitudinalGapMapper()
    low, high = mapper.safe_gap_range(speed_mps=120 / 3.6)

    assert 12.0 <= low < high <= 80.0
    assert mapper.map_action(-3.0, speed_mps=120 / 3.6) == low
    assert mapper.map_action(3.0, speed_mps=120 / 3.6) == high
    assert high - low >= mapper.min_range_width_m
