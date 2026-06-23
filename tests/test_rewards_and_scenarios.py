from src.rewards.density_reward import density_efficiency_reward
from src.rewards.combined_reward import RewardBreakdown, combined_reward
from src.scenarios.scenario_registry import get_scenario, list_scenarios


def test_density_reward_peaks_near_critical_density():
    assert density_efficiency_reward(27.0, 27.0) == 1.0
    assert density_efficiency_reward(35.0, 27.0) == 0.0
    assert 0.0 < density_efficiency_reward(20.0, 27.0) < 1.0


def test_combined_reward_returns_breakdown():
    reward, breakdown = combined_reward(
        density=25.0,
        critical_density=27.0,
        speed_mps=20.0,
        free_flow_speed_mps=30.0,
        queue_m=10.0,
        ttc_s=3.0,
        tet_s=1.0,
        tit_s=0.5,
        action_delta=0.1,
    )

    assert isinstance(breakdown, RewardBreakdown)
    assert reward == breakdown.total
    assert breakdown.density > 0.0
    assert breakdown.queue_penalty <= 0.0


def test_extrapolation_1_matches_thesis_parameters():
    scenario = get_scenario("extrapolation_1")

    assert scenario.lane_count == 3
    assert scenario.bottleneck_length_m == 150
    assert scenario.cav_ratio == 0.10
    assert scenario.main_flow == 3500
    assert scenario.ramp_flow == 1800


def test_scenario_registry_lists_all_thesis_scenarios():
    names = set(list_scenarios())

    assert {
        "base",
        "interpolation_1",
        "interpolation_2",
        "interpolation_3",
        "extrapolation_1",
        "extrapolation_2",
    }.issubset(names)
