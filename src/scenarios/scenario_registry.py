from __future__ import annotations

from src.scenarios.scenario_config import ScenarioConfig


_SCENARIOS: dict[str, ScenarioConfig] = {
    "base": ScenarioConfig(
        name="base",
        lane_count=5,
        bottleneck_length_m=150,
        cav_ratio=0.50,
        main_flow=4292,
        ramp_flow=2446,
        description="5-lane M25 merge baseline evening peak scenario.",
        is_meta_train=True,
        seed=2026062301,
    ),
    "interpolation_1": ScenarioConfig(
        name="interpolation_1",
        lane_count=4,
        bottleneck_length_m=180,
        cav_ratio=0.30,
        main_flow=4100,
        ramp_flow=2200,
        description="Lane-reduction interpolation scenario.",
        is_meta_train=True,
        seed=2026062302,
    ),
    "interpolation_2": ScenarioConfig(
        name="interpolation_2",
        lane_count=5,
        bottleneck_length_m=150,
        cav_ratio=0.70,
        main_flow=4500,
        ramp_flow=2700,
        description="Higher CAV-ratio interpolation scenario.",
        is_meta_train=True,
        seed=2026062303,
    ),
    "interpolation_3": ScenarioConfig(
        name="interpolation_3",
        lane_count=4,
        bottleneck_length_m=180,
        cav_ratio=0.50,
        main_flow=5200,
        ramp_flow=2800,
        description="High-demand interpolation scenario.",
        is_meta_train=True,
        seed=2026062304,
    ),
    "extrapolation_1": ScenarioConfig(
        name="extrapolation_1",
        lane_count=3,
        bottleneck_length_m=150,
        cav_ratio=0.10,
        main_flow=3500,
        ramp_flow=1800,
        description="Low CAV-ratio OOD accident-like lane closure scenario.",
        is_meta_train=False,
        seed=2026062305,
    ),
    "extrapolation_2": ScenarioConfig(
        name="extrapolation_2",
        lane_count=5,
        bottleneck_length_m=150,
        cav_ratio=1.00,
        main_flow=5100,
        ramp_flow=3100,
        description="Full-CAV continuous downstream bottleneck OOD scenario.",
        is_meta_train=False,
        seed=2026062306,
        extra={"downstream_lane_count": 3, "continuous_bottleneck": True},
    ),
}


def get_scenario(name: str) -> ScenarioConfig:
    try:
        return _SCENARIOS[name]
    except KeyError as exc:
        raise KeyError(f"unknown scenario '{name}', available: {', '.join(list_scenarios())}") from exc


def list_scenarios() -> list[str]:
    return sorted(_SCENARIOS)


def meta_train_scenarios() -> list[ScenarioConfig]:
    return [scenario for scenario in _SCENARIOS.values() if scenario.is_meta_train]


def meta_test_scenarios() -> list[ScenarioConfig]:
    return [scenario for scenario in _SCENARIOS.values() if not scenario.is_meta_train]
