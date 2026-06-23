from src.algorithms.registry import algorithm_names


def test_registry_exposes_thesis_comparison_algorithms():
    assert {
        "classic_vsl",
        "traditional_drl_vsl",
        "vanilla_ppo",
        "continuous_ppo",
        "beta_ppo",
        "dr_ppo",
        "trans_beta_ppo",
        "td3",
    }.issubset(set(algorithm_names()))
