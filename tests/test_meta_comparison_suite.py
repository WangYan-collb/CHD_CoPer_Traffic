from src.cli.common import load_config


def test_chapter5_suite_uses_reasonable_meta_rl_baselines():
    suite = load_config("configs/meta_rl/comparison_suite.yaml")

    assert suite["meta_config"] == "configs/meta_rl/maml_trans_beta_ppo.yaml"
    assert suite["baseline_configs"] == [
        "configs/baselines/classic_vsl.yaml",
        "configs/rl/traditional_drl_vsl.yaml",
        "configs/rl/vanilla_ppo.yaml",
        "configs/rl/dr_ppo.yaml",
        "configs/rl/trans_beta_ppo.yaml",
    ]
