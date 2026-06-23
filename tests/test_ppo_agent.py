import numpy as np

from src.algorithms.trans_beta_ppo.agent import TransBetaPPOAgent


def test_trans_beta_ppo_selects_bounded_action():
    agent = TransBetaPPOAgent(state_dim=8, action_dim=3, sequence_length=4)

    action, info = agent.select_action(np.zeros((4, 8), dtype=np.float32))

    assert action.shape == (3,)
    assert np.all(action >= 0.0)
    assert np.all(action <= 1.0)
    assert "log_prob" in info
    assert "value" in info
