import torch

from src.models.beta_actor_critic import BetaActorCritic


def test_beta_actor_outputs_positive_distribution_params_and_bounded_actions():
    model = BetaActorCritic(
        state_dim=8,
        action_dim=3,
        sequence_length=4,
        embed_dim=32,
        num_heads=4,
    )
    states = torch.zeros(2, 4, 8)

    action, log_prob, entropy, value = model.act(states)
    alpha, beta = model.distribution_params(states)

    assert torch.all(alpha > 1.0)
    assert torch.all(beta > 1.0)
    assert torch.all(action >= 0.0)
    assert torch.all(action <= 1.0)
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2,)


def test_beta_actor_evaluates_supplied_actions():
    model = BetaActorCritic(
        state_dim=8,
        action_dim=3,
        sequence_length=4,
        embed_dim=32,
        num_heads=4,
    )
    states = torch.zeros(2, 4, 8)
    actions = torch.full((2, 3), 0.5)

    log_prob, entropy, value = model.evaluate_actions(states, actions)

    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2,)
