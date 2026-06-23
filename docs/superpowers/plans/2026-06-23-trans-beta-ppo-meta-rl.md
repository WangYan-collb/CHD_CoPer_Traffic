# Trans-Beta-PPO Meta-RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable thesis reproduction project with Trans-Beta-PPO, MAML-Trans-Beta-PPO, SUMO scenario configuration, realistic highway longitudinal gap mapping, training/testing CLIs, logging, and tests.

**Architecture:** The project is a Python package under `src/` with focused modules for models, algorithms, SUMO environment wrappers, moving-bottleneck control, rewards, scenarios, logging, evaluation, and command-line entry points. The first implementation provides deterministic unit-tested core behavior and SUMO-compatible interfaces; full SUMO runs depend on local `SUMO_HOME`.

**Tech Stack:** Python 3.9+, PyTorch, NumPy, PyYAML, pandas, pytest, optional SUMO TraCI.

---

### Task 1: Core Tests and Package Skeleton

**Files:**
- Create: `tests/test_longitudinal_gap.py`
- Create: `tests/test_beta_actor_critic.py`
- Create: `tests/test_rewards_and_scenarios.py`
- Create: `src/__init__.py`

- [ ] **Step 1: Write failing tests for physical gap mapping**

```python
from src.control.longitudinal_gap import LongitudinalGapMapper


def test_gap_mapping_uses_highway_time_headway_at_80_kmh():
    mapper = LongitudinalGapMapper()
    low, high = mapper.safe_gap_range(speed_mps=80 / 3.6)
    assert 28.0 <= low <= 30.0
    assert 46.0 <= high <= 48.0
    assert mapper.map_action(0.0, speed_mps=80 / 3.6) == low
    assert mapper.map_action(1.0, speed_mps=80 / 3.6) == high
```

- [ ] **Step 2: Write failing tests for Beta actor bounds**

```python
import torch
from src.models.beta_actor_critic import BetaActorCritic


def test_beta_actor_outputs_positive_distribution_params_and_bounded_actions():
    model = BetaActorCritic(state_dim=8, action_dim=3, sequence_length=4, embed_dim=32, num_heads=4)
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
```

- [ ] **Step 3: Write failing tests for rewards and thesis scenarios**

```python
from src.rewards.density_reward import density_efficiency_reward
from src.scenarios.scenario_registry import get_scenario


def test_density_reward_peaks_near_critical_density():
    assert density_efficiency_reward(27.0, 27.0) == 1.0
    assert density_efficiency_reward(35.0, 27.0) == 0.0
    assert 0.0 < density_efficiency_reward(20.0, 27.0) < 1.0


def test_extrapolation_1_matches_thesis_parameters():
    scenario = get_scenario("extrapolation_1")
    assert scenario.lane_count == 3
    assert scenario.bottleneck_length_m == 150
    assert scenario.cav_ratio == 0.10
    assert scenario.main_flow == 3500
    assert scenario.ramp_flow == 1800
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_longitudinal_gap.py tests/test_beta_actor_critic.py tests/test_rewards_and_scenarios.py -q`
Expected: FAIL because modules do not exist.

### Task 2: Implement Models, Rewards, Scenarios, and Control Mapping

**Files:**
- Create: `src/control/longitudinal_gap.py`
- Create: `src/models/transformer_encoder.py`
- Create: `src/models/beta_actor_critic.py`
- Create: `src/rewards/density_reward.py`
- Create: `src/rewards/safety_reward.py`
- Create: `src/rewards/efficiency_reward.py`
- Create: `src/rewards/combined_reward.py`
- Create: `src/scenarios/scenario_config.py`
- Create: `src/scenarios/scenario_registry.py`

- [ ] **Step 1: Implement the physical longitudinal gap mapper**

Use `gap = vehicle_length + standstill_gap + speed * headway`, clamp to `12-80 m`, and expose `safe_gap_range()` plus `map_action()`.

- [ ] **Step 2: Implement Transformer encoder and Beta actor-critic**

Use sinusoidal position encoding, `nn.TransformerEncoder`, `softplus + 1.0` alpha/beta heads, `torch.distributions.Beta`, and value output.

- [ ] **Step 3: Implement reward helpers**

Use density reward with peak at critical density and zero above critical density, plus speed, queue, safety, and smoothness terms in a configurable combined reward.

- [ ] **Step 4: Implement scenario registry**

Register base, interpolation, and extrapolation scenarios from the thesis with deterministic seeds and route-generation parameters.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_longitudinal_gap.py tests/test_beta_actor_critic.py tests/test_rewards_and_scenarios.py -q`
Expected: all selected tests pass.

### Task 3: Implement PPO and Meta-RL Training Core

**Files:**
- Create: `src/algorithms/trans_beta_ppo/buffer.py`
- Create: `src/algorithms/trans_beta_ppo/agent.py`
- Create: `src/algorithms/meta_trans_beta_ppo/maml.py`
- Create: `tests/test_ppo_agent.py`

- [ ] **Step 1: Write failing PPO buffer/update tests**

```python
import numpy as np
from src.algorithms.trans_beta_ppo.agent import TransBetaPPOAgent


def test_trans_beta_ppo_selects_bounded_action():
    agent = TransBetaPPOAgent(state_dim=8, action_dim=3, sequence_length=4)
    action, info = agent.select_action(np.zeros((4, 8), dtype=np.float32))
    assert action.shape == (3,)
    assert np.all(action >= 0.0)
    assert np.all(action <= 1.0)
    assert "log_prob" in info
```

- [ ] **Step 2: Implement rollout buffer and Trans-Beta-PPO agent**

Support action selection, transition storage, GAE returns, PPO update, checkpoint save/load.

- [ ] **Step 3: Implement first-order MAML wrapper**

Support cloning policy parameters for tasks, inner adaptation, outer meta-update over task batches, and meta-test adaptation hooks.

- [ ] **Step 4: Run algorithm tests**

Run: `pytest tests/test_ppo_agent.py -q`
Expected: pass.

### Task 4: SUMO-Compatible Environment, Logging, and CLI

**Files:**
- Create: `src/envs/sumo/env.py`
- Create: `src/envs/sumo/traci_client.py`
- Create: `src/envs/sumo/observations.py`
- Create: `src/envs/sumo/metrics.py`
- Create: `src/control/moving_bottleneck.py`
- Create: `src/control/conflict_resolution.py`
- Create: `src/control/speed_limit_mapper.py`
- Create: `src/logging_utils/experiment_logger.py`
- Create: `src/logging_utils/checkpoint.py`
- Create: `src/cli/train.py`
- Create: `src/cli/test.py`
- Create: `src/cli/meta_train.py`
- Create: `src/cli/meta_test.py`
- Create: `configs/rl/trans_beta_ppo.yaml`
- Create: `configs/meta_rl/maml_trans_beta_ppo.yaml`

- [ ] **Step 1: Implement SUMO adapter with graceful missing-SUMO failure**

If `SUMO_HOME` or `traci` is missing, raise a clear `SumoUnavailableError`. Unit tests and smoke tests can then skip SUMO-dependent execution.

- [ ] **Step 2: Implement moving bottleneck controller**

Select CAV candidates across lanes, apply mapped speed/gap controls, and return selected CAV IDs plus control diagnostics.

- [ ] **Step 3: Implement experiment logger and checkpoints**

Create per-run directories, write resolved config, `metrics.csv`, `actions.csv`, `summary.json`, and checkpoints.

- [ ] **Step 4: Implement CLIs**

Provide `train`, `test`, `meta_train`, and `meta_test` modules with `--config`, `--checkpoint`, `--episodes`, and `--smoke` options.

- [ ] **Step 5: Run CLI import smoke tests**

Run: `python -m src.cli.train --help`
Run: `python -m src.cli.meta_train --help`
Expected: both commands print usage and exit 0.

### Task 5: Project Metadata and Verification

**Files:**
- Create: `README.md`
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `.gitignore`

- [ ] **Step 1: Document setup and reproduction commands**

README must include dependency installation, `SUMO_HOME`, RL training/testing commands, Meta-RL training/testing commands, and output locations.

- [ ] **Step 2: Add minimal dependencies**

Use clean dependencies: `torch`, `numpy`, `pandas`, `pyyaml`, `pytest`, `matplotlib`, `sumolib`, `traci`.

- [ ] **Step 3: Run full verification**

Run: `pytest -q`
Run: `python -m src.cli.train --help`
Run: `python -m src.cli.meta_train --help`
Expected: tests pass and CLI help commands exit 0.
