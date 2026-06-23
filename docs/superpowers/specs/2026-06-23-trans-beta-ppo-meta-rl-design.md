# Trans-Beta-PPO and Meta-RL Reproduction Design

## Goal

Build a clean, runnable reproduction project for the thesis codebase around SUMO-based moving bottleneck variable speed limit control. The project will use the original GitHub `WangYan-collb/ShareCode/new_mvsl_pro_DL` code as a reference for SUMO assets, traffic-flow generation, detectors, and moving-bottleneck interaction, but it will reorganize the implementation into focused modules and implement the thesis algorithms directly:

- Trans-Beta-PPO for multi-lane moving bottleneck VSL control.
- MAML-Trans-Beta-PPO for meta reinforcement learning across heterogeneous traffic scenarios.
- SUMO training and testing flows with reproducible records, checkpoints, and evaluation outputs.

## Source Context

The reference project contains useful SUMO assets and environment logic, but it mixes IDE files, algorithms, logs, model weights, generated XML outputs, and analysis scripts. The PPO implementations also still use Gaussian continuous policies in key files, while the thesis requires Beta-distribution bounded continuous action modeling. The reproduction project will therefore migrate useful assets and behavior while replacing algorithm internals with a clean implementation.

The thesis requires these core constraints:

- CAVs actively construct a controllable moving bottleneck upstream of the merge area.
- Multi-lane control uses state sensing, flexible control, intelligent decision-making, local search, and conflict resolution.
- Transformer encodes traffic-state history for long-term temporal traffic evolution.
- Beta policy outputs bounded continuous actions in `[0, 1]`.
- Meta-RL uses MAML-style inner adaptation and outer meta-training over heterogeneous scenarios.

## Project Structure

```text
CHD_CoPer_Traffic/
  configs/
    rl/
      trans_beta_ppo.yaml
    meta_rl/
      maml_trans_beta_ppo.yaml
    scenarios/
      base.yaml
      interpolation_1.yaml
      interpolation_2.yaml
      interpolation_3.yaml
      extrapolation_1.yaml
      extrapolation_2.yaml
  data/
    sumo/
      base_network/
      generated_routes/
  docs/
    superpowers/
      specs/
      plans/
  experiments/
    rl/
    meta_rl/
  scripts/
    train_trans_beta_ppo.sh
    test_trans_beta_ppo.sh
    train_maml_trans_beta_ppo.sh
    test_maml_trans_beta_ppo.sh
  src/
    algorithms/
      trans_beta_ppo/
      meta_trans_beta_ppo/
    control/
      moving_bottleneck.py
      longitudinal_gap.py
      conflict_resolution.py
      speed_limit_mapper.py
    envs/
      sumo/
        env.py
        traci_client.py
        observations.py
        metrics.py
    evaluation/
      evaluator.py
      plots.py
      scenario_metrics.py
    logging_utils/
      experiment_logger.py
      checkpoint.py
    models/
      transformer_encoder.py
      beta_actor_critic.py
    rewards/
      density_reward.py
      safety_reward.py
      efficiency_reward.py
      combined_reward.py
    scenarios/
      scenario_config.py
      route_generator.py
      scenario_registry.py
    cli/
      train.py
      test.py
      meta_train.py
      meta_test.py
  tests/
```

## Algorithm Design

### Trans-Beta-PPO

The single-task RL algorithm will use an actor-critic architecture:

- Input: fixed-length traffic-state sequence, default 30 control substeps.
- Encoder: linear state embedding, sinusoidal position encoding, Transformer encoder layers, temporal mean pooling.
- Actor: two heads output `alpha` and `beta` for each action dimension. A `softplus + 1.0` transform keeps parameters positive and avoids unstable near-zero Beta parameters.
- Distribution: `torch.distributions.Beta(alpha, beta)`.
- Critic: value estimate from the Transformer feature vector.
- PPO update: GAE advantage estimation, clipped policy objective, value loss, entropy bonus, gradient clipping, checkpointing.

The action vector will be bounded in `[0, 1]`, then mapped into physically meaningful control commands:

- Speed limit.
- Moving bottleneck start position.
- Moving bottleneck longitudinal search gap.

### MAML-Trans-Beta-PPO

The meta-RL algorithm will wrap the Trans-Beta-PPO policy:

- Outer loop samples source scenarios from the base and interpolation scenario set.
- Inner loop adapts policy parameters with a small number of PPO gradient steps on one scenario.
- First-order MAML will be used by default to avoid second-order gradient cost during SUMO training.
- Outer update optimizes the post-adaptation policy performance across sampled tasks.
- Meta-test evaluates fast adaptation on interpolation and extrapolation scenarios.

The default meta-learning settings follow the thesis:

- Meta learning rate: `0.0005`.
- Inner steps: `3`.
- Meta batch size: `4`.
- Meta training episodes: configurable, default `500` for full reproduction and a smaller smoke-test profile for verification.

## SUMO Environment Design

The environment will expose a Gymnasium-style API:

```python
state, info = env.reset(scenario=scenario_config)
next_state, reward, terminated, truncated, info = env.step(action)
```

Environment responsibilities:

- Start and stop SUMO through TraCI.
- Generate route files from scenario configuration.
- Collect detector state, lane density, speed, queue length, TET, TIT, emissions, and throughput.
- Maintain a rolling state-history buffer for Transformer input.
- Convert algorithm actions into CAV speed and lane-change commands.
- Record per-step metrics when configured.

The original SUMO network and detector files from `new_mvsl_pro_DL/configuration` will be migrated into `data/sumo/base_network`, with generated route files separated under `data/sumo/generated_routes`.

## Scenario Design

The scenario registry will include the thesis scenario library.

| Scenario | Bottleneck | CAV Ratio | Main Flow | Ramp Flow | Purpose |
| --- | --- | ---: | ---: | ---: | --- |
| Base | 5-lane merge, 150 m | 0.50 | 4292 | 2446 | Meta-training source |
| Interpolation 1 | 4-lane merge, 180 m | 0.30 | 4100 | 2200 | Meta-training source |
| Interpolation 2 | 5-lane merge, 150 m | 0.70 | 4500 | 2700 | Meta-training source |
| Interpolation 3 | 4-lane merge, 180 m | 0.50 | 5200 | 2800 | Meta-training source |
| Extrapolation 1 | 3-lane merge, 150 m | 0.10 | 3500 | 1800 | OOD validation |
| Extrapolation 2 | 5-lane plus downstream 3-lane continuous bottleneck | 1.00 | 5100 | 3100 | OOD validation |

Every scenario will use explicit random seeds and route-generation parameters so training and testing can be repeated.

## Realistic Longitudinal Search Gap Design

The longitudinal search gap must be realistic for highway CAV moving-bottleneck construction. It will not be treated as an arbitrary 5-30 m slider.

The executable search gap will be derived from speed, vehicle length, minimum standstill gap, and time headway:

```text
gap_m = vehicle_length_m + standstill_gap_m + speed_mps * time_headway_s
```

Default physical parameters:

- Passenger car length: `4.5 m`.
- Minimum standstill gap: `2.0 m`.
- Absolute minimum executable gap: `12 m`.
- Absolute maximum executable gap: `80 m`.
- Normal highway control time-headway range: `0.8 s` to `2.0 s`.
- Conservative default range: `1.0 s` to `1.8 s`.

For example:

- At 60 km/h (`16.7 m/s`), a 1.0-1.8 s headway gives about `23-37 m`.
- At 80 km/h (`22.2 m/s`), a 1.0-1.8 s headway gives about `29-47 m`.
- At 100 km/h (`27.8 m/s`), a 1.0-1.8 s headway gives about `34-57 m`.

The Beta action for longitudinal search will map to a scenario-specific safe interval computed from the current control speed:

```text
gap_min = clamp(vehicle_length + standstill_gap + speed * min_headway, 12, 80)
gap_max = clamp(vehicle_length + standstill_gap + speed * max_headway, gap_min + 5, 80)
gap = gap_min + action * (gap_max - gap_min)
```

Additional realism checks before applying CAV control:

- Reject or relax gaps that put adjacent controlled CAVs within the emergency minimum spacing.
- Prefer gaps in the 15-60 m range for normal highway operation.
- Allow 60-80 m only for sparse CAV or high-speed conditions where closer control would be unsafe.
- Store both raw action and mapped physical gap in logs.

This keeps the thesis sensitivity range of 15-25 m available under lower-speed congestion control, while preventing unrealistic spacing at higher speed.

## Moving Bottleneck Control Design

The moving bottleneck module will be split into:

- `longitudinal_gap.py`: maps normalized actions to realistic highway gaps.
- `moving_bottleneck.py`: selects CAVs across lanes in the control area.
- `conflict_resolution.py`: detects invalid CAV formations and repairs them.
- `speed_limit_mapper.py`: maps normalized speed actions into speed commands.

The controller will:

1. Locate candidate CAVs in the upstream control area.
2. Select one controlled CAV per target lane where possible.
3. Use the realistic longitudinal gap to form a staggered cross-lane bottleneck.
4. Apply speed limits through TraCI only to selected CAVs.
5. Restore normal lane-change and speed behavior after the vehicle leaves the control area.

## Reward Design

Reward functions will be separate files under `src/rewards`.

Default combined reward:

- Density efficiency reward: piecewise reward around critical density, matching the thesis Chapter 4 idea.
- Speed reward: encourages higher bottleneck and network speeds without rewarding unsafe acceleration.
- Queue penalty: penalizes ramp and bottleneck queue growth.
- Safety penalty: penalizes high TET/TIT and low TTC conditions.
- Smoothness penalty: penalizes large action changes between control periods.

Weights will be configured in YAML so experiments can reproduce thesis settings or run ablations.

## Logging and Records

The logging layer will create one experiment directory per run:

```text
experiments/rl/<timestamp>_<scenario>_<seed>/
experiments/meta_rl/<timestamp>_<run_name>/
```

Each run will include:

- `config.yaml`: resolved configuration.
- `metrics.csv`: per-step and per-episode metrics.
- `actions.csv`: raw and mapped actions.
- `checkpoints/`: model weights.
- `summary.json`: final evaluation metrics.
- `plots/`: generated reward and metric curves.

Meta-RL runs will additionally record:

- task-level adaptation curves;
- inner-loop losses;
- outer-loop reward;
- interpolation and extrapolation performance;
- performance decay and OOD robustness score.

## CLI Acceptance Commands

The finished project must support:

```bash
python -m src.cli.train --config configs/rl/trans_beta_ppo.yaml
python -m src.cli.test --config configs/rl/trans_beta_ppo.yaml --checkpoint <path>
python -m src.cli.meta_train --config configs/meta_rl/maml_trans_beta_ppo.yaml
python -m src.cli.meta_test --config configs/meta_rl/maml_trans_beta_ppo.yaml --checkpoint <path>
```

For development verification, the project will also include smoke-test configs that run for a very short simulation horizon without requiring full 500-episode training.

## Testing Strategy

Unit tests will cover:

- Beta actor output parameter positivity and action bounds.
- Longitudinal gap mapping under 60, 80, and 100 km/h.
- Reward function behavior around critical density and unsafe TTC.
- Scenario registry values.
- Route-generation XML validity for representative scenarios.
- Checkpoint save/load.

SUMO smoke tests will run only when `SUMO_HOME` and SUMO binaries are available.

## Implementation Assumptions

- Python 3.9+ will be used.
- SUMO must be installed locally and `SUMO_HOME` must point to the installation.
- The project will prefer CPU-compatible training by default, with CUDA used automatically if available.
- Original GitHub assets can be migrated into this workspace.
- Generated outputs and checkpoints will not be committed by default.

## Open Risks

- The original SUMO network may require adjustment for 3-lane and continuous-bottleneck extrapolation scenarios. If network geometry cannot represent a scenario directly, the scenario generator will create the closest reproducible approximation and document it in the run config.
- Full 500-episode SUMO training may take a long time. Smoke tests will verify software correctness; thesis-scale runs will be launched separately.
- Some exact thesis figures depend on stochastic traffic generation and hardware/runtime conditions. The project will reproduce the methodology and metrics, not hard-code thesis tables.
