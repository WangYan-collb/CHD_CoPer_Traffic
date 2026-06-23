# CHD CoPer Traffic

This project reproduces the thesis workflow for moving-bottleneck variable speed limit control under mixed traffic flow:

- `Trans-Beta-PPO`: Transformer temporal encoder plus Beta-distribution bounded continuous PPO.
- `MAML-Trans-Beta-PPO`: first-order MAML orchestration for multi-scenario meta reinforcement learning.
- SUMO-compatible environment interfaces for CAV moving bottleneck control.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Recommended versions for reproducing the thesis experiments:

- Python: `3.9.17`
- SUMO: `1.25.0`
- PyTorch: `2.1.0`
- pandas: `1.5.3`
- matplotlib: `3.7.1`

For real SUMO runs, install SUMO 1.25.0 and export `SUMO_HOME`:

```bash
export SUMO_HOME=/path/to/sumo
```

On Windows, set `SUMO_HOME` in system environment variables, for example:

```text
SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
```

Copy the SUMO network assets from the original thesis project into:

```text
data/sumo/base_network/test1.net.xml
data/sumo/base_network/E2_info.xml
```

At runtime the project generates scenario route files and `.sumocfg` files under `data/sumo/generated_routes/`.

## PyCharm Configuration

1. Open this folder as the PyCharm project: `CHD_CoPer_Traffic`.
2. Create or select a Python 3.9.17 virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Add the project root as a source root if PyCharm does not detect it automatically.
5. Set environment variable `SUMO_HOME` in each run configuration.
6. Prefer the Python files under `run/` for right-click execution in PyCharm:

```text
run/check_python_environment.py
run/check_sumo_assets.py
run/train_trans_beta_ppo.py
run/evaluate_trans_beta_ppo.py
run/train_meta_trans_beta_ppo.py
run/evaluate_meta_trans_beta_ppo.py
run/run_chapter4_comparison.py
run/run_chapter5_baselines.py
```

Each file has editable constants such as `SMOKE`, `CONFIG_PATH`, `CHECKPOINT`, and `EPISODES` near the top.

Module mode is also supported:

```text
Module name: src.cli.train
Parameters: --config configs/rl/trans_beta_ppo.yaml --smoke
Working directory: <project root>
```

For full SUMO training, remove `--smoke`.

## Smoke Runs

For a step-by-step guide that explains what to train, test, compare, and where outputs are saved, see [docs/RUNBOOK.md](docs/RUNBOOK.md). For the folder/module layout, see [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

Smoke mode validates the Python/algorithm/logging path without launching SUMO:

```bash
.venv/bin/python -m src.cli.train --config configs/rl/trans_beta_ppo.yaml --smoke
.venv/bin/python -m src.cli.test --config configs/rl/trans_beta_ppo.yaml --smoke
.venv/bin/python -m src.cli.meta_train --config configs/meta_rl/maml_trans_beta_ppo.yaml --smoke
.venv/bin/python -m src.cli.meta_test --config configs/meta_rl/maml_trans_beta_ppo.yaml --smoke
```

## Full Reproduction Commands

```bash
.venv/bin/python -m src.cli.train --config configs/rl/trans_beta_ppo.yaml
.venv/bin/python -m src.cli.test --config configs/rl/trans_beta_ppo.yaml --checkpoint experiments/rl/<run>/checkpoints/trans_beta_ppo.pth
.venv/bin/python -m src.cli.meta_train --config configs/meta_rl/maml_trans_beta_ppo.yaml
.venv/bin/python -m src.cli.meta_test --config configs/meta_rl/maml_trans_beta_ppo.yaml --checkpoint experiments/meta_rl/<run>/checkpoints/maml_trans_beta_ppo.pth
```

## Experiment Comparison Design

Chapter 4 RL comparison configs:

- `configs/rl/continuous_ppo.yaml`: continuous Gaussian PPO baseline.
- `configs/rl/beta_ppo.yaml`: Beta-distribution PPO without Transformer.
- `configs/rl/trans_beta_ppo.yaml`: thesis Trans-Beta-PPO.
- `configs/rl/td3.yaml`: TD3 continuous-control baseline.

Batch smoke run:

```bash
.venv/bin/python -m src.cli.run_suite --suite configs/rl/comparison_suite.yaml --smoke
```

Full batch run:

```bash
.venv/bin/python -m src.cli.run_suite --suite configs/rl/comparison_suite.yaml
```

Chapter 5 Meta-RL comparison:

Recommended Chapter 5 baselines:

- `configs/baselines/classic_vsl.yaml`: non-learning rule-based VSL lower bound.
- `configs/rl/traditional_drl_vsl.yaml`: plain MLP DRL-VSL baseline.
- `configs/rl/vanilla_ppo.yaml`: standard continuous PPO.
- `configs/rl/dr_ppo.yaml`: distribution-robust PPO baseline with traffic-demand and CAV-ratio perturbation settings.
- `configs/rl/trans_beta_ppo.yaml`: strongest non-meta baseline from Chapter 4.
- `configs/meta_rl/maml_trans_beta_ppo.yaml`: proposed MAML-Trans-Beta-PPO.

This comparison is stronger than only comparing against weak DRL models: it separates rule control, generic DRL, PPO, robust PPO, Transformer/Beta-PPO, and the added MAML meta-learning module. Test all methods on extrapolation scenarios and compare reward, average speed, queue length, density, TET, TIT, performance decay rate, adaptation steps, and OOD robustness score.

## Realistic Longitudinal Search Gap

The CAV longitudinal search gap is mapped from the Beta policy action through a highway time-headway model:

```text
gap = vehicle_length + standstill_gap + speed * time_headway
```

The default executable range uses 1.0-1.8 s headway, clamps to 12-80 m, and records both raw actions and mapped physical gaps. This keeps the thesis 15-25 m congestion-control region available while preventing unsafe high-speed spacing.

## Outputs

Each run creates an experiment folder containing:

- resolved config;
- `metrics.csv`;
- `actions.csv`;
- checkpoints;
- `summary.json`.
