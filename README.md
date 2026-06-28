# CHD CoPer Traffic

This project reproduces the thesis workflow for moving-bottleneck variable speed limit control under mixed traffic flow:

- `Trans-Beta-PPO`: Transformer temporal encoder plus Beta-distribution bounded continuous PPO.
- `Topology-Aware-Trans-Beta-PPO`: Trans-Beta-PPO with moving-bottleneck configuration, CAV position/gap feedback, and reward shaping.
- `MAML-Trans-Beta-PPO`: first-order MAML orchestration for multi-scenario meta reinforcement learning.
- `Reptile-Trans-Beta-PPO`: lighter first-order meta-RL variant that is often more stable for SUMO-in-the-loop training.
- `Context-Meta-Trans-Beta-PPO`: context-conditioned meta-RL variant for fast online task inference.
- `SAC`: maximum-entropy off-policy continuous-control baseline.
- SUMO-compatible environment interfaces for CAV moving bottleneck control.
- Default control timing: one episode is `3600` SUMO seconds, one RL step/control cycle is `120` SUMO seconds.
- Control action: `[speed_limit, control_start, control_end, longitudinal_gap]`.
- Spatial control-zone actions follow an upstream/downstream constrained mapping: the default bottleneck is at `7500 m`, with `2500 m` upstream control length, `300 m` recovery length, `0.40/0.35` start/end candidate fractions, and `1500 m` minimum control length.

## Setup

Windows + PyCharm + NVIDIA GPU is the primary target environment. See [docs/WINDOWS_PYCHARM_GPU.md](docs/WINDOWS_PYCHARM_GPU.md).

```bat
py -3.9 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip install -r requirements-windows-gpu.txt
```

Recommended versions for reproducing the thesis experiments:

- Python: `3.9.17`
- SUMO: `1.25.0`
- PyTorch: `2.1.0`
- pandas: `1.5.3`
- matplotlib: `3.7.1`

Install SUMO 1.25.0 and set `SUMO_HOME` in Windows system environment variables, for example:

```text
SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
```

Generate thesis SUMO network, route and simulation files from PyCharm:

```text
run/build_sumo_network.py
```

This creates `data/sumo/base_network/*.net.xml`, `data/sumo/routes/*.rou.xml`, and `data/sumo/configs/*.sumocfg`.

## PyCharm Configuration

1. Open this folder as the PyCharm project: `CHD_CoPer_Traffic`.
2. Create or select a Python 3.9.17 virtual environment at `.venv`.
3. Install GPU dependencies from `requirements-windows-gpu.txt`.
4. Add the project root as a source root if PyCharm does not detect it automatically.
5. Set environment variable `SUMO_HOME` in each run configuration.
6. Prefer the Python files under `run/` for right-click execution in PyCharm:

```text
run/check_python_environment.py
run/check_sumo_assets.py
run/build_sumo_network.py
run/train_trans_beta_ppo.py
run/evaluate_trans_beta_ppo.py
run/train_meta_trans_beta_ppo.py
run/evaluate_meta_trans_beta_ppo.py
run/run_chapter4_comparison.py
run/run_chapter5_baselines.py
run/train_sac.py
run/evaluate_sac.py
run/train_topology_aware_trans_beta_ppo.py
run/evaluate_topology_aware_trans_beta_ppo.py
run/train_reptile_trans_beta_ppo.py
run/evaluate_reptile_trans_beta_ppo.py
run/train_context_meta_trans_beta_ppo.py
run/evaluate_context_meta_trans_beta_ppo.py
```

Each file has editable constants such as `SMOKE`, `CONFIG_PATH`, `CHECKPOINT`, and `EPISODES` near the top.

Module mode is also supported:

```text
Module name: src.cli.train
Parameters: --config configs/rl/trans_beta_ppo.yaml
Working directory: <project root>
```

Training entries default to real SUMO mode. Use `SMOKE = True` only for debugging without SUMO.

## Windows PyCharm Runs

For a step-by-step guide that explains what to train, test, compare, and where outputs are saved, see [docs/RUNBOOK.md](docs/RUNBOOK.md). For a detailed Chinese guide covering modules, model design, experiment design, SUMO simulation, meta-RL, hyperparameters, and extension steps, see [docs/DETAILED_CHINESE_GUIDE.md](docs/DETAILED_CHINESE_GUIDE.md). For the folder/module layout, see [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

Recommended PyCharm order:

```text
run/check_python_environment.py
run/build_sumo_network.py
run/train_trans_beta_ppo.py
run/evaluate_trans_beta_ppo.py
run/run_chapter4_comparison.py
run/train_meta_trans_beta_ppo.py
run/evaluate_meta_trans_beta_ppo.py
```

## Full Reproduction Commands

```bat
.venv\Scripts\python.exe -m src.cli.train --config configs\rl\trans_beta_ppo.yaml
.venv\Scripts\python.exe -m src.cli.test --config configs\rl\trans_beta_ppo.yaml --checkpoint experiments\rl\<run>\checkpoints\trans_beta_ppo.pth
.venv\Scripts\python.exe -m src.cli.meta_train --config configs\meta_rl\maml_trans_beta_ppo.yaml
.venv\Scripts\python.exe -m src.cli.meta_test --config configs\meta_rl\maml_trans_beta_ppo.yaml --checkpoint experiments\meta_rl\<run>\checkpoints\maml_trans_beta_ppo.pth
.venv\Scripts\python.exe -m src.cli.meta_train --config configs\meta_rl\reptile_trans_beta_ppo.yaml
.venv\Scripts\python.exe -m src.cli.meta_test --config configs\meta_rl\reptile_trans_beta_ppo.yaml --checkpoint experiments\meta_rl\<run>\checkpoints\reptile_trans_beta_ppo.pth
.venv\Scripts\python.exe -m src.cli.train --config configs\rl\sac.yaml
.venv\Scripts\python.exe -m src.cli.meta_train --config configs\meta_rl\context_meta_trans_beta_ppo.yaml
```

## Experiment Comparison Design

Chapter 4 RL comparison configs:

- `configs/rl/continuous_ppo.yaml`: continuous Gaussian PPO baseline.
- `configs/rl/beta_ppo.yaml`: Beta-distribution PPO without Transformer.
- `configs/rl/trans_beta_ppo.yaml`: thesis Trans-Beta-PPO.
- `configs/rl/topology_aware_trans_beta_ppo.yaml`: Trans-Beta-PPO plus moving-bottleneck topology state/reward.
- `configs/rl/td3.yaml`: TD3 continuous-control baseline.
- `configs/rl/sac.yaml`: maximum-entropy off-policy continuous-control baseline.

Full batch run:

```bat
.venv\Scripts\python.exe -m src.cli.run_suite --suite configs\rl\comparison_suite.yaml
```

Chapter 5 Meta-RL comparison:

Recommended Chapter 5 baselines:

- `configs/baselines/feedback_vsl.yaml`: density-feedback VSL controller.
- `configs/baselines/mpc_vsl.yaml`: lightweight MPC-style VSL controller.
- `configs/rl/traditional_drl_vsl.yaml`: plain MLP DRL-VSL baseline.
- `configs/rl/vanilla_ppo.yaml`: standard continuous PPO.
- `configs/rl/dr_ppo.yaml`: distribution-robust PPO baseline with traffic-demand and CAV-ratio perturbation settings.
- `configs/rl/trans_beta_ppo.yaml`: strongest non-meta baseline from Chapter 4.
- `configs/meta_rl/maml_trans_beta_ppo.yaml`: proposed first-order MAML-Trans-Beta-PPO.
- `configs/meta_rl/reptile_trans_beta_ppo.yaml`: recommended lightweight Reptile-Trans-Beta-PPO meta-RL comparison.
- `configs/meta_rl/context_meta_trans_beta_ppo.yaml`: context-conditioned meta-RL variant for online task inference.

This comparison separates feedback/MPC control, generic DRL, PPO, robust PPO, Transformer/Beta-PPO, and the added meta-learning modules. Test all methods on extrapolation scenarios and compare reward, average speed, queue length, density, TET, TIT, performance decay rate, adaptation steps, and OOD robustness score.

## Realistic Longitudinal Search Gap

The CAV longitudinal search gap is mapped from the Beta policy action through a highway time-headway model:

```text
gap = vehicle_length + standstill_gap + speed * time_headway
```

The default executable range uses a CAV-parameter IDM/time-headway gap model with 1.1-2.0 s headway, clamps to 12-95 m, and records both raw actions and mapped physical gaps. This keeps low-speed congestion-control gaps available while preventing unsafe high-speed spacing.

## Outputs

Each run creates an experiment folder containing:

- resolved config;
- `metrics.csv`;
- `actions.csv`;
- checkpoints;
- `summary.json`.
