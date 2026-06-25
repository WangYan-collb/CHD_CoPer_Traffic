# Project Structure / 项目结构

更详细的中文模块说明、模型设计、实验设计、SUMO 运行流程和扩展方法见：

```text
docs/DETAILED_CHINESE_GUIDE.md
```

This project is organized so each layer has one responsibility.

```text
configs/
  rl/                         Chapter 4 RL hyperparameter configs
  meta_rl/                    Chapter 5 meta-RL configs and comparison suite
  baselines/                  Rule-based baseline configs
  scenarios/                  Scenario notes/config snapshots

data/
  sumo/base_network/          Put test1.net.xml and E2_info.xml here
  sumo/generated_routes/      Runtime-generated rou.xml and sumocfg files

run/
  *.py                        PyCharm right-click runnable entry files

src/algorithms/
  trans_beta_ppo/             Transformer + Beta PPO
  beta_ppo/                   Beta PPO without Transformer
  continuous_ppo/             Gaussian continuous PPO
  td3/                        TD3 baseline
  dr_ppo/                     Robust PPO baseline wrapper
  meta_trans_beta_ppo/        First-order MAML/Reptile-style meta update
  rule_based/                 Feedback VSL, MPC-VSL and legacy rule baselines

src/models/
  transformer_encoder.py      Temporal sequence encoder
  beta_actor_critic.py        Transformer Beta actor-critic
  mlp_beta_actor_critic.py    Non-Transformer Beta actor-critic
  gaussian_actor_critic.py    Gaussian continuous actor-critic

src/envs/sumo/
  env.py                      RL environment and TraCI interaction loop
  config_builder.py           sumocfg generation
  traci_client.py             SUMO binary and TraCI loading
  state_calculator.py         Density, speed, queue and state-vector calculation
  metrics.py                  Traffic metric dataclass
  observations.py             Sequence state buffer for Transformer input

src/control/
  moving_bottleneck.py        CAV moving bottleneck command construction
  longitudinal_gap.py         Highway time-headway gap mapping
  speed_limit_mapper.py       Normalized action to physical speed limit
  conflict_resolution.py      CAV selection conflict filtering

src/scenarios/
  scenario_registry.py        Base, interpolation and extrapolation scenarios
  traffic_flow.py             Normal-distribution traffic flow sampling
  route_generator.py          SUMO route file generation
  vehicle_params.py           HDV/CAV SUMO vehicle type parameters
  road_network.py             Route edge names and road design constants
  sumo_network.py             PlainXML network, detector and sumocfg builder

src/rewards/
  combined_reward.py          Main reward aggregation
  density_reward.py           Density component
  efficiency_reward.py        Speed and queue component
  safety_reward.py            TTC/TET/TIT and action smoothness component

src/cli/
  train.py                    Generic training entry
  test.py                     Generic checkpoint test entry
  meta_train.py               Meta-RL training entry
  meta_test.py                Meta-RL evaluation entry
  run_suite.py                Batch comparison entry

src/logging_utils/
  experiment_logger.py        config.json, metrics.csv, actions.csv and summary.json

src/project/
  paths.py                    Project root, SUMO asset and output paths
```

For daily use, start from the files in `run/`. The `src/cli/` modules are still available for terminal/module-mode execution, but PyCharm users do not need to remember command-line arguments.

## 中文模块速览

```text
configs/        实验配置和超参数，不建议在代码里硬改参数
data/sumo/      SUMO 路网、route、sumocfg 和检测器文件
run/            PyCharm 右键运行入口
src/algorithms/ 强化学习、元强化学习和规则控制算法
src/models/     Transformer、Beta Actor-Critic、上下文编码网络
src/envs/sumo/  SUMO TraCI 交互、状态采样、拥堵预判、step 循环
src/control/    CAV 移动瓶颈构建、限速映射、纵向间隙映射
src/scenarios/  场景注册、车流生成、车辆参数、路网生成
src/rewards/    密度、速度、排队、安全、构型奖励
src/cli/        命令行训练、测试、元训练、批量实验入口
src/logging_utils/ 实验结果记录
```
