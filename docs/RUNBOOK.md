# CHD_CoPer_Traffic 运行手册

这份文档面向第一次接手项目的同学，按“先确认 Windows/PyCharm/GPU 环境、生成 SUMO 路网和车流、再跑训练和对比分析”的顺序执行。

如果需要了解每个模块做什么、模型和实验为什么这样设计、元强化学习如何扩展、新模型如何增加，请看更详细的中文说明：

```text
docs/DETAILED_CHINESE_GUIDE.md
```

## 1. 项目目标

本项目用于复现论文中基于智能网联车移动瓶颈控制的高速公路车流控制实验，核心内容包括：

- 第四章：连续动作强化学习对比实验，重点模型是 `Trans-Beta-PPO`。
- 第五章：在 `Trans-Beta-PPO` 基础上加入元学习，形成 `MAML-Trans-Beta-PPO`，并提供更轻量的 `Reptile-Trans-Beta-PPO` 和上下文条件化 `Context-Meta-Trans-Beta-PPO` 作为元强化学习对比模型。
- 仿真环境：SUMO 路网、车辆类型、正态分布车流、CAV 主动构建移动瓶颈。

当前默认仿真节奏：

```text
episode = 3600 仿真秒
RL step / control cycle = 120 仿真秒
每个 step 内每 1 仿真秒采集一次交通状态
step 返回状态 = 120 秒逐秒样本的平均交通状态
```

动作空间为 4 维连续动作：

```text
a0: CAV 限速值映射
a1: 控制区域起点
a2: 控制区域终点
a3: 纵向搜索间隙
```

## 2. 环境准备

推荐版本：

```text
Python: 3.9.17
SUMO: 1.25.0
PyTorch: 2.1.0
numpy: 1.26.3
pandas: 1.5.3
matplotlib: 3.7.1
```

Windows PyCharm + GPU 的详细安装说明见：

```text
docs/WINDOWS_PYCHARM_GPU.md
```

如果在 PyCharm 里运行：

1. Open Project 选择项目根目录 `CHD_CoPer_Traffic`。
2. Interpreter 选择 `.venv\Scripts\python.exe`。
3. Working directory 设置为项目根目录。
4. 环境变量里设置 `SUMO_HOME`。
5. 优先右键运行 `run/` 目录下的 Python 文件；命令行入口使用 `scripts/*.bat`。
6. 训练入口默认是真实 SUMO 模式，只有调试时才把 `SMOKE = True`。

工程模块说明见：

```text
docs/PROJECT_STRUCTURE.md
```

Windows 示例：

```text
SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
```

## 3. SUMO 文件放置

正式 SUMO 实验需要路网文件、车流文件和仿真配置文件。本项目通过 `run/build_sumo_network.py` 生成：

```text
data/sumo/base_network/*.net.xml
data/sumo/base_network/*.add.xml
data/sumo/routes/*.rou.xml
data/sumo/configs/*.sumocfg
```

基础场景同时兼容旧配置文件名：

```text
data/sumo/base_network/test1.net.xml
data/sumo/base_network/E2_info.xml
```

第一次在 Windows 上运行：

```text
run/build_sumo_network.py
```

该入口按论文 M25 合流区设置生成 SUMO PlainXML 路网源文件，并在本机安装 SUMO 后调用 `netconvert` 生成可运行的 `.net.xml`。生成的场景路网包括：

```text
data/sumo/base_network/base.net.xml
data/sumo/base_network/interpolation_1.net.xml
data/sumo/base_network/interpolation_2.net.xml
data/sumo/base_network/interpolation_3.net.xml
data/sumo/base_network/extrapolation_1.net.xml
data/sumo/base_network/extrapolation_2.net.xml
```

其中 `base.net.xml` 会同步复制为 `test1.net.xml`，`base.add.xml` 会同步复制为 `E2_info.xml`，兼容默认训练配置。第五章不同场景会优先使用对应的场景路网，例如 `extrapolation_1.net.xml` 是 3 车道低 CAV 渗透率外推场景，`extrapolation_2.net.xml` 包含下游车道收缩设置。

## 4. PyCharm 快速运行入口

推荐直接右键运行这些文件：

```text
run/check_python_environment.py
run/build_sumo_network.py
run/check_sumo_assets.py
run/generate_sumo_routes.py
run/train_trans_beta_ppo.py
run/evaluate_trans_beta_ppo.py
run/run_chapter4_comparison.py
run/train_sac.py
run/evaluate_sac.py
run/train_topology_aware_trans_beta_ppo.py
run/evaluate_topology_aware_trans_beta_ppo.py
run/train_meta_trans_beta_ppo.py
run/evaluate_meta_trans_beta_ppo.py
run/train_reptile_trans_beta_ppo.py
run/evaluate_reptile_trans_beta_ppo.py
run/train_context_meta_trans_beta_ppo.py
run/evaluate_context_meta_trans_beta_ppo.py
run/run_chapter5_baselines.py
```

每个入口文件顶部都有可改参数：

```python
CONFIG_PATH = "configs/rl/trans_beta_ppo.yaml"
SMOKE = False
EPISODES = None
CHECKPOINT = None
```

`SMOKE = False` 表示真实 SUMO 交互训练。只有在不想启动 SUMO、仅检查 Python 逻辑时才临时改成 `True`。

控制周期在配置文件里修改：

```yaml
environment:
  simulation_time_s: 3600
  control_cycle_s: 120
  aggregation_time_s: 120
  bottleneck_position_m: 7500.0
  upstream_control_length_m: 2500.0
  recovery_length_m: 300.0
  start_position_fraction: 0.40
  end_position_fraction: 0.35
  min_control_length_m: 1500.0
  route_randomization_enabled: true
```

其中 `control_cycle_s` 决定一个强化学习 step 持续多少仿真秒；`aggregation_time_s` 保持同值，用于表示该周期内状态聚合窗口。

控制区起点和终点参考论文中的空间动作映射方式：不是在整条上游路段任意取点，而是在瓶颈上游有效控制区内分段映射。默认设置表示瓶颈位于 7500 m，向上游取 2500 m 作为可控区域，并在瓶颈前保留 300 m 速度适应/恢复区，因此有效控制长度为 2200 m。起点动作映射到更靠上游的候选段，终点动作映射到靠近瓶颈但不进入瓶颈的候选段，并保证控制区长度不少于 1500 m。训练时 `route_randomization_enabled=true`，每个 episode 会用不同 route seed 采样车流；测试入口不传 episode seed，默认固定车流，便于复现实验对比。

## 5. 第四章强化学习训练

主模型是 `Trans-Beta-PPO`：

```bat
.venv\Scripts\python.exe -m src.cli.train --config configs\rl\trans_beta_ppo.yaml
```

训练完成后终端会输出：

```text
Run directory: experiments/rl/<timestamp>_trans_beta_ppo_base
```

常用输出文件：

```text
config.json
metrics.csv
actions.csv
summary.json
checkpoints/trans_beta_ppo.pth
```

测试训练好的 checkpoint：

```bat
.venv\Scripts\python.exe -m src.cli.test --config configs\rl\trans_beta_ppo.yaml --checkpoint experiments\rl\<run>\checkpoints\trans_beta_ppo.pth
```

## 6. 第四章对比实验

推荐对比顺序：

1. `continuous_ppo`: 高斯连续 PPO。
2. `beta_ppo`: Beta 连续 PPO，不加 Transformer。
3. `trans_beta_ppo`: 本文第四章主模型。
4. `td3`: 连续控制 off-policy 对比模型。
5. `sac`: 最大熵 off-policy 连续控制模型，用于比较 PPO 系列和更样本高效的连续控制方法。
6. `topology_aware_trans_beta_ppo`: 在主模型基础上加入移动瓶颈构型状态和构型奖励，用于验证 CAV 构型信息是否提升控制推理。

单独训练某个模型：

```bat
.venv\Scripts\python.exe -m src.cli.train --config configs\rl\beta_ppo.yaml
.venv\Scripts\python.exe -m src.cli.train --config configs\rl\continuous_ppo.yaml
.venv\Scripts\python.exe -m src.cli.train --config configs\rl\td3.yaml
.venv\Scripts\python.exe -m src.cli.train --config configs\rl\sac.yaml
.venv\Scripts\python.exe -m src.cli.train --config configs\rl\topology_aware_trans_beta_ppo.yaml
```

批量跑第四章对比：

```bat
.venv\Scripts\python.exe -m src.cli.run_suite --suite configs\rl\comparison_suite.yaml
```

对比分析建议看这些指标：

```text
reward
density
speed_mps
queue_m
speed_limit_kmh
longitudinal_gap_m
selected_cav_count
```

其中 `actions.csv` 用于分析模型控制行为，`metrics.csv` 用于画训练收敛和交通效率指标。

## 7. 第五章元强化学习训练

元强化学习主配置：

```text
configs/meta_rl/maml_trans_beta_ppo.yaml
```

训练场景由 `src/scenarios/scenario_registry.py` 自动加载：

```text
meta-train: base, interpolation_1, interpolation_2, interpolation_3
meta-test: extrapolation_1, extrapolation_2
```

也就是说，直接运行元训练入口时，会加载基准场景和三个插值场景。`meta_batch_size=4` 时每个 meta episode 都使用这 4 个任务；以后如果增加更多插值场景，代码会按随机种子从任务池里抽取一个 meta batch。

训练：

```bat
.venv\Scripts\python.exe -m src.cli.meta_train --config configs\meta_rl\maml_trans_beta_ppo.yaml
```

训练完成后会生成：

```text
experiments/meta_rl/<timestamp>_maml_trans_beta_ppo/
```

关键文件：

```text
metrics.csv
summary.json
checkpoints/maml_trans_beta_ppo.pth
```

测试：

```bat
.venv\Scripts\python.exe -m src.cli.meta_test --config configs\meta_rl\maml_trans_beta_ppo.yaml --checkpoint experiments\meta_rl\<run>\checkpoints\maml_trans_beta_ppo.pth
```

当前元学习实现是一阶 MAML/Reptile 风格：每个任务从同一个初始策略出发做少量内循环适应，然后把多个任务适应后的策略变化汇总到 meta policy。这样比二阶 MAML 更适合 SUMO，因为仿真成本低一些、实现更稳定。

如果想跑更稳定的一阶元学习对比，推荐使用 `Reptile-Trans-Beta-PPO`：

```bat
.venv\Scripts\python.exe -m src.cli.meta_train --config configs\meta_rl\reptile_trans_beta_ppo.yaml
.venv\Scripts\python.exe -m src.cli.meta_test --config configs\meta_rl\reptile_trans_beta_ppo.yaml --checkpoint experiments\meta_rl\<run>\checkpoints\reptile_trans_beta_ppo.pth
```

它的训练流程是：

```text
1. 读取 meta-train 场景池。
2. 每个 meta episode 选择一个 meta batch。
3. 对每个场景，从同一 Trans-Beta-PPO 初始化开始。
4. 在该场景内跑 inner_steps 次 SUMO episode，并执行 PPO 内循环更新。
5. 收集所有场景适应后的策略参数。
6. 将 meta 初始化朝这些适应后参数的平均方向移动。
7. 在 extrapolation_1、extrapolation_2 上看少步适应能力和 OOD 泛化。
```

如果想加入近几年常用的“上下文/隐变量任务推断”思想，运行：

```bat
.venv\Scripts\python.exe -m src.cli.meta_train --config configs\meta_rl\context_meta_trans_beta_ppo.yaml
.venv\Scripts\python.exe -m src.cli.meta_test --config configs\meta_rl\context_meta_trans_beta_ppo.yaml --checkpoint experiments\meta_rl\<run>\checkpoints\context_meta_trans_beta_ppo.pth
```

这个模型保留 Transformer + Beta-PPO 主体，同时从最近一个状态序列的均值和波动中编码 traffic context。它适合 VSL 的原因是新场景通常体现为需求强度、CAV 渗透率、瓶颈形态和拥堵传播状态的变化，context 分支可以在线推断这些差异，再配合少步 Reptile 适应。

## 9. 第五章对比实验

推荐对比模型：

1. `feedback_vsl`: 基于密度/排队反馈的传统 VSL 控制。
2. `mpc_vsl`: 简化 MPC 式 VSL 控制。
3. `traditional_drl_vsl`: 传统 MLP DRL-VSL。
4. `vanilla_ppo`: 标准 PPO。
5. `dr_ppo`: 鲁棒 PPO，对车流和 CAV 渗透率扰动。
6. `trans_beta_ppo`: 不加元学习的第四章最强模型。
7. `maml_trans_beta_ppo`: 第五章主模型。
8. `reptile_trans_beta_ppo`: 一阶 Reptile 元强化学习对比模型，适合 SUMO 高成本训练。
9. `context_meta_trans_beta_ppo`: 上下文条件化元强化学习模型，用于验证在线任务识别是否提升少步适应。

配置入口：

```text
configs/meta_rl/comparison_suite.yaml
```

先训练非元学习基线：

```bat
.venv\Scripts\python.exe -m src.cli.run_suite --suite configs\meta_rl\comparison_suite.yaml
```

再训练元强化学习模型：

```bat
.venv\Scripts\python.exe -m src.cli.meta_train --config configs\meta_rl\maml_trans_beta_ppo.yaml
.venv\Scripts\python.exe -m src.cli.meta_train --config configs\meta_rl\reptile_trans_beta_ppo.yaml
.venv\Scripts\python.exe -m src.cli.meta_train --config configs\meta_rl\context_meta_trans_beta_ppo.yaml
```

最后在外推场景上测试：

```bat
.venv\Scripts\python.exe -m src.cli.meta_test --config configs\meta_rl\maml_trans_beta_ppo.yaml
```

第五章建议重点比较：

```text
OOD 场景 reward
适应前后 reward 提升
adaptation_steps
performance_decay_rate
ood_robustness_score
mean_speed
queue_length
density
```

## 10. 场景和车流设置

场景注册在：

```text
src/scenarios/scenario_registry.py
```

车流生成逻辑在：

```text
src/scenarios/traffic_flow.py
src/scenarios/route_generator.py
```

项目按 10 分钟分段生成 1 小时车流，每段主线和匝道流量使用正态分布采样。随机性由每个 scenario 的 `seed` 控制，保证同一个场景重复运行时车流可复现。

车辆参数在：

```text
src/scenarios/vehicle_params.py
```

移动瓶颈控制在：

```text
src/control/moving_bottleneck.py
src/control/longitudinal_gap.py
```

纵向搜索间隙使用高速公路时距模型：

```text
gap = vehicle_length + standstill_gap + speed * time_headway
```

默认时距范围是 1.0-1.8 s，并限制在 12-80 m，避免高速状态下出现不真实的小间隙。

当前代码已把纵向搜索间隙改成基于 CAV 车辆参数和 IDM/时距思想的安全间隙模型。它使用 CAV 车长、最小停车间隙、舒适加减速和当前速度计算可行间隙范围，动作 `a3` 只是在这个范围内选择。默认时距范围为 1.1-2.0 s，绝对间隙限制为 12-95 m。CAV 离开控制区或不再被选中时，会执行 `setSpeed(-1)` 和恢复默认换道模式，让车辆根据 SUMO 跟驰模型、道路限速和下游拥堵状态自然恢复，不写死固定恢复速度。

拥堵预判逻辑在：

```text
src/envs/sumo/congestion_prediction.py
```

每个仿真秒都会根据密度、平均速度、排队长度、流量衰减和排队增长计算拥堵分数。只有预测为拥堵时，当前控制周期才执行移动瓶颈可变限速；如果移动瓶颈链式构建失败，则自动改为控制当前控制区域内所有 `CAV.*` 车辆。

移动瓶颈主动构建采用分级策略：

```text
1. 每个 120 秒控制周期内，动作保持不变。
2. 每 1 个仿真秒重新读取主线 CAV 位置，避免车辆驶出控制区后仍使用旧选择。
3. CAV 位置使用主线绝对里程坐标，不直接使用 SUMO edge 内局部 lanePosition。
4. 控制区起点从瓶颈上游有效区间的上游候选段选择。
5. 控制区终点从瓶颈上游有效区间的下游候选段选择。
6. 控制区终点不会直接伸入瓶颈，默认保留 300 m 恢复区。
7. 优先按车道构建 staggered moving bottleneck 主链。
8. 主链附近补选支撑 CAV，提高限速控制覆盖率。
9. 主链构建不足时，降级控制当前控制区内 CAV。
10. 控制区内没有 CAV 时，选择控制区上下游 500 m 内最近 CAV 作为兜底。
```

日志中会记录：

```text
construction_mode
chain_coverage
active_control_seconds
control_coverage_ratio
selected_cav_count
fallback_used
```

其中 `control_coverage_ratio` 越接近 1，表示该控制周期内越多仿真秒成功找到了可控 CAV 并施加限速。

构型感知强化学习配置在：

```text
configs/rl/topology_aware_trans_beta_ppo.yaml
```

它相对 `trans_beta_ppo` 增加了两类信息：

```text
state: chain_coverage, control_coverage_ratio, fallback_used, active_control_seconds,
       target_vehicle_count, speed_limit_delta_kmh,
       controlled_position_mean_m, controlled_position_std_m,
       actual_gap_mean_m, gap_error_m,
       congestion_score_delta, queue_delta_m
reward: topology_reward
```

这让 DRL 不只根据交通流状态和限速动作学习，也能根据“移动瓶颈是否真的构建成功、控制周期内是否持续控制到 CAV、是否频繁兜底”来调整策略。原 `trans_beta_ppo.yaml` 不打开这些项，方便做消融对比。

其中 CAV 控制位置和间隙反馈的含义是：

```text
controlled_position_mean_m: 上一周期被控 CAV 的平均主线位置
controlled_position_std_m: 被控 CAV 位置离散度，反映控制构型是否过散
actual_gap_mean_m: 被控 CAV 实际平均间隙
gap_error_m: 实际平均间隙与动作目标间隙的偏差
congestion_score_delta: 周期末拥堵分数 - 周期初拥堵分数
queue_delta_m: 周期末排队长度 - 周期初排队长度
```

这些量会进入下一个控制周期的状态，指导智能体判断“上个周期把 CAV 控制在这个位置、这个间隙后，拥堵有没有缓解”，从而调整下一周期的控制区、限速和间隙。

## 11. 结果整理建议

每次实验结束后，把不同模型的 `metrics.csv` 和 `actions.csv` 汇总到表格里。

第四章建议画：

- episode-reward 收敛曲线；
- 平均速度对比；
- 排队长度对比；
- 密度变化对比；
- 控制速度和纵向间隙变化曲线。

第五章建议画：

- 元训练 mean_reward 曲线；
- 不同外推场景的适应前后指标对比；
- 非元学习模型和元学习模型在 OOD 场景下的性能下降率；
- 适应步数相同时的 reward 提升。

## 12. 常见问题

`ModuleNotFoundError: traci`

说明没有安装 SUMO Python 包，重新执行：

```bat
.venv\Scripts\python.exe -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip install -r requirements-windows-gpu.txt
```

`SUMO_HOME is not set`

说明没有设置 SUMO 路径。设置 `SUMO_HOME` 后重新打开终端或 PyCharm。

`SUMO net file not found`

说明缺少：

```text
data/sumo/base_network/test1.net.xml
```

`SUMO additional file not found`

说明缺少：

```text
data/sumo/base_network/E2_info.xml
```

训练很慢

先减少配置里的 `episodes`，确认流程没问题后再跑完整实验。

## 13. 推荐执行顺序

第一次接手项目时按下面顺序执行：

```text
run/check_python_environment.py
run/build_sumo_network.py
run/check_sumo_assets.py
run/train_trans_beta_ppo.py
run/run_chapter4_comparison.py
run/train_meta_trans_beta_ppo.py
run/evaluate_meta_trans_beta_ppo.py
```

## 14. PyCharm 中训练和测试到底调用什么

以 `run/train_trans_beta_ppo.py` 为例，PyCharm 右键运行后会调用：

```text
run/train_trans_beta_ppo.py
-> src.cli.train.main()
-> load_config(configs/rl/trans_beta_ppo.yaml)
-> create_agent("trans_beta_ppo")
-> SumoMovingBottleneckEnv(...)
-> env.reset(route_seed_offset=episode)
-> agent.select_action(state)
-> env.step(action)
-> agent.store_transition(...)
-> agent.update()
-> agent.save(checkpoint)
```

其中 `env.reset()` 会生成或刷新当前 episode 的 route 文件，并启动 SUMO；`env.step()` 会推进一个控制周期，也就是 120 个仿真秒。测试入口 `run/evaluate_trans_beta_ppo.py` 调用 `src.cli.test.main()`，它会加载 checkpoint，只执行动作推理和 SUMO 交互，不做参数更新。

训练时 route 随机性由 `route_randomization_enabled` 控制。打开后，第 `episode` 回合使用 `scenario.seed + episode` 生成 route，使同一场景下的车流需求和 CAV 发车时刻有轻微变化，提高模型鲁棒性。测试时保持固定 seed，保证不同模型在同一测试车流上比较。
