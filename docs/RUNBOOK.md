# CHD_CoPer_Traffic 运行手册

这份文档面向第一次接手项目的同学，按“先确认 Windows/PyCharm/GPU 环境、生成 SUMO 路网和车流、再跑训练和对比分析”的顺序执行。

## 1. 项目目标

本项目用于复现论文中基于智能网联车移动瓶颈控制的高速公路车流控制实验，核心内容包括：

- 第四章：连续动作强化学习对比实验，重点模型是 `Trans-Beta-PPO`。
- 第五章：在 `Trans-Beta-PPO` 基础上加入元学习，形成 `MAML-Trans-Beta-PPO`。
- 仿真环境：SUMO 路网、车辆类型、正态分布车流、CAV 主动构建移动瓶颈。

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
run/train_meta_trans_beta_ppo.py
run/evaluate_meta_trans_beta_ppo.py
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

单独训练某个模型：

```bat
.venv\Scripts\python.exe -m src.cli.train --config configs\rl\beta_ppo.yaml
.venv\Scripts\python.exe -m src.cli.train --config configs\rl\continuous_ppo.yaml
.venv\Scripts\python.exe -m src.cli.train --config configs\rl\td3.yaml
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

## 9. 第五章对比实验

推荐对比模型：

1. `classic_vsl`: 规则控制基线。
2. `traditional_drl_vsl`: 传统 MLP DRL-VSL。
3. `vanilla_ppo`: 标准 PPO。
4. `dr_ppo`: 鲁棒 PPO，对车流和 CAV 渗透率扰动。
5. `trans_beta_ppo`: 不加元学习的第四章最强模型。
6. `maml_trans_beta_ppo`: 第五章主模型。

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
