# CHD_CoPer_Traffic 运行手册

这份文档面向第一次接手项目的同学，按“先确认环境、再跑 smoke、再跑正式训练、最后做对比分析”的顺序执行。

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

创建虚拟环境并安装依赖：

```bash
python3.9 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

如果在 PyCharm 里运行：

1. Open Project 选择项目根目录 `CHD_CoPer_Traffic`。
2. Interpreter 选择 `.venv/bin/python`。
3. Working directory 设置为项目根目录。
4. 环境变量里设置 `SUMO_HOME`。
5. 优先右键运行 `run/` 目录下的 Python 文件，不需要运行 `.sh`。
6. 第一次运行保持文件顶部的 `SMOKE = True`，正式 SUMO 实验再改成 `False`。

工程模块说明见：

```text
docs/PROJECT_STRUCTURE.md
```

macOS/Linux 示例：

```bash
export SUMO_HOME=/path/to/sumo
```

Windows 示例：

```text
SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
```

## 3. SUMO 文件放置

正式 SUMO 实验必须放入原始路网和检测器文件：

```text
data/sumo/base_network/test1.net.xml
data/sumo/base_network/E2_info.xml
```

程序运行时会自动生成：

```text
data/sumo/generated_routes/<scenario>.rou.xml
data/sumo/generated_routes/<scenario>.sumocfg
```

`generated_routes` 是运行产物，不需要提交到 GitHub。

## 4. PyCharm 快速运行入口

推荐直接右键运行这些文件：

```text
run/check_sumo_assets.py
run/check_python_environment.py
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
SMOKE = True
EPISODES = None
CHECKPOINT = None
```

新机器先保持 `SMOKE = True`。路网、SUMO、依赖都确认以后，再把 `SMOKE` 改成 `False` 跑正式实验。

## 5. 先跑 Smoke 检查

Smoke 模式不启动 SUMO，只检查 Python、模型、动作、奖励、日志路径是否能跑通。新机器第一次运行必须先跑 smoke。

训练主模型：

```bash
.venv/bin/python -m src.cli.train --config configs/rl/trans_beta_ppo.yaml --smoke
```

测试主模型：

```bash
.venv/bin/python -m src.cli.test --config configs/rl/trans_beta_ppo.yaml --smoke
```

训练元强化学习模型：

```bash
.venv/bin/python -m src.cli.meta_train --config configs/meta_rl/maml_trans_beta_ppo.yaml --smoke
```

测试元强化学习模型：

```bash
.venv/bin/python -m src.cli.meta_test --config configs/meta_rl/maml_trans_beta_ppo.yaml --smoke
```

如果 smoke 都不能跑，先不要开 SUMO，优先检查 Python 版本、依赖安装和 working directory。

## 6. 第四章强化学习训练

主模型是 `Trans-Beta-PPO`：

```bash
.venv/bin/python -m src.cli.train --config configs/rl/trans_beta_ppo.yaml
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

```bash
.venv/bin/python -m src.cli.test \
  --config configs/rl/trans_beta_ppo.yaml \
  --checkpoint experiments/rl/<run>/checkpoints/trans_beta_ppo.pth
```

## 7. 第四章对比实验

推荐对比顺序：

1. `continuous_ppo`: 高斯连续 PPO。
2. `beta_ppo`: Beta 连续 PPO，不加 Transformer。
3. `trans_beta_ppo`: 本文第四章主模型。
4. `td3`: 连续控制 off-policy 对比模型。

单独训练某个模型：

```bash
.venv/bin/python -m src.cli.train --config configs/rl/beta_ppo.yaml
.venv/bin/python -m src.cli.train --config configs/rl/continuous_ppo.yaml
.venv/bin/python -m src.cli.train --config configs/rl/td3.yaml
```

批量跑第四章对比：

```bash
.venv/bin/python -m src.cli.run_suite --suite configs/rl/comparison_suite.yaml
```

先快速检查批量流程：

```bash
.venv/bin/python -m src.cli.run_suite --suite configs/rl/comparison_suite.yaml --smoke
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

## 8. 第五章元强化学习训练

元强化学习主配置：

```text
configs/meta_rl/maml_trans_beta_ppo.yaml
```

训练：

```bash
.venv/bin/python -m src.cli.meta_train --config configs/meta_rl/maml_trans_beta_ppo.yaml
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

```bash
.venv/bin/python -m src.cli.meta_test \
  --config configs/meta_rl/maml_trans_beta_ppo.yaml \
  --checkpoint experiments/meta_rl/<run>/checkpoints/maml_trans_beta_ppo.pth
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

```bash
.venv/bin/python -m src.cli.run_suite --suite configs/meta_rl/comparison_suite.yaml
```

再训练元强化学习模型：

```bash
.venv/bin/python -m src.cli.meta_train --config configs/meta_rl/maml_trans_beta_ppo.yaml
```

最后在外推场景上测试：

```bash
.venv/bin/python -m src.cli.meta_test --config configs/meta_rl/maml_trans_beta_ppo.yaml
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

```bash
.venv/bin/python -m pip install -r requirements.txt
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

先用 `--smoke` 验证代码，再减少配置里的 `episodes`，确认流程没问题后再跑完整实验。

## 13. 推荐执行顺序

第一次接手项目时按下面顺序执行：

```bash
.venv/bin/python -m src.cli.train --config configs/rl/trans_beta_ppo.yaml --smoke
.venv/bin/python -m src.cli.meta_train --config configs/meta_rl/maml_trans_beta_ppo.yaml --smoke
.venv/bin/python -m src.cli.run_suite --suite configs/rl/comparison_suite.yaml --smoke
.venv/bin/python -m src.cli.train --config configs/rl/trans_beta_ppo.yaml
.venv/bin/python -m src.cli.run_suite --suite configs/rl/comparison_suite.yaml
.venv/bin/python -m src.cli.meta_train --config configs/meta_rl/maml_trans_beta_ppo.yaml
.venv/bin/python -m src.cli.meta_test --config configs/meta_rl/maml_trans_beta_ppo.yaml
```
