# CHD_CoPer_Traffic 中文详细说明

本文档面向后续接手项目的同学，说明项目每个模块做什么、模型如何设计、实验如何组织、SUMO 路网和车流如何生成、元强化学习如何训练测试、超参数如何修改，以及如何增加新的模型。

## 1. 项目整体目标

本项目用于复现论文中“智能网联车主动构建移动瓶颈的高速公路可变限速控制”实验。核心思想是：在混合交通流中，利用 CAV 作为可控车辆，通过强化学习输出连续控制动作，决定限速值、控制区域和 CAV 纵向搜索间隙，使 CAV 形成移动瓶颈，提前抑制拥堵扩散。

项目包含两条主线：

```text
第四章：单场景/多基线深度强化学习 VSL 控制
第五章：多场景元强化学习，提高新场景少步适应能力
```

默认仿真设置：

```text
一个 episode = 3600 仿真秒
一个 RL step = 一个控制周期 = 120 仿真秒
每个 step 内每 1 秒采集一次 SUMO 状态
step 返回状态 = 该控制周期内逐秒状态的聚合结果
```

动作空间为 4 维连续动作：

```text
a0: CAV 限速值，映射为 km/h
a1: 控制区域起点，映射为主线绝对里程
a2: 控制区域终点，映射为主线绝对里程
a3: CAV 纵向搜索间隙，映射为真实高速时距间隙
```

## 2. 推荐阅读顺序

第一次接手项目建议按下面顺序阅读：

```text
README.md                         项目入口和快速运行
docs/WINDOWS_PYCHARM_GPU.md       Windows + PyCharm + GPU 环境配置
docs/PROJECT_STRUCTURE.md         文件夹和模块结构
docs/RUNBOOK.md                   训练、测试、对比实验运行手册
docs/DETAILED_CHINESE_GUIDE.md    当前这份详细设计说明
```

如果只是想先跑起来，先看 `README.md` 和 `docs/RUNBOOK.md`。如果要改模型、改奖励、改场景，重点看本文档。

## 3. 文件夹结构说明

### 3.1 configs

`configs/` 放所有实验配置。项目不建议在代码里硬改超参数，优先改 YAML。

```text
configs/rl/
```

第四章强化学习模型配置，包括：

```text
trans_beta_ppo.yaml                 主模型 Transformer + Beta-PPO
topology_aware_trans_beta_ppo.yaml  构型感知 Trans-Beta-PPO
beta_ppo.yaml                       不加 Transformer 的 Beta-PPO
continuous_ppo.yaml                 高斯连续 PPO
td3.yaml                            TD3 连续控制基线
sac.yaml                            SAC 连续控制基线
traditional_drl_vsl.yaml            传统 MLP DRL-VSL
dr_ppo.yaml                         鲁棒 PPO
vanilla_ppo.yaml                    标准 PPO
comparison_suite.yaml               第四章批量对比配置
```

```text
configs/meta_rl/
```

第五章元强化学习配置，包括：

```text
maml_trans_beta_ppo.yaml            一阶 MAML 风格元强化学习
reptile_trans_beta_ppo.yaml         Reptile 元强化学习
context_meta_trans_beta_ppo.yaml    上下文条件化元强化学习
comparison_suite.yaml               第五章对比配置
```

```text
configs/baselines/
```

规则式和传统控制基线：

```text
feedback_vsl.yaml                   基于密度/排队反馈的 VSL
mpc_vsl.yaml                        简化 MPC 式 VSL
classic_vsl.yaml                    旧版规则 VSL，保留但不作为主推荐
```

### 3.2 run

`run/` 是给 PyCharm 右键运行准备的入口。师弟不需要记命令，直接打开对应 Python 文件运行。

常用入口：

```text
run/check_python_environment.py                 检查 Python、torch、CUDA、SUMO 包
run/build_sumo_network.py                       生成 SUMO 路网、检测器、route、sumocfg
run/check_sumo_assets.py                        检查 SUMO 文件是否存在
run/train_trans_beta_ppo.py                     训练第四章主模型
run/evaluate_trans_beta_ppo.py                  测试第四章主模型
run/train_topology_aware_trans_beta_ppo.py      训练构型感知模型
run/evaluate_topology_aware_trans_beta_ppo.py   测试构型感知模型
run/train_sac.py                                训练 SAC 基线
run/evaluate_sac.py                             测试 SAC 基线
run/run_chapter4_comparison.py                  批量跑第四章对比模型
run/train_meta_trans_beta_ppo.py                训练 MAML-Trans-Beta-PPO
run/evaluate_meta_trans_beta_ppo.py             测试 MAML-Trans-Beta-PPO
run/train_reptile_trans_beta_ppo.py             训练 Reptile-Trans-Beta-PPO
run/evaluate_reptile_trans_beta_ppo.py          测试 Reptile-Trans-Beta-PPO
run/train_context_meta_trans_beta_ppo.py        训练 Context-Meta-Trans-Beta-PPO
run/evaluate_context_meta_trans_beta_ppo.py     测试 Context-Meta-Trans-Beta-PPO
run/run_chapter5_baselines.py                   批量跑第五章非元学习基线
```

每个 `run/*.py` 顶部都有这些可改变量：

```python
CONFIG_PATH = "configs/rl/trans_beta_ppo.yaml"
SMOKE = False
EPISODES = None
CHECKPOINT = None
```

说明：

```text
CONFIG_PATH: 指定使用哪个 YAML 配置
SMOKE: True 表示不启动 SUMO，只做快速逻辑调试；正式实验用 False
EPISODES: 临时覆盖训练回合数；None 表示使用 YAML 中的 episodes
CHECKPOINT: 测试时指定模型权重路径
```

### 3.3 src/algorithms

`src/algorithms/` 放强化学习算法和规则控制器。

```text
src/algorithms/trans_beta_ppo/
```

第四章主模型。核心文件：

```text
agent.py          Trans-Beta-PPO 训练、动作采样、PPO update、模型保存加载
buffer.py         PPO rollout buffer
context_agent.py  构型感知/上下文条件化版本使用的 agent
```

```text
src/algorithms/beta_ppo/
```

不加 Transformer 的 Beta-PPO。用于证明 Transformer 时序编码的作用。

```text
src/algorithms/continuous_ppo/
```

高斯连续 PPO。用于和 Beta 分布动作策略对比，说明 Beta 分布更适合 `[0,1]` 有界连续动作。

```text
src/algorithms/td3/
src/algorithms/sac/
```

off-policy 连续控制基线。SAC 是最大熵连续控制算法，用于比较 PPO 系列和更样本高效的连续控制方法。

```text
src/algorithms/meta_trans_beta_ppo/
```

第五章元强化学习封装：

```text
maml.py       一阶 MAML 风格元更新
reptile.py    Reptile 元更新
```

这两个文件不重新实现底层 PPO，而是包装 `TransBetaPPOAgent` 或构型感知 agent，在多个场景上进行少步内循环适应，再更新元初始化。

```text
src/algorithms/rule_based/
```

规则式控制基线：

```text
feedback_vsl.py  密度/排队反馈 VSL
mpc_vsl.py       简化 MPC 式 VSL
classic_vsl.py   旧版规则 VSL
```

```text
src/algorithms/registry.py
```

算法注册表。新增算法时必须在这里注册 `algorithm.name`，否则训练入口找不到。

### 3.4 src/models

`src/models/` 放神经网络结构。

```text
transformer_encoder.py
```

交通状态时序编码器。输入形状为：

```text
(sequence_length, state_dim)
```

默认 `sequence_length=30`，表示策略看到最近 30 个控制周期的状态序列。

```text
beta_actor_critic.py
```

主模型网络：Transformer 编码后分别进入 actor 和 critic。

actor 输出 Beta 分布参数：

```text
alpha > 1
beta > 1
```

然后从 Beta 分布采样 `[0,1]` 动作。选择 Beta 分布的原因是 VSL 动作天然有界，例如限速、控制区起终点、搜索间隙都可以先归一化到 `[0,1]`，再映射到物理量。

```text
context_beta_actor_critic.py
```

上下文条件化模型。它除了 Transformer 时序特征，还从状态序列的均值和波动中编码 traffic context，用于近似识别当前场景的车流强度、CAV 渗透率、瓶颈形态和拥堵传播状态。

### 3.5 src/envs/sumo

`src/envs/sumo/` 是强化学习和 SUMO 交互的核心。

```text
env.py
```

最重要的环境文件，负责：

```text
启动/关闭 SUMO TraCI
每个 episode reset
每个 RL step 执行 120 个仿真秒
每秒采集交通流状态
每秒判断拥堵预警
每秒刷新 CAV 候选车辆
对 CAV 施加限速
计算 reward
构造下一个 state
返回 info 日志
```

```text
state_calculator.py
```

计算交通流状态向量。基础状态包括：

```text
density
speed_mps
queue_m
speed_limit_kmh
longitudinal_gap_m
selected_cav_count
throughput
congestion_score
control_start_m
control_end_m
```

构型感知模型还会加入：

```text
chain_coverage
control_coverage_ratio
fallback_used
active_control_seconds
target_vehicle_count
speed_limit_delta_kmh
controlled_position_mean_m
controlled_position_std_m
actual_gap_mean_m
gap_error_m
congestion_score_delta
queue_delta_m
```

```text
congestion_prediction.py
```

拥堵超前预判。使用密度、速度、排队长度、流量衰减、排队增长构造拥堵分数。默认 `control_activation_score=0.45`，表示还没完全达到拥堵阈值但已有拥堵趋势时，也可以提前控制。

```text
config_builder.py
traci_client.py
metrics.py
observations.py
```

分别负责 SUMO 配置生成、TraCI 加载、交通指标数据结构、Transformer 状态序列缓存。

### 3.6 src/control

`src/control/` 是移动瓶颈主动构建模块。

```text
moving_bottleneck.py
```

根据智能体动作和当前 CAV 分布构建控制命令。当前采用分级策略：

```text
1. 按动作确定限速、控制区起点、控制区终点、目标搜索间隙
2. 每条主线车道选择接近目标位置的 CAV，构建 staggered moving bottleneck 主链
3. 主链附近补选支撑 CAV，提高控制覆盖率
4. 主链不足时降级为区域内 CAV 控制
5. 区域内无 CAV 时选择上下游附近 CAV 兜底
```

输出 `MovingBottleneckCommand`：

```text
speed_limit_kmh
longitudinal_gap_m
start_position_m
end_position_m
selected_vehicles
target_vehicle_count
chain_coverage
construction_mode
fallback_used
```

```text
longitudinal_gap.py
```

把动作中的间隙维度映射为真实高速行车间隙：

```text
gap = vehicle_length + standstill_gap + speed * time_headway
```

默认时距范围：

```text
1.0 s - 1.8 s
```

并把绝对间隙限制在：

```text
12 m - 80 m
```

这样既保留论文中 15-25 m 的低速控制区域，也避免高速下不真实的小间隙。

```text
speed_limit_mapper.py
```

把动作 `a0` 从 `[0,1]` 映射为真实限速值。

```text
conflict_resolution.py
```

过滤同车道内间距过近的 CAV，避免选择多个过近车辆造成不真实控制。

### 3.7 src/scenarios

`src/scenarios/` 放场景、车流、车辆参数和路网生成逻辑。

```text
scenario_registry.py
```

注册论文实验场景：

```text
base                 基准场景
interpolation_1      插值训练场景
interpolation_2      插值训练场景
interpolation_3      插值训练场景
extrapolation_1      外推测试场景
extrapolation_2      外推测试场景
```

元训练默认使用：

```text
base + interpolation_1 + interpolation_2 + interpolation_3
```

元测试默认使用：

```text
extrapolation_1 + extrapolation_2
```

```text
road_network.py
sumo_network.py
```

生成论文高速合流区路网。默认几何：

```text
上游主线控制区 7500 m
匝道 500 m
瓶颈/合流区 150 m 或场景指定长度
下游恢复区 2000 m
```

```text
traffic_flow.py
route_generator.py
```

生成 route 文件。项目按 10 分钟分段生成 1 小时车流，每段主线和匝道流量按正态分布采样，并由场景 seed 控制随机性。

```text
vehicle_params.py
```

定义 HDV 和 CAV 车辆类型参数，例如加速度、减速度、车长、跟驰模型、CAV 渗透率等。

### 3.8 src/rewards

奖励函数拆成多个文件，方便修改。

```text
combined_reward.py
```

总奖励聚合。基础项包括：

```text
density reward
speed reward
queue penalty
safety reward
smoothness penalty
topology reward
```

默认 `Trans-Beta-PPO` 中 `topology=0`，不启用构型奖励。`Topology-Aware-Trans-Beta-PPO` 中打开构型奖励。

```text
topology_reward.py
```

构型奖励，用于把移动瓶颈构建质量和 DRL 训练目标关联起来：

```text
chain_coverage 高，加分
control_coverage_ratio 高，加分
selected_cav_count 足够，加分
fallback_used，扣分
没有控制到 CAV，扣分
限速跳变大，扣分
实际间隙偏离目标间隙，扣分
控制后拥堵分数上升，扣分
控制后排队增长，扣分
```

## 4. SUMO 路网、车流和仿真文件

正式运行前必须有 SUMO 的三类文件：

```text
net.xml      路网
rou.xml      车辆、路线、发车流
sumocfg      SUMO 仿真配置
```

本项目通过代码自动生成：

```text
run/build_sumo_network.py
```

生成目录：

```text
data/sumo/base_network/
data/sumo/routes/
data/sumo/configs/
```

基础场景兼容旧代码文件名：

```text
data/sumo/base_network/test1.net.xml
data/sumo/base_network/E2_info.xml
```

在 Windows 上必须设置：

```text
SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
```

如果 `SUMO_HOME` 没设置，项目可以生成 PlainXML 和 route 源文件，但不能调用 `netconvert` 生成真正可运行的 `.net.xml`。

## 5. 如何让项目跑起来

### 5.1 Windows + PyCharm 环境

推荐版本：

```text
Python 3.9.17
SUMO 1.25.0
PyTorch 2.1.0 + CUDA 12.1
numpy 1.26.3
pandas 1.5.3
matplotlib 3.7.1
```

安装顺序：

```bat
py -3.9 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip install -r requirements-windows-gpu.txt
```

PyCharm 设置：

```text
Open Project: CHD_CoPer_Traffic
Interpreter: .venv\Scripts\python.exe
Working directory: 项目根目录
Environment variables: SUMO_HOME=你的 SUMO 安装路径
```

### 5.2 第一次运行顺序

建议顺序：

```text
1. run/check_python_environment.py
2. run/build_sumo_network.py
3. run/check_sumo_assets.py
4. run/train_trans_beta_ppo.py
5. run/evaluate_trans_beta_ppo.py
6. run/run_chapter4_comparison.py
7. run/train_meta_trans_beta_ppo.py
8. run/evaluate_meta_trans_beta_ppo.py
```

如果训练太慢，可以在对应 `run/*.py` 里临时设置：

```python
SMOKE = True
EPISODES = 2
```

注意：`SMOKE=True` 只用于检查代码逻辑，不代表真实 SUMO 实验结果。

## 6. 第四章模型设计

### 6.1 Trans-Beta-PPO

主模型设计：

```text
状态序列 -> Transformer Encoder -> Actor-Critic
Actor -> Beta(alpha, beta) -> 4 维连续动作
Critic -> 状态价值 V(s)
```

为什么用 Transformer：

```text
VSL 控制不是只看当前秒的状态，而是要看拥堵形成趋势。
Transformer 用最近多个控制周期的状态序列提取时间相关性。
```

为什么用 Beta 分布：

```text
动作天然有界：
限速动作在 [0,1]
控制起点动作在 [0,1]
控制终点动作在 [0,1]
间隙动作在 [0,1]
Beta 分布天然定义在 [0,1]，比高斯分布后裁剪更合理。
```

### 6.2 Topology-Aware-Trans-Beta-PPO

这是对主模型的增强，不替代原模型，用于消融实验。

新增信息：

```text
移动瓶颈主链覆盖率
每周期实际控制 CAV 的秒数比例
是否 fallback
被控 CAV 平均位置
被控 CAV 位置离散度
实际平均间隙
目标间隙偏差
控制后拥堵分数变化
控制后排队长度变化
```

设计目的：

```text
让智能体知道：上个周期这样布置 CAV 后，拥堵有没有缓解。
下一周期智能体可以据此调整控制区、限速和间隙。
```

对应配置：

```text
configs/rl/topology_aware_trans_beta_ppo.yaml
```

关键开关：

```yaml
environment:
  state_dim: 22
  topology_state_enabled: true
  topology_reward_enabled: true
  topology_reward_weight: 0.10
```

原始主模型配置 `configs/rl/trans_beta_ppo.yaml` 不打开这些项，因此可以比较：

```text
Trans-Beta-PPO
vs
Topology-Aware-Trans-Beta-PPO
```

来验证移动瓶颈构型信息是否有贡献。

### 6.3 对比模型

第四章建议对比：

```text
continuous_ppo      验证 Beta 分布是否优于高斯动作分布
beta_ppo            验证 Transformer 是否有贡献
trans_beta_ppo      第四章主模型
topology_aware      验证移动瓶颈构型反馈是否有贡献
td3                 off-policy 连续控制基线
sac                 最大熵 off-policy 连续控制基线
```

实验分析建议：

```text
reward 收敛曲线
平均速度
平均密度
排队长度
限速变化曲线
CAV 控制数量
移动瓶颈间隙
chain_coverage
control_coverage_ratio
gap_error_m
congestion_score_delta
queue_delta_m
```

## 7. 第五章元强化学习设计

### 7.1 元强化学习要解决的问题

普通强化学习模型通常在一个固定场景训练。如果换成新场景，例如：

```text
CAV 渗透率变低
主线需求变高
匝道流量变化
车道数变化
瓶颈位置或长度变化
```

普通模型可能需要重新训练很多回合。元强化学习的目标是：在多个训练场景中学到一个容易适应的新初始化，使模型遇到外推场景时只需要少量 episode 微调。

第五章实验逻辑：

```text
meta-train: base + interpolation_1 + interpolation_2 + interpolation_3
meta-test: extrapolation_1 + extrapolation_2
```

评价重点不是“重新训练 500 回合”，而是：

```text
适应前表现
少步适应后的表现
适应提升幅度
外推性能衰减率
OOD 鲁棒性
```

### 7.2 MAML-Trans-Beta-PPO

配置：

```text
configs/meta_rl/maml_trans_beta_ppo.yaml
```

训练入口：

```text
run/train_meta_trans_beta_ppo.py
```

测试入口：

```text
run/evaluate_meta_trans_beta_ppo.py
```

当前实现是一阶 MAML 风格，不做昂贵的二阶梯度。训练流程：

```text
1. 保存当前 meta policy 参数
2. 对每个训练场景复制同一个初始化
3. 在该场景中跑 inner_steps 次 PPO 适应
4. 得到每个场景适应后的参数
5. 聚合多个任务的适应方向
6. 更新 meta policy
```

### 7.3 Reptile-Trans-Beta-PPO

配置：

```text
configs/meta_rl/reptile_trans_beta_ppo.yaml
```

Reptile 更简单、更稳定，适合 SUMO 这种仿真成本高的任务。思想是：

```text
让初始化参数朝多个任务少步适应后的平均参数移动
```

如果 MAML 训练不稳定，可以优先使用 Reptile 做第五章主实验或补充实验。

### 7.4 Context-Meta-Trans-Beta-PPO

配置：

```text
configs/meta_rl/context_meta_trans_beta_ppo.yaml
```

这个模型加入上下文条件化思想。它从最近状态序列中提取：

```text
状态均值
状态波动
拥堵变化趋势
控制响应特征
```

用于推断当前场景隐含任务信息。适合 VSL 的原因是不同场景的差异经常体现在车流需求、CAV 渗透率、瓶颈形态和拥堵传播速度上。

## 8. 如何修改超参数

所有超参数优先在 YAML 里改。

### 8.1 环境超参数

示例：

```yaml
environment:
  sequence_length: 30
  state_dim: 17
  action_dim: 4
  simulation_time_s: 3600
  control_cycle_s: 120
  aggregation_time_s: 120
  control_activation_score: 0.45
```

说明：

```text
sequence_length: Transformer 看多少个历史控制周期
state_dim: 状态向量维度，普通模型 17，构型感知模型 22
action_dim: 动作维度，当前为 4
simulation_time_s: 一个 episode 的仿真秒数
control_cycle_s: 一个 RL step 持续多少仿真秒
aggregation_time_s: 状态聚合窗口，通常和 control_cycle_s 保持一致
control_activation_score: 拥堵预警控制触发分数
```

### 8.2 PPO 超参数

示例：

```yaml
algorithm:
  lr: 0.0001
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.1
  value_coef: 0.5
  entropy_coef: 0.01
  update_epochs: 4
```

建议：

```text
训练震荡大：降低 lr 或 clip_epsilon
探索不足：提高 entropy_coef
价值函数误差大：调整 value_coef
收敛慢：增加 episodes 或 update_epochs
```

### 8.3 元学习超参数

示例：

```yaml
meta:
  algorithm: reptile
  meta_lr: 0.05
  inner_steps: 3
  meta_batch_size: 4
  episodes: 500
```

说明：

```text
algorithm: maml 或 reptile
meta_lr: 元更新步长
inner_steps: 新场景少步适应次数
meta_batch_size: 每个 meta episode 使用多少个场景
episodes: 元训练回合数
```

建议：

```text
SUMO 很慢时先把 episodes 调小验证流程
新场景适应不足时增加 inner_steps
元训练不稳定时降低 meta_lr
场景数量增加后增大 meta_batch_size
```

### 8.4 移动瓶颈构型超参数

配置在环境和控制模块里。常用 YAML 参数：

```yaml
environment:
  topology_state_enabled: true
  topology_reward_enabled: true
  topology_reward_weight: 0.10
```

如果想改间隙范围，看：

```text
src/control/longitudinal_gap.py
```

如果想改控制区选车策略，看：

```text
src/control/moving_bottleneck.py
```

## 9. 如何设计实验分析

### 9.1 第四章实验

目标：证明模型结构逐步改进有效。

推荐实验链条：

```text
continuous_ppo
-> beta_ppo
-> trans_beta_ppo
-> topology_aware_trans_beta_ppo
-> td3 / sac
```

可解释逻辑：

```text
continuous_ppo vs beta_ppo:
  验证 Beta 有界动作分布对连续 VSL 控制更合适

beta_ppo vs trans_beta_ppo:
  验证 Transformer 时序状态编码有效

trans_beta_ppo vs topology_aware_trans_beta_ppo:
  验证移动瓶颈构型反馈和构型奖励有效

trans_beta_ppo vs td3/sac:
  与其他连续控制算法比较
```

建议图表：

```text
训练 reward 曲线
平均速度柱状图
平均密度柱状图
排队长度柱状图
TET/TIT 安全指标
限速动作变化曲线
控制区起终点变化曲线
CAV 实际间隙与目标间隙偏差
control_coverage_ratio 曲线
```

### 9.2 第五章实验

目标：证明元学习提升新场景泛化和少步适应能力。

推荐实验链条：

```text
非元学习模型在 extrapolation 场景直接测试
MAML/Reptile/Context-Meta 在 extrapolation 场景适应前测试
MAML/Reptile/Context-Meta 少步适应后测试
比较适应提升和性能衰减
```

重点指标：

```text
before_reward
after_reward
reward_improvement
adaptation_steps
performance_decay_rate
ood_robustness_score
mean_speed
queue_length
density
```

结论写法：

```text
如果元学习模型适应前已经不差，说明 meta initialization 有泛化能力。
如果适应 1-3 步后明显提升，说明模型具备快速适应能力。
如果外推场景性能下降率更低，说明 OOD 鲁棒性更好。
```

## 10. 如何增加新场景

修改：

```text
src/scenarios/scenario_registry.py
```

新增一个 `ScenarioConfig`：

```python
"new_scenario": ScenarioConfig(
    name="new_scenario",
    lane_count=5,
    bottleneck_length_m=150,
    cav_ratio=0.50,
    main_flow=4500,
    ramp_flow=2500,
    description="new scenario description",
    is_meta_train=True,
    seed=2026062401,
)
```

说明：

```text
is_meta_train=True: 加入元训练场景
is_meta_train=False: 作为外推测试场景
```

然后重新运行：

```text
run/build_sumo_network.py
```

生成该场景对应的 `.net.xml`、`.rou.xml` 和 `.sumocfg`。

## 11. 如何增加新模型

新增模型通常需要 4 步。

### 11.1 新增算法目录

例如：

```text
src/algorithms/my_algorithm/
  __init__.py
  agent.py
```

`agent.py` 至少实现：

```python
select_action(state) -> action, info
store_transition(...)
update()
save(path)
load(path)
```

### 11.2 注册算法

修改：

```text
src/algorithms/registry.py
```

添加算法名称：

```python
"my_algorithm"
```

添加创建逻辑：

```python
if name == "my_algorithm":
    from src.algorithms.my_algorithm.agent import MyAlgorithmAgent
    return MyAlgorithmAgent(state_dim, action_dim, sequence_length, **config)
```

### 11.3 新增 YAML 配置

例如：

```text
configs/rl/my_algorithm.yaml
```

配置中必须有：

```yaml
algorithm:
  name: my_algorithm
```

### 11.4 新增 PyCharm 入口

复制一个已有文件：

```text
run/train_trans_beta_ppo.py
```

改成：

```python
CONFIG_PATH = "configs/rl/my_algorithm.yaml"
```

测试入口同理复制 `run/evaluate_trans_beta_ppo.py`。

## 12. 输出文件怎么看

每次训练都会生成：

```text
experiments/rl/<timestamp>_<run_name>_<scenario>/
experiments/meta_rl/<timestamp>_<run_name>/
```

主要文件：

```text
config.json      本次运行实际使用的配置
metrics.csv      每个 episode 的 reward、loss、交通指标
actions.csv      每个 step 的动作、限速、控制区、CAV 构型
summary.json     训练总结和 checkpoint 路径
checkpoints/     模型权重
```

建议分析：

```text
metrics.csv: 画 reward 收敛、平均速度、密度、排队长度
actions.csv: 画限速动作、控制区、纵向间隙、CAV 控制覆盖率
summary.json: 记录最终模型路径和运行摘要
```

## 13. 常见问题

### 13.1 `SUMO_HOME is not set`

说明 Windows 环境变量没有设置 SUMO 路径。设置后重启 PyCharm。

### 13.2 `traci` 或 `sumolib` 找不到

重新安装依赖：

```bat
.venv\Scripts\python.exe -m pip install -r requirements-windows-gpu.txt
```

并确认 `SUMO_HOME/tools` 能被 Python 找到。

### 13.3 训练很慢

SUMO + DRL 本来就慢。调试时：

```text
把 run 文件中的 SMOKE 改成 True
或把 EPISODES 改成 2
或在 YAML 中减少 episodes
```

正式实验再恢复。

### 13.4 checkpoint 路径怎么填

训练结束后终端会打印：

```text
Run directory: experiments/rl/<timestamp>_<run_name>
```

模型一般在：

```text
experiments/rl/<timestamp>_<run_name>/checkpoints/<algorithm_name>.pth
```

把这个路径填到对应 `run/evaluate_*.py` 的 `CHECKPOINT` 即可。

## 14. 推荐提交前检查

每次改完代码建议运行：

```bat
.venv\Scripts\python.exe -m compileall src run
```

如果在 Mac/Linux 当前开发机临时检查，也可以用：

```text
python3 -m compileall src run
```

再检查配置是否能解析。Windows 上可以直接运行：

```text
run/check_python_environment.py
```

正式训练前运行：

```text
run/check_sumo_assets.py
```

确认 SUMO 路网、route 和 sumocfg 文件都存在。
