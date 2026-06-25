# Windows PyCharm GPU Setup

This project is intended to run from PyCharm on Windows with SUMO and an NVIDIA GPU.

## 1. Install Runtime

Recommended versions:

```text
Windows: Windows 10/11 64-bit
Python: 3.9.17 x64
SUMO: 1.25.0
PyTorch: 2.1.0 CUDA 12.1 wheel
```

Install SUMO and set the Windows environment variable:

```text
SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
```

Restart PyCharm after setting `SUMO_HOME`.

## 2. Create PyCharm Interpreter

Open the project folder in PyCharm:

```text
CHD_CoPer_Traffic
```

Create a virtual environment under the project root:

```text
.venv
```

Install GPU dependencies in PyCharm Terminal:

```bat
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip install -r requirements-windows-gpu.txt
```

Run this file from PyCharm to confirm CUDA:

```text
run/check_python_environment.py
```

It should print `torch cuda available: True`.

## 3. Generate SUMO Assets

Right-click and run:

```text
run/build_sumo_network.py
```

This generates:

```text
data/sumo/base_network/*.nod.xml
data/sumo/base_network/*.edg.xml
data/sumo/base_network/*.con.xml
data/sumo/base_network/*.add.xml
data/sumo/base_network/*.net.xml
data/sumo/routes/*.rou.xml
data/sumo/configs/*.sumocfg
```

The base scenario also writes legacy-compatible files:

```text
data/sumo/base_network/test1.net.xml
data/sumo/base_network/E2_info.xml
```

## 4. Run Training

Right-click in PyCharm:

```text
run/train_trans_beta_ppo.py
```

For Chapter 4 comparison:

```text
run/run_chapter4_comparison.py
```

For Chapter 5 meta-RL:

```text
run/train_meta_trans_beta_ppo.py
run/evaluate_meta_trans_beta_ppo.py
run/run_chapter5_baselines.py
```

All main configs set `algorithm.device: cuda`.
