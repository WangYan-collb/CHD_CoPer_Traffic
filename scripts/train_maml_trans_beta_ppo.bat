@echo off
setlocal
cd /d "%~dp0\.."
.venv\Scripts\python.exe -m src.cli.meta_train --config configs\meta_rl\maml_trans_beta_ppo.yaml %*
