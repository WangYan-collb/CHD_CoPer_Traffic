@echo off
setlocal
cd /d "%~dp0\.."
.venv\Scripts\python.exe -m src.cli.test --config configs\rl\trans_beta_ppo.yaml %*
