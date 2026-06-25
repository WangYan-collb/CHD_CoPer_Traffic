@echo off
setlocal
cd /d "%~dp0\.."
.venv\Scripts\python.exe run\build_sumo_network.py %*
