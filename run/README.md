# PyCharm Run Entries

These files are intended to be run directly from PyCharm with right click -> Run.

Recommended first run order:

1. `check_sumo_assets.py`
2. `check_python_environment.py`
3. `build_sumo_network.py`
4. `generate_sumo_routes.py`
5. `train_trans_beta_ppo.py`
6. `evaluate_trans_beta_ppo.py`
7. `run_chapter4_comparison.py`
8. `train_meta_trans_beta_ppo.py`
9. `evaluate_meta_trans_beta_ppo.py`
10. `run_chapter5_baselines.py`

Each entry has editable constants at the top:

- `CONFIG_PATH`
- `SUITE_PATH`
- `CHECKPOINT`
- `SMOKE`
- `EPISODES`

Keep `SMOKE = True` for first checks. Change it to `False` only after Python dependencies, SUMO, `SUMO_HOME`, and base network files are ready.
