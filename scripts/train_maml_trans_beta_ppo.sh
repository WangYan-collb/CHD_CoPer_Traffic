#!/usr/bin/env bash
set -euo pipefail
.venv/bin/python -m src.cli.meta_train --config configs/meta_rl/maml_trans_beta_ppo.yaml "$@"
