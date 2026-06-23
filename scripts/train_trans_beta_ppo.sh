#!/usr/bin/env bash
set -euo pipefail
.venv/bin/python -m src.cli.train --config configs/rl/trans_beta_ppo.yaml "$@"
