#!/usr/bin/env bash
set -euo pipefail
.venv/bin/python -m src.cli.test --config configs/rl/trans_beta_ppo.yaml "$@"
