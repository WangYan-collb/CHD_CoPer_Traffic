from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(self, root: str | Path, run_name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(root) / f"{timestamp}_{run_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.actions_path = self.run_dir / "actions.csv"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

    def write_config(self, config: dict[str, Any]) -> None:
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def append_csv(self, path: Path, row: dict[str, Any]) -> None:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def log_metric(self, row: dict[str, Any]) -> None:
        self.append_csv(self.metrics_path, row)

    def log_action(self, row: dict[str, Any]) -> None:
        self.append_csv(self.actions_path, row)

    def write_summary(self, summary: dict[str, Any]) -> None:
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
