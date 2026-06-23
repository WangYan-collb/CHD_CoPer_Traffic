from __future__ import annotations

from pathlib import Path


def checkpoint_path(run_dir: str | Path, name: str = "model.pth") -> Path:
    path = Path(run_dir) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path / name
