from __future__ import annotations

from pathlib import Path


def ensure_plot_dir(path: str | Path) -> Path:
    plot_dir = Path(path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir
