from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
SUMO_DIR = DATA_DIR / "sumo"
BASE_NETWORK_DIR = SUMO_DIR / "base_network"
GENERATED_ROUTES_DIR = SUMO_DIR / "generated_routes"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


@dataclass(frozen=True)
class SumoAssetPaths:
    net_file: Path = BASE_NETWORK_DIR / "test1.net.xml"
    additional_file: Path = BASE_NETWORK_DIR / "E2_info.xml"
    generated_routes_dir: Path = GENERATED_ROUTES_DIR

    def missing_required_files(self) -> list[Path]:
        return [path for path in (self.net_file, self.additional_file) if not path.exists()]


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate
