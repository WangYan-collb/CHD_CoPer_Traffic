from __future__ import annotations

import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from _bootstrap import activate_project_root

activate_project_root()

from src.project.paths import SumoAssetPaths


def main() -> None:
    assets = SumoAssetPaths()
    missing = assets.missing_required_files()
    if missing:
        print("Missing SUMO assets:")
        for path in missing:
            print(f"  - {path}")
        print("Copy test1.net.xml and E2_info.xml into data/sumo/base_network before real SUMO runs.")
        return
    print("SUMO assets are ready:")
    print(f"  net file: {assets.net_file}")
    print(f"  additional file: {assets.additional_file}")


if __name__ == "__main__":
    main()
