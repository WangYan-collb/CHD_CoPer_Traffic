from __future__ import annotations

import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from _bootstrap import activate_project_root

activate_project_root()

from src.project.paths import GENERATED_ROUTES_DIR
from src.scenarios.route_generator import generate_route_file
from src.scenarios.scenario_registry import get_scenario, list_scenarios


SCENARIOS = list_scenarios()


def main() -> None:
    for scenario_name in SCENARIOS:
        scenario = get_scenario(scenario_name)
        path = generate_route_file(scenario, GENERATED_ROUTES_DIR / f"{scenario.name}.rou.xml")
        print(f"generated {scenario.name}: {path}")


if __name__ == "__main__":
    main()
