from __future__ import annotations

import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from _bootstrap import activate_project_root

activate_project_root()

from src.envs.sumo.traci_client import SumoUnavailableError
from src.envs.sumo.config_builder import SumoConfigPaths, build_sumocfg
from src.project.paths import BASE_NETWORK_DIR, SUMO_CONFIGS_DIR, SUMO_ROUTES_DIR
from src.scenarios.route_generator import generate_route_file
from src.scenarios.scenario_registry import get_scenario, list_scenarios
from src.scenarios.sumo_network import build_sumo_network, write_plain_network_files


BUILD_ALL_SCENARIOS = True
SCENARIO_NAME = "base"


def main() -> None:
    scenario_names = list_scenarios() if BUILD_ALL_SCENARIOS else [SCENARIO_NAME]
    for scenario_name in scenario_names:
        scenario = get_scenario(scenario_name)
        try:
            result = build_sumo_network(scenario_name)
        except SumoUnavailableError as exc:
            node_file, edge_file, connection_file, detector_file = write_plain_network_files(scenario)
            route_file = generate_route_file(scenario, SUMO_ROUTES_DIR / f"{scenario.name}.rou.xml")
            sumocfg_file = SUMO_CONFIGS_DIR / f"{scenario.name}.sumocfg"
            build_sumocfg(
                SumoConfigPaths(
                    net_file=BASE_NETWORK_DIR / f"{scenario.name}.net.xml",
                    route_file=route_file,
                    additional_file=detector_file,
                    output_file=sumocfg_file,
                ),
                validate_inputs=False,
            )
            print(f"[plain xml only] {scenario_name}")
            print(f"  reason: {exc}")
            print(f"  nodes: {node_file}")
            print(f"  edges: {edge_file}")
            print(f"  connections: {connection_file}")
            print(f"  detectors: {detector_file}")
            print(f"  route: {route_file}")
            print(f"  sumocfg: {sumocfg_file}")
            print("  after SUMO is installed, rerun this file to create net.xml and sumocfg files")
            continue
        print(f"[built] {scenario_name}")
        print(f"  net: {result.net_file}")
        print(f"  additional: {result.detector_file}")
        print(f"  route: {result.route_file}")
        print(f"  sumocfg: {result.sumocfg_file}")


if __name__ == "__main__":
    main()
