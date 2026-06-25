from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from src.envs.sumo.config_builder import SumoConfigPaths, build_sumocfg
from src.envs.sumo.traci_client import SumoUnavailableError, netconvert_binary
from src.project.paths import BASE_NETWORK_DIR, PROJECT_ROOT, SUMO_CONFIGS_DIR, SUMO_ROUTES_DIR
from src.scenarios.road_network import RoadNetworkDesign
from src.scenarios.scenario_config import ScenarioConfig
from src.scenarios.scenario_registry import get_scenario, list_scenarios
from src.scenarios.route_generator import generate_route_file


@dataclass(frozen=True)
class SumoNetworkBuildResult:
    scenario_name: str
    node_file: Path
    edge_file: Path
    connection_file: Path
    detector_file: Path
    net_file: Path
    route_file: Path
    sumocfg_file: Path


def write_plain_network_files(
    scenario: ScenarioConfig,
    output_dir: Path = BASE_NETWORK_DIR,
) -> tuple[Path, Path, Path, Path]:
    road = RoadNetworkDesign(bottleneck_length_m=scenario.bottleneck_length_m)
    prefix = scenario.name
    output_dir.mkdir(parents=True, exist_ok=True)
    node_file = output_dir / f"{prefix}.nod.xml"
    edge_file = output_dir / f"{prefix}.edg.xml"
    connection_file = output_dir / f"{prefix}.con.xml"
    detector_file = output_dir / f"{prefix}.add.xml"

    _write_nodes(node_file, road)
    _write_edges(edge_file, road, scenario)
    _write_connections(connection_file, scenario)
    _write_detectors(detector_file, road, scenario)
    return node_file, edge_file, connection_file, detector_file


def build_sumo_network(
    scenario_name: str = "base",
    output_dir: Path = BASE_NETWORK_DIR,
    write_legacy_names: bool = True,
) -> SumoNetworkBuildResult:
    scenario = get_scenario(scenario_name)
    node_file, edge_file, connection_file, detector_file = write_plain_network_files(scenario, output_dir)
    net_file = output_dir / f"{scenario.name}.net.xml"
    route_file = generate_route_file(scenario, SUMO_ROUTES_DIR / f"{scenario.name}.rou.xml")
    sumocfg_file = SUMO_CONFIGS_DIR / f"{scenario.name}.sumocfg"
    build_sumocfg(
        SumoConfigPaths(
            net_file=net_file,
            route_file=route_file,
            additional_file=detector_file,
            output_file=sumocfg_file,
        ),
        validate_inputs=False,
    )
    command = [
        netconvert_binary(),
        "--node-files",
        str(node_file),
        "--edge-files",
        str(edge_file),
        "--connection-files",
        str(connection_file),
        "--output-file",
        str(net_file),
        "--no-turnarounds",
        "true",
        "--junctions.join",
        "true",
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)

    build_sumocfg(
        SumoConfigPaths(
            net_file=net_file,
            route_file=route_file,
            additional_file=detector_file,
            output_file=sumocfg_file,
        )
    )
    if write_legacy_names and scenario.name == "base":
        _copy_text(net_file, output_dir / "test1.net.xml")
        _copy_text(detector_file, output_dir / "E2_info.xml")
    return SumoNetworkBuildResult(
        scenario_name=scenario.name,
        node_file=node_file,
        edge_file=edge_file,
        connection_file=connection_file,
        detector_file=detector_file,
        net_file=net_file,
        route_file=route_file,
        sumocfg_file=sumocfg_file,
    )


def build_all_sumo_networks(output_dir: Path = BASE_NETWORK_DIR) -> list[SumoNetworkBuildResult]:
    return [build_sumo_network(name, output_dir=output_dir) for name in list_scenarios()]


def _write_nodes(path: Path, road: RoadNetworkDesign) -> None:
    ramp_join_x = road.upstream_length_m - road.ramp_length_m * 0.35
    root = ET.Element("nodes")
    nodes = [
        ("main_start", 0.0, 0.0),
        ("merge_start", road.upstream_length_m, 0.0),
        ("merge_end", road.upstream_length_m + road.bottleneck_length_m, 0.0),
        ("downstream_end", road.upstream_length_m + road.bottleneck_length_m + road.downstream_length_m, 0.0),
        ("ramp_start", ramp_join_x - road.ramp_length_m, -road.ramp_length_m),
        ("ramp_mid", ramp_join_x, -road.lane_width_m),
    ]
    for node_id, x, y in nodes:
        ET.SubElement(root, "node", {"id": node_id, "x": f"{x:.2f}", "y": f"{y:.2f}", "type": "priority"})
    _write_xml(path, root)


def _write_edges(path: Path, road: RoadNetworkDesign, scenario: ScenarioConfig) -> None:
    downstream_lanes = int(scenario.extra.get("downstream_lane_count", scenario.lane_count))
    root = ET.Element("edges")
    for edge_id, from_node, to_node, lanes in [
        (road.upstream_edge, "main_start", "merge_start", scenario.lane_count),
        (road.merge_edge, "merge_start", "merge_end", scenario.lane_count),
        (road.downstream_edge, "merge_end", "downstream_end", downstream_lanes),
        (road.ramp_edge, "ramp_start", "ramp_mid", 1),
        (road.ramp_merge_edge, "ramp_mid", "merge_start", 1),
    ]:
        ET.SubElement(
            root,
            "edge",
            {
                "id": edge_id,
                "from": from_node,
                "to": to_node,
                "numLanes": str(lanes),
                "speed": f"{road.speed_mps:.2f}",
                "priority": "1",
            },
        )
    _write_xml(path, root)


def _write_connections(path: Path, scenario: ScenarioConfig) -> None:
    downstream_lanes = int(scenario.extra.get("downstream_lane_count", scenario.lane_count))
    root = ET.Element("connections")
    for lane in range(scenario.lane_count):
        ET.SubElement(root, "connection", {"from": "E1", "to": "E4", "fromLane": str(lane), "toLane": str(lane)})
    for lane in range(min(scenario.lane_count, downstream_lanes)):
        ET.SubElement(root, "connection", {"from": "E4", "to": "E6", "fromLane": str(lane), "toLane": str(lane)})
    if downstream_lanes < scenario.lane_count:
        for lane in range(downstream_lanes, scenario.lane_count):
            ET.SubElement(
                root,
                "connection",
                {"from": "E4", "to": "E6", "fromLane": str(lane), "toLane": str(downstream_lanes - 1)},
            )
    ET.SubElement(root, "connection", {"from": "E2", "to": "E3", "fromLane": "0", "toLane": "0"})
    ET.SubElement(root, "connection", {"from": "E3", "to": "E4", "fromLane": "0", "toLane": str(scenario.lane_count - 1)})
    _write_xml(path, root)


def _write_detectors(path: Path, road: RoadNetworkDesign, scenario: ScenarioConfig) -> None:
    root = ET.Element("additional")
    detector_specs = [
        ("upstream", road.upstream_edge, max(50.0, road.upstream_length_m - 600.0), max(100.0, road.upstream_length_m - 100.0)),
        ("merge", road.merge_edge, 0.0, road.bottleneck_length_m),
        ("downstream", road.downstream_edge, 50.0, min(600.0, road.downstream_length_m)),
        ("ramp", road.ramp_merge_edge, 0.0, road.ramp_length_m * 0.35),
    ]
    for region, edge_id, start_pos, end_pos in detector_specs:
        lanes = 1 if edge_id in {"E2", "E3"} else scenario.lane_count
        for lane in range(lanes):
            ET.SubElement(
                root,
                "laneAreaDetector",
                {
                    "id": f"e2_{scenario.name}_{region}_{lane}",
                    "lane": f"{edge_id}_{lane}",
                    "pos": f"{start_pos:.1f}",
                    "endPos": f"{end_pos:.1f}",
                    "freq": "30",
                    "file": f"data/sumo/detectors/{scenario.name}_{region}_{lane}.xml",
                },
            )
    _write_xml(path, root)


def _write_xml(path: Path, root: ET.Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _copy_text(source: Path, target: Path) -> None:
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
