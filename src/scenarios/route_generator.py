from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

from src.scenarios.scenario_config import ScenarioConfig
from src.scenarios.road_network import RoadNetworkDesign
from src.scenarios.traffic_flow import NormalTrafficFlowSampler
from src.scenarios.vehicle_params import default_vehicle_types


def _add_vehicle_types(root: ET.Element) -> None:
    for vehicle_type in default_vehicle_types():
        ET.SubElement(root, "vType", vehicle_type.sumo_attributes())


def generate_route_file(scenario: ScenarioConfig, output_path: str | Path, seed_offset: int = 0) -> Path:
    """Generate a simple SUMO route file from a thesis scenario config."""

    rng = random.Random(scenario.seed + int(seed_offset))
    road = RoadNetworkDesign(bottleneck_length_m=scenario.bottleneck_length_m)
    sampler = NormalTrafficFlowSampler(segment_seconds=600)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("routes")
    _add_vehicle_types(root)
    ET.SubElement(root, "route", {"id": "main", "edges": road.main_route_edges})
    ET.SubElement(root, "route", {"id": "ramp", "edges": road.ramp_route_edges})

    cav_index = 0
    for segment_index, segment in enumerate(sampler.sample_segments(scenario, seed_offset=seed_offset)):
        split = sampler.split_vehicle_flows(segment, scenario.cav_ratio)
        for idx, vehicle_type in enumerate(("HDV_0", "HDV_1", "HDV_2")):
            ET.SubElement(
                root,
                "flow",
                {
                    "id": f"main_hdv_{segment_index}_{idx}",
                    "type": vehicle_type,
                    "route": "main",
                    "vehsPerHour": str(split.hdv_main_vph[idx]),
                    "begin": str(segment.begin_s),
                    "end": str(segment.end_s),
                    "departLane": "random",
                    "departSpeed": "speedLimit",
                    "departPos": "base",
                },
            )
            ET.SubElement(
                root,
                "flow",
                {
                    "id": f"ramp_hdv_{segment_index}_{idx}",
                    "type": vehicle_type,
                    "route": "ramp",
                    "vehsPerHour": str(split.hdv_ramp_vph[idx]),
                    "begin": str(segment.begin_s),
                    "end": str(segment.end_s),
                    "departLane": "random",
                    "departSpeed": "speedLimit",
                    "departPos": "free",
                },
            )

        segment_hours = (segment.end_s - segment.begin_s) / 3600
        cav_count = max(1, int(split.cav_main_vph * segment_hours))
        for _ in range(cav_count):
            cav_index += 1
            depart = round(rng.uniform(segment.begin_s, segment.end_s), 1)
            ET.SubElement(
                root,
                "vehicle",
                {
                    "id": f"CAV.{cav_index}",
                    "type": "CAV",
                    "route": "main",
                    "depart": str(depart),
                    "departLane": "random",
                    "departPos": "base",
                    "departSpeed": "speedLimit",
                },
            )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path
