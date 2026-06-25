from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class SumoConfigPaths:
    net_file: Path
    route_file: Path
    output_file: Path
    additional_file: Path | None = None
    step_length: float = 1.0
    begin_s: int = 0
    end_s: int = 3600


def build_sumocfg(paths: SumoConfigPaths, validate_inputs: bool = True) -> SumoConfigPaths:
    if validate_inputs and not paths.net_file.exists():
        raise FileNotFoundError(
            f"SUMO net file not found: {paths.net_file}. "
            "Place the thesis net.xml under data/sumo/base_network or update the config."
        )
    if validate_inputs and not paths.route_file.exists():
        raise FileNotFoundError(f"SUMO route file not found: {paths.route_file}")

    root = ET.Element("configuration")
    input_node = ET.SubElement(root, "input")
    ET.SubElement(input_node, "net-file", {"value": _rel(paths.output_file, paths.net_file)})
    ET.SubElement(input_node, "route-files", {"value": _rel(paths.output_file, paths.route_file)})
    if validate_inputs and paths.additional_file and not paths.additional_file.exists():
        raise FileNotFoundError(
            f"SUMO additional file not found: {paths.additional_file}. "
            "Place E2_info.xml under data/sumo/base_network or set additional_file to null."
        )
    if paths.additional_file:
        ET.SubElement(input_node, "additional-files", {"value": _rel(paths.output_file, paths.additional_file)})

    time_node = ET.SubElement(root, "time")
    ET.SubElement(time_node, "begin", {"value": str(paths.begin_s)})
    ET.SubElement(time_node, "end", {"value": str(paths.end_s)})
    ET.SubElement(time_node, "step-length", {"value": str(paths.step_length)})

    processing_node = ET.SubElement(root, "processing")
    ET.SubElement(processing_node, "ignore-route-errors", {"value": "true"})
    ET.SubElement(processing_node, "time-to-teleport", {"value": "-1"})

    paths.output_file.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(paths.output_file, encoding="utf-8", xml_declaration=True)
    return paths


def _rel(base_file: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), base_file.parent.resolve())
