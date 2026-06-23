from pathlib import Path

import pytest

from src.envs.sumo.config_builder import SumoConfigPaths, build_sumocfg


def test_build_sumocfg_references_net_route_and_additional(tmp_path: Path):
    net = tmp_path / "test.net.xml"
    route = tmp_path / "traffic.rou.xml"
    additional = tmp_path / "E2_info.xml"
    net.write_text("<net/>", encoding="utf-8")
    route.write_text("<routes/>", encoding="utf-8")
    additional.write_text("<additional/>", encoding="utf-8")

    paths = build_sumocfg(
        SumoConfigPaths(
            net_file=net,
            route_file=route,
            additional_file=additional,
            output_file=tmp_path / "run.sumocfg",
        )
    )

    content = paths.output_file.read_text(encoding="utf-8")
    assert "test.net.xml" in content
    assert "traffic.rou.xml" in content
    assert "E2_info.xml" in content


def test_build_sumocfg_rejects_missing_additional_file(tmp_path: Path):
    net = tmp_path / "test.net.xml"
    route = tmp_path / "traffic.rou.xml"
    net.write_text("<net/>", encoding="utf-8")
    route.write_text("<routes/>", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="additional file"):
        build_sumocfg(
            SumoConfigPaths(
                net_file=net,
                route_file=route,
                additional_file=tmp_path / "missing_E2_info.xml",
                output_file=tmp_path / "run.sumocfg",
            )
        )
