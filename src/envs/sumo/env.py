from __future__ import annotations

import numpy as np
from pathlib import Path

from src.control.conflict_resolution import ControlledVehicle
from src.control.moving_bottleneck import MovingBottleneckController
from src.envs.sumo.config_builder import SumoConfigPaths, build_sumocfg
from src.envs.sumo.state_calculator import average_metrics, build_state_vector, sample_network_metrics, smoke_metrics
from src.envs.sumo.observations import StateHistory
from src.envs.sumo.traci_client import SumoUnavailableError, build_sumo_command, load_traci
from src.project.paths import BASE_NETWORK_DIR, GENERATED_ROUTES_DIR
from src.rewards.combined_reward import combined_reward
from src.scenarios.scenario_config import ScenarioConfig
from src.scenarios.route_generator import generate_route_file


class SumoMovingBottleneckEnv:
    """SUMO-compatible environment shell.

    The class is intentionally conservative: it refuses to start real simulation
    when SUMO is not installed, while still exposing a deterministic smoke mode.
    """

    def __init__(
        self,
        scenario: ScenarioConfig,
        sequence_length: int = 30,
        state_dim: int = 17,
        smoke: bool = False,
        aggregation_time_s: int = 30,
        net_file: str | Path = "data/sumo/base_network/test1.net.xml",
        additional_file: str | Path = "data/sumo/base_network/E2_info.xml",
        use_gui: bool = False,
    ):
        self.scenario = scenario
        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self.smoke = smoke
        self.aggregation_time_s = aggregation_time_s
        self.net_file = self._scenario_asset_path(Path(net_file), ".net.xml", "test1.net.xml")
        self.additional_file = self._scenario_asset_path(Path(additional_file), ".add.xml", "E2_info.xml")
        self.use_gui = use_gui
        self.history = StateHistory(sequence_length, state_dim)
        self.step_count = 0
        self.traci = None
        self.route_file = None
        self.sumocfg_file = None
        self.controller = MovingBottleneckController()
        self.last_action = np.zeros(3, dtype=np.float32)

    def reset(self) -> tuple[np.ndarray, dict[str, object]]:
        self.step_count = 0
        self.route_file = generate_route_file(
            self.scenario,
            GENERATED_ROUTES_DIR / f"{self.scenario.name}.rou.xml",
        )
        if not self.smoke:
            self.traci = load_traci()
            self.sumocfg_file = GENERATED_ROUTES_DIR / f"{self.scenario.name}.sumocfg"
            build_sumocfg(
                SumoConfigPaths(
                    net_file=self.net_file,
                    route_file=Path(self.route_file),
                    additional_file=self.additional_file,
                    output_file=self.sumocfg_file,
                )
            )
            self.traci.start(build_sumo_command(self.sumocfg_file, gui=self.use_gui))
        return self.history.reset(), {
            "scenario": self.scenario.name,
            "smoke": self.smoke,
            "route_file": str(self.route_file),
            "sumocfg_file": None if self.sumocfg_file is None else str(self.sumocfg_file),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.step_count += 1
        if self.smoke:
            metrics = smoke_metrics(self.step_count)
            reward, breakdown = combined_reward(
                density=metrics.density,
                critical_density=27.0,
                speed_mps=metrics.speed_mps,
                free_flow_speed_mps=30.0,
                queue_m=metrics.queue_m,
                ttc_s=metrics.ttc_s,
                tet_s=metrics.tet_s,
                tit_s=metrics.tit_s,
                action_delta=float(np.abs(action).mean()),
            )
            state = build_state_vector(metrics, self.state_dim)
            next_state = self.history.append(state)
            terminated = self.step_count >= 5
            return next_state, reward, terminated, False, {
                "reward": breakdown,
                "density": metrics.density,
                "speed_mps": metrics.speed_mps,
                "queue_m": metrics.queue_m,
            }
        if self.traci is None:
            raise SumoUnavailableError("TraCI is not started; call reset() before step()")
        return self._real_sumo_step(action)

    def close(self) -> None:
        if self.traci is not None:
            self.traci.close(False)
            self.traci = None

    def _scenario_asset_path(self, configured: Path, suffix: str, legacy_name: str) -> Path:
        scenario_file = BASE_NETWORK_DIR / f"{self.scenario.name}{suffix}"
        if scenario_file.exists() and configured.name == legacy_name:
            return scenario_file
        return configured

    def _real_sumo_step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        traci = self.traci
        speed_mps = max(1.0, self._network_mean_speed())
        candidates = self._cav_candidates()
        command = self.controller.build_command(
            list(np.asarray(action, dtype=np.float32)),
            speed_mps=speed_mps,
            candidates_by_lane=candidates,
        )
        speed_limit_mps = command.speed_limit_kmh / 3.6
        for vehicle in command.selected_vehicles:
            if vehicle.vehicle_id in traci.vehicle.getIDList():
                traci.vehicle.setSpeed(vehicle.vehicle_id, speed_limit_mps)
                traci.vehicle.setLaneChangeMode(vehicle.vehicle_id, 255)

        metric_samples = []
        for _ in range(self.aggregation_time_s):
            if traci.simulation.getMinExpectedNumber() <= 0:
                break
            traci.simulationStep()
            metric_samples.append(sample_network_metrics(traci))

        metrics = average_metrics(metric_samples)
        action_delta = float(np.abs(np.asarray(action, dtype=np.float32) - self.last_action).mean())
        self.last_action = np.asarray(action, dtype=np.float32)
        reward, breakdown = combined_reward(
            density=metrics.density,
            critical_density=27.0,
            speed_mps=metrics.speed_mps,
            free_flow_speed_mps=30.0,
            queue_m=metrics.queue_m,
            ttc_s=metrics.ttc_s,
            tet_s=metrics.tet_s,
            tit_s=metrics.tit_s,
            action_delta=action_delta,
        )
        state = build_state_vector(
            metrics,
            self.state_dim,
            speed_limit_kmh=command.speed_limit_kmh,
            longitudinal_gap_m=command.longitudinal_gap_m,
            selected_cav_count=len(command.selected_vehicles),
        )
        next_state = self.history.append(state)
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        info = {
            "reward": breakdown,
            "speed_limit_kmh": command.speed_limit_kmh,
            "longitudinal_gap_m": command.longitudinal_gap_m,
            "selected_cav_count": len(command.selected_vehicles),
            "density": metrics.density,
            "speed_mps": metrics.speed_mps,
            "queue_m": metrics.queue_m,
            "throughput": metrics.throughput,
        }
        return next_state, reward, terminated, False, info

    def _network_mean_speed(self) -> float:
        speeds = []
        for vehicle_id in self.traci.vehicle.getIDList():
            speeds.append(self.traci.vehicle.getSpeed(vehicle_id))
        return float(np.mean(speeds)) if speeds else 20.0

    def _cav_candidates(self) -> dict[str, list[ControlledVehicle]]:
        candidates: dict[str, list[ControlledVehicle]] = {}
        for lane_id in self.traci.lane.getIDList():
            if lane_id.startswith(":"):
                continue
            lane_candidates = []
            for vehicle_id in self.traci.lane.getLastStepVehicleIDs(lane_id):
                if not vehicle_id.startswith("CAV."):
                    continue
                lane_candidates.append(
                    ControlledVehicle(
                        vehicle_id=vehicle_id,
                        lane_id=lane_id,
                        position_m=float(self.traci.vehicle.getLanePosition(vehicle_id)),
                    )
                )
            if lane_candidates:
                candidates[lane_id] = lane_candidates
        return candidates
