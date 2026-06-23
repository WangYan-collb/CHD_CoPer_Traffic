from __future__ import annotations

import numpy as np
from pathlib import Path

from src.control.conflict_resolution import ControlledVehicle
from src.control.moving_bottleneck import MovingBottleneckController
from src.envs.sumo.config_builder import SumoConfigPaths, build_sumocfg
from src.envs.sumo.observations import StateHistory
from src.envs.sumo.traci_client import SumoUnavailableError, build_sumo_command, load_traci
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
        self.net_file = Path(net_file)
        self.additional_file = Path(additional_file)
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
            f"data/sumo/generated_routes/{self.scenario.name}.rou.xml",
        )
        if not self.smoke:
            self.traci = load_traci()
            self.sumocfg_file = Path(f"data/sumo/generated_routes/{self.scenario.name}.sumocfg")
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
            density = min(35.0, 20.0 + self.step_count * 0.2)
            speed = max(5.0, 28.0 - self.step_count * 0.05)
            queue = max(0.0, density - 27.0) * 2.0
            reward, breakdown = combined_reward(
                density=density,
                critical_density=27.0,
                speed_mps=speed,
                free_flow_speed_mps=30.0,
                queue_m=queue,
                ttc_s=3.0,
                tet_s=0.0,
                tit_s=0.0,
                action_delta=float(np.abs(action).mean()),
            )
            state = np.zeros(self.state_dim, dtype=np.float32)
            state[0] = density / 100.0
            state[1] = speed / 30.0
            state[2] = queue / 100.0
            next_state = self.history.append(state)
            terminated = self.step_count >= 5
            return next_state, reward, terminated, False, {"reward": breakdown}
        if self.traci is None:
            raise SumoUnavailableError("TraCI is not started; call reset() before step()")
        return self._real_sumo_step(action)

    def close(self) -> None:
        if self.traci is not None:
            self.traci.close(False)
            self.traci = None

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

        density_values: list[float] = []
        speed_values: list[float] = []
        queue_values: list[float] = []
        for _ in range(self.aggregation_time_s):
            if traci.simulation.getMinExpectedNumber() <= 0:
                break
            traci.simulationStep()
            density, speed, queue = self._sample_network_metrics()
            density_values.append(density)
            speed_values.append(speed)
            queue_values.append(queue)

        density = float(np.mean(density_values)) if density_values else 0.0
        speed = float(np.mean(speed_values)) if speed_values else 0.0
        queue = float(np.mean(queue_values)) if queue_values else 0.0
        action_delta = float(np.abs(np.asarray(action, dtype=np.float32) - self.last_action).mean())
        self.last_action = np.asarray(action, dtype=np.float32)
        reward, breakdown = combined_reward(
            density=density,
            critical_density=27.0,
            speed_mps=speed,
            free_flow_speed_mps=30.0,
            queue_m=queue,
            ttc_s=3.0,
            tet_s=0.0,
            tit_s=0.0,
            action_delta=action_delta,
        )
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0] = min(density / 100.0, 1.0)
        state[1] = min(speed / 30.0, 1.0)
        state[2] = min(queue / 100.0, 1.0)
        state[3] = command.speed_limit_kmh / 120.0
        state[4] = min(command.longitudinal_gap_m / 100.0, 1.0)
        next_state = self.history.append(state)
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        info = {
            "reward": breakdown,
            "speed_limit_kmh": command.speed_limit_kmh,
            "longitudinal_gap_m": command.longitudinal_gap_m,
            "selected_cav_count": len(command.selected_vehicles),
            "density": density,
            "speed_mps": speed,
            "queue_m": queue,
        }
        return next_state, reward, terminated, False, info

    def _network_mean_speed(self) -> float:
        speeds = []
        for vehicle_id in self.traci.vehicle.getIDList():
            speeds.append(self.traci.vehicle.getSpeed(vehicle_id))
        return float(np.mean(speeds)) if speeds else 20.0

    def _sample_network_metrics(self) -> tuple[float, float, float]:
        lane_ids = list(self.traci.lane.getIDList())
        valid_lanes = [lane for lane in lane_ids if not lane.startswith(":")]
        densities = []
        speeds = []
        queues = []
        for lane_id in valid_lanes:
            length = max(self.traci.lane.getLength(lane_id), 1.0)
            vehicle_count = self.traci.lane.getLastStepVehicleNumber(lane_id)
            densities.append(vehicle_count / length * 1000.0)
            speeds.append(self.traci.lane.getLastStepMeanSpeed(lane_id))
            queues.append(self.traci.lane.getLastStepHaltingNumber(lane_id) * 7.5)
        return (
            float(np.mean(densities)) if densities else 0.0,
            float(np.mean(speeds)) if speeds else 0.0,
            float(np.mean(queues)) if queues else 0.0,
        )

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
