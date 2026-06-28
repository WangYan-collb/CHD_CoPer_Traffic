from __future__ import annotations

import numpy as np
from pathlib import Path

from src.control.conflict_resolution import ControlledVehicle
from src.control.moving_bottleneck import MovingBottleneckController, SpatialControlMapping
from src.envs.sumo.congestion_prediction import CongestionPredictor
from src.envs.sumo.config_builder import SumoConfigPaths, build_sumocfg
from src.envs.sumo.state_calculator import average_metrics, build_state_vector, sample_network_metrics, smoke_metrics
from src.envs.sumo.observations import StateHistory
from src.envs.sumo.traci_client import SumoUnavailableError, build_sumo_command, load_traci
from src.project.paths import BASE_NETWORK_DIR, SUMO_CONFIGS_DIR, SUMO_ROUTES_DIR
from src.rewards.combined_reward import RewardWeights, combined_reward
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
        aggregation_time_s: int = 120,
        control_cycle_s: int | None = None,
        simulation_time_s: int = 3600,
        congestion_prediction_enabled: bool = True,
        control_activation_score: float = 0.45,
        topology_state_enabled: bool = False,
        topology_reward_enabled: bool = False,
        topology_reward_weight: float = 0.10,
        bottleneck_position_m: float = 7500.0,
        upstream_control_length_m: float = 1900.0,
        recovery_length_m: float = 300.0,
        start_position_fraction: float = 0.25,
        end_position_fraction: float = 0.25,
        min_control_length_m: float = 1200.0,
        net_file: str | Path = "data/sumo/base_network/test1.net.xml",
        additional_file: str | Path = "data/sumo/base_network/E2_info.xml",
        use_gui: bool = False,
    ):
        self.scenario = scenario
        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self.smoke = smoke
        self.control_cycle_s = int(control_cycle_s or aggregation_time_s)
        self.simulation_time_s = int(simulation_time_s)
        self.congestion_prediction_enabled = congestion_prediction_enabled
        self.control_activation_score = float(control_activation_score)
        self.topology_state_enabled = bool(topology_state_enabled)
        self.topology_reward_enabled = bool(topology_reward_enabled)
        self.topology_reward_weight = float(topology_reward_weight)
        self.net_file = self._scenario_asset_path(Path(net_file), ".net.xml", "test1.net.xml")
        self.additional_file = self._scenario_asset_path(Path(additional_file), ".add.xml", "E2_info.xml")
        self.use_gui = use_gui
        self.history = StateHistory(sequence_length, state_dim)
        self.step_count = 0
        self.traci = None
        self.route_file = None
        self.sumocfg_file = None
        self.controller = MovingBottleneckController(
            spatial_mapping=SpatialControlMapping(
                bottleneck_position_m=float(bottleneck_position_m),
                upstream_control_length_m=float(upstream_control_length_m),
                recovery_length_m=float(recovery_length_m),
                start_fraction=float(start_position_fraction),
                end_fraction=float(end_position_fraction),
                min_control_length_m=float(min_control_length_m),
            )
        )
        self.congestion_predictor = CongestionPredictor()
        self.last_action = np.zeros(4, dtype=np.float32)
        self.currently_controlled_cavs: set[str] = set()

    def reset(self, route_seed_offset: int = 0) -> tuple[np.ndarray, dict[str, object]]:
        self.step_count = 0
        self.currently_controlled_cavs.clear()
        self.route_file = generate_route_file(
            self.scenario,
            SUMO_ROUTES_DIR / f"{self.scenario.name}.rou.xml",
            seed_offset=route_seed_offset,
        )
        if not self.smoke:
            self.traci = load_traci()
            self.sumocfg_file = SUMO_CONFIGS_DIR / f"{self.scenario.name}.sumocfg"
            build_sumocfg(
                SumoConfigPaths(
                    net_file=self.net_file,
                    route_file=Path(self.route_file),
                    additional_file=self.additional_file,
                    output_file=self.sumocfg_file,
                    end_s=self.simulation_time_s,
                )
            )
            self.traci.start(build_sumo_command(self.sumocfg_file, gui=self.use_gui))
        return self.history.reset(), {
            "scenario": self.scenario.name,
            "smoke": self.smoke,
            "route_file": str(self.route_file),
            "route_seed_offset": route_seed_offset,
            "sumocfg_file": None if self.sumocfg_file is None else str(self.sumocfg_file),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.step_count += 1
        if self.smoke:
            metrics = smoke_metrics(self.step_count)
            prediction = self.congestion_predictor.predict([metrics])
            reward, breakdown = combined_reward(
                density=metrics.density,
                critical_density=27.0,
                speed_mps=metrics.speed_mps,
                free_flow_speed_mps=30.0,
                queue_m=metrics.queue_m,
                ttc_s=metrics.ttc_s,
                tet_s=metrics.tet_s,
                tit_s=metrics.tit_s,
                action_delta=float(np.abs(_action4(action)).mean()),
            )
            state = build_state_vector(metrics, self.state_dim, congestion_score=prediction.score)
            next_state = self.history.append(state)
            terminated = self.step_count * self.control_cycle_s >= self.simulation_time_s
            return next_state, reward, terminated, False, {
                "reward": breakdown,
                "density": metrics.density,
                "speed_mps": metrics.speed_mps,
                "queue_m": metrics.queue_m,
                "congestion_score": prediction.score,
                "is_congested": prediction.is_congested,
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
        action4 = _action4(action)
        command = self.controller.build_command(
            list(action4),
            speed_mps=speed_mps,
            candidates_by_lane=self._cav_candidates(),
        )
        metric_samples = []
        prediction = self.congestion_predictor.predict([])
        controlled_vehicle_ids: set[str] = set()
        fallback_used = False
        active_control_seconds = 0
        command_mode_counts: dict[str, int] = {}
        controlled_position_samples: list[float] = []
        actual_gap_samples: list[float] = []
        for _ in range(self.control_cycle_s):
            if traci.simulation.getMinExpectedNumber() <= 0:
                break
            current_metrics = sample_network_metrics(traci)
            metric_samples.append(current_metrics)
            prediction = self.congestion_predictor.predict(metric_samples)
            apply_control = (
                prediction.is_congested or prediction.score >= self.control_activation_score
                if self.congestion_prediction_enabled
                else True
            )
            if apply_control:
                speed_mps = max(1.0, self._network_mean_speed())
                candidates = self._cav_candidates()
                command = self.controller.build_command(
                    list(action4),
                    speed_mps=speed_mps,
                    candidates_by_lane=candidates,
                )
                speed_limit_mps = command.speed_limit_kmh / 3.6
                selected = command.selected_vehicles
                if not selected:
                    selected = self._cavs_in_control_area(command.start_position_m, command.end_position_m)
                    fallback_used = True
                if not selected:
                    selected = self._nearest_cavs_to_control_area(command.start_position_m, command.end_position_m)
                    fallback_used = True
                selected_ids = {vehicle.vehicle_id for vehicle in selected}
                self._release_cav_speed_controls(self.currently_controlled_cavs - selected_ids)
                for vehicle in selected:
                    if vehicle.vehicle_id in traci.vehicle.getIDList():
                        traci.vehicle.setSpeed(vehicle.vehicle_id, speed_limit_mps)
                        traci.vehicle.setLaneChangeMode(vehicle.vehicle_id, 0)
                        controlled_vehicle_ids.add(vehicle.vehicle_id)
                self.currently_controlled_cavs = selected_ids
                if selected:
                    active_control_seconds += 1
                    positions = [vehicle.position_m for vehicle in selected]
                    controlled_position_samples.extend(positions)
                    actual_gap_samples.extend(_position_gaps(positions))
                fallback_used = fallback_used or command.fallback_used
                command_mode_counts[command.construction_mode] = command_mode_counts.get(command.construction_mode, 0) + 1
            else:
                self._release_cav_speed_controls()
            traci.simulationStep()

        metrics = average_metrics(metric_samples)
        action_delta = float(np.abs(action4 - self.last_action).mean())
        speed_limit_delta_kmh = 0.0
        if self.last_action.size >= 1:
            speed_limit_delta_kmh = abs(command.speed_limit_kmh - self._last_speed_limit_kmh())
        self.last_action = action4
        control_coverage_ratio = round(active_control_seconds / max(self.control_cycle_s, 1), 4)
        controlled_position_mean_m = float(np.mean(controlled_position_samples)) if controlled_position_samples else 0.0
        controlled_position_std_m = float(np.std(controlled_position_samples)) if controlled_position_samples else 0.0
        actual_gap_mean_m = float(np.mean(actual_gap_samples)) if actual_gap_samples else 0.0
        gap_error_m = (
            abs(actual_gap_mean_m - command.longitudinal_gap_m)
            if actual_gap_mean_m > 0.0
            else 0.0
        )
        congestion_score_delta = _congestion_delta(metric_samples, self.congestion_predictor, prediction.score)
        queue_delta_m = metric_samples[-1].queue_m - metric_samples[0].queue_m if len(metric_samples) >= 2 else 0.0
        reward_weights = None
        if self.topology_reward_enabled:
            reward_weights = RewardWeights(topology=self.topology_reward_weight)
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
            chain_coverage=command.chain_coverage,
            control_coverage_ratio=control_coverage_ratio,
            fallback_used=fallback_used,
            selected_cav_count=len(controlled_vehicle_ids),
            target_vehicle_count=command.target_vehicle_count,
            speed_limit_delta_kmh=speed_limit_delta_kmh,
            actual_gap_mean_m=actual_gap_mean_m,
            target_gap_m=command.longitudinal_gap_m,
            congestion_score_delta=congestion_score_delta,
            queue_delta_m=queue_delta_m,
            weights=reward_weights,
        )
        state = build_state_vector(
            metrics,
            self.state_dim,
            speed_limit_kmh=command.speed_limit_kmh,
            longitudinal_gap_m=command.longitudinal_gap_m,
            selected_cav_count=len(controlled_vehicle_ids),
            congestion_score=prediction.score,
            control_start_m=command.start_position_m,
            control_end_m=command.end_position_m,
            chain_coverage=command.chain_coverage if self.topology_state_enabled else 0.0,
            control_coverage_ratio=control_coverage_ratio if self.topology_state_enabled else 0.0,
            fallback_used=fallback_used if self.topology_state_enabled else False,
            active_control_seconds=active_control_seconds if self.topology_state_enabled else 0,
            target_vehicle_count=command.target_vehicle_count if self.topology_state_enabled else 0,
            speed_limit_delta_kmh=speed_limit_delta_kmh if self.topology_state_enabled else 0.0,
            controlled_position_mean_m=controlled_position_mean_m if self.topology_state_enabled else 0.0,
            controlled_position_std_m=controlled_position_std_m if self.topology_state_enabled else 0.0,
            actual_gap_mean_m=actual_gap_mean_m if self.topology_state_enabled else 0.0,
            gap_error_m=gap_error_m if self.topology_state_enabled else 0.0,
            congestion_score_delta=congestion_score_delta if self.topology_state_enabled else 0.0,
            queue_delta_m=queue_delta_m if self.topology_state_enabled else 0.0,
        )
        next_state = self.history.append(state)
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        info = {
            "reward": breakdown,
            "speed_limit_kmh": command.speed_limit_kmh,
            "longitudinal_gap_m": command.longitudinal_gap_m,
            "control_start_m": command.start_position_m,
            "control_end_m": command.end_position_m,
            "selected_cav_count": len(controlled_vehicle_ids),
            "fallback_used": fallback_used,
            "construction_mode": command.construction_mode,
            "chain_coverage": command.chain_coverage,
            "target_vehicle_count": command.target_vehicle_count,
            "active_control_seconds": active_control_seconds,
            "control_coverage_ratio": control_coverage_ratio,
            "command_mode_counts": command_mode_counts,
            "topology_reward": breakdown.topology,
            "speed_limit_delta_kmh": speed_limit_delta_kmh,
            "controlled_position_mean_m": controlled_position_mean_m,
            "controlled_position_std_m": controlled_position_std_m,
            "actual_gap_mean_m": actual_gap_mean_m,
            "gap_error_m": gap_error_m,
            "congestion_score_delta": congestion_score_delta,
            "queue_delta_m": queue_delta_m,
            "is_congested": prediction.is_congested,
            "congestion_score": prediction.score,
            "density_score": prediction.density_score,
            "speed_score": prediction.speed_score,
            "queue_score": prediction.queue_score,
            "flow_decay_score": prediction.flow_decay_score,
            "queue_growth_score": prediction.queue_growth_score,
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

    def _last_speed_limit_kmh(self) -> float:
        if self.last_action.size == 0:
            return 0.0
        from src.control.speed_limit_mapper import map_speed_action

        return float(map_speed_action(float(self.last_action[0])))

    def _cav_candidates(self) -> dict[str, list[ControlledVehicle]]:
        candidates: dict[str, list[ControlledVehicle]] = {}
        for lane_id in self.traci.lane.getIDList():
            if lane_id.startswith(":") or not self._is_mainline_lane(lane_id):
                continue
            lane_candidates = []
            for vehicle_id in self.traci.lane.getLastStepVehicleIDs(lane_id):
                if not vehicle_id.startswith("CAV."):
                    continue
                position = self._vehicle_absolute_position(vehicle_id)
                if position is None:
                    continue
                lane_candidates.append(
                    ControlledVehicle(
                        vehicle_id=vehicle_id,
                        lane_id=lane_id,
                        position_m=position,
                    )
                )
            if lane_candidates:
                candidates[lane_id] = lane_candidates
        return candidates

    def _cavs_in_control_area(self, start_position_m: float, end_position_m: float) -> list[ControlledVehicle]:
        selected = []
        for lane_id in self.traci.lane.getIDList():
            if lane_id.startswith(":") or not self._is_mainline_lane(lane_id):
                continue
            for vehicle_id in self.traci.lane.getLastStepVehicleIDs(lane_id):
                if not vehicle_id.startswith("CAV."):
                    continue
                position = self._vehicle_absolute_position(vehicle_id)
                if position is None:
                    continue
                if start_position_m <= position <= end_position_m:
                    selected.append(ControlledVehicle(vehicle_id, lane_id, position))
        return selected

    def _nearest_cavs_to_control_area(
        self,
        start_position_m: float,
        end_position_m: float,
        max_distance_m: float = 500.0,
    ) -> list[ControlledVehicle]:
        center = 0.5 * (start_position_m + end_position_m)
        candidates = [
            vehicle
            for lane_candidates in self._cav_candidates().values()
            for vehicle in lane_candidates
        ]
        lane_count = len({vehicle.lane_id for vehicle in candidates})
        nearby = [
            vehicle
            for vehicle in candidates
            if start_position_m - max_distance_m <= vehicle.position_m <= end_position_m + max_distance_m
        ]
        return sorted(nearby, key=lambda vehicle: abs(vehicle.position_m - center))[: max(1, lane_count)]

    def _is_mainline_lane(self, lane_id: str) -> bool:
        edge_id = lane_id.split("_", 1)[0]
        return edge_id in {"E1", "E4", "E6"}

    def _vehicle_absolute_position(self, vehicle_id: str) -> float | None:
        road_id = self.traci.vehicle.getRoadID(vehicle_id)
        if road_id.startswith(":"):
            return None
        lane_position = float(self.traci.vehicle.getLanePosition(vehicle_id))
        edge_offsets = {
            "E1": 0.0,
            "E4": 7500.0,
            "E6": 7500.0 + float(self.scenario.bottleneck_length_m),
        }
        if road_id not in edge_offsets:
            return None
        return edge_offsets[road_id] + lane_position

    def _release_cav_speed_controls(self, vehicle_ids: set[str] | None = None) -> None:
        if vehicle_ids is None:
            vehicle_ids = {vehicle_id for vehicle_id in self.traci.vehicle.getIDList() if vehicle_id.startswith("CAV.")}
        for vehicle_id in vehicle_ids:
            if vehicle_id in self.traci.vehicle.getIDList():
                self.traci.vehicle.setSpeed(vehicle_id, -1)
                self.traci.vehicle.setLaneChangeMode(vehicle_id, 1621)
        if vehicle_ids:
            self.currently_controlled_cavs.difference_update(vehicle_ids)


def _action4(action: np.ndarray) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size >= 4:
        return arr[:4]
    if arr.size == 3:
        return np.asarray([arr[0], arr[1], min(1.0, arr[1] + 0.25), arr[2]], dtype=np.float32)
    raise ValueError("action must contain at least 3 dimensions")


def _position_gaps(positions: list[float]) -> list[float]:
    ordered = sorted(set(float(position) for position in positions))
    return [
        right - left
        for left, right in zip(ordered, ordered[1:])
        if right > left
    ]


def _congestion_delta(metric_samples, predictor, final_score: float) -> float:
    if not metric_samples:
        return 0.0
    initial_score = predictor.predict([metric_samples[0]]).score
    return float(final_score - initial_score)
