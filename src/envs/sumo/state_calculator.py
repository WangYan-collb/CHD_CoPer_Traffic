from __future__ import annotations

import numpy as np

from src.envs.sumo.metrics import TrafficMetrics


def smoke_metrics(step_count: int) -> TrafficMetrics:
    density = min(35.0, 20.0 + step_count * 0.2)
    speed = max(5.0, 28.0 - step_count * 0.05)
    queue = max(0.0, density - 27.0) * 2.0
    return TrafficMetrics(
        density=density,
        speed_mps=speed,
        queue_m=queue,
        ttc_s=3.0,
        tet_s=0.0,
        tit_s=0.0,
        throughput=0.0,
    )


def sample_network_metrics(traci) -> TrafficMetrics:
    lane_ids = list(traci.lane.getIDList())
    valid_lanes = [lane for lane in lane_ids if not lane.startswith(":")]
    densities = []
    speeds = []
    queues = []
    throughput = 0.0
    for lane_id in valid_lanes:
        length = max(float(traci.lane.getLength(lane_id)), 1.0)
        vehicle_count = float(traci.lane.getLastStepVehicleNumber(lane_id))
        mean_speed = float(traci.lane.getLastStepMeanSpeed(lane_id))
        halting_count = float(traci.lane.getLastStepHaltingNumber(lane_id))
        densities.append(vehicle_count / length * 1000.0)
        speeds.append(max(mean_speed, 0.0))
        queues.append(halting_count * 7.5)
        throughput += vehicle_count
    return TrafficMetrics(
        density=float(np.mean(densities)) if densities else 0.0,
        speed_mps=float(np.mean(speeds)) if speeds else 0.0,
        queue_m=float(np.mean(queues)) if queues else 0.0,
        ttc_s=3.0,
        tet_s=0.0,
        tit_s=0.0,
        throughput=throughput,
    )


def average_metrics(samples: list[TrafficMetrics]) -> TrafficMetrics:
    if not samples:
        return TrafficMetrics(0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0)
    return TrafficMetrics(
        density=float(np.mean([item.density for item in samples])),
        speed_mps=float(np.mean([item.speed_mps for item in samples])),
        queue_m=float(np.mean([item.queue_m for item in samples])),
        ttc_s=float(np.mean([item.ttc_s for item in samples])),
        tet_s=float(np.mean([item.tet_s for item in samples])),
        tit_s=float(np.mean([item.tit_s for item in samples])),
        throughput=float(np.sum([item.throughput for item in samples])),
    )


def build_state_vector(
    metrics: TrafficMetrics,
    state_dim: int,
    speed_limit_kmh: float = 0.0,
    longitudinal_gap_m: float = 0.0,
    selected_cav_count: int = 0,
    congestion_score: float = 0.0,
    control_start_m: float = 0.0,
    control_end_m: float = 0.0,
) -> np.ndarray:
    state = np.zeros(state_dim, dtype=np.float32)
    state[0] = min(metrics.density / 100.0, 1.0)
    state[1] = min(metrics.speed_mps / 30.0, 1.0)
    state[2] = min(metrics.queue_m / 100.0, 1.0)
    if state_dim > 3:
        state[3] = min(speed_limit_kmh / 120.0, 1.0)
    if state_dim > 4:
        state[4] = min(longitudinal_gap_m / 100.0, 1.0)
    if state_dim > 5:
        state[5] = min(selected_cav_count / 10.0, 1.0)
    if state_dim > 6:
        state[6] = min(metrics.throughput / 1000.0, 1.0)
    if state_dim > 7:
        state[7] = min(max(congestion_score, 0.0), 1.0)
    if state_dim > 8:
        state[8] = min(control_start_m / 7500.0, 1.0)
    if state_dim > 9:
        state[9] = min(control_end_m / 7500.0, 1.0)
    return state
