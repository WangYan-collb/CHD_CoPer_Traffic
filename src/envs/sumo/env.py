from __future__ import annotations

import numpy as np

from src.envs.sumo.observations import StateHistory
from src.envs.sumo.traci_client import SumoUnavailableError, load_traci
from src.rewards.combined_reward import combined_reward
from src.scenarios.scenario_config import ScenarioConfig
from src.scenarios.route_generator import generate_route_file


class SumoMovingBottleneckEnv:
    """SUMO-compatible environment shell.

    The class is intentionally conservative: it refuses to start real simulation
    when SUMO is not installed, while still exposing a deterministic smoke mode.
    """

    def __init__(self, scenario: ScenarioConfig, sequence_length: int = 30, state_dim: int = 17, smoke: bool = False):
        self.scenario = scenario
        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self.smoke = smoke
        self.history = StateHistory(sequence_length, state_dim)
        self.step_count = 0
        self.traci = None
        self.route_file = None

    def reset(self) -> tuple[np.ndarray, dict[str, object]]:
        self.step_count = 0
        self.route_file = generate_route_file(
            self.scenario,
            f"data/sumo/generated_routes/{self.scenario.name}.rou.xml",
        )
        if not self.smoke:
            self.traci = load_traci()
        return self.history.reset(), {
            "scenario": self.scenario.name,
            "smoke": self.smoke,
            "route_file": str(self.route_file),
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
        raise SumoUnavailableError("real SUMO stepping is not implemented until SUMO assets are configured")

    def close(self) -> None:
        if self.traci is not None:
            self.traci.close(False)
