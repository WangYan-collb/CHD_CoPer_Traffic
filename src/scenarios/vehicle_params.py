from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleType:
    id: str
    length_m: float
    color: str
    car_follow_model: str
    accel: float
    decel: float
    tau: float
    min_gap_m: float
    lane_change_strategic: str = "0"
    sigma: float | None = None
    cc1: float | None = None
    speed_factor: str | None = None

    def sumo_attributes(self) -> dict[str, str]:
        attrs = {
            "id": self.id,
            "color": self.color,
            "length": str(self.length_m),
            "carFollowModel": self.car_follow_model,
            "accel": str(self.accel),
            "decel": str(self.decel),
            "tau": str(self.tau),
            "minGap": str(self.min_gap_m),
            "lcStrategic": self.lane_change_strategic,
        }
        if self.sigma is not None:
            attrs["sigma"] = str(self.sigma)
        if self.cc1 is not None:
            attrs["cc1"] = str(self.cc1)
        if self.speed_factor is not None:
            attrs["speedFactor"] = self.speed_factor
        return attrs


def default_vehicle_types() -> list[VehicleType]:
    """Vehicle parameters aligned with the thesis reference SUMO project."""

    return [
        VehicleType(
            id="HDV_0",
            length_m=3.5,
            color="127,255,0",
            car_follow_model="Krauss",
            accel=2.6,
            decel=4.5,
            sigma=0.5,
            tau=1.0,
            min_gap_m=2.0,
            lane_change_strategic="-1",
            speed_factor="normc(1,0.1,0.2,2)",
        ),
        VehicleType(
            id="HDV_1",
            length_m=3.5,
            color="255,0,255",
            car_follow_model="W99",
            accel=2.6,
            decel=4.5,
            cc1=1.5,
            tau=2.0,
            min_gap_m=2.0,
            lane_change_strategic="0",
            speed_factor="normc(1,0.1,0.2,2)",
        ),
        VehicleType(
            id="HDV_2",
            length_m=7.0,
            color="30,144,255",
            car_follow_model="W99",
            accel=1.5,
            decel=3.5,
            tau=1.5,
            min_gap_m=3.0,
            lane_change_strategic="-1",
            speed_factor="normc(1,0.1,0.2,2)",
        ),
        VehicleType(
            id="CAV",
            length_m=3.5,
            color="255,0,0",
            car_follow_model="IDM",
            accel=3.0,
            decel=5.5,
            tau=1.0,
            min_gap_m=1.4,
            lane_change_strategic="0",
        ),
    ]
