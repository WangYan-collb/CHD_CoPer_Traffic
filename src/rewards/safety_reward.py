from __future__ import annotations


def ttc_safety_reward(ttc_s: float, safe_ttc_s: float = 3.0) -> float:
    if ttc_s <= 0:
        return -1.0
    if ttc_s >= safe_ttc_s:
        return 1.0
    return (ttc_s / safe_ttc_s) * 2.0 - 1.0


def risk_penalty(tet_s: float, tit_s: float, tet_scale: float = 60.0, tit_scale: float = 10.0) -> float:
    return -(max(0.0, tet_s) / tet_scale + max(0.0, tit_s) / tit_scale)
