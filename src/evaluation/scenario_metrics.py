from __future__ import annotations


def performance_decay_rate(source_score: float, target_score: float) -> float:
    if source_score == 0:
        return 0.0
    return (source_score - target_score) / source_score * 100.0


def ood_robustness_score(interpolation_score: float, extrapolation_score: float) -> float:
    if interpolation_score == 0:
        return 0.0
    return extrapolation_score / interpolation_score * 100.0
