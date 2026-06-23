from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetaScenarioScore:
    scenario: str
    split: str
    before: float
    after: float
    adaptation_steps: int


@dataclass(frozen=True)
class MetaGeneralizationSummary:
    interpolation_mean_after: float
    extrapolation_mean_after: float
    performance_decay_rate: float
    ood_robustness_score: float
    mean_adaptation_steps: float


def summarize_meta_generalization(scores: list[MetaScenarioScore]) -> MetaGeneralizationSummary:
    interpolation = [score for score in scores if score.split == "interpolation"]
    extrapolation = [score for score in scores if score.split == "extrapolation"]
    interpolation_mean = _mean([score.after for score in interpolation])
    extrapolation_mean = _mean([score.after for score in extrapolation])
    if interpolation_mean == 0:
        decay = 0.0
        robustness = 0.0
    else:
        decay = max(0.0, (interpolation_mean - extrapolation_mean) / interpolation_mean * 100.0)
        robustness = extrapolation_mean / interpolation_mean * 100.0
    return MetaGeneralizationSummary(
        interpolation_mean_after=round(interpolation_mean, 4),
        extrapolation_mean_after=round(extrapolation_mean, 4),
        performance_decay_rate=round(decay, 4),
        ood_robustness_score=round(robustness, 4),
        mean_adaptation_steps=round(_mean([score.adaptation_steps for score in scores]), 4),
    )


def _mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
