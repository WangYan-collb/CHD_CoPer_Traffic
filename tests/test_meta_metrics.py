from src.evaluation.meta_metrics import MetaScenarioScore, summarize_meta_generalization


def test_meta_summary_reports_decay_and_ood_robustness():
    scores = [
        MetaScenarioScore("interpolation_1", "interpolation", before=0.70, after=0.86, adaptation_steps=3),
        MetaScenarioScore("interpolation_2", "interpolation", before=0.72, after=0.88, adaptation_steps=3),
        MetaScenarioScore("extrapolation_1", "extrapolation", before=0.50, after=0.78, adaptation_steps=3),
    ]

    summary = summarize_meta_generalization(scores)

    assert summary.interpolation_mean_after == 0.87
    assert summary.extrapolation_mean_after == 0.78
    assert summary.ood_robustness_score > 0.0
    assert summary.performance_decay_rate > 0.0
