"""
TDD: Failing tests for ablation analysis pure functions.
These functions will live in src/utils/ablation_analysis.py (Functional Core).
"""
import pytest


def test_rank_configs_by_sharpe():
    """Best Sharpe should be ranked first."""
    from src.utils.ablation_analysis import rank_configs

    results = {
        "baseline": {"sharpe": 0.5, "roi": 2.0, "mdd": -3.0},
        "nextgen":  {"sharpe": -0.2, "roi": -1.0, "mdd": -8.0},
        "dsr_only": {"sharpe": 1.2, "roi": 5.0, "mdd": -2.0},
    }
    ranked = rank_configs(results, metric="sharpe")
    assert ranked[0][0] == "dsr_only"
    assert ranked[-1][0] == "nextgen"


def test_identify_harmful_components():
    """Components whose addition makes Sharpe worse than baseline are harmful."""
    from src.utils.ablation_analysis import identify_harmful_components

    results = {
        "baseline":          {"sharpe": 0.5},
        "baseline+nextgen":  {"sharpe": -0.3},
        "baseline+cvar":     {"sharpe": 0.8},
        "baseline+feat_gate":{"sharpe": 0.1},
    }
    harmful = identify_harmful_components(results, baseline_key="baseline")
    assert "baseline+nextgen" in harmful
    assert "baseline+feat_gate" in harmful
    assert "baseline+cvar" not in harmful


def test_compute_component_delta():
    """Delta = config_sharpe - baseline_sharpe."""
    from src.utils.ablation_analysis import compute_component_delta

    assert compute_component_delta(1.5, 1.0) == pytest.approx(0.5)
    assert compute_component_delta(-0.5, 1.0) == pytest.approx(-1.5)
