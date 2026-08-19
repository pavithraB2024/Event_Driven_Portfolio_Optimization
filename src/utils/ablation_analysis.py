"""
Functional Core: Pure analysis functions for ablation study results.
No I/O, no side effects — just data in, data out.
"""
from typing import Dict, List, Tuple


def compute_component_delta(config_sharpe: float, baseline_sharpe: float) -> float:
    """Compute the marginal contribution of a component vs baseline."""
    return config_sharpe - baseline_sharpe


def rank_configs(
    results: Dict[str, Dict[str, float]],
    metric: str = "sharpe",
) -> List[Tuple[str, Dict[str, float]]]:
    """Rank ablation configs by a given metric (descending)."""
    return sorted(results.items(), key=lambda kv: kv[1][metric], reverse=True)


def identify_harmful_components(
    results: Dict[str, Dict[str, float]],
    baseline_key: str = "baseline",
) -> List[str]:
    """Return config names whose Sharpe is strictly worse than baseline."""
    baseline_sharpe = results[baseline_key]["sharpe"]
    return [
        name for name, metrics in results.items()
        if name != baseline_key and metrics["sharpe"] < baseline_sharpe
    ]
