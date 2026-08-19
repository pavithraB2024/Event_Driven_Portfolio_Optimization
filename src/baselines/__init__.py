"""
Baseline Portfolio Strategies

This module provides classical portfolio optimization strategies for benchmarking:
- Merton Solution (closed-form optimal allocation)
- Mean-Variance Optimization (Markowitz)
- Naive Strategies (Equal-weight, Buy-and-Hold, Risk Parity)
"""

from .merton_strategy import MertonStrategy
from .mean_variance import MeanVarianceStrategy
from .naive_strategies import (
    EqualWeightStrategy,
    BuyAndHoldStrategy,
    RiskParityStrategy
)

__all__ = [
    'MertonStrategy',
    'MeanVarianceStrategy',
    'EqualWeightStrategy',
    'BuyAndHoldStrategy',
    'RiskParityStrategy'
]
