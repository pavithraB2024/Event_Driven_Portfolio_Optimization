"""
Strategy Adapters for Backtesting Engine

Adapters to connect baseline strategies to the BacktestEngine interface.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from typing import Optional

from src.backtesting.backtest_engine import Strategy
from src.baselines import (
    MertonStrategy,
    MeanVarianceStrategy,
    EqualWeightStrategy,
    BuyAndHoldStrategy,
    RiskParityStrategy
)


class MertonStrategyAdapter(Strategy):
    """Adapter for Merton strategy."""

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        estimation_window: int = 252,
        risk_aversion: float = 1.0
    ):
        self.strategy = MertonStrategy(
            risk_free_rate=risk_free_rate,
            estimation_window=estimation_window,
            rebalance_freq=1  # Rebalancing handled by engine
        )
        self.risk_aversion = risk_aversion

    @property
    def name(self) -> str:
        return "Merton"

    def allocate(
        self,
        data: pd.DataFrame,
        current_idx: int,
        current_weights: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Compute Merton optimal weights."""
        # Extract returns from data
        return_cols = [col for col in data.columns if col.startswith('return_')]
        if not return_cols:
            # Fallback: equal weights
            return np.ones(len(current_weights)) / len(current_weights)

        returns = data[return_cols].copy()
        returns.columns = [col.replace('return_', '') for col in returns.columns]

        # Allocate using Merton strategy
        weights = self.strategy.allocate(
            returns=returns,
            risk_aversion=self.risk_aversion,
            current_date_idx=current_idx
        )

        return weights


class MeanVarianceAdapter(Strategy):
    """Adapter for Mean-Variance strategy."""

    def __init__(
        self,
        estimation_window: int = 252,
        risk_aversion: float = 2.0,
        allow_short: bool = False
    ):
        self.strategy = MeanVarianceStrategy(
            estimation_window=estimation_window,
            rebalance_freq=1,  # Rebalancing handled by engine
            risk_aversion=risk_aversion,
            allow_short=allow_short
        )

    @property
    def name(self) -> str:
        return "Mean-Variance"

    def allocate(
        self,
        data: pd.DataFrame,
        current_idx: int,
        current_weights: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Compute mean-variance optimal weights."""
        # Extract returns from data
        return_cols = [col for col in data.columns if col.startswith('return_')]
        if not return_cols:
            return np.ones(len(current_weights)) / len(current_weights)

        returns = data[return_cols].copy()
        returns.columns = [col.replace('return_', '') for col in returns.columns]

        # Allocate
        weights = self.strategy.allocate(
            returns=returns,
            current_date_idx=current_idx
        )

        return weights


class EqualWeightAdapter(Strategy):
    """Adapter for Equal-Weight strategy."""

    def __init__(self):
        self.strategy = EqualWeightStrategy(rebalance_freq=1)

    @property
    def name(self) -> str:
        return "Equal-Weight"

    def allocate(
        self,
        data: pd.DataFrame,
        current_idx: int,
        current_weights: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Compute equal weights."""
        n_assets = len(current_weights)
        return self.strategy.allocate(n_assets)


class BuyAndHoldAdapter(Strategy):
    """Adapter for Buy-and-Hold strategy."""

    def __init__(self, target_allocation: Optional[np.ndarray] = None):
        self.strategy = BuyAndHoldStrategy(target_allocation=target_allocation)
        self._initial_allocation = None

    @property
    def name(self) -> str:
        return "Buy-and-Hold"

    def allocate(
        self,
        data: pd.DataFrame,
        current_idx: int,
        current_weights: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Return initial allocation on first call, then maintain current weights.

        Note: Buy-and-hold doesn't rebalance, so we return current drifted weights
        after the initial allocation.
        """
        if self._initial_allocation is None:
            # First call: set initial allocation
            n_assets = len(current_weights)
            self._initial_allocation = self.strategy.allocate(n_assets)
            return self._initial_allocation
        else:
            # Subsequent calls: maintain current weights (let them drift)
            return current_weights


class RiskParityAdapter(Strategy):
    """Adapter for Risk Parity strategy."""

    def __init__(self, estimation_window: int = 60):
        self.strategy = RiskParityStrategy(
            estimation_window=estimation_window,
            rebalance_freq=1  # Rebalancing handled by engine
        )

    @property
    def name(self) -> str:
        return "Risk Parity"

    def allocate(
        self,
        data: pd.DataFrame,
        current_idx: int,
        current_weights: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Compute risk parity weights."""
        # Extract returns from data
        return_cols = [col for col in data.columns if col.startswith('return_')]
        if not return_cols:
            return np.ones(len(current_weights)) / len(current_weights)

        returns = data[return_cols].copy()
        returns.columns = [col.replace('return_', '') for col in returns.columns]

        # Allocate
        weights = self.strategy.allocate(
            returns=returns,
            current_date_idx=current_idx
        )

        return weights


# Factory function for convenience
def create_strategy_adapter(
    strategy_name: str,
    **kwargs
) -> Strategy:
    """
    Factory function to create strategy adapters.

    Args:
        strategy_name: Name of strategy ("merton", "mean_variance", "equal_weight", "buy_hold", "risk_parity")
        **kwargs: Strategy-specific parameters

    Returns:
        Strategy adapter instance
    """
    strategy_map = {
        'merton': MertonStrategyAdapter,
        'mean_variance': MeanVarianceAdapter,
        'equal_weight': EqualWeightAdapter,
        'buy_hold': BuyAndHoldAdapter,
        'buy_and_hold': BuyAndHoldAdapter,
        'risk_parity': RiskParityAdapter
    }

    strategy_class = strategy_map.get(strategy_name.lower())
    if strategy_class is None:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategy_map.keys())}")

    return strategy_class(**kwargs)


if __name__ == "__main__":
    # Test adapters
    print("Testing Strategy Adapters...")

    # Generate mock data
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    data = pd.DataFrame({
        'return_SPY': np.random.normal(0.0005, 0.01, n_days),
        'return_TLT': np.random.normal(0.0002, 0.005, n_days),
        'return_GLD': np.random.normal(0.0003, 0.008, n_days)
    }, index=dates)

    # Test each adapter
    adapters = {
        'Merton': MertonStrategyAdapter(),
        'Mean-Variance': MeanVarianceAdapter(),
        'Equal-Weight': EqualWeightAdapter(),
        'Buy-and-Hold': BuyAndHoldAdapter(),
        'Risk Parity': RiskParityAdapter()
    }

    print("\nTesting allocation at t=300:")
    for name, adapter in adapters.items():
        weights = adapter.allocate(
            data=data,
            current_idx=300,
            current_weights=np.array([0.33, 0.33, 0.34])
        )
        print(f"{name:15s}: {weights}")

    # Test factory
    print("\nTesting factory function:")
    strategy = create_strategy_adapter('equal_weight')
    print(f"Created: {strategy.name}")
