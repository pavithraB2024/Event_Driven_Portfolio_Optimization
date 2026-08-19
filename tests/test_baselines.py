"""
Unit Tests for Baseline Portfolio Strategies
"""

import pytest
import numpy as np
import pandas as pd
from src.baselines import (
    MertonStrategy,
    MeanVarianceStrategy,
    EqualWeightStrategy,
    BuyAndHoldStrategy,
    RiskParityStrategy
)


@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    returns = pd.DataFrame({
        'SPY': np.random.normal(0.0005, 0.01, n_days),
        'TLT': np.random.normal(0.0002, 0.005, n_days),
        'GLD': np.random.normal(0.0003, 0.008, n_days)
    }, index=dates)

    return returns


@pytest.fixture
def four_asset_returns():
    """Generate 4-asset returns for testing."""
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    returns = pd.DataFrame({
        'SPY': np.random.normal(0.0005, 0.01, n_days),
        'TLT': np.random.normal(0.0002, 0.005, n_days),
        'GLD': np.random.normal(0.0003, 0.008, n_days),
        'BTC': np.random.normal(0.001, 0.03, n_days)
    }, index=dates)

    return returns


class TestMertonStrategy:
    """Test cases for Merton Strategy."""

    def test_initialization(self):
        """Test Merton strategy initialization."""
        strategy = MertonStrategy(
            risk_free_rate=0.02,
            estimation_window=252,
            rebalance_freq=20
        )
        assert strategy.risk_free_rate == 0.02
        assert strategy.estimation_window == 252
        assert strategy.rebalance_freq == 20

    def test_calculate_optimal_weight(self, sample_returns):
        """Test optimal weight calculation."""
        strategy = MertonStrategy()
        weight = strategy.calculate_optimal_weight(
            sample_returns['SPY'],
            risk_aversion=1.0
        )
        assert 0.0 <= weight <= 1.0
        assert isinstance(weight, float)

    def test_allocate(self, sample_returns):
        """Test portfolio allocation."""
        strategy = MertonStrategy()
        weights = strategy.allocate(sample_returns)
        assert len(weights) == len(sample_returns.columns)
        assert np.isclose(weights.sum(), 1.0)
        assert np.all(weights >= 0.0)
        assert np.all(weights <= 1.0)

    def test_backtest(self, sample_returns):
        """Test backtesting functionality."""
        strategy = MertonStrategy()
        results = strategy.backtest(
            sample_returns,
            initial_value=100000.0,
            risk_aversion=1.0,
            transaction_cost=0.0025
        )

        # Check required keys
        assert 'portfolio_values' in results
        assert 'portfolio_returns' in results
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'avg_turnover' in results
        assert 'final_value' in results

        # Check values
        assert results['final_value'] > 0
        assert isinstance(results['sharpe_ratio'], float)
        assert 0.0 <= results['max_drawdown'] <= 1.0
        assert results['avg_turnover'] >= 0.0

    def test_edge_case_zero_variance(self):
        """Test behavior with zero variance."""
        strategy = MertonStrategy()
        zero_var_returns = pd.Series([0.0] * 100)
        weight = strategy.calculate_optimal_weight(zero_var_returns)
        assert weight == 0.0


class TestMeanVarianceStrategy:
    """Test cases for Mean-Variance Strategy."""

    def test_initialization(self):
        """Test Mean-Variance strategy initialization."""
        strategy = MeanVarianceStrategy(
            estimation_window=252,
            rebalance_freq=20,
            risk_aversion=2.0,
            allow_short=False
        )
        assert strategy.estimation_window == 252
        assert strategy.rebalance_freq == 20
        assert strategy.risk_aversion == 2.0
        assert strategy.allow_short is False

    def test_estimate_parameters(self, sample_returns):
        """Test parameter estimation."""
        strategy = MeanVarianceStrategy()
        mu, Sigma = strategy.estimate_parameters(sample_returns)

        assert len(mu) == len(sample_returns.columns)
        assert Sigma.shape == (len(sample_returns.columns), len(sample_returns.columns))
        assert np.allclose(Sigma, Sigma.T)  # Covariance matrix should be symmetric

    def test_optimize_portfolio(self, sample_returns):
        """Test portfolio optimization."""
        strategy = MeanVarianceStrategy()
        mu, Sigma = strategy.estimate_parameters(sample_returns)
        weights = strategy.optimize_portfolio(mu, Sigma)

        assert len(weights) == len(sample_returns.columns)
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        assert np.all(weights >= -1e-6)  # Allow small negative due to numerical precision
        assert np.all(weights <= 1.0 + 1e-6)

    def test_allocate(self, sample_returns):
        """Test allocation."""
        strategy = MeanVarianceStrategy()
        weights = strategy.allocate(sample_returns, current_date_idx=300)

        assert len(weights) == len(sample_returns.columns)
        assert np.isclose(weights.sum(), 1.0)
        assert np.all(weights >= 0.0)

    def test_backtest(self, sample_returns):
        """Test backtesting."""
        strategy = MeanVarianceStrategy(risk_aversion=2.0)
        results = strategy.backtest(
            sample_returns,
            initial_value=100000.0
        )

        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert results['final_value'] > 0

    def test_efficient_frontier(self, sample_returns):
        """Test efficient frontier calculation."""
        strategy = MeanVarianceStrategy()
        risks, returns, weights = strategy.efficient_frontier(sample_returns, n_points=10)

        assert len(risks) == 10
        assert len(returns) == 10
        assert weights.shape == (10, len(sample_returns.columns))


class TestEqualWeightStrategy:
    """Test cases for Equal-Weight Strategy."""

    def test_initialization(self):
        """Test Equal-Weight initialization."""
        strategy = EqualWeightStrategy(rebalance_freq=20)
        assert strategy.rebalance_freq == 20

    def test_allocate(self):
        """Test equal allocation."""
        strategy = EqualWeightStrategy()
        weights = strategy.allocate(n_assets=4)

        assert len(weights) == 4
        assert np.allclose(weights, 0.25)
        assert np.isclose(weights.sum(), 1.0)

    def test_backtest(self, four_asset_returns):
        """Test backtesting."""
        strategy = EqualWeightStrategy(rebalance_freq=20)
        results = strategy.backtest(
            four_asset_returns,
            initial_value=100000.0
        )

        assert results['final_value'] > 0
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert results['avg_turnover'] >= 0.0


class TestBuyAndHoldStrategy:
    """Test cases for Buy-and-Hold Strategy."""

    def test_initialization(self):
        """Test Buy-and-Hold initialization."""
        strategy = BuyAndHoldStrategy()
        assert strategy.target_allocation is None

    def test_initialization_with_allocation(self):
        """Test initialization with custom allocation."""
        custom = np.array([0.6, 0.4])
        strategy = BuyAndHoldStrategy(target_allocation=custom)
        assert np.array_equal(strategy.target_allocation, custom)

    def test_allocate_default_two_assets(self):
        """Test default 60/40 allocation."""
        strategy = BuyAndHoldStrategy()
        weights = strategy.allocate(n_assets=2)

        assert len(weights) == 2
        assert weights[0] == 0.6
        assert weights[1] == 0.4
        assert np.isclose(weights.sum(), 1.0)

    def test_allocate_default_four_assets(self):
        """Test default 4-asset allocation."""
        strategy = BuyAndHoldStrategy()
        weights = strategy.allocate(n_assets=4)

        assert len(weights) == 4
        assert np.isclose(weights.sum(), 1.0)
        assert weights[0] == 0.50  # SPY
        assert weights[1] == 0.30  # TLT
        assert weights[2] == 0.15  # GLD
        assert weights[3] == 0.05  # BTC

    def test_backtest(self, four_asset_returns):
        """Test backtesting."""
        strategy = BuyAndHoldStrategy()
        results = strategy.backtest(
            four_asset_returns,
            initial_value=100000.0
        )

        assert results['final_value'] > 0
        assert results['avg_turnover'] == 0.0  # No rebalancing
        assert 'sharpe_ratio' in results


class TestRiskParityStrategy:
    """Test cases for Risk Parity Strategy."""

    def test_initialization(self):
        """Test Risk Parity initialization."""
        strategy = RiskParityStrategy(
            estimation_window=60,
            rebalance_freq=20
        )
        assert strategy.estimation_window == 60
        assert strategy.rebalance_freq == 20

    def test_allocate(self, sample_returns):
        """Test allocation."""
        strategy = RiskParityStrategy()
        weights = strategy.allocate(sample_returns, current_date_idx=100)

        assert len(weights) == len(sample_returns.columns)
        assert np.isclose(weights.sum(), 1.0)
        assert np.all(weights >= 0.0)
        assert np.all(weights <= 1.0)

    def test_backtest(self, four_asset_returns):
        """Test backtesting."""
        strategy = RiskParityStrategy()
        results = strategy.backtest(
            four_asset_returns,
            initial_value=100000.0
        )

        assert results['final_value'] > 0
        assert 'sharpe_ratio' in results
        assert results['avg_turnover'] >= 0.0

    def test_inverse_volatility_weighting(self, sample_returns):
        """Test that higher volatility assets get lower weights."""
        strategy = RiskParityStrategy()

        # Create returns with different volatilities
        high_vol = pd.DataFrame({
            'low_vol': np.random.normal(0.001, 0.005, 200),
            'high_vol': np.random.normal(0.001, 0.02, 200)
        })

        weights = strategy.allocate(high_vol, current_date_idx=100)

        # Low volatility asset should have higher weight
        assert weights[0] > weights[1]


class TestStrategyComparison:
    """Integration tests comparing strategies."""

    def test_all_strategies_produce_results(self, four_asset_returns):
        """Test that all strategies can run successfully."""
        strategies = {
            'Merton': MertonStrategy(),
            'Mean-Variance': MeanVarianceStrategy(),
            'Equal-Weight': EqualWeightStrategy(),
            'Buy-and-Hold': BuyAndHoldStrategy(),
            'Risk Parity': RiskParityStrategy()
        }

        results = {}
        for name, strategy in strategies.items():
            results[name] = strategy.backtest(
                four_asset_returns,
                initial_value=100000.0
            )

        # All should complete
        assert len(results) == 5

        # All should have positive final values
        for name, res in results.items():
            assert res['final_value'] > 0, f"{name} failed"

    def test_metrics_consistency(self, sample_returns):
        """Test that metrics are consistent across strategies."""
        strategy = EqualWeightStrategy()
        results = strategy.backtest(sample_returns)

        # Sharpe ratio should be reasonable
        assert -5.0 <= results['sharpe_ratio'] <= 10.0

        # Max drawdown should be between 0 and 100%
        assert 0.0 <= results['max_drawdown'] <= 1.0

        # Total return should be reasonable
        assert -0.99 <= results['total_return'] <= 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
