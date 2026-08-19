"""
Classical Merton Solution for Portfolio Optimization
Closed-form optimal allocation under log utility
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict


class MertonStrategy:
    """
    Merton's optimal portfolio allocation.

    Under log utility and geometric Brownian motion assumption:
    w* = (μ - r) / σ²

    where:
        w* = optimal fraction in risky asset
        μ = expected return of risky asset
        r = risk-free rate
        σ² = variance of risky asset
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        estimation_window: int = 252,
        rebalance_freq: int = 20
    ):
        """
        Initialize Merton strategy.

        Args:
            risk_free_rate: Annual risk-free rate
            estimation_window: Days to estimate μ and σ
            rebalance_freq: Rebalancing frequency in days
        """
        self.risk_free_rate = risk_free_rate
        self.estimation_window = estimation_window
        self.rebalance_freq = rebalance_freq

    def calculate_optimal_weight(
        self,
        returns: pd.Series,
        risk_aversion: float = 1.0
    ) -> float:
        """
        Calculate Merton optimal weight for risky asset.

        Args:
            returns: Historical returns of risky asset
            risk_aversion: Risk aversion coefficient (γ)

        Returns:
            Optimal weight for risky asset [0, 1]
        """
        # Estimate expected return (annualized)
        mu = returns.mean() * 252

        # Estimate variance (annualized)
        sigma_squared = returns.var() * 252

        # Risk-free rate (annual)
        r = self.risk_free_rate

        # Merton's formula
        if sigma_squared > 0:
            optimal_weight = (mu - r) / (risk_aversion * sigma_squared)
        else:
            optimal_weight = 0.0

        # Constrain to [0, 1] (no shorting, no leverage)
        optimal_weight = np.clip(optimal_weight, 0.0, 1.0)

        return optimal_weight

    def allocate(
        self,
        returns: pd.DataFrame,
        risk_aversion: float = 1.0,
        current_date_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate portfolio allocation across multiple assets.

        Args:
            returns: DataFrame of asset returns
            risk_aversion: Risk aversion coefficient
            current_date_idx: Current time index (for rolling estimation)

        Returns:
            Array of portfolio weights
        """
        n_assets = returns.shape[1]
        weights = np.zeros(n_assets)

        if current_date_idx is None:
            current_date_idx = len(returns) - 1

        # Use rolling window for parameter estimation
        start_idx = max(0, current_date_idx - self.estimation_window)
        end_idx = current_date_idx

        historical_returns = returns.iloc[start_idx:end_idx]

        # Calculate optimal weight for each asset separately
        # (Simplified: not considering correlation in this baseline)
        for i, col in enumerate(returns.columns):
            asset_returns = historical_returns[col].dropna()

            if len(asset_returns) > 20:  # Minimum data requirement
                weight = self.calculate_optimal_weight(
                    asset_returns,
                    risk_aversion=risk_aversion
                )
                weights[i] = weight

        # Normalize to sum to 1
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # Equal weight if no valid weights
            weights = np.ones(n_assets) / n_assets

        return weights

    def backtest(
        self,
        returns: pd.DataFrame,
        initial_value: float = 100000.0,
        risk_aversion: float = 1.0,
        transaction_cost: float = 0.001
    ) -> Dict:
        """
        Backtest Merton strategy.

        Args:
            returns: DataFrame of asset returns
            initial_value: Initial portfolio value
            risk_aversion: Risk aversion coefficient
            transaction_cost: Transaction cost rate

        Returns:
            Dictionary with backtest results
        """
        portfolio_values = [initial_value]
        portfolio_weights_history = []
        turnover_history = []

        current_weights = np.ones(returns.shape[1]) / returns.shape[1]
        current_value = initial_value

        for t in range(self.estimation_window, len(returns)):
            # Rebalance if needed
            if (t - self.estimation_window) % self.rebalance_freq == 0:
                # Calculate new optimal weights
                new_weights = self.allocate(
                    returns,
                    risk_aversion=risk_aversion,
                    current_date_idx=t
                )

                # Calculate turnover
                turnover = np.abs(new_weights - current_weights).sum()
                turnover_history.append(turnover)

                # Apply transaction costs
                transaction_costs = transaction_cost * turnover * current_value
                current_value -= transaction_costs

                # Update weights
                current_weights = new_weights
            else:
                turnover_history.append(0.0)

            # Calculate portfolio return
            period_returns = returns.iloc[t].values
            portfolio_return = np.dot(current_weights, period_returns)

            # Update portfolio value
            current_value *= (1 + portfolio_return)

            portfolio_values.append(current_value)
            portfolio_weights_history.append(current_weights.copy())

        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]

        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        avg_turnover = np.mean(turnover_history)

        results = {
            'portfolio_values': portfolio_values,
            'portfolio_weights': portfolio_weights_history,
            'portfolio_returns': portfolio_returns,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_turnover': avg_turnover,
            'final_value': portfolio_values[-1]
        }

        return results

    @staticmethod
    def _calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = returns - daily_rf
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return sharpe

    @staticmethod
    def _calculate_max_drawdown(values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax
        return abs(drawdown.min())


if __name__ == "__main__":
    # Test Merton strategy
    print("Testing Merton Strategy...")

    # Generate mock returns
    np.random.seed(42)
    n_days = 1000
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    returns = pd.DataFrame({
        'SPY': np.random.normal(0.0005, 0.01, n_days),  # 12.5% annual return, 16% vol
        'TLT': np.random.normal(0.0002, 0.005, n_days)   # 5% annual return, 8% vol
    }, index=dates)

    # Initialize strategy
    strategy = MertonStrategy(risk_free_rate=0.02)

    # Backtest
    results = strategy.backtest(
        returns=returns,
        initial_value=100000.0,
        risk_aversion=1.0
    )

    print("\nBacktest Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Avg Turnover: {results['avg_turnover']:.3f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
