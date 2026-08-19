"""
Naive Portfolio Allocation Strategies
Simple baseline strategies for benchmarking
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict


class EqualWeightStrategy:
    """
    Equal-Weight (1/N) Portfolio Strategy.

    Allocates equal weight to all assets regardless of market conditions.
    This is a surprisingly robust benchmark that often outperforms
    more sophisticated strategies due to diversification benefits.
    """

    def __init__(self, rebalance_freq: int = 20):
        """
        Initialize Equal-Weight strategy.

        Args:
            rebalance_freq: Rebalancing frequency in days
        """
        self.rebalance_freq = rebalance_freq

    def allocate(self, n_assets: int) -> np.ndarray:
        """
        Calculate equal-weight allocation.

        Args:
            n_assets: Number of assets

        Returns:
            Array of portfolio weights (all equal)
        """
        return np.ones(n_assets) / n_assets

    def backtest(
        self,
        returns: pd.DataFrame,
        initial_value: float = 100000.0,
        transaction_cost: float = 0.001
    ) -> Dict:
        """
        Backtest Equal-Weight strategy.

        Args:
            returns: DataFrame of asset returns
            initial_value: Initial portfolio value
            transaction_cost: Transaction cost rate

        Returns:
            Dictionary with backtest results
        """
        n_assets = returns.shape[1]
        portfolio_values = [initial_value]
        portfolio_weights_history = []
        turnover_history = []

        # Equal weights throughout
        equal_weights = self.allocate(n_assets)
        current_weights = equal_weights.copy()
        current_value = initial_value

        for t in range(len(returns)):
            # Rebalance if needed
            if t % self.rebalance_freq == 0 and t > 0:
                # Equal weights don't change, but we rebalance due to drift
                new_weights = equal_weights.copy()

                # Calculate turnover (drift from equal weights)
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

            # Weights drift due to different asset returns
            # w_new = w_old * (1 + r) / (1 + r_portfolio)
            current_weights = current_weights * (1 + period_returns) / (1 + portfolio_return)

            portfolio_values.append(current_value)
            portfolio_weights_history.append(current_weights.copy())

        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]

        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        avg_turnover = np.mean(turnover_history) if turnover_history else 0.0

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


class BuyAndHoldStrategy:
    """
    Buy-and-Hold Portfolio Strategy.

    Sets initial allocation and never rebalances (except for initial allocation).
    Classic strategies include 60/40 stocks/bonds or custom allocations.
    """

    def __init__(self, target_allocation: Optional[np.ndarray] = None):
        """
        Initialize Buy-and-Hold strategy.

        Args:
            target_allocation: Initial target allocation (if None, use 60/40 for 2 assets or equal for more)
        """
        self.target_allocation = target_allocation

    def allocate(self, n_assets: int) -> np.ndarray:
        """
        Calculate initial allocation.

        Args:
            n_assets: Number of assets

        Returns:
            Array of portfolio weights
        """
        if self.target_allocation is not None:
            # Use provided allocation
            weights = self.target_allocation.copy()
            # Normalize
            if weights.sum() > 0:
                weights = weights / weights.sum()
            return weights
        elif n_assets == 2:
            # Default 60/40 for two assets (stocks/bonds)
            return np.array([0.6, 0.4])
        elif n_assets == 4:
            # For typical 4-asset portfolio: SPY, TLT, GLD, BTC
            # 50% stocks, 30% bonds, 15% gold, 5% crypto
            return np.array([0.50, 0.30, 0.15, 0.05])
        else:
            # Equal weight for other cases
            return np.ones(n_assets) / n_assets

    def backtest(
        self,
        returns: pd.DataFrame,
        initial_value: float = 100000.0,
        transaction_cost: float = 0.001
    ) -> Dict:
        """
        Backtest Buy-and-Hold strategy.

        Args:
            returns: DataFrame of asset returns
            initial_value: Initial portfolio value
            transaction_cost: Transaction cost rate (applied once at start)

        Returns:
            Dictionary with backtest results
        """
        n_assets = returns.shape[1]
        portfolio_values = [initial_value]
        portfolio_weights_history = []
        turnover_history = [0.0]  # No turnover after initial allocation

        # Initial allocation
        initial_weights = self.allocate(n_assets)
        current_weights = initial_weights.copy()

        # Apply initial transaction cost (from cash to assets)
        initial_turnover = initial_weights.sum()  # Should be 1.0
        initial_transaction_cost = transaction_cost * initial_turnover * initial_value
        current_value = initial_value - initial_transaction_cost

        for t in range(len(returns)):
            # No rebalancing in buy-and-hold

            # Calculate portfolio return
            period_returns = returns.iloc[t].values
            portfolio_return = np.dot(current_weights, period_returns)

            # Update portfolio value
            current_value *= (1 + portfolio_return)

            # Weights drift due to different asset returns
            # w_new = w_old * (1 + r) / (1 + r_portfolio)
            current_weights = current_weights * (1 + period_returns) / (1 + portfolio_return)

            portfolio_values.append(current_value)
            portfolio_weights_history.append(current_weights.copy())
            turnover_history.append(0.0)

        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]

        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        avg_turnover = 0.0  # No turnover after initial allocation

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


class RiskParityStrategy:
    """
    Risk Parity Portfolio Strategy.

    Allocates capital such that each asset contributes equally to portfolio risk.
    This is a more sophisticated naive strategy that balances risk rather than capital.
    """

    def __init__(self, estimation_window: int = 60, rebalance_freq: int = 20):
        """
        Initialize Risk Parity strategy.

        Args:
            estimation_window: Days to estimate volatility
            rebalance_freq: Rebalancing frequency in days
        """
        self.estimation_window = estimation_window
        self.rebalance_freq = rebalance_freq

    def allocate(self, returns: pd.DataFrame, current_date_idx: Optional[int] = None) -> np.ndarray:
        """
        Calculate risk parity allocation (inverse volatility weighting).

        Args:
            returns: DataFrame of asset returns
            current_date_idx: Current time index

        Returns:
            Array of portfolio weights
        """
        n_assets = returns.shape[1]

        if current_date_idx is None:
            current_date_idx = len(returns) - 1

        # Use rolling window for volatility estimation
        start_idx = max(0, current_date_idx - self.estimation_window)
        end_idx = current_date_idx

        historical_returns = returns.iloc[start_idx:end_idx]

        if len(historical_returns) < 10:
            # Fallback to equal weights
            return np.ones(n_assets) / n_assets

        # Calculate volatilities (annualized)
        volatilities = historical_returns.std().values * np.sqrt(252)

        # Inverse volatility weighting
        # w_i = (1/σ_i) / Σ(1/σ_j)
        if np.all(volatilities > 0):
            inv_vol = 1.0 / volatilities
            weights = inv_vol / inv_vol.sum()
        else:
            weights = np.ones(n_assets) / n_assets

        return weights

    def backtest(
        self,
        returns: pd.DataFrame,
        initial_value: float = 100000.0,
        transaction_cost: float = 0.001
    ) -> Dict:
        """
        Backtest Risk Parity strategy.

        Args:
            returns: DataFrame of asset returns
            initial_value: Initial portfolio value
            transaction_cost: Transaction cost rate

        Returns:
            Dictionary with backtest results
        """
        n_assets = returns.shape[1]
        portfolio_values = [initial_value]
        portfolio_weights_history = []
        turnover_history = []

        current_weights = np.ones(n_assets) / n_assets
        current_value = initial_value

        for t in range(len(returns)):
            # Rebalance if needed
            if t >= self.estimation_window and t % self.rebalance_freq == 0:
                # Calculate new risk parity weights
                new_weights = self.allocate(returns, current_date_idx=t)

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

            # Weights drift
            current_weights = current_weights * (1 + period_returns) / (1 + portfolio_return)

            portfolio_values.append(current_value)
            portfolio_weights_history.append(current_weights.copy())

        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]

        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        avg_turnover = np.mean(turnover_history) if turnover_history else 0.0

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
    # Test naive strategies
    print("Testing Naive Strategies...")

    # Generate mock returns
    np.random.seed(42)
    n_days = 1000
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    returns = pd.DataFrame({
        'SPY': np.random.normal(0.0005, 0.01, n_days),
        'TLT': np.random.normal(0.0002, 0.005, n_days),
        'GLD': np.random.normal(0.0003, 0.008, n_days),
        'BTC': np.random.normal(0.001, 0.03, n_days)
    }, index=dates)

    print("\n" + "="*60)
    print("1. EQUAL-WEIGHT STRATEGY (1/N)")
    print("="*60)
    eq_strategy = EqualWeightStrategy(rebalance_freq=20)
    eq_results = eq_strategy.backtest(returns, initial_value=100000.0)

    print(f"Total Return: {eq_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {eq_results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {eq_results['max_drawdown']:.2%}")
    print(f"Avg Turnover: {eq_results['avg_turnover']:.3f}")
    print(f"Final Value: ${eq_results['final_value']:,.2f}")

    print("\n" + "="*60)
    print("2. BUY-AND-HOLD STRATEGY (50/30/15/5)")
    print("="*60)
    bh_strategy = BuyAndHoldStrategy()
    bh_results = bh_strategy.backtest(returns, initial_value=100000.0)

    print(f"Initial Allocation: {bh_strategy.allocate(4)}")
    print(f"Total Return: {bh_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {bh_results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {bh_results['max_drawdown']:.2%}")
    print(f"Avg Turnover: {bh_results['avg_turnover']:.3f}")
    print(f"Final Value: ${bh_results['final_value']:,.2f}")

    print("\n" + "="*60)
    print("3. RISK PARITY STRATEGY (Inverse Volatility)")
    print("="*60)
    rp_strategy = RiskParityStrategy(estimation_window=60, rebalance_freq=20)
    rp_results = rp_strategy.backtest(returns, initial_value=100000.0)

    print(f"Total Return: {rp_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {rp_results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {rp_results['max_drawdown']:.2%}")
    print(f"Avg Turnover: {rp_results['avg_turnover']:.3f}")
    print(f"Final Value: ${rp_results['final_value']:,.2f}")

    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    comparison = pd.DataFrame({
        'Equal-Weight': [
            eq_results['total_return'],
            eq_results['sharpe_ratio'],
            eq_results['max_drawdown'],
            eq_results['avg_turnover']
        ],
        'Buy-and-Hold': [
            bh_results['total_return'],
            bh_results['sharpe_ratio'],
            bh_results['max_drawdown'],
            bh_results['avg_turnover']
        ],
        'Risk Parity': [
            rp_results['total_return'],
            rp_results['sharpe_ratio'],
            rp_results['max_drawdown'],
            rp_results['avg_turnover']
        ]
    }, index=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Avg Turnover'])

    print(comparison)
