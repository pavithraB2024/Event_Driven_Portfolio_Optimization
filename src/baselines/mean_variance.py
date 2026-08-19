"""
Mean-Variance Optimization (Markowitz Portfolio Theory)
Quadratic programming approach to portfolio allocation
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from scipy.optimize import minimize
import warnings


class MeanVarianceStrategy:
    """
    Markowitz Mean-Variance Optimization.

    Solves the quadratic programming problem:
        minimize: (1/2) * w^T * Σ * w - γ * μ^T * w
        subject to: w^T * 1 = 1, w >= 0

    where:
        w = portfolio weights
        Σ = covariance matrix
        μ = expected returns
        γ = risk aversion coefficient
    """

    def __init__(
        self,
        estimation_window: int = 252,
        rebalance_freq: int = 20,
        risk_aversion: float = 1.0,
        allow_short: bool = False,
        target_return: Optional[float] = None
    ):
        """
        Initialize Mean-Variance strategy.

        Args:
            estimation_window: Days to estimate μ and Σ
            rebalance_freq: Rebalancing frequency in days
            risk_aversion: Risk aversion coefficient (higher = more conservative)
            allow_short: Allow short selling (negative weights)
            target_return: Target return (if None, use risk aversion approach)
        """
        self.estimation_window = estimation_window
        self.rebalance_freq = rebalance_freq
        self.risk_aversion = risk_aversion
        self.allow_short = allow_short
        self.target_return = target_return

    def estimate_parameters(
        self,
        returns: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate expected returns and covariance matrix.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Tuple of (expected_returns, covariance_matrix)
        """
        # Expected returns (annualized mean)
        mu = returns.mean().values * 252

        # Covariance matrix (annualized)
        Sigma = returns.cov().values * 252

        return mu, Sigma

    def optimize_portfolio(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        risk_aversion: Optional[float] = None
    ) -> np.ndarray:
        """
        Solve mean-variance optimization problem.

        Args:
            mu: Expected returns vector
            Sigma: Covariance matrix
            risk_aversion: Risk aversion (if None, use self.risk_aversion)

        Returns:
            Optimal portfolio weights
        """
        n_assets = len(mu)

        if risk_aversion is None:
            risk_aversion = self.risk_aversion

        # Objective function: minimize (1/2) * w^T * Σ * w - γ * μ^T * w
        def objective(w):
            portfolio_variance = w.T @ Sigma @ w
            portfolio_return = mu.T @ w
            # Negative because we minimize (want to maximize return - risk)
            return 0.5 * portfolio_variance - risk_aversion * portfolio_return

        # Gradient
        def gradient(w):
            return Sigma @ w - risk_aversion * mu

        # Constraints: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]

        # If target return is specified, add as constraint
        if self.target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: mu.T @ w - self.target_return
            })

        # Bounds: no short selling unless allowed
        if self.allow_short:
            bounds = [(-1.0, 1.0) for _ in range(n_assets)]
        else:
            bounds = [(0.0, 1.0) for _ in range(n_assets)]

        # Initial guess: equal weights
        w0 = np.ones(n_assets) / n_assets

        # Solve optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective,
                w0,
                method='SLSQP',
                jac=gradient,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )

        if result.success:
            weights = result.x
            # Ensure weights are non-negative and sum to 1
            weights = np.maximum(weights, 0.0)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(n_assets) / n_assets
            return weights
        else:
            # Fallback to equal weights
            return np.ones(n_assets) / n_assets

    def allocate(
        self,
        returns: pd.DataFrame,
        current_date_idx: Optional[int] = None,
        risk_aversion: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate portfolio allocation using mean-variance optimization.

        Args:
            returns: DataFrame of asset returns
            current_date_idx: Current time index (for rolling estimation)
            risk_aversion: Override default risk aversion

        Returns:
            Array of portfolio weights
        """
        n_assets = returns.shape[1]

        if current_date_idx is None:
            current_date_idx = len(returns) - 1

        # Use rolling window for parameter estimation
        start_idx = max(0, current_date_idx - self.estimation_window)
        end_idx = current_date_idx

        historical_returns = returns.iloc[start_idx:end_idx]

        # Need sufficient data
        if len(historical_returns) < 20:
            return np.ones(n_assets) / n_assets

        # Estimate parameters
        mu, Sigma = self.estimate_parameters(historical_returns)

        # Regularize covariance matrix (add small diagonal term for stability)
        Sigma += np.eye(n_assets) * 1e-8

        # Optimize
        weights = self.optimize_portfolio(mu, Sigma, risk_aversion)

        return weights

    def backtest(
        self,
        returns: pd.DataFrame,
        initial_value: float = 100000.0,
        risk_aversion: Optional[float] = None,
        transaction_cost: float = 0.001
    ) -> Dict:
        """
        Backtest Mean-Variance strategy.

        Args:
            returns: DataFrame of asset returns
            initial_value: Initial portfolio value
            risk_aversion: Risk aversion coefficient (override default)
            transaction_cost: Transaction cost rate

        Returns:
            Dictionary with backtest results
        """
        if risk_aversion is None:
            risk_aversion = self.risk_aversion

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
                    current_date_idx=t,
                    risk_aversion=risk_aversion
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

    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate efficient frontier.

        Args:
            returns: DataFrame of asset returns
            n_points: Number of points on frontier

        Returns:
            Tuple of (risks, returns, weights_matrix)
        """
        mu, Sigma = self.estimate_parameters(returns)

        # Range of target returns
        min_return = mu.min()
        max_return = mu.max()
        target_returns = np.linspace(min_return, max_return, n_points)

        risks = []
        actual_returns = []
        weights_list = []

        for target in target_returns:
            # Set target return
            self.target_return = target

            # Optimize
            weights = self.optimize_portfolio(mu, Sigma)

            # Calculate risk and return
            portfolio_return = mu.T @ weights
            portfolio_risk = np.sqrt(weights.T @ Sigma @ weights)

            risks.append(portfolio_risk)
            actual_returns.append(portfolio_return)
            weights_list.append(weights)

        # Reset target return
        self.target_return = None

        return np.array(risks), np.array(actual_returns), np.array(weights_list)

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
    # Test Mean-Variance strategy
    print("Testing Mean-Variance Optimization Strategy...")

    # Generate mock returns
    np.random.seed(42)
    n_days = 1000
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    # Simulate correlated returns
    mean_returns = [0.0005, 0.0002, 0.0003]  # SPY, TLT, GLD
    cov_matrix = [
        [0.01**2, 0.00002, 0.00003],
        [0.00002, 0.005**2, 0.00001],
        [0.00003, 0.00001, 0.008**2]
    ]

    returns_data = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    returns = pd.DataFrame(
        returns_data,
        columns=['SPY', 'TLT', 'GLD'],
        index=dates
    )

    # Initialize strategy
    strategy = MeanVarianceStrategy(
        estimation_window=252,
        rebalance_freq=20,
        risk_aversion=2.0,
        allow_short=False
    )

    # Backtest
    results = strategy.backtest(
        returns=returns,
        initial_value=100000.0,
        risk_aversion=2.0
    )

    print("\nBacktest Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Avg Turnover: {results['avg_turnover']:.3f}")
    print(f"Final Value: ${results['final_value']:,.2f}")

    # Calculate efficient frontier
    print("\nCalculating Efficient Frontier...")
    risks, rets, weights = strategy.efficient_frontier(returns.iloc[:252], n_points=10)

    print("\nEfficient Frontier (first 5 points):")
    for i in range(min(5, len(risks))):
        print(f"Risk: {risks[i]:.4f}, Return: {rets[i]:.4f}, Weights: {weights[i]}")
