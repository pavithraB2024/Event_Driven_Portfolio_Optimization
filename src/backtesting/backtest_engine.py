"""
Backtesting Engine for Portfolio Strategies

Core framework for simulating portfolio strategies with:
- Position tracking and rebalancing
- Transaction cost accounting
- Slippage modeling
- Performance metrics calculation
- Multi-strategy comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.0025  # 0.25% per trade
    slippage: float = 0.0005  # 0.05% slippage
    rebalance_frequency: int = 1  # Daily rebalancing
    risk_free_rate: float = 0.02  # Annual risk-free rate
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class TradeRecord:
    """Record of a single trade."""
    date: pd.Timestamp
    asset: str
    old_weight: float
    new_weight: float
    turnover: float
    transaction_cost: float
    slippage_cost: float
    portfolio_value: float


@dataclass
class BacktestResults:
    """Results from a backtest run."""
    # Time series
    portfolio_values: pd.Series
    portfolio_returns: pd.Series
    weights_history: pd.DataFrame
    drawdown_series: pd.Series

    # Scalar metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    avg_turnover: float
    total_transaction_costs: float

    # Trade history
    trades: List[TradeRecord]

    # Config
    config: BacktestConfig
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_periods: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'win_rate': self.win_rate,
            'avg_turnover': self.avg_turnover,
            'total_transaction_costs': self.total_transaction_costs,
            'final_value': self.portfolio_values.iloc[-1],
            'n_periods': self.n_periods,
            'n_trades': len(self.trades)
        }


class Strategy(ABC):
    """Abstract base class for portfolio strategies."""

    @abstractmethod
    def allocate(
        self,
        data: pd.DataFrame,
        current_idx: int,
        current_weights: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute target portfolio weights.

        Args:
            data: Historical market data
            current_idx: Current time index
            current_weights: Current portfolio weights
            **kwargs: Additional strategy-specific parameters

        Returns:
            Array of target weights (must sum to 1)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass


class BacktestEngine:
    """
    Core backtesting engine for portfolio strategies.

    Handles:
    - Portfolio value tracking
    - Weight updates and rebalancing
    - Transaction costs and slippage
    - Performance metrics calculation
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtesting engine.

        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()

    def _calculate_transaction_costs(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float
    ) -> Tuple[float, float]:
        """
        Calculate transaction costs and slippage.

        Args:
            old_weights: Current portfolio weights
            new_weights: Target portfolio weights
            portfolio_value: Current portfolio value

        Returns:
            Tuple of (transaction_cost, slippage_cost)
        """
        # Turnover (sum of absolute weight changes)
        turnover = np.abs(new_weights - old_weights).sum()

        # Transaction costs (proportional to turnover)
        transaction_cost = self.config.transaction_cost * turnover * portfolio_value

        # Slippage (proportional to turnover)
        slippage_cost = self.config.slippage * turnover * portfolio_value

        return transaction_cost, slippage_cost

    def _calculate_metrics(
        self,
        portfolio_values: pd.Series,
        portfolio_returns: pd.Series,
        weights_history: pd.DataFrame,
        trades: List[TradeRecord]
    ) -> Dict:
        """Calculate performance metrics."""

        # Return metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        n_years = len(portfolio_values) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        # Risk metrics
        volatility = portfolio_returns.std() * np.sqrt(252)

        # Sharpe ratio
        daily_rf = (1 + self.config.risk_free_rate) ** (1/252) - 1
        excess_returns = portfolio_returns - daily_rf
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0.0

        # Sortino ratio
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            downside_dev = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (excess_returns.mean() * 252) / downside_dev
        else:
            sortino_ratio = 0.0

        # Drawdown
        cumulative = portfolio_values
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

        # Win rate
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns) if len(portfolio_returns) > 0 else 0.0

        # Turnover
        turnovers = [trade.turnover for trade in trades]
        avg_turnover = np.mean(turnovers) if turnovers else 0.0

        # Transaction costs
        total_transaction_costs = sum(trade.transaction_cost + trade.slippage_cost for trade in trades)

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'avg_turnover': avg_turnover,
            'total_transaction_costs': total_transaction_costs,
            'drawdown_series': drawdown
        }

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        returns: pd.DataFrame,
        **strategy_kwargs
    ) -> BacktestResults:
        """
        Run backtest for a given strategy.

        Args:
            strategy: Strategy instance
            data: Historical market data (prices, features, etc.)
            returns: Asset returns DataFrame
            **strategy_kwargs: Additional strategy-specific parameters

        Returns:
            BacktestResults object
        """
        # Initialize
        n_assets = len(returns.columns)
        n_periods = len(returns)

        # Filter date range if specified
        if self.config.start_date:
            returns = returns[returns.index >= self.config.start_date]
            data = data[data.index >= self.config.start_date]
        if self.config.end_date:
            returns = returns[returns.index <= self.config.end_date]
            data = data[data.index <= self.config.end_date]

        n_periods = len(returns)

        # Storage
        portfolio_values = [self.config.initial_capital]
        weights_history = []
        trades = []

        # Initial allocation
        current_weights = strategy.allocate(
            data=data,
            current_idx=0,
            current_weights=np.ones(n_assets) / n_assets,
            **strategy_kwargs
        )

        # Initial transaction costs
        current_weights.sum()  # From cash to assets
        initial_tc, initial_slip = self._calculate_transaction_costs(
            np.zeros(n_assets),
            current_weights,
            self.config.initial_capital
        )

        current_value = self.config.initial_capital - initial_tc - initial_slip
        weights_history.append(current_weights.copy())

        # Record initial trade
        for i, asset in enumerate(returns.columns):
            trades.append(TradeRecord(
                date=returns.index[0],
                asset=asset,
                old_weight=0.0,
                new_weight=current_weights[i],
                turnover=current_weights[i],
                transaction_cost=initial_tc * (current_weights[i] / current_weights.sum()),
                slippage_cost=initial_slip * (current_weights[i] / current_weights.sum()),
                portfolio_value=current_value
            ))

        # Simulation loop
        for t in range(n_periods):
            # Get period returns
            period_returns = returns.iloc[t].values

            # Calculate portfolio return
            portfolio_return = np.dot(current_weights, period_returns)

            # Update portfolio value
            current_value *= (1 + portfolio_return)

            # Weights drift due to different asset returns
            # w_new = w_old * (1 + r) / (1 + r_portfolio)
            drifted_weights = current_weights * (1 + period_returns) / (1 + portfolio_return)

            # Check if rebalancing is needed
            should_rebalance = (t % self.config.rebalance_frequency == 0) and (t > 0)

            if should_rebalance:
                # Get new target weights from strategy
                new_weights = strategy.allocate(
                    data=data,
                    current_idx=t,
                    current_weights=drifted_weights,
                    **strategy_kwargs
                )

                # Calculate costs
                tc, slip = self._calculate_transaction_costs(
                    drifted_weights,
                    new_weights,
                    current_value
                )

                # Apply costs
                current_value -= (tc + slip)

                # Record trade
                turnover = np.abs(new_weights - drifted_weights).sum()
                for i, asset in enumerate(returns.columns):
                    if abs(new_weights[i] - drifted_weights[i]) > 1e-6:
                        trades.append(TradeRecord(
                            date=returns.index[t],
                            asset=asset,
                            old_weight=drifted_weights[i],
                            new_weight=new_weights[i],
                            turnover=abs(new_weights[i] - drifted_weights[i]),
                            transaction_cost=tc * abs(new_weights[i] - drifted_weights[i]) / turnover,
                            slippage_cost=slip * abs(new_weights[i] - drifted_weights[i]) / turnover,
                            portfolio_value=current_value
                        ))

                # Update weights
                current_weights = new_weights
            else:
                # No rebalancing, weights drift
                current_weights = drifted_weights

            # Store
            portfolio_values.append(current_value)
            weights_history.append(current_weights.copy())

        # Convert to Series/DataFrame
        portfolio_values = pd.Series(portfolio_values, index=returns.index.insert(0, returns.index[0]))
        portfolio_returns = portfolio_values.pct_change().dropna()
        weights_df = pd.DataFrame(
            weights_history,
            columns=returns.columns,
            index=returns.index.insert(0, returns.index[0])
        )

        # Calculate metrics
        metrics = self._calculate_metrics(
            portfolio_values,
            portfolio_returns,
            weights_df,
            trades
        )

        # Create results object
        results = BacktestResults(
            portfolio_values=portfolio_values,
            portfolio_returns=portfolio_returns,
            weights_history=weights_df,
            drawdown_series=metrics['drawdown_series'],
            total_return=metrics['total_return'],
            annualized_return=metrics['annualized_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            calmar_ratio=metrics['calmar_ratio'],
            max_drawdown=metrics['max_drawdown'],
            volatility=metrics['volatility'],
            win_rate=metrics['win_rate'],
            avg_turnover=metrics['avg_turnover'],
            total_transaction_costs=metrics['total_transaction_costs'],
            trades=trades,
            config=self.config,
            start_date=returns.index[0],
            end_date=returns.index[-1],
            n_periods=n_periods
        )

        return results

    def compare_strategies(
        self,
        strategies: Dict[str, Strategy],
        data: pd.DataFrame,
        returns: pd.DataFrame,
        **strategy_kwargs
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Args:
            strategies: Dictionary of {name: strategy} pairs
            data: Historical market data
            returns: Asset returns
            **strategy_kwargs: Strategy-specific parameters

        Returns:
            DataFrame with comparison metrics
        """
        results = {}

        for name, strategy in strategies.items():
            print(f"Running backtest for {name}...")
            result = self.run(strategy, data, returns, **strategy_kwargs)
            results[name] = result.to_dict()

        comparison = pd.DataFrame(results).T

        return comparison


if __name__ == "__main__":
    # Test backtesting engine
    print("Testing Backtesting Engine...")

    # Generate mock data
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    returns = pd.DataFrame({
        'SPY': np.random.normal(0.0005, 0.01, n_days),
        'TLT': np.random.normal(0.0002, 0.005, n_days),
        'GLD': np.random.normal(0.0003, 0.008, n_days)
    }, index=dates)

    # Simple equal-weight strategy
    class SimpleEqualWeight(Strategy):
        @property
        def name(self):
            return "Equal Weight"

        def allocate(self, data, current_idx, current_weights, **kwargs):
            return np.ones(len(current_weights)) / len(current_weights)

    # Run backtest
    config = BacktestConfig(
        initial_capital=100000.0,
        transaction_cost=0.0025,
        rebalance_frequency=20
    )

    engine = BacktestEngine(config)
    strategy = SimpleEqualWeight()

    results = engine.run(strategy, returns, returns)

    print("\nBacktest Results:")
    print(f"Total Return:       {results.total_return:.2%}")
    print(f"Annualized Return:  {results.annualized_return:.2%}")
    print(f"Sharpe Ratio:       {results.sharpe_ratio:.3f}")
    print(f"Max Drawdown:       {results.max_drawdown:.2%}")
    print(f"Volatility:         {results.volatility:.2%}")
    print(f"Win Rate:           {results.win_rate:.2%}")
    print(f"Avg Turnover:       {results.avg_turnover:.3f}")
    print(f"Total Costs:        ${results.total_transaction_costs:,.2f}")
    print(f"Final Value:        ${results.portfolio_values.iloc[-1]:,.2f}")
    print(f"Number of Trades:   {len(results.trades)}")
