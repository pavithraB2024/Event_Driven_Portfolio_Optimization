"""
Comprehensive Baseline Strategy Backtesting

Uses the BacktestEngine to evaluate all baseline strategies on real market data.
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig
from src.backtesting.strategy_adapters import (
    MertonStrategyAdapter,
    MeanVarianceAdapter,
    EqualWeightAdapter,
    BuyAndHoldAdapter,
    RiskParityAdapter
)


def load_data(data_path: str = 'data/processed/28stocks_dataset.csv'):
    """Load processed market data."""
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    return data


def extract_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Extract return columns."""
    return_cols = [col for col in data.columns if col.startswith('return_')]
    returns = data[return_cols].copy()
    returns.columns = [col.replace('return_', '') for col in returns.columns]
    return returns


def main(portfolio_name: str = '28stocks',
         start_date: str = '2013-11-04',
         end_date: str = '2014-03-14'):
    """Main execution function."""

    print("="*80)
    print("COMPREHENSIVE BASELINE STRATEGY BACKTESTING")
    print(f"Portfolio: {portfolio_name}")
    print(f"Window:    {start_date} -> {end_date}")
    print("Using BacktestEngine Framework")
    print("="*80)

    # Load data
    print("\nLoading data...")
    data_path = f'data/processed/{portfolio_name}_dataset.csv'
    data = load_data(data_path)
    data = data.loc[start_date:end_date]
    returns = extract_returns(data)

    print(f"Data: {data.index[0]} to {data.index[-1]} ({len(data)} days)")
    print(f"Assets: {', '.join(returns.columns)}")

    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000.0,
        transaction_cost=0.0025,  # 0.25% (matches Sun2024 & Soleymani2021)
        slippage=0.0005,  # 0.05%
        rebalance_frequency=20,  # Every 20 days
        risk_free_rate=0.02
    )

    # Initialize engine
    engine = BacktestEngine(config)

    # Define strategies
    strategies = {
        'Merton': MertonStrategyAdapter(risk_aversion=1.0),
        'Mean-Variance': MeanVarianceAdapter(risk_aversion=2.0),
        'Equal-Weight': EqualWeightAdapter(),
        'Buy-and-Hold': BuyAndHoldAdapter(),
        'Risk Parity': RiskParityAdapter()
    }

    # Run backtests
    results = {}
    print("\n" + "="*80)
    print("RUNNING BACKTESTS")
    print("="*80)

    for name, strategy in strategies.items():
        print(f"\n{name}:")
        print("-" * 80)
        result = engine.run(strategy, data, returns)
        results[name] = result

        print(f"Total Return:       {result.total_return:>8.2%}")
        print(f"Annualized Return:  {result.annualized_return:>8.2%}")
        print(f"Sharpe Ratio:       {result.sharpe_ratio:>8.3f}")
        print(f"Sortino Ratio:      {result.sortino_ratio:>8.3f}")
        print(f"Calmar Ratio:       {result.calmar_ratio:>8.3f}")
        print(f"Max Drawdown:       {result.max_drawdown:>8.2%}")
        print(f"Volatility:         {result.volatility:>8.2%}")
        print(f"Win Rate:           {result.win_rate:>8.2%}")
        print(f"Avg Turnover:       {result.avg_turnover:>8.3f}")
        print(f"Transaction Costs:  ${result.total_transaction_costs:>11,.2f}")
        print(f"Final Value:        ${result.portfolio_values.iloc[-1]:>11,.2f}")
        print(f"Number of Trades:   {len(result.trades):>8,}")

    # Create comparison DataFrame
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)

    comparison = pd.DataFrame({
        name: res.to_dict()
        for name, res in results.items()
    }).T

    print(comparison.to_string())

    # Save results
    output_dir = Path(f'simulations/backtesting_results/{portfolio_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison table
    comparison.to_csv(output_dir / 'strategy_comparison.csv')
    print(f"\nComparison saved to: {output_dir / 'strategy_comparison.csv'}")

    # Save detailed results for each strategy
    for name, result in results.items():
        strategy_dir = output_dir / name.replace(' ', '_').replace('-', '_').lower()
        strategy_dir.mkdir(exist_ok=True)

        # Portfolio values
        result.portfolio_values.to_csv(strategy_dir / 'portfolio_values.csv')

        # Weights history
        result.weights_history.to_csv(strategy_dir / 'weights_history.csv')

        # Drawdown series
        result.drawdown_series.to_csv(strategy_dir / 'drawdown.csv')

        # Trades
        if result.trades:
            trades_df = pd.DataFrame([
                {
                    'date': trade.date,
                    'asset': trade.asset,
                    'old_weight': trade.old_weight,
                    'new_weight': trade.new_weight,
                    'turnover': trade.turnover,
                    'transaction_cost': trade.transaction_cost,
                    'slippage_cost': trade.slippage_cost,
                    'portfolio_value': trade.portfolio_value
                }
                for trade in result.trades
            ])
            trades_df.to_csv(strategy_dir / 'trades.csv', index=False)

    print(f"Detailed results saved to: {output_dir}")

    # Create visualizations
    print("\nGenerating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Wealth trajectories
    ax = axes[0, 0]
    for name, result in results.items():
        ax.plot(result.portfolio_values.index, result.portfolio_values.values,
                label=name, linewidth=2, alpha=0.8)
    ax.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # 2. Drawdown comparison
    ax = axes[0, 1]
    for name, result in results.items():
        ax.plot(result.drawdown_series.index, result.drawdown_series.values * 100,
                label=name, linewidth=2, alpha=0.8)
    ax.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # 3. Performance metrics
    ax = axes[1, 0]
    metrics_to_plot = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
    x = np.arange(len(results))
    width = 0.25

    for i, metric in enumerate(metrics_to_plot):
        metric_key = metric.lower().replace(' ', '_')
        values = [comparison.loc[name, metric_key] for name in results.keys()]
        ax.bar(x + i*width, values, width, label=metric, alpha=0.8)

    ax.set_title('Risk-Adjusted Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(results.keys(), rotation=15, ha='right')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Risk-Return scatter
    ax = axes[1, 1]
    for name, result in results.items():
        annual_return = result.annualized_return * 100
        annual_vol = result.volatility * 100
        ax.scatter(annual_vol, annual_return, s=200, alpha=0.7, label=name)
        ax.annotate(name, (annual_vol, annual_return),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax.set_xlabel('Annualized Volatility (%)')
    ax.set_ylabel('Annualized Return (%)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'backtest_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_dir / 'backtest_comparison.png'}")

    # Identify best strategies
    print("\n" + "="*80)
    print("BEST STRATEGIES BY METRIC")
    print("="*80)

    best_return = comparison['annualized_return'].idxmax()
    best_sharpe = comparison['sharpe_ratio'].idxmax()
    best_sortino = comparison['sortino_ratio'].idxmax()
    best_calmar = comparison['calmar_ratio'].idxmax()
    best_drawdown = comparison['max_drawdown'].idxmin()

    print(f"Best Annualized Return:  {best_return:15s} ({comparison.loc[best_return, 'annualized_return']:.2%})")
    print(f"Best Sharpe Ratio:       {best_sharpe:15s} ({comparison.loc[best_sharpe, 'sharpe_ratio']:.3f})")
    print(f"Best Sortino Ratio:      {best_sortino:15s} ({comparison.loc[best_sortino, 'sortino_ratio']:.3f})")
    print(f"Best Calmar Ratio:       {best_calmar:15s} ({comparison.loc[best_calmar, 'calmar_ratio']:.3f})")
    print(f"Best Drawdown Control:   {best_drawdown:15s} ({comparison.loc[best_drawdown, 'max_drawdown']:.2%})")

    print("\n" + "="*80)
    print("BACKTESTING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline strategy backtests for a given portfolio")
    parser.add_argument("--portfolio_name", type=str, default="28stocks",
                        help="Portfolio identifier; loads data/processed/{portfolio_name}_dataset.csv "
                             "and writes to simulations/backtesting_results/{portfolio_name}/")
    parser.add_argument("--start_date", type=str, default="2013-11-04",
                        help="Test window start (default = Sun et al. benchmark start).")
    parser.add_argument("--end_date", type=str, default="2014-03-14",
                        help="Test window end (default = Sun et al. benchmark end).")
    args = parser.parse_args()
    main(portfolio_name=args.portfolio_name,
         start_date=args.start_date,
         end_date=args.end_date)
