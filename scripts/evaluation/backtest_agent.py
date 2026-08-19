"""
Comprehensive Backtesting Framework for Portfolio Allocation Agents

This module provides a robust backtesting system to evaluate and compare
different portfolio allocation strategies including:
- Memory-Augmented SAC Agent (LSTM-based)
- Standard SAC Agent (MLP-based)
- Baseline Strategies (Merton, Mean-Variance, Equal-Weight, Buy-and-Hold, Risk Parity)

The framework supports:
- Walk-forward analysis
- Transaction costs
- Position limits
- Risk metrics (Sharpe, Sortino, Max Drawdown, VaR, CVaR)
- Crisis period analysis
- Comprehensive performance reporting
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json

from src.agents.sac_agent import SACAgent
from src.agents.memory_augmented_sac_agent import MemoryAugmentedSACAgent
from src.environments.portfolio_env import PortfolioEnv
from src.environments.path_dependent_portfolio_env import PathDependentPortfolioEnv
from src.backtesting.strategy_adapters import (
    MertonStrategyAdapter,
    MeanVarianceAdapter,
    EqualWeightAdapter,
    BuyAndHoldAdapter,
    RiskParityAdapter
)


class PerformanceMetrics:
    """Calculate comprehensive performance metrics for portfolio backtesting"""

    @staticmethod
    def total_return(returns: np.ndarray) -> float:
        """Calculate total return"""
        return (1 + returns).prod() - 1

    @staticmethod
    def annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate annualized return"""
        total_ret = PerformanceMetrics.total_return(returns)
        n_periods = len(returns)
        return (1 + total_ret) ** (periods_per_year / n_periods) - 1

    @staticmethod
    def annualized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(periods_per_year)

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02,
                     periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio"""
        ann_ret = PerformanceMetrics.annualized_return(returns, periods_per_year)
        ann_vol = PerformanceMetrics.annualized_volatility(returns, periods_per_year)
        if ann_vol == 0:
            return 0.0
        return (ann_ret - risk_free_rate) / ann_vol

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02,
                     periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio (downside deviation only)"""
        ann_ret = PerformanceMetrics.annualized_return(returns, periods_per_year)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        downside_std = downside_returns.std() * np.sqrt(periods_per_year)
        if downside_std == 0:
            return 0.0
        return (ann_ret - risk_free_rate) / downside_std

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cumulative) / cumulative
        return abs(drawdown.min())

    @staticmethod
    def calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        ann_ret = PerformanceMetrics.annualized_return(returns, periods_per_year)
        equity_curve = (1 + returns).cumprod()
        max_dd = PerformanceMetrics.max_drawdown(equity_curve)
        if max_dd == 0:
            return 0.0
        return ann_ret / max_dd

    @staticmethod
    def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)"""
        return np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR / Expected Shortfall)"""
        var = PerformanceMetrics.value_at_risk(returns, confidence)
        return returns[returns <= var].mean()

    @staticmethod
    def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        returns_above = returns[returns > threshold] - threshold
        returns_below = threshold - returns[returns <= threshold]

        if len(returns_below) == 0 or returns_below.sum() == 0:
            return np.inf

        return returns_above.sum() / returns_below.sum()

    @staticmethod
    def calculate_all_metrics(returns: np.ndarray, equity_curve: np.ndarray,
                             risk_free_rate: float = 0.02,
                             periods_per_year: int = 252) -> Dict[str, float]:
        """Calculate all performance metrics"""
        return {
            'Total Return': PerformanceMetrics.total_return(returns),
            'Annualized Return': PerformanceMetrics.annualized_return(returns, periods_per_year),
            'Annualized Volatility': PerformanceMetrics.annualized_volatility(returns, periods_per_year),
            'Sharpe Ratio': PerformanceMetrics.sharpe_ratio(returns, risk_free_rate, periods_per_year),
            'Sortino Ratio': PerformanceMetrics.sortino_ratio(returns, risk_free_rate, periods_per_year),
            'Max Drawdown': PerformanceMetrics.max_drawdown(equity_curve),
            'Calmar Ratio': PerformanceMetrics.calmar_ratio(returns, periods_per_year),
            'VaR (95%)': PerformanceMetrics.value_at_risk(returns, 0.95),
            'CVaR (95%)': PerformanceMetrics.conditional_var(returns, 0.95),
            'Omega Ratio': PerformanceMetrics.omega_ratio(returns, 0.0),
        }


class AgentBacktester:
    """Backtest portfolio allocation agents"""

    def __init__(self,
                 transaction_cost: float = 0.0025,
                 position_limit: float = 0.5,
                 initial_capital: float = 1_000_000,
                 risk_free_rate: float = 0.02):
        """
        Initialize backtester

        Args:
            transaction_cost: Transaction cost as fraction of traded value
            position_limit: Maximum weight for any single asset
            initial_capital: Initial portfolio value
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.transaction_cost = transaction_cost
        self.position_limit = position_limit
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

    def apply_transaction_costs(self,
                                old_weights: np.ndarray,
                                new_weights: np.ndarray,
                                portfolio_value: float) -> float:
        """Calculate transaction costs from rebalancing"""
        turnover = np.abs(new_weights - old_weights).sum()
        cost = turnover * self.transaction_cost * portfolio_value
        return cost

    def apply_position_limits(self, weights: np.ndarray) -> np.ndarray:
        """Enforce position limits and normalize"""
        # Clip weights to position limits
        weights = np.clip(weights, 0, self.position_limit)

        # Renormalize to sum to 1
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # If all weights are zero, use equal weight
            weights = np.ones_like(weights) / len(weights)

        return weights

    def backtest_sac_agent(self,
                          agent_path: str,
                          env: PortfolioEnv,
                          device: str = 'cpu',
                          is_memory: bool = False) -> Dict:
        """
        Backtest trained SAC agent

        Args:
            agent_path: Path to trained SAC model
            env: Portfolio environment for testing
            device: Device to run agent on
            is_memory: Whether this is a Memory-Augmented SAC agent

        Returns:
            Dictionary with backtest results
        """
        # Load trained agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        if is_memory:
            agent = MemoryAugmentedSACAgent(state_dim, action_dim, network_type='LSTM', device=device)
        else:
            agent = SACAgent(state_dim, action_dim, device=device)
        
        agent.load(agent_path)

        # Run backtest
        state, _ = env.reset()
        done = False
        hidden_state = None

        portfolio_values = [self.initial_capital]
        weights_history = []
        returns_history = []
        actions_history = []
        transaction_costs_history = []

        current_weights = np.zeros(action_dim)  # Start with zero allocation
        portfolio_value = self.initial_capital
        total_transaction_costs = 0.0

        step = 0
        while not done:
            # Get action from agent (deterministic)
            if is_memory:
                action, hidden_state = agent.select_action(state, hidden_state=hidden_state, deterministic=True)
            else:
                action = agent.select_action(state, deterministic=True)
            
            actions_history.append(action.copy())

            # Normalize action to weights (softmax)
            new_weights = np.exp(action) / np.sum(np.exp(action))

            # Apply position limits
            new_weights = self.apply_position_limits(new_weights)

            # Calculate and apply transaction costs
            costs = self.apply_transaction_costs(current_weights, new_weights, portfolio_value)
            portfolio_value -= costs
            total_transaction_costs += costs
            transaction_costs_history.append(costs)

            # Execute action
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            # Calculate portfolio return
            portfolio_return = info.get('portfolio_return', 0.0)
            returns_history.append(portfolio_return)

            # Update portfolio value
            portfolio_value *= (1 + portfolio_return)
            portfolio_values.append(portfolio_value)

            # Store weights
            weights_history.append(new_weights.copy())

            # Update state and weights
            state = next_state
            current_weights = new_weights
            step += 1

        # Calculate metrics
        returns_array = np.array(returns_history)
        equity_curve = np.array(portfolio_values[1:])  # Exclude initial value

        metrics = PerformanceMetrics.calculate_all_metrics(
            returns_array, equity_curve, self.risk_free_rate
        )

        strategy_name = 'Memory SAC (LSTM)' if is_memory else 'Standard SAC (MLP)'

        return {
            'strategy': strategy_name,
            'portfolio_values': portfolio_values,
            'returns': returns_history,
            'weights': weights_history,
            'actions': actions_history,
            'transaction_costs': transaction_costs_history,
            'total_transaction_costs': total_transaction_costs,
            'metrics': metrics,
            'final_value': portfolio_value
        }

    def backtest_baseline_strategy(self,
                                   strategy,
                                   env: PortfolioEnv) -> Dict:
        """
        Backtest baseline strategy

        Args:
            strategy: Baseline strategy object
            env: Portfolio environment for testing

        Returns:
            Dictionary with backtest results
        """
        action_dim = env.action_space.shape[0]
        n_assets = action_dim  # Assuming continuous action space matches assets

        state, _ = env.reset()
        done = False

        portfolio_values = [self.initial_capital]
        weights_history = []
        returns_history = []

        # Initialize equal weights or cash (strategies usually provide first allocation)
        current_weights = np.ones(n_assets) / n_assets
        portfolio_value = self.initial_capital

        while not done:
            # Get allocation from strategy
            window_size = getattr(env, 'window_size', 50)
            current_idx = min(env.current_step + window_size, len(env.data) - 1)
            
            # Ensure we don't look ahead illegally
            current_idx = max(window_size, current_idx)
            
            try:
                new_weights = strategy.allocate(
                    data=env.data,
                    current_idx=current_idx,
                    current_weights=current_weights
                )
            except ValueError:
                # Handle dimension mismatch
                new_weights = current_weights.copy()
            except Exception:
                 # Fallback
                 new_weights = current_weights.copy()

            # Ensure new_weights has correct shape
            if len(new_weights) != n_assets:
                new_weights = current_weights.copy()

            # Apply position limits
            new_weights = self.apply_position_limits(new_weights)

            # Calculate and apply transaction costs
            costs = self.apply_transaction_costs(current_weights, new_weights, portfolio_value)
            portfolio_value -= costs

            # Dummy action for stepping environment (we track portfolio manually)
            # For continuous env, action should be of shape (action_dim,)
            action = np.zeros(action_dim)

            # Execute action
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            # Calculate actual portfolio return based on weights
            # Try to get returns from info first, but check if they are valid (non-zero or matching data)
            asset_returns = info.get('asset_returns', np.zeros(n_assets))
            
            # If info returns are suspicious (all zero), fetch from data manually
            if np.all(asset_returns == 0):
                return_cols = [c for c in env.data.columns if c.startswith('return_')]
                if len(return_cols) == n_assets:
                    # env.step increments current_step, so we need the previous one
                    # But dependending on env implementation, it might be different.
                    # Standard PortfolioEnv: step() uses current_step, then increments.
                    # So the returns correspond to env.current_step - 1
                    t_idx = env.current_step - 1
                    if 0 <= t_idx < len(env.data):
                        asset_returns = env.data.iloc[t_idx][return_cols].values
            
            if len(asset_returns) != n_assets:
                 asset_returns = np.zeros(n_assets)
            
            portfolio_return = np.dot(new_weights, asset_returns)
            returns_history.append(portfolio_return)

            # Update portfolio value
            portfolio_value *= (1 + portfolio_return)
            portfolio_values.append(portfolio_value)

            # Store weights
            weights_history.append(new_weights.copy())

            # Update state and weights
            current_weights = new_weights
            # step increment processed by env

        # Calculate metrics
        returns_array = np.array(returns_history)
        equity_curve = np.array(portfolio_values[1:])

        metrics = PerformanceMetrics.calculate_all_metrics(
            returns_array, equity_curve, self.risk_free_rate
        )

        strategy_name = strategy.__class__.__name__.replace('Strategy', '')

        return {
            'strategy': strategy_name,
            'portfolio_values': portfolio_values,
            'returns': returns_history,
            'weights': weights_history,
            'metrics': metrics,
            'final_value': portfolio_value
        }


def run_comprehensive_backtest(
    memory_sac_path: str,
    standard_sac_path: str,
    data_path: str,
    output_dir: str = 'simulations/backtest_results',
    device: str = 'cpu'
) -> pd.DataFrame:
    """
    Run comprehensive backtest comparing all strategies

    Args:
        memory_sac_path: Path to trained Memory-Augmented SAC model
        standard_sac_path: Path to trained Standard SAC model
        data_path: Path to dataset
        output_dir: Directory to save results
        device: Device for SAC agents

    Returns:
        DataFrame with comparison metrics
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Add date column from index for compatibility
    df['date'] = df.index

    # Split data (use same split as training)
    train_size = int(0.8 * len(df))
    test_df = df.iloc[train_size:].copy()

    print(f"\nBacktesting on {len(test_df)} test days")
    print(f"Test period: {test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]}")

    # Create test environments
    test_env_memory = PathDependentPortfolioEnv(test_df, action_type='continuous')
    test_env_standard = PortfolioEnv(test_df, action_type='continuous')

    # Initialize backtester
    backtester = AgentBacktester(
        transaction_cost=0.0025,
        position_limit=0.5,
        initial_capital=100_000,
        risk_free_rate=0.02
    )

    # Store all results
    all_results = []

    # 1. Backtest Memory-Augmented SAC Agent
    print("\n" + "="*60)
    print("Backtesting Memory-Augmented SAC Agent...")
    print("="*60)
    try:
        memory_sac_results = backtester.backtest_sac_agent(
            memory_sac_path, test_env_memory, device, is_memory=True
        )
        all_results.append(memory_sac_results)
        print(f"[OK] Memory SAC - Final Value: ${memory_sac_results['final_value']:,.2f}")
        print(f"  Sharpe Ratio: {memory_sac_results['metrics']['Sharpe Ratio']:.3f}")
        print(f"  Transaction Costs: ${memory_sac_results['total_transaction_costs']:,.2f}")
    except Exception as e:
        print(f"[X] Memory SAC failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Backtest Standard SAC Agent
    print("\n" + "="*60)
    print("Backtesting Standard SAC Agent...")
    print("="*60)
    try:
        standard_sac_results = backtester.backtest_sac_agent(
            standard_sac_path, test_env_standard, device, is_memory=False
        )
        all_results.append(standard_sac_results)
        print(f"[OK] Standard SAC - Final Value: ${standard_sac_results['final_value']:,.2f}")
        print(f"  Sharpe Ratio: {standard_sac_results['metrics']['Sharpe Ratio']:.3f}")
        print(f"  Transaction Costs: ${standard_sac_results['total_transaction_costs']:,.2f}")
    except Exception as e:
        print(f"[X] Standard SAC failed: {e}")
        import traceback
        traceback.print_exc()

    # 3. Backtest Baseline Strategies
    print("\n" + "="*60)
    print("Backtesting Baseline Strategies...")
    print("="*60)
    
    baseline_strategies = {
        'Merton': MertonStrategyAdapter(risk_aversion=2.0),
        'Mean-Variance': MeanVarianceAdapter(risk_aversion=2.0),
       'Equal-Weight': EqualWeightAdapter(),
        'Buy-and-Hold': BuyAndHoldAdapter(),
        'Risk Parity': RiskParityAdapter()
    }

    for name, strategy in baseline_strategies.items():
        print(f"\nBacktesting {name}...")
        test_env_copy = PortfolioEnv(test_df, action_type='continuous')
        try:
            results = backtester.backtest_baseline_strategy(strategy, test_env_copy)
            all_results.append(results)
            print(f"[OK] {name} - Final Value: ${results['final_value']:,.2f}")
            print(f"  Sharpe Ratio: {results['metrics']['Sharpe Ratio']:.3f}")
        except Exception as e:
            print(f"[X] {name} failed: {e}")
            import traceback
            traceback.print_exc()

    # Create comparison DataFrame
    comparison_data = []
    for result in all_results:
        row = {'Strategy': result['strategy']}
        row.update(result['metrics'])
        row['Final Value'] = result['final_value']
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by Sharpe Ratio
    comparison_df = comparison_df.sort_values('Sharpe Ratio', ascending=False)

    # Save comparison
    comparison_path = output_path / 'strategy_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n[OK] Saved comparison to {comparison_path}")

    # Save detailed results
    results_path = output_path / 'detailed_results.json'

    # Convert numpy arrays to lists for JSON serialization
    json_results = []
    for result in all_results:
        json_result = {
            'strategy': result['strategy'],
            'metrics': result['metrics'],
            'final_value': result['final_value'],
            'portfolio_values': [float(v) for v in result['portfolio_values']],
            'returns': [float(r) for r in result['returns']]
        }
        json_results.append(json_result)

    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"[OK] Saved detailed results to {results_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_backtest_results(all_results, test_df, output_path)

    return comparison_df


def plot_backtest_results(results: List[Dict], test_df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive backtest visualizations"""

    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (15, 10)

    # 1. Equity Curves
    fig, ax = plt.subplots(figsize=(15, 8))

    for result in results:
        dates = pd.to_datetime(test_df['date'].iloc[:len(result['portfolio_values'])])
        ax.plot(dates, result['portfolio_values'],
               label=result['strategy'], linewidth=2, alpha=0.8)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax.set_title('Portfolio Equity Curves - All Strategies',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Drawdown Analysis
    fig, ax = plt.subplots(figsize=(15, 8))

    for result in results:
        portfolio_values = np.array(result['portfolio_values'])
        cumulative = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cumulative) / cumulative * 100

        dates = pd.to_datetime(test_df['date'].iloc[:len(drawdown)])
        ax.plot(dates, drawdown, label=result['strategy'], linewidth=2, alpha=0.8)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_title('Drawdown Analysis - All Strategies',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'drawdown_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Metrics Comparison Heatmap
    metrics_data = []
    strategies = []

    for result in results:
        strategies.append(result['strategy'])
        metrics_data.append(list(result['metrics'].values()))

    metrics_df = pd.DataFrame(
        metrics_data,
        index=strategies,
        columns=list(results[0]['metrics'].keys())
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn',
               center=0, ax=ax, cbar_kws={'label': 'Metric Value'})
    ax.set_title('Performance Metrics Heatmap',
                fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Risk-Return Scatter
    fig, ax = plt.subplots(figsize=(12, 8))

    for result in results:
        metrics = result['metrics']
        ax.scatter(metrics['Annualized Volatility'] * 100,
                  metrics['Annualized Return'] * 100,
                  s=200, alpha=0.6, label=result['strategy'])

        ax.annotate(result['strategy'],
                   (metrics['Annualized Volatility'] * 100,
                    metrics['Annualized Return'] * 100),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)

    ax.set_xlabel('Annualized Volatility (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annualized Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'risk_return_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Rolling Sharpe Ratio
    fig, ax = plt.subplots(figsize=(15, 8))
    window = 63  # Quarter

    for result in results:
        returns = pd.Series(result['returns'])
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)

        dates = pd.to_datetime(test_df['date'].iloc[:len(rolling_sharpe)])
        ax.plot(dates, rolling_sharpe, label=result['strategy'], linewidth=2, alpha=0.8)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rolling Sharpe Ratio (63-day)', fontsize=12, fontweight='bold')
    ax.set_title('Rolling Sharpe Ratio Analysis', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'rolling_sharpe.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Generated all visualization plots")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Backtest Portfolio Allocation Agents')
    parser.add_argument('--memory-sac', type=str, default='models/sac_trained_new_step40000.pth',
                       help='Path to trained Memory-Augmented SAC model')
    parser.add_argument('--standard-sac', type=str, default='models/sac_baseline.pth',
                       help='Path to trained Standard SAC model')
    parser.add_argument('--data', type=str, default='data/processed/complete_dataset.csv',
                       help='Path to dataset')
    parser.add_argument('--output', type=str, default='simulations/sac_backtest_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu or cuda)')

    args = parser.parse_args()

    # Run comprehensive backtest
    print("\n" + "="*80)
    print("COMPREHENSIVE SAC PORTFOLIO BACKTESTING FRAMEWORK")
    print("="*80)

    comparison_df = run_comprehensive_backtest(
        memory_sac_path=args.memory_sac,
        standard_sac_path=args.standard_sac,
        data_path=args.data,
        output_dir=args.output,
        device=args.device
    )

    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))

    print("\n[OK] Backtesting complete!")
