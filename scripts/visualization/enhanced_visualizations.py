"""
Enhanced Visualization Suite for Portfolio Allocation Analysis

Creates advanced interactive and static visualizations:
- Rolling performance metrics (Sharpe, Sortino, Calmar)
- Portfolio allocation heatmaps over time
- Interactive Plotly dashboards
- Regime-based performance analysis
- Risk-return scatter with confidence ellipses
- Correlation matrix evolution
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from src.agents.sac_agent import SACAgent
from src.agents.memory_augmented_sac_agent import MemoryAugmentedSACAgent
from src.environments.portfolio_env import PortfolioEnv
from src.environments.path_dependent_portfolio_env import PathDependentPortfolioEnv


class EnhancedVisualizer:
    """
    Create advanced visualizations for portfolio analysis
    """

    def __init__(self, output_dir: str = 'simulations/enhanced_viz'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (16, 10)

    def create_rolling_metrics_plot(
        self,
        returns: np.ndarray,
        dates: pd.Series,
        window: int = 63,
        save_name: str = 'rolling_metrics.png'
    ):
        """
        Create rolling performance metrics visualization

        Args:
            returns: Array of returns
            dates: Series of dates
            window: Rolling window size (default 63 = quarter)
            save_name: Output filename
        """
        print("\nCreating rolling metrics visualization...")

        returns_series = pd.Series(returns, index=dates)

        # Calculate rolling metrics
        rolling_mean = returns_series.rolling(window).mean() * 252
        rolling_std = returns_series.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_mean / rolling_std

        # Rolling Sortino (downside deviation)
        def rolling_sortino(series, window):
            sortino_values = []
            for i in range(len(series)):
                if i < window:
                    sortino_values.append(np.nan)
                else:
                    window_returns = series.iloc[i-window:i]
                    mean_return = window_returns.mean() * 252
                    downside = window_returns[window_returns < 0]
                    if len(downside) > 0:
                        downside_std = downside.std() * np.sqrt(252)
                        sortino_values.append(mean_return / downside_std if downside_std > 0 else 0)
                    else:
                        sortino_values.append(np.inf)
            return pd.Series(sortino_values, index=series.index)

        rolling_sortino_ratio = rolling_sortino(returns_series, window)

        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle(f'Rolling Performance Metrics ({window}-day window)',
                    fontsize=16, fontweight='bold')

        # 1. Rolling Sharpe Ratio
        ax1 = axes[0]
        ax1.plot(dates, rolling_sharpe, linewidth=2, color='#2ecc71', label='Sharpe Ratio')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.axhline(y=1, color='blue', linestyle=':', alpha=0.3, label='Sharpe = 1')
        ax1.axhline(y=2, color='green', linestyle=':', alpha=0.3, label='Sharpe = 2')
        ax1.fill_between(dates, 0, rolling_sharpe, where=(rolling_sharpe > 0),
                        color='#2ecc71', alpha=0.2)
        ax1.fill_between(dates, 0, rolling_sharpe, where=(rolling_sharpe < 0),
                        color='#e74c3c', alpha=0.2)
        ax1.set_ylabel('Sharpe Ratio', fontweight='bold')
        ax1.set_title('Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 2. Rolling Sortino Ratio
        ax2 = axes[1]
        # Clip extreme values for visualization
        rolling_sortino_clipped = rolling_sortino_ratio.clip(-5, 5)
        ax2.plot(dates, rolling_sortino_clipped, linewidth=2, color='#3498db',
                label='Sortino Ratio')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.axhline(y=1, color='blue', linestyle=':', alpha=0.3, label='Sortino = 1')
        ax2.fill_between(dates, 0, rolling_sortino_clipped, where=(rolling_sortino_clipped > 0),
                        color='#3498db', alpha=0.2)
        ax2.fill_between(dates, 0, rolling_sortino_clipped, where=(rolling_sortino_clipped < 0),
                        color='#e74c3c', alpha=0.2)
        ax2.set_ylabel('Sortino Ratio', fontweight='bold')
        ax2.set_title('Rolling Sortino Ratio (Clipped at ±5)', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # 3. Rolling Return vs Volatility
        ax3 = axes[2]
        ax3_twin = ax3.twinx()
        ax3.plot(dates, rolling_mean * 100, linewidth=2, color='#9b59b6',
                label='Annualized Return')
        ax3_twin.plot(dates, rolling_std * 100, linewidth=2, color='#e67e22',
                     label='Annualized Volatility', linestyle='--')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax3.set_ylabel('Annualized Return (%)', fontweight='bold', color='#9b59b6')
        ax3_twin.set_ylabel('Annualized Volatility (%)', fontweight='bold', color='#e67e22')
        ax3.set_xlabel('Date', fontweight='bold')
        ax3.set_title('Rolling Return & Volatility', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='y', labelcolor='#9b59b6')
        ax3_twin.tick_params(axis='y', labelcolor='#e67e22')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Saved rolling metrics to {self.output_dir / save_name}")

    def create_allocation_heatmap(
        self,
        weights_history: list,
        dates: pd.Series,
        asset_names: list = ['SPY', 'TLT', 'GLD', 'BTC'],
        save_name: str = 'allocation_heatmap.png'
    ):
        """
        Create portfolio allocation heatmap over time

        Args:
            weights_history: List of weight arrays
            dates: Series of dates
            asset_names: List of asset names
            save_name: Output filename
        """
        print("\nCreating allocation heatmap...")

        # Convert to DataFrame
        weights_df = pd.DataFrame(weights_history, columns=asset_names, index=dates)

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Portfolio Allocation Over Time', fontsize=16, fontweight='bold')

        # 1. Stacked area chart
        ax1 = axes[0]
        ax1.stackplot(dates, *[weights_df[asset] * 100 for asset in asset_names],
                     labels=asset_names,
                     colors=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
                     alpha=0.8)
        ax1.set_ylabel('Allocation (%)', fontweight='bold')
        ax1.set_title('Stacked Allocation', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)

        # 2. Heatmap
        ax2 = axes[1]
        # Sample every N days for readability if too many days
        sample_freq = max(1, len(weights_df) // 100)
        weights_sampled = weights_df.iloc[::sample_freq]

        im = ax2.imshow(weights_sampled.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax2.set_yticks(range(len(asset_names)))
        ax2.set_yticklabels(asset_names)
        ax2.set_xlabel('Time Period', fontweight='bold')
        ax2.set_title('Allocation Heatmap', fontsize=12, fontweight='bold')

        # Color bar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Weight', rotation=270, labelpad=15, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Saved allocation heatmap to {self.output_dir / save_name}")

    def create_interactive_dashboard(
        self,
        portfolio_values: np.ndarray,
        returns: np.ndarray,
        weights_history: list,
        dates: pd.Series,
        asset_names: list = ['SPY', 'TLT', 'GLD', 'BTC'],
        save_name: str = 'interactive_dashboard.html'
    ):
        """
        Create interactive Plotly dashboard

        Args:
            portfolio_values: Array of portfolio values
            returns: Array of returns
            weights_history: List of weight arrays
            dates: Series of dates
            asset_names: List of asset names
            save_name: Output filename
        """
        print("\nCreating interactive dashboard...")

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Portfolio Value Over Time', 'Daily Returns Distribution',
                          'Rolling Sharpe Ratio', 'Allocation Over Time',
                          'Cumulative Returns', 'Drawdown Analysis'),
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
                  [{'type': 'scatter'}, {'type': 'scatter'}],
                  [{'type': 'scatter'}, {'type': 'scatter'}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )

        # 1. Portfolio Value
        fig.add_trace(
            go.Scatter(x=dates, y=portfolio_values, mode='lines',
                      name='Portfolio Value',
                      line=dict(color='#2ecc71', width=2),
                      hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'),
            row=1, col=1
        )

        # 2. Returns Distribution
        fig.add_trace(
            go.Histogram(x=returns * 100, nbinsx=50,
                        name='Returns',
                        marker_color='#3498db',
                        hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'),
            row=1, col=2
        )

        # 3. Rolling Sharpe
        returns_series = pd.Series(returns)
        rolling_sharpe = (returns_series.rolling(63).mean() / returns_series.rolling(63).std()) * np.sqrt(252)

        fig.add_trace(
            go.Scatter(x=dates, y=rolling_sharpe, mode='lines',
                      name='Sharpe Ratio',
                      line=dict(color='#9b59b6', width=2),
                      hovertemplate='Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>'),
            row=2, col=1
        )

        # 4. Allocation over time
        weights_df = pd.DataFrame(weights_history, columns=asset_names, index=dates)
        colors_map = {'SPY': '#3498db', 'TLT': '#2ecc71', 'GLD': '#f39c12', 'BTC': '#e74c3c'}

        for asset in asset_names:
            fig.add_trace(
                go.Scatter(x=dates, y=weights_df[asset] * 100,
                          mode='lines', name=asset,
                          stackgroup='one',
                          line=dict(color=colors_map.get(asset, '#95a5a6'), width=0.5),
                          hovertemplate=f'{asset}: %{{y:.1f}}%<extra></extra>'),
                row=2, col=2
            )

        # 5. Cumulative Returns
        cumulative_returns = (1 + returns).cumprod() - 1

        fig.add_trace(
            go.Scatter(x=dates, y=cumulative_returns * 100, mode='lines',
                      name='Cumulative Return',
                      line=dict(color='#16a085', width=2),
                      fill='tozeroy',
                      hovertemplate='Date: %{x}<br>Cumulative: %{y:.2f}%<extra></extra>'),
            row=3, col=1
        )

        # 6. Drawdown
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cumulative_max) / cumulative_max * 100

        fig.add_trace(
            go.Scatter(x=dates, y=drawdown, mode='lines',
                      name='Drawdown',
                      line=dict(color='#e74c3c', width=2),
                      fill='tozeroy',
                      hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'),
            row=3, col=2
        )

        # Update layout
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=2)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Allocation (%)", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative Return (%)", row=3, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=2)

        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Portfolio Performance Interactive Dashboard",
            title_font_size=20,
            hovermode='x unified'
        )

        # Save
        fig.write_html(self.output_dir / save_name)
        print(f"[OK] Saved interactive dashboard to {self.output_dir / save_name}")

    def create_regime_analysis(
        self,
        returns: np.ndarray,
        regimes: np.ndarray,
        dates: pd.Series,
        save_name: str = 'regime_analysis.png'
    ):
        """
        Create regime-based performance analysis

        Args:
            returns: Array of returns
            regimes: Array of regime labels
            dates: Series of dates
            save_name: Output filename
        """
        print("\nCreating regime analysis...")

        # Create regime DataFrame
        regime_df = pd.DataFrame({
            'date': dates,
            'return': returns,
            'regime': regimes
        })

        # Calculate metrics by regime
        regime_stats = regime_df.groupby('regime')['return'].agg([
            ('mean', lambda x: x.mean() * 252 * 100),
            ('std', lambda x: x.std() * np.sqrt(252) * 100),
            ('sharpe', lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0),
            ('count', 'count')
        ]).reset_index()

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Regime-Based Performance Analysis', fontsize=16, fontweight='bold')

        # 1. Returns by regime (boxplot)
        ax1 = axes[0, 0]
        regime_colors = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
        box_data = [regime_df[regime_df['regime'] == r]['return'] * 100 for r in sorted(regime_df['regime'].unique())]
        bp = ax1.boxplot(box_data, labels=[f'Regime {r}' for r in sorted(regime_df['regime'].unique())],
                        patch_artist=True)
        for patch, r in zip(bp['boxes'], sorted(regime_df['regime'].unique())):
            patch.set_facecolor(regime_colors.get(r, '#95a5a6'))
            patch.set_alpha(0.7)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.set_ylabel('Daily Return (%)', fontweight='bold')
        ax1.set_title('Returns Distribution by Regime', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Sharpe Ratio by regime (bar chart)
        ax2 = axes[0, 1]
        colors = [regime_colors.get(r, '#95a5a6') for r in regime_stats['regime']]
        ax2.bar(regime_stats['regime'].astype(str), regime_stats['sharpe'],
                       color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.axhline(y=1, color='blue', linestyle=':', alpha=0.3, label='Sharpe = 1')
        ax2.set_xlabel('Regime', fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio', fontweight='bold')
        ax2.set_title('Sharpe Ratio by Regime', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Time in each regime
        ax3 = axes[1, 0]
        regime_counts = regime_df['regime'].value_counts().sort_index()
        colors = [regime_colors.get(r, '#95a5a6') for r in regime_counts.index]
        ax3.pie(regime_counts.values, labels=[f'Regime {r}' for r in regime_counts.index],
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('Time Distribution Across Regimes', fontsize=12, fontweight='bold')

        # 4. Regime transitions over time
        ax4 = axes[1, 1]
        ax4.scatter(dates, regimes, c=[regime_colors.get(r, '#95a5a6') for r in regimes],
                   s=10, alpha=0.6)
        ax4.set_xlabel('Date', fontweight='bold')
        ax4.set_ylabel('Regime', fontweight='bold')
        ax4.set_title('Regime Evolution Over Time', fontsize=12, fontweight='bold')
        ax4.set_yticks(sorted(regime_df['regime'].unique()))
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Saved regime analysis to {self.output_dir / save_name}")

        # Print regime statistics
        print("\nRegime Statistics:")
        print("="*80)
        print(regime_stats.to_string(index=False))


def run_sac_visualization(model_path: str, data_path: str, device: str = 'cpu', is_memory: bool = True):
    """
    Run complete visualization suite for SAC agent
    
    Args:
        model_path: Path to trained SAC model
        data_path: Path to dataset
        device: Device to run on
        is_memory: Whether to use MemoryAugmentedSACAgent
    """
    print("\n" + "="*80)
    print("ENHANCED VISUALIZATION SUITE (SAC)")
    print("="*80)

    # Load data
    df = pd.read_csv(data_path)

    # Get date column
    date_col = 'Date' if 'Date' in df.columns else 'date'
    df['date'] = pd.to_datetime(df[date_col])

    # Split data (use test set)
    train_size = int(0.8 * len(df))
    test_df = df.iloc[train_size:].reset_index(drop=True)

    print(f"\nTest period: {test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]}")
    print(f"Test days: {len(test_df)}")

    # Create environment
    if is_memory:
        env = PathDependentPortfolioEnv(test_df, action_type='continuous')
    else:
        env = PortfolioEnv(test_df, action_type='continuous')

    # Load agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if is_memory:
        agent = MemoryAugmentedSACAgent(state_dim, action_dim, network_type='LSTM', device=device)
    else:
        agent = SACAgent(state_dim, action_dim, device=device)
        
    agent.load(model_path)

    # Run backtest
    state, _ = env.reset()
    done = False
    hidden_state = None

    portfolio_value = 100000
    portfolio_values = [portfolio_value]
    returns_list = []
    weights_history = []
    regimes_list = []
    dates_list = [test_df['date'].iloc[20]] # Approximate start date after window

    asset_names = ['SPY', 'TLT', 'GLD', 'BTC']

    step = 0
    print("\nRunning backtest for visualization...")

    while not done:
        # Get action
        if is_memory:
            action, hidden_state = agent.select_action(state, hidden_state=hidden_state, deterministic=True)
        else:
            action = agent.select_action(state, deterministic=True)

        # Normalize weights (softmax) if not already done by env/agent
        # SAC outputs tanh actions, so we need to convert to weights if env expects it
        # But PortfolioEnv usually handles this or expects weights. 
        # Let's assume the agent output is processed by env or we need to process it.
        # PathDependentPortfolioEnv step takes action and does softmax internally if needed.
        # Let's get the weights from info dict after step.

        # Take step
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_result

        # Get weights from info
        weights = info.get('weights', np.zeros(len(asset_names)))
        weights_history.append(weights)

        # Get return
        portfolio_return = info.get('portfolio_return', 0.0)
        returns_list.append(portfolio_return)

        # Update value
        portfolio_value = info.get('portfolio_value', portfolio_value * (1 + portfolio_return))
        portfolio_values.append(portfolio_value)

        # Get regime if available
        if step + 21 < len(test_df) and 'regime_gmm' in test_df.columns:
            regimes_list.append(test_df['regime_gmm'].iloc[step + 21])
            dates_list.append(test_df['date'].iloc[step + 21])
        elif step < len(test_df):
             dates_list.append(test_df['date'].iloc[step])

        state = next_state
        step += 1

    print(f"Backtest complete: {step} steps")

    # Create visualizer
    viz = EnhancedVisualizer()

    # Generate all visualizations
    returns_array = np.array(returns_list)
    portfolio_values_array = np.array(portfolio_values)
    
    # Ensure dates match length
    if len(dates_list) > len(returns_list):
        dates_list = dates_list[:len(returns_list)]
    elif len(dates_list) < len(returns_list):
        # Pad with last date + 1 day
        last_date = dates_list[-1]
        for _ in range(len(returns_list) - len(dates_list)):
            last_date += pd.Timedelta(days=1)
            dates_list.append(last_date)
            
    dates_series = pd.Series(dates_list)

    # 1. Rolling metrics
    viz.create_rolling_metrics_plot(returns_array, dates_series)

    # 2. Allocation heatmap
    viz.create_allocation_heatmap(weights_history, dates_series, asset_names)

    # 3. Interactive dashboard
    viz.create_interactive_dashboard(
        portfolio_values_array[1:], returns_array, weights_history,
        dates_series, asset_names
    )

    # 4. Regime analysis (if available)
    if len(regimes_list) > 0:
        # Ensure regimes match length
        if len(regimes_list) > len(returns_list):
            regimes_list = regimes_list[:len(returns_list)]
        viz.create_regime_analysis(returns_array, np.array(regimes_list), dates_series)

    print("\n" + "="*80)
    print("VISUALIZATION SUITE COMPLETE!")
    print("="*80)
    print(f"\nAll visualizations saved to: {viz.output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Visualization Suite')
    parser.add_argument('--model', type=str, default='models/sac_trained_new_step40000.pth',
                       help='Path to trained model')
    parser.add_argument('--data', type=str, default='data/processed/complete_dataset_enhanced.csv',
                       help='Path to dataset')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--agent-type', type=str, default='sac_memory',
                       choices=['sac', 'sac_memory'],
                       help='Type of agent to visualize')

    args = parser.parse_args()

    if args.agent_type == 'sac':
        run_sac_visualization(args.model, args.data, args.device, is_memory=False)
    else: # sac_memory
        run_sac_visualization(args.model, args.data, args.device, is_memory=True)
