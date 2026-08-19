"""
Visualization Module
Creates plots for regime analysis, performance comparison, and portfolio analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class PortfolioVisualizer:
    """Create visualizations for portfolio analysis."""

    def __init__(self, save_dir: str = "docs/figures"):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/regimes", exist_ok=True)
        os.makedirs(f"{save_dir}/performance", exist_ok=True)
        os.makedirs(f"{save_dir}/eda", exist_ok=True)

    def plot_price_trajectories(
        self,
        data: pd.DataFrame,
        save_name: str = "price_trajectories.png"
    ) -> None:
        """
        Plot asset price trajectories.

        Args:
            data: DataFrame with price columns
            save_name: Filename to save plot
        """
        price_cols = [col for col in data.columns if col.startswith('price_')]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, col in enumerate(price_cols):
            asset_name = col.replace('price_', '')
            axes[i].plot(data.index, data[col], linewidth=1.5, color=f'C{i}')
            axes[i].set_title(f'{asset_name} Price History', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Price ($)')
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "eda", save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()

    def plot_regime_colored_prices(
        self,
        data: pd.DataFrame,
        regime_col: str = 'regime_gmm',
        price_col: str = 'price_SPY',
        save_name: str = "regime_colored_prices.png"
    ) -> None:
        """
        Plot prices colored by market regime.

        Args:
            data: DataFrame with prices and regimes
            regime_col: Name of regime column
            price_col: Name of price column
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        regime_colors = {0: 'green', 1: 'red', 2: 'orange'}
        regime_names = {0: 'Bull', 1: 'Bear', 2: 'Volatile'}

        for regime_id in sorted(data[regime_col].unique()):
            mask = data[regime_col] == regime_id
            ax.scatter(
                data.index[mask],
                data[price_col][mask],
                c=regime_colors.get(regime_id, 'gray'),
                label=regime_names.get(regime_id, f'Regime {regime_id}'),
                alpha=0.6,
                s=10
            )

        asset_name = price_col.replace('price_', '')
        ax.set_title(f'{asset_name} Prices Colored by Market Regime ({regime_col.upper()})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "regimes", save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()

    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        save_name: str = "correlation_matrix.png"
    ) -> None:
        """
        Plot correlation matrix heatmap.

        Args:
            data: DataFrame with return columns
            save_name: Filename to save plot
        """
        return_cols = [col for col in data.columns if col.startswith('return_')]
        returns = data[return_cols].copy()
        returns.columns = [col.replace('return_', '') for col in returns.columns]

        corr = returns.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        ax.set_title('Asset Return Correlation Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "eda", save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()

    def plot_volatility_timeseries(
        self,
        data: pd.DataFrame,
        save_name: str = "volatility_timeseries.png"
    ) -> None:
        """
        Plot volatility time series with VIX overlay.

        Args:
            data: DataFrame with volatility and VIX
            save_name: Filename to save plot
        """
        vol_cols = [col for col in data.columns if col.startswith('volatility_')]

        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Plot asset volatilities
        for col in vol_cols[:2]:  # Plot first 2 assets
            asset_name = col.replace('volatility_', '')
            ax1.plot(data.index, data[col], label=asset_name, alpha=0.7, linewidth=1.5)

        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Asset Volatility (Annualized)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # Overlay VIX
        if 'VIX' in data.columns:
            ax2 = ax1.twinx()
            ax2.plot(data.index, data['VIX'], color='red', label='VIX', alpha=0.5, linewidth=2)
            ax2.set_ylabel('VIX Level', fontsize=12, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(loc='upper right')

        plt.title('Asset Volatility with VIX Overlay', fontsize=14, fontweight='bold')
        plt.tight_layout()

        filepath = os.path.join(self.save_dir, "eda", save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()

    def plot_regime_statistics(
        self,
        regime_stats: pd.DataFrame,
        save_name: str = "regime_statistics.png"
    ) -> None:
        """
        Plot regime statistics as bar chart.

        Args:
            regime_stats: DataFrame with regime statistics
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Count
        axes[0].bar(regime_stats['Regime'], regime_stats['Count'], color=['green', 'red', 'orange'])
        axes[0].set_title('Regime Frequency', fontweight='bold')
        axes[0].set_ylabel('Number of Days')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Average Return
        colors = ['green' if x > 0 else 'red' for x in regime_stats['Avg_Return']]
        axes[1].bar(regime_stats['Regime'], regime_stats['Avg_Return'] * 252, color=colors)
        axes[1].set_title('Annualized Average Return by Regime', fontweight='bold')
        axes[1].set_ylabel('Return (%)')
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[1].grid(True, alpha=0.3, axis='y')

        # Average Volatility
        axes[2].bar(regime_stats['Regime'], regime_stats['Avg_Volatility'] * 100,
                   color=['green', 'red', 'orange'])
        axes[2].set_title('Average Volatility by Regime', fontweight='bold')
        axes[2].set_ylabel('Volatility (%)')
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "regimes", save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()

    def plot_wealth_comparison(
        self,
        strategies: Dict[str, np.ndarray],
        dates: pd.DatetimeIndex,
        save_name: str = "wealth_comparison.png"
    ) -> None:
        """
        Plot wealth trajectories for multiple strategies.

        Args:
            strategies: Dict of strategy_name -> portfolio_values
            dates: DatetimeIndex for x-axis
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for i, (name, values) in enumerate(strategies.items()):
            ax.plot(dates[:len(values)], values, label=name,
                   linewidth=2, color=colors[i % len(colors)])

        ax.set_title('Portfolio Wealth Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # Add shaded regions for bear markets if available
        ax.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Value')

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "performance", save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()

    def plot_drawdown_comparison(
        self,
        strategies: Dict[str, np.ndarray],
        dates: pd.DatetimeIndex,
        save_name: str = "drawdown_comparison.png"
    ) -> None:
        """
        Plot drawdown comparison.

        Args:
            strategies: Dict of strategy_name -> portfolio_values
            dates: DatetimeIndex for x-axis
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        for name, values in strategies.items():
            running_max = np.maximum.accumulate(values)
            drawdown = (values - running_max) / running_max * 100
            ax.plot(dates[:len(drawdown)], drawdown, label=name, linewidth=2)

        ax.fill_between(dates[:len(drawdown)], 0, drawdown, alpha=0.3)
        ax.set_title('Drawdown Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "performance", save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()

    def plot_risk_return_scatter(
        self,
        strategies_metrics: Dict[str, Dict[str, float]],
        save_name: str = "risk_return_scatter.png"
    ) -> None:
        """
        Plot risk-return scatter plot.

        Args:
            strategies_metrics: Dict of strategy -> {'return': x, 'volatility': y}
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        for name, metrics in strategies_metrics.items():
            ax.scatter(
                metrics['volatility'] * 100,
                metrics['return'] * 100,
                s=200,
                alpha=0.7,
                label=name
            )
            ax.annotate(
                name,
                (metrics['volatility'] * 100, metrics['return'] * 100),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10
            )

        ax.set_title('Risk-Return Profile', fontsize=16, fontweight='bold')
        ax.set_xlabel('Volatility (Annualized %)', fontsize=12)
        ax.set_ylabel('Return (Annualized %)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True, shadow=True)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "performance", save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()


if __name__ == "__main__":
    print("Visualization module loaded. Use PortfolioVisualizer class to create plots.")
