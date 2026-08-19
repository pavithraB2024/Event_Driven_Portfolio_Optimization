"""
Feature Engineering Module
Computes technical indicators and momentum signals
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Generate technical indicators and trading signals."""

    @staticmethod
    def calculate_rsi(
        prices: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            prices: Price series
            window: Look-back period

        Returns:
            RSI values (0-100)
        """
        delta = prices.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with MACD, signal line, and histogram
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Price series
            window: Look-back period
            num_std: Number of standard deviations

        Returns:
            DataFrame with upper, middle, and lower bands
        """
        middle_band = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)

        return pd.DataFrame({
            'BB_Upper': upper_band,
            'BB_Middle': middle_band,
            'BB_Lower': lower_band
        })

    @staticmethod
    def calculate_momentum(
        prices: pd.Series,
        window: int = 10
    ) -> pd.Series:
        """
        Calculate price momentum.

        Args:
            prices: Price series
            window: Look-back period

        Returns:
            Momentum values
        """
        return prices.diff(window)

    @staticmethod
    def calculate_moving_averages(
        prices: pd.Series,
        windows: list = [20, 50, 200]
    ) -> pd.DataFrame:
        """
        Calculate multiple moving averages.

        Args:
            prices: Price series
            windows: List of MA periods

        Returns:
            DataFrame with all MAs
        """
        mas = pd.DataFrame(index=prices.index)

        for window in windows:
            mas[f'MA_{window}'] = prices.rolling(window=window).mean()

        return mas

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        window: int = 60,
        risk_free_rate: float = 0.02
    ) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.

        Args:
            returns: Return series
            window: Rolling window
            risk_free_rate: Annual risk-free rate

        Returns:
            Rolling Sharpe ratio
        """
        # Convert annual rate to daily
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1

        excess_returns = returns - daily_rf
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()

        sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

        return sharpe

    @staticmethod
    def calculate_drawdown(prices: pd.Series) -> pd.DataFrame:
        """
        Calculate drawdown metrics.

        Args:
            prices: Price series

        Returns:
            DataFrame with running max and drawdown
        """
        running_max = prices.cummax()
        drawdown = (prices - running_max) / running_max

        return pd.DataFrame({
            'Running_Max': running_max,
            'Drawdown': drawdown
        })

    def generate_all_features(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate complete feature set for all assets.

        Args:
            prices: Asset prices
            returns: Asset returns

        Returns:
            DataFrame with all technical features
        """
        features = pd.DataFrame(index=prices.index)

        for asset in prices.columns:
            # RSI
            features[f'RSI_{asset}'] = self.calculate_rsi(prices[asset])

            # MACD
            macd = self.calculate_macd(prices[asset])
            features[f'MACD_{asset}'] = macd['MACD']
            features[f'MACD_Signal_{asset}'] = macd['Signal']

            # Bollinger Bands
            bb = self.calculate_bollinger_bands(prices[asset])
            features[f'BB_Upper_{asset}'] = bb['BB_Upper']
            features[f'BB_Lower_{asset}'] = bb['BB_Lower']

            # Momentum
            features[f'Momentum_10_{asset}'] = self.calculate_momentum(prices[asset], window=10)
            features[f'Momentum_30_{asset}'] = self.calculate_momentum(prices[asset], window=30)

            # Moving averages
            mas = self.calculate_moving_averages(prices[asset], windows=[20, 50])
            features[f'MA_20_{asset}'] = mas['MA_20']
            features[f'MA_50_{asset}'] = mas['MA_50']

            # Sharpe ratio
            features[f'Sharpe_60_{asset}'] = self.calculate_sharpe_ratio(returns[asset], window=60)

            # Drawdown
            dd = self.calculate_drawdown(prices[asset])
            features[f'Drawdown_{asset}'] = dd['Drawdown']

        return features.dropna()


if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()

    # Mock data
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
    prices = pd.DataFrame({
        'SPY': 300 + np.cumsum(np.random.randn(len(dates)) * 2),
        'TLT': 140 + np.cumsum(np.random.randn(len(dates)) * 1)
    }, index=dates)

    returns = prices.pct_change()

    # Generate features
    features = engineer.generate_all_features(prices, returns)

    print("Generated features:")
    print(features.columns.tolist())
    print(f"\nFeature shape: {features.shape}")
