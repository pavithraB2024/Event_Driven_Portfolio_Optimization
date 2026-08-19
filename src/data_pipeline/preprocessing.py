"""
Data Preprocessing Module
Cleans, normalizes, and prepares data for RL training
"""

import pandas as pd
import numpy as np
from typing import Tuple
import os


class DataPreprocessor:
    """Preprocess financial data for RL environment."""

    def __init__(self, processed_dir: str = "data/processed"):
        """
        Initialize preprocessor.

        Args:
            processed_dir: Directory to save processed data
        """
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

    def compute_returns(
        self,
        prices: pd.DataFrame,
        method: str = 'log'
    ) -> pd.DataFrame:
        """
        Calculate asset returns.

        Args:
            prices: DataFrame of asset prices
            method: 'log' for log returns or 'simple' for simple returns

        Returns:
            DataFrame of returns
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        elif method == 'simple':
            returns = prices.pct_change()
        else:
            raise ValueError(f"Unknown method: {method}")

        return returns.dropna()

    def calculate_rolling_volatility(
        self,
        returns: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility.

        Args:
            returns: DataFrame of returns
            window: Rolling window size in days

        Returns:
            DataFrame of rolling volatility (annualized)
        """
        # Annualization factor (252 trading days)
        annualization_factor = np.sqrt(252)

        volatility = returns.rolling(window=window).std() * annualization_factor
        return volatility.dropna()

    def calculate_rolling_correlation(
        self,
        returns: pd.DataFrame,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Calculate rolling correlation matrix.

        Args:
            returns: DataFrame of returns
            window: Rolling window size

        Returns:
            DataFrame with rolling correlations
        """
        # Calculate rolling correlation between first asset and others
        correlations = {}

        for col in returns.columns[1:]:
            correlations[f'corr_{returns.columns[0]}_{col}'] = (
                returns[returns.columns[0]]
                .rolling(window=window)
                .corr(returns[col])
            )

        return pd.DataFrame(correlations).dropna()

    def normalize_features(
        self,
        data: pd.DataFrame,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Normalize features for RL input.

        Args:
            data: DataFrame to normalize
            method: 'zscore' or 'minmax'

        Returns:
            Normalized DataFrame
        """
        if method == 'zscore':
            normalized = (data - data.mean()) / data.std()
        elif method == 'minmax':
            normalized = (data - data.min()) / (data.max() - data.min())
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized

    def align_data(
        self,
        *dataframes: pd.DataFrame
    ) -> Tuple[pd.DataFrame, ...]:
        """
        Align multiple dataframes by index (date).

        Args:
            *dataframes: Variable number of DataFrames to align

        Returns:
            Tuple of aligned DataFrames
        """
        # Find common index
        common_index = dataframes[0].index

        for df in dataframes[1:]:
            common_index = common_index.intersection(df.index)

        # Align all dataframes
        aligned = tuple(df.loc[common_index] for df in dataframes)

        return aligned

    def handle_missing_data(
        self,
        data: pd.DataFrame,
        method: str = 'forward_fill'
    ) -> pd.DataFrame:
        """
        Handle missing values in data.

        Args:
            data: DataFrame with potential missing values
            method: 'forward_fill', 'backward_fill', or 'drop'

        Returns:
            DataFrame with handled missing values
        """
        if method == 'forward_fill':
            return data.fillna(method='ffill')
        elif method == 'backward_fill':
            return data.fillna(method='bfill')
        elif method == 'drop':
            return data.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")

    def create_lagged_features(
        self,
        data: pd.DataFrame,
        lags: list = [1, 5, 20]
    ) -> pd.DataFrame:
        """
        Create lagged features for time series.

        Args:
            data: DataFrame of features
            lags: List of lag periods

        Returns:
            DataFrame with original and lagged features
        """
        lagged_data = data.copy()

        for col in data.columns:
            for lag in lags:
                lagged_data[f'{col}_lag_{lag}'] = data[col].shift(lag)

        return lagged_data.dropna()

    def prepare_dataset(
        self,
        prices: pd.DataFrame,
        vix: pd.Series,
        treasury: pd.Series,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Prepare complete dataset for RL training.

        Args:
            prices: Asset prices
            vix: VIX volatility index
            treasury: Treasury rates
            save: Whether to save processed data

        Returns:
            Complete prepared dataset
        """
        print("Preparing dataset...")

        # Calculate returns
        returns = self.compute_returns(prices, method='log')

        # Calculate rolling volatility
        volatility = self.calculate_rolling_volatility(returns, window=20)

        # Align all data
        vix_aligned = vix.to_frame(name='VIX')
        treasury_aligned = treasury.to_frame(name='Treasury_10Y')

        aligned_data = self.align_data(
            prices,
            returns,
            volatility,
            vix_aligned,
            treasury_aligned
        )

        prices_aligned, returns_aligned, vol_aligned, vix_aligned, treasury_aligned = aligned_data

        # Combine all features
        dataset = pd.DataFrame(index=returns_aligned.index)

        # Add prices
        for col in prices_aligned.columns:
            dataset[f'price_{col}'] = prices_aligned[col]

        # Add returns
        for col in returns_aligned.columns:
            dataset[f'return_{col}'] = returns_aligned[col]

        # Add volatility
        for col in vol_aligned.columns:
            dataset[f'volatility_{col}'] = vol_aligned[col]

        # Add macro indicators
        dataset['VIX'] = vix_aligned['VIX']

        # Only add Treasury if it has data for the common date range
        # If treasury_aligned has wrong dates, we'll need to resample it to match our index
        treasury_resampled = treasury_aligned['Treasury_10Y'].reindex(dataset.index, method='ffill')
        dataset['Treasury_10Y'] = treasury_resampled

        print(f"Dataset shape before missing data handling: {dataset.shape}")

        # Handle any remaining missing values
        dataset = self.handle_missing_data(dataset, method='forward_fill')
        print(f"Dataset shape after forward fill: {dataset.shape}")

        # Check if there are still any columns with all NaN values (shouldn't happen after ffill but just in case)
        all_nan_cols = dataset.columns[dataset.isna().all()]
        if len(all_nan_cols) > 0:
            print(f"Warning: Columns with all NaN values: {list(all_nan_cols)}")
            dataset = dataset.drop(columns=all_nan_cols)

        # Only drop rows where ALL feature values are NaN, keep rows with at least some data
        dataset = dataset.dropna(how='all')
        print(f"Dataset shape after dropping all-NaN rows: {dataset.shape}")

        print(f"Dataset shape: {dataset.shape}")
        print(f"Date range: {dataset.index[0]} to {dataset.index[-1]}")

        if save:
            filepath = os.path.join(self.processed_dir, "complete_dataset.csv")
            dataset.to_csv(filepath)
            print(f"Saved to {filepath}")

        return dataset


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()

    # Load raw data (assuming it exists)
    try:
        prices = pd.read_csv("data/raw/asset_prices_1d.csv", index_col=0, parse_dates=True)
        vix = pd.read_csv("data/raw/vix.csv", index_col=0, parse_dates=True).squeeze()
        treasury = pd.read_csv("data/raw/treasury_10y.csv", index_col=0, parse_dates=True).squeeze()

        # Prepare dataset
        dataset = preprocessor.prepare_dataset(prices, vix, treasury)

        print("\nDataset columns:")
        print(dataset.columns.tolist())

    except FileNotFoundError:
        print("Raw data not found. Please run download.py first.")
