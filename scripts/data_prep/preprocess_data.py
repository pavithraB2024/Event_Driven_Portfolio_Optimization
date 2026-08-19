"""
Data Preprocessing Script
Prepares complete dataset for RL training
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.data_pipeline.preprocessing import DataPreprocessor
from src.data_pipeline.features import FeatureEngineer

def main():
    print("=" * 80)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 80)

    # Load raw data
    print("\n1. Loading raw data...")
    try:
        prices = pd.read_csv("data/raw/asset_prices_1d.csv", index_col=0, parse_dates=True)
        vix = pd.read_csv("data/raw/vix.csv", index_col=0, parse_dates=True).squeeze()

        # Try to load treasury, generate mock if not available
        try:
            treasury = pd.read_csv("data/raw/treasury_10y.csv", index_col=0, parse_dates=True).squeeze()
        except FileNotFoundError:
            print("   ! Treasury data not found, generating mock data...")
            treasury = pd.Series(
                data=2.5 + 0.5 * pd.Series(range(len(prices))).pct_change().fillna(0).cumsum(),
                index=prices.index,
                name='DGS10'
            ).clip(lower=0.5, upper=5.0)
            treasury.to_csv("data/raw/treasury_10y.csv")

        print(f"   OK Prices: {prices.shape}")
        print(f"   OK VIX: {vix.shape}")
        print(f"   OK Treasury: {treasury.shape}")
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        print("   Please run: python src/data_pipeline/download.py first")
        return

    # Prepare dataset
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    dataset = preprocessor.prepare_dataset(prices, vix, treasury, save=True)

    print(f"   OK Complete dataset: {dataset.shape}")
    print(f"   OK Date range: {dataset.index[0]} to {dataset.index[-1]}")
    print(f"   OK Trading days: {len(dataset)}")

    # Generate additional features
    print("\n3. Generating technical features...")
    engineer = FeatureEngineer()

    # Extract price and return columns
    price_cols = [col for col in dataset.columns if col.startswith('price_')]
    return_cols = [col for col in dataset.columns if col.startswith('return_')]

    prices_for_features = dataset[price_cols].copy()
    prices_for_features.columns = [col.replace('price_', '') for col in prices_for_features.columns]

    returns_for_features = dataset[return_cols].copy()
    returns_for_features.columns = [col.replace('return_', '') for col in returns_for_features.columns]

    # Generate features
    features = engineer.generate_all_features(prices_for_features, returns_for_features)

    print(f"   OK Technical features generated: {features.shape[1]} features")

    # Combine with original dataset
    dataset_with_features = pd.concat([dataset, features], axis=1)
    dataset_with_features = dataset_with_features.loc[:, ~dataset_with_features.columns.duplicated()]
    dataset_with_features = dataset_with_features.dropna()

    # Save enhanced dataset
    dataset_with_features.to_csv("data/processed/complete_dataset_enhanced.csv")
    print(f"   OK Enhanced dataset saved: {dataset_with_features.shape}")

    # Display summary statistics
    print("\n4. Dataset Summary:")
    print("-" * 80)
    print(f"Total features: {dataset_with_features.shape[1]}")
    print(f"Total observations: {dataset_with_features.shape[0]}")
    print("\nColumn categories:")

    price_count = len([c for c in dataset_with_features.columns if c.startswith('price_')])
    return_count = len([c for c in dataset_with_features.columns if c.startswith('return_')])
    vol_count = len([c for c in dataset_with_features.columns if c.startswith('volatility_')])
    rsi_count = len([c for c in dataset_with_features.columns if c.startswith('RSI_')])
    macd_count = len([c for c in dataset_with_features.columns if c.startswith('MACD_')])

    print(f"   - Prices: {price_count}")
    print(f"   - Returns: {return_count}")
    print(f"   - Volatility: {vol_count}")
    print(f"   - RSI: {rsi_count}")
    print(f"   - MACD: {macd_count}")
    print("   - Macro indicators: 2 (VIX, Treasury)")

    print("\n5. Data splits for training:")
    train_size = int(len(dataset_with_features) * 0.8)
    test_size = len(dataset_with_features) - train_size

    print(f"   - Training set: {train_size} days ({train_size/252:.1f} years)")
    print(f"   - Test set: {test_size} days ({test_size/252:.1f} years)")
    print(f"   - Train end date: {dataset_with_features.index[train_size-1]}")
    print(f"   - Test start date: {dataset_with_features.index[train_size]}")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)
    print("\nOutput files:")
    print("   - data/processed/complete_dataset.csv")
    print("   - data/processed/complete_dataset_enhanced.csv")
    print("\nNext steps:")
    print("   1. Train regime detection: python scripts/train_regime_models.py")
    print("   2. Train DQN agent: python scripts/train_dqn.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
