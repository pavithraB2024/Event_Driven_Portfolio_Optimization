"""
Train Regime Detection Models (GMM and HMM)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.regime_detection.gmm_classifier import GMMRegimeDetector
try:
    from src.regime_detection.hmm_classifier import HMMRegimeDetector
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not installed. HMM training will be skipped.")

def main():
    print("=" * 80)
    print("REGIME DETECTION MODEL TRAINING")
    print("=" * 80)

    # Load processed data
    print("\n1. Loading processed dataset...")
    try:
        data = pd.read_csv("data/processed/complete_dataset.csv", index_col=0, parse_dates=True)
        print(f"   Loaded: {data.shape}")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    except FileNotFoundError:
        print("   ERROR: Processed dataset not found!")
        print("   Please run: python scripts/simple_preprocess.py first")
        return

    # Extract returns and VIX
    return_cols = [col for col in data.columns if col.startswith('return_')]
    returns = data[return_cols].copy()
    vix = data['VIX'].copy()

    print("\n2. Features for regime detection:")
    print(f"   - Returns: {returns.shape}")
    print(f"   - VIX: {vix.shape}")

    # Train GMM
    print("\n" + "=" * 80)
    print("TRAINING GAUSSIAN MIXTURE MODEL (GMM)")
    print("=" * 80)

    gmm_detector = GMMRegimeDetector(n_regimes=3, random_state=42)
    # Use expanding window for causal regime detection (prevents look-ahead bias)
    print("   Running expanding window GMM (causal)...")
    gmm_regimes = gmm_detector.fit_predict_expanding(
        returns, 
        vix, 
        min_window=252, 
        refit_freq=63
    )

    # Fit final model on all data for deployment/saving
    print("   Fitting final model on full dataset for deployment...")
    gmm_detector.fit(returns, vix)

    # Get statistics
    gmm_stats = gmm_detector.get_regime_statistics(returns, vix)
    print("\nGMM Regime Statistics:")
    print(gmm_stats.to_string(index=False))

    # Save GMM model
    os.makedirs("models", exist_ok=True)
    gmm_detector.save("models/gmm_regime_detector.pkl")

    # Save regime labels
    os.makedirs("data/regime_labels", exist_ok=True)
    gmm_regimes.to_csv("data/regime_labels/gmm_regimes.csv")
    print("\nGMM regimes saved to data/regime_labels/gmm_regimes.csv")

    # Train HMM
    hmm_regimes = None
    if HMM_AVAILABLE:
        print("\n" + "=" * 80)
        print("TRAINING HIDDEN MARKOV MODEL (HMM)")
        print("=" * 80)

        hmm_detector = HMMRegimeDetector(n_regimes=3, n_iter=100, random_state=42)
        hmm_detector.fit(returns)

        # Predict regimes
        hmm_regimes = hmm_detector.predict(returns)

        # Get statistics
        hmm_stats = hmm_detector.get_regime_statistics(returns)
        print("\nHMM Regime Statistics:")
        print(hmm_stats.to_string(index=False))

        # Transition matrix
        print("\nHMM Transition Matrix:")
        transition_matrix = hmm_detector.get_transition_matrix()
        print(transition_matrix.round(3).to_string())

        # Save HMM model
        hmm_detector.save("models/hmm_regime_detector.pkl")

        # Save regime labels
        hmm_regimes.to_csv("data/regime_labels/hmm_regimes.csv")
        print("\nHMM regimes saved to data/regime_labels/hmm_regimes.csv")

        # Compare regime assignments
        print("\n" + "=" * 80)
        print("REGIME COMPARISON (GMM vs HMM)")
        print("=" * 80)

        # Align regimes
        common_idx = gmm_regimes.index.intersection(hmm_regimes.index)
        gmm_aligned = gmm_regimes.loc[common_idx]
        hmm_aligned = hmm_regimes.loc[common_idx]

        # Agreement percentage
        agreement = (gmm_aligned == hmm_aligned).sum() / len(common_idx) * 100
        print(f"\nRegime agreement: {agreement:.1f}%")

        # Cross-tabulation
        print("\nCross-tabulation (GMM vs HMM):")
        crosstab = pd.crosstab(gmm_aligned, hmm_aligned, margins=True)
        print(crosstab)
    else:
        print("\nSkipping HMM training and comparison (hmmlearn not installed).")

    # Add regimes to dataset
    print("\n" + "=" * 80)
    print("ADDING REGIMES TO DATASET")
    print("=" * 80)

    data_with_regimes = data.copy()
    data_with_regimes['regime_gmm'] = gmm_regimes
    
    # Fill any missing regime values with mode (for the initial warm-up period)
    if 'regime_gmm' in data_with_regimes.columns and not data_with_regimes['regime_gmm'].dropna().empty:
         data_with_regimes['regime_gmm'].fillna(data_with_regimes['regime_gmm'].mode()[0], inplace=True)
    
    if hmm_regimes is not None:
        data_with_regimes['regime_hmm'] = hmm_regimes
        if 'regime_hmm' in data_with_regimes.columns and not data_with_regimes['regime_hmm'].dropna().empty:
             data_with_regimes['regime_hmm'].fillna(data_with_regimes['regime_hmm'].mode()[0], inplace=True)
    else:
        # If HMM not available, just fill with 0 or skip
        data_with_regimes['regime_hmm'] = 0

    # Save enhanced dataset
    data_with_regimes.to_csv("data/processed/dataset_with_regimes.csv")
    print(f"\nEnhanced dataset saved: {data_with_regimes.shape}")
    print("File: data/processed/dataset_with_regimes.csv")

    print("\n" + "=" * 80)
    print("REGIME TRAINING COMPLETE!")
    print("=" * 80)
    print("\nSaved files:")
    print("  - models/gmm_regime_detector.pkl")
    print("  - models/hmm_regime_detector.pkl")
    print("  - data/regime_labels/gmm_regimes.csv")
    print("  - data/regime_labels/hmm_regimes.csv")
    print("  - data/processed/dataset_with_regimes.csv")
    print("\nNext step:")
    print("  python scripts/train_dqn.py --episodes 500")
    print("=" * 80)

if __name__ == "__main__":
    main()
