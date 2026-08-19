"""
Gaussian Mixture Model for Market Regime Detection
Identifies bull/bear/volatile market states
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Optional
import joblib
import os


class GMMRegimeDetector:
    """Detect market regimes using Gaussian Mixture Models."""

    def __init__(
        self,
        n_regimes: int = 3,
        random_state: int = 42
    ):
        """
        Initialize GMM regime detector.

        Args:
            n_regimes: Number of market regimes (default: 3 for Bull/Bear/Volatile)
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = GaussianMixture(
            n_components=n_regimes,
            covariance_type='full',
            random_state=random_state,
            max_iter=200
        )
        self.regime_names = None
        self.is_fitted = False

    def prepare_features(
        self,
        returns: pd.DataFrame,
        vix: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Prepare features for regime detection.

        Args:
            returns: Asset returns
            vix: VIX volatility index (optional)

        Returns:
            Feature matrix for GMM
        """
        features_list = []

        # Mean return
        mean_return = returns.mean(axis=1)
        features_list.append(mean_return.values.reshape(-1, 1))

        # Volatility (rolling std)
        volatility = returns.std(axis=1)
        features_list.append(volatility.values.reshape(-1, 1))

        # VIX if available
        if vix is not None:
            vix_aligned = vix.reindex(returns.index)
            features_list.append(vix_aligned.values.reshape(-1, 1))

        # Combine features
        features = np.hstack(features_list)

        # Remove NaN values
        features = features[~np.isnan(features).any(axis=1)]

        return features

    def fit(
        self,
        returns: pd.DataFrame,
        vix: Optional[pd.Series] = None
    ) -> 'GMMRegimeDetector':
        """
        Fit GMM model to detect regimes.

        Args:
            returns: Asset returns
            vix: VIX volatility index

        Returns:
            self
        """
        print(f"Fitting GMM with {self.n_regimes} regimes...")

        # Prepare features
        features = self.prepare_features(returns, vix)

        # Fit GMM
        self.model.fit(features)

        # Assign regime names based on characteristics
        self.regime_names = self._assign_regime_names(features)

        self.is_fitted = True
        print("GMM fitting complete!")

        return self

    def predict(
        self,
        returns: pd.DataFrame,
        vix: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Predict market regimes.

        Args:
            returns: Asset returns
            vix: VIX volatility index

        Returns:
            Series of regime labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Prepare features
        features = self.prepare_features(returns, vix)

        # Predict regimes
        regimes = self.model.predict(features)

        # Create series with proper index
        regime_series = pd.Series(
            regimes,
            index=returns.index[:len(regimes)],
            name='regime'
        )

        return regime_series

    def predict_proba(
        self,
        returns: pd.DataFrame,
        vix: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Predict regime probabilities.

        Args:
            returns: Asset returns
            vix: VIX volatility index

        Returns:
            DataFrame with probabilities for each regime
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        features = self.prepare_features(returns, vix)
        probabilities = self.model.predict_proba(features)

        prob_df = pd.DataFrame(
            probabilities,
            index=returns.index[:len(probabilities)],
            columns=[f'Regime_{i}' for i in range(self.n_regimes)]
        )

        return prob_df

    def _assign_regime_names(self, features: np.ndarray) -> dict:
        """
        Assign human-readable names to regimes based on characteristics.

        Args:
            features: Feature matrix

        Returns:
            Dictionary mapping regime index to name
        """
        regimes = self.model.predict(features)

        # Calculate mean characteristics for each regime
        regime_characteristics = {}

        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            mean_return = features[mask, 0].mean()
            mean_volatility = features[mask, 1].mean()

            regime_characteristics[regime_id] = {
                'mean_return': mean_return,
                'mean_volatility': mean_volatility
            }

        # Assign names based on characteristics
        regime_names = {}

        # Sort regimes by return
        sorted_regimes = sorted(
            regime_characteristics.items(),
            key=lambda x: x[1]['mean_return']
        )

        # Lowest return = Bear
        # Highest return = Bull
        # Middle or high volatility = Volatile
        if self.n_regimes == 3:
            regime_names[sorted_regimes[0][0]] = 'Bear'
            regime_names[sorted_regimes[2][0]] = 'Bull'
            regime_names[sorted_regimes[1][0]] = 'Volatile'
        else:
            # Generic naming for other regime counts
            for idx, (regime_id, _) in enumerate(sorted_regimes):
                regime_names[regime_id] = f'Regime_{idx}'

        return regime_names

    def get_regime_statistics(
        self,
        returns: pd.DataFrame,
        vix: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Get statistics for each detected regime.

        Args:
            returns: Asset returns
            vix: VIX volatility index

        Returns:
            DataFrame with regime statistics
        """
        regimes = self.predict(returns, vix)

        stats = []

        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            regime_returns = returns[mask]

            regime_name = self.regime_names.get(regime_id, f'Regime_{regime_id}')

            stats.append({
                'Regime': regime_name,
                'ID': regime_id,
                'Count': mask.sum(),
                'Percentage': f"{mask.sum() / len(regimes) * 100:.1f}%",
                'Avg_Return': regime_returns.mean().mean(),
                'Avg_Volatility': regime_returns.std().mean()
            })

        return pd.DataFrame(stats)

    def fit_predict_expanding(
        self,
        returns: pd.DataFrame,
        vix: Optional[pd.Series] = None,
        min_window: int = 252,
        refit_freq: int = 63
    ) -> pd.Series:
        """
        Predict regimes using an expanding window to prevent look-ahead bias.

        Args:
            returns: Asset returns
            vix: VIX volatility index
            min_window: Minimum data points required before first prediction
            refit_freq: Number of days between model re-fits (optimization for speed)

        Returns:
            Series of regime labels (aligned with input index, NaN for initial window)
        """
        features = self.prepare_features(returns, vix)
        n_samples = len(features)
        regime_labels = np.full(n_samples, np.nan)

        # Initialize model
        current_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=self.random_state,
            max_iter=100
        )

        # We need to map clusters to semantic labels (Bull, Bear, Volatile) consistently
        last_refit = -1
        cluster_map = {}

        print(f"Starting expanding window GMM (min_window={min_window}, refit_freq={refit_freq})...")

        for t in range(min_window, n_samples):
            # Re-fit model periodically
            if t - last_refit >= refit_freq or last_refit == -1:
                # Train on data up to t (exclusive of t, or inclusive? strictly past data)
                # features[:t] includes indices 0 to t-1.
                train_data = features[:t]
                current_model.fit(train_data)
                last_refit = t

                # Determine mapping for this fit
                train_preds = current_model.predict(train_data)

                # Calculate mean return for each cluster
                cluster_stats = {}
                for c in range(self.n_regimes):
                    mask = train_preds == c
                    if mask.sum() > 0:
                        cluster_stats[c] = {
                            'mean_ret': train_data[mask, 0].mean()  # Assuming 0 is mean_return
                        }
                    else:
                        cluster_stats[c] = {'mean_ret': 0}

                # Sort clusters by Return (Lowest = Bear, Highest = Bull)
                sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1]['mean_ret'])

                # Create mapping: Cluster ID -> Label ID (0=Bear, 1=Volatile, 2=Bull)
                if self.n_regimes == 3:
                     cluster_map[sorted_clusters[0][0]] = 0 # Bear
                     cluster_map[sorted_clusters[1][0]] = 1 # Volatile
                     cluster_map[sorted_clusters[2][0]] = 2 # Bull
                else:
                    for i, (c_id, _) in enumerate(sorted_clusters):
                        cluster_map[c_id] = i

            # Predict for current step t
            current_feat = features[t].reshape(1, -1)
            cluster_pred = current_model.predict(current_feat)[0]

            # Map cluster to consistent label
            regime_labels[t] = cluster_map.get(cluster_pred, 1) # Default to 1 (Volatile) if issue

        return pd.Series(regime_labels, index=returns.index[:n_samples], name='regime')

    def save(self, filepath: str) -> None:
        """
        Save model to disk.

        Args:
            filepath: Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'regime_names': self.regime_names,
            'n_regimes': self.n_regimes,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> 'GMMRegimeDetector':
        """
        Load model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            self
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.regime_names = data['regime_names']
        self.n_regimes = data['n_regimes']
        self.is_fitted = data['is_fitted']
        print(f"Model loaded from {filepath}")
        return self


if __name__ == "__main__":
    # Example usage

    # Mock data
    dates = pd.date_range(start='2015-01-01', end='2023-01-01', freq='D')
    returns = pd.DataFrame({
        'SPY': np.random.randn(len(dates)) * 0.01,
        'TLT': np.random.randn(len(dates)) * 0.008
    }, index=dates)

    # Simulate VIX
    vix = pd.Series(
        15 + 10 * np.abs(np.random.randn(len(dates))),
        index=dates,
        name='VIX'
    )

    # Fit GMM
    detector = GMMRegimeDetector(n_regimes=3)
    detector.fit(returns, vix)

    # Predict regimes
    regimes = detector.predict(returns, vix)

    # Get statistics
    stats = detector.get_regime_statistics(returns, vix)
    print("\nRegime Statistics:")
    print(stats)

    # Save model
    detector.save("models/gmm_regime_detector.pkl")
