"""
Hidden Markov Model for Market Regime Detection
Models regime transitions as a Markov process
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
import joblib
import os


class HMMRegimeDetector:
    """Detect market regimes using Hidden Markov Models."""

    def __init__(
        self,
        n_regimes: int = 3,
        n_iter: int = 100,
        random_state: int = 42
    ):
        """
        Initialize HMM regime detector.

        Args:
            n_regimes: Number of hidden states (regimes)
            n_iter: Maximum number of EM iterations
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state

        # Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state
        )

        self.regime_names = None
        self.is_fitted = False

    def prepare_features(
        self,
        returns: pd.DataFrame
    ) -> np.ndarray:
        """
        Prepare features for HMM.

        Args:
            returns: Asset returns

        Returns:
            Feature matrix
        """
        # Use returns as observable states
        features = returns.values

        # Remove NaN
        features = features[~np.isnan(features).any(axis=1)]

        return features

    def fit(
        self,
        returns: pd.DataFrame
    ) -> 'HMMRegimeDetector':
        """
        Fit HMM to returns data.

        Args:
            returns: Asset returns

        Returns:
            self
        """
        print(f"Fitting HMM with {self.n_regimes} hidden states...")

        # Prepare features
        features = self.prepare_features(returns)

        # Fit HMM
        self.model.fit(features)

        # Assign regime names
        self.regime_names = self._assign_regime_names(features)

        self.is_fitted = True
        print("HMM fitting complete!")
        print(f"Converged: {self.model.monitor_.converged}")

        return self

    def predict(
        self,
        returns: pd.DataFrame
    ) -> pd.Series:
        """
        Predict most likely regime sequence using Viterbi algorithm.

        Args:
            returns: Asset returns

        Returns:
            Series of regime labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        features = self.prepare_features(returns)

        # Viterbi algorithm for most likely state sequence
        regimes = self.model.predict(features)

        regime_series = pd.Series(
            regimes,
            index=returns.index[:len(regimes)],
            name='regime'
        )

        return regime_series

    def predict_proba(
        self,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict regime probabilities (forward-backward algorithm).

        Args:
            returns: Asset returns

        Returns:
            DataFrame with probabilities for each regime
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        features = self.prepare_features(returns)

        # Forward-backward algorithm
        probabilities = self.model.predict_proba(features)

        prob_df = pd.DataFrame(
            probabilities,
            index=returns.index[:len(probabilities)],
            columns=[f'Regime_{i}' for i in range(self.n_regimes)]
        )

        return prob_df

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get regime transition probability matrix.

        Returns:
            DataFrame of transition probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        transition_df = pd.DataFrame(
            self.model.transmat_,
            index=[self.regime_names.get(i, f'Regime_{i}') for i in range(self.n_regimes)],
            columns=[self.regime_names.get(i, f'Regime_{i}') for i in range(self.n_regimes)]
        )

        return transition_df

    def _assign_regime_names(self, features: np.ndarray) -> dict:
        """
        Assign names to regimes based on mean returns.

        Args:
            features: Feature matrix

        Returns:
            Dictionary mapping regime ID to name
        """
        regimes = self.model.predict(features)

        # Calculate mean return for each regime
        regime_means = []

        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            mean_return = features[mask].mean()
            regime_means.append((regime_id, mean_return))

        # Sort by mean return
        sorted_regimes = sorted(regime_means, key=lambda x: x[1])

        regime_names = {}

        if self.n_regimes == 3:
            regime_names[sorted_regimes[0][0]] = 'Bear'
            regime_names[sorted_regimes[2][0]] = 'Bull'
            regime_names[sorted_regimes[1][0]] = 'Volatile'
        else:
            for idx, (regime_id, _) in enumerate(sorted_regimes):
                regime_names[regime_id] = f'Regime_{idx}'

        return regime_names

    def get_regime_statistics(
        self,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get statistics for each regime.

        Args:
            returns: Asset returns

        Returns:
            DataFrame with regime statistics
        """
        regimes = self.predict(returns)

        stats = []

        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            regime_returns = returns[mask]

            regime_name = self.regime_names.get(regime_id, f'Regime_{regime_id}')

            # Calculate average duration
            regime_array = regimes.values
            durations = []
            current_duration = 0

            for i in range(len(regime_array)):
                if regime_array[i] == regime_id:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                    current_duration = 0

            avg_duration = np.mean(durations) if durations else 0

            stats.append({
                'Regime': regime_name,
                'ID': regime_id,
                'Count': mask.sum(),
                'Percentage': f"{mask.sum() / len(regimes) * 100:.1f}%",
                'Avg_Return': regime_returns.mean().mean(),
                'Avg_Volatility': regime_returns.std().mean(),
                'Avg_Duration_Days': f"{avg_duration:.1f}"
            })

        return pd.DataFrame(stats)

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'regime_names': self.regime_names,
            'n_regimes': self.n_regimes,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> 'HMMRegimeDetector':
        """Load model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.regime_names = data['regime_names']
        self.n_regimes = data['n_regimes']
        self.is_fitted = data['is_fitted']
        print(f"Model loaded from {filepath}")
        return self


if __name__ == "__main__":
    # Example usage
    dates = pd.date_range(start='2015-01-01', end='2023-01-01', freq='D')

    # Simulate regime-switching returns
    np.random.seed(42)
    returns_data = []

    for i in range(len(dates)):
        # Simple regime switching
        if i < len(dates) // 3:
            # Bull market
            ret = np.random.normal(0.001, 0.01, 2)
        elif i < 2 * len(dates) // 3:
            # Bear market
            ret = np.random.normal(-0.001, 0.015, 2)
        else:
            # Volatile
            ret = np.random.normal(0.0, 0.02, 2)

        returns_data.append(ret)

    returns = pd.DataFrame(
        returns_data,
        index=dates,
        columns=['SPY', 'TLT']
    )

    # Fit HMM
    detector = HMMRegimeDetector(n_regimes=3)
    detector.fit(returns)

    # Predict regimes
    regimes = detector.predict(returns)

    # Get statistics
    stats = detector.get_regime_statistics(returns)
    print("\nRegime Statistics:")
    print(stats)

    # Transition matrix
    print("\nTransition Matrix:")
    print(detector.get_transition_matrix())

    # Save model
    detector.save("models/hmm_regime_detector.pkl")
