"""State feature extraction for the portfolio environment."""
import numpy as np
import pandas as pd
from typing import List, Optional
from abc import ABC, abstractmethod


# ───────────────────────────────────────────────────────────────────
# Pure Functions — Feature Engineering
# ───────────────────────────────────────────────────────────────────

def inject_observation_noise(
    obs: np.ndarray,
    noise_std: float = 0.001,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add Gaussian noise to observation vector for regularisation.
    Breaks deterministic state transitions → reduces overfitting.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, noise_std, size=obs.shape).astype(obs.dtype)
    return obs + noise


def calculate_regime_features(
    returns_window: np.ndarray,
    current_value: float,
    peak_value: float,
) -> np.ndarray:
    """
    Compute 4 macro-regime features for richer state representation.
    """
    if len(returns_window) < 2:
        return np.zeros(4, dtype=np.float32)

    momentum = float(np.mean(returns_window))
    dispersion = float(np.std(returns_window))

    n = len(returns_window)
    mean_r = np.mean(returns_window)
    std_r = np.std(returns_window)
    if std_r > 1e-8 and n > 2:
        skewness = float(
            (n / ((n - 1) * (n - 2)))
            * np.sum(((returns_window - mean_r) / std_r) ** 3)
        )
    else:
        skewness = 0.0

    dd_norm = (peak_value - current_value) / peak_value if peak_value > 0 else 0.0

    features = np.array([momentum, dispersion, skewness, dd_norm], dtype=np.float32)
    return np.clip(features, -10.0, 10.0)


def apply_deeppocket_normalization(
    close_prices: np.ndarray,
    open_prices: np.ndarray,
) -> np.ndarray:
    """
    Normalize close prices relative to open prices (Soleymani, 2021).
    """
    safe_open = np.where(open_prices == 0, 1e-8, open_prices)
    return close_prices / safe_open


# ───────────────────────────────────────────────────────────────────
# Feature Extractor Interface
# ───────────────────────────────────────────────────────────────────

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        idx: int,
        data: pd.DataFrame,
        return_cols: List[str],
        portfolio_weights: np.ndarray,
        portfolio_value: float,
        initial_balance: float,
        peak_value: float,
        return_history_window: List[float],
        allocation_history: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        pass


class StandardFeatureExtractor(FeatureExtractor):
    def __init__(self, noise_augmentation: bool = False, noise_std: float = 0.001):
        self.noise_augmentation = noise_augmentation
        self.noise_std = noise_std
        self._rng = np.random.default_rng(42)

    def extract(
        self,
        idx: int,
        data: pd.DataFrame,
        return_cols: List[str],
        portfolio_weights: np.ndarray,
        portfolio_value: float,
        initial_balance: float,
        peak_value: float,
        return_history_window: List[float],
        allocation_history: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        state_components = [portfolio_weights]

        # Recent returns (last 5 days)
        recent_returns = []
        for col in return_cols:
            returns = data[col].iloc[idx-5:idx].values
            recent_returns.extend(returns)
        state_components.append(np.array(recent_returns))

        # Current volatility
        volatility = []
        for col in return_cols:
            vol = data[col].iloc[idx-20:idx].std()
            volatility.append(vol)
        state_components.append(np.array(volatility))

        # Market regime
        regime_col = None
        if 'regime' in data.columns:
            regime_col = 'regime'
        elif 'regime_gmm' in data.columns:
            regime_col = 'regime_gmm'
        elif 'regime_hmm' in data.columns:
            regime_col = 'regime_hmm'

        if regime_col:
            regime = data[regime_col].iloc[idx]
            regime_one_hot = np.zeros(3)
            if not np.isnan(regime):
                regime_one_hot[int(regime)] = 1.0
            state_components.append(regime_one_hot)

        # Macro indicators
        if 'VIX' in data.columns:
            vix = data['VIX'].iloc[idx]
            state_components.append(np.array([vix if not np.isnan(vix) else 20.0]))

        if 'Treasury_10Y' in data.columns:
            treasury = data['Treasury_10Y'].iloc[idx]
            state_components.append(np.array([treasury if not np.isnan(treasury) else 2.5]))

        # Portfolio value (normalized)
        portfolio_value_norm = portfolio_value / initial_balance
        state_components.append(np.array([portfolio_value_norm]))

        # Phase 3: Extra macro-regime features
        recent_portfolio_returns = np.array(return_history_window[-20:])
        regime_extra = calculate_regime_features(
            recent_portfolio_returns, portfolio_value, peak_value
        )
        state_components.append(regime_extra)

        # Concatenate
        state = np.concatenate([comp.flatten() for comp in state_components])
        state = np.nan_to_num(state, nan=0.0, posinf=1e30, neginf=-1e30)
        state = state.astype(np.float32)

        if self.noise_augmentation:
            state = inject_observation_noise(state, self.noise_std, self._rng)

        return state


class PathDependentFeatureExtractor(FeatureExtractor):
    def __init__(self, max_history_length: int = 20, n_assets: int = 1):
        self.max_history_length = max_history_length
        self.n_assets = n_assets

    def extract(
        self,
        idx: int,
        data: pd.DataFrame,
        return_cols: List[str],
        portfolio_weights: np.ndarray,
        portfolio_value: float,
        initial_balance: float,
        peak_value: float, # unused but included for interface
        return_history_window: List[float], # unused
        allocation_history: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        
        if allocation_history is None:
            allocation_history = []
            
        state_components = [portfolio_weights]

        # Recent returns (last 5 days)
        recent_returns = []
        for col in return_cols:
            returns = data[col].iloc[idx-5:idx].values
            recent_returns.extend(returns)
        state_components.append(np.array(recent_returns))

        # Current volatility
        volatility = []
        for col in return_cols:
            vol = data[col].iloc[idx-20:idx].std()
            volatility.append(vol)
        state_components.append(np.array(volatility))

        # Market regime 
        regime_col = None
        if 'regime' in data.columns:
            regime_col = 'regime'
        elif 'regime_gmm' in data.columns:
            regime_col = 'regime_gmm'
        elif 'regime_hmm' in data.columns:
            regime_col = 'regime_hmm'

        if regime_col:
            regime = data[regime_col].iloc[idx]
            regime_one_hot = np.zeros(3)
            if not np.isnan(regime):
                regime_one_hot[int(regime)] = 1.0
            state_components.append(regime_one_hot)

        # Macro indicators
        if 'VIX' in data.columns:
            vix = data['VIX'].iloc[idx]
            state_components.append(np.array([vix if not np.isnan(vix) else 20.0]))

        if 'Treasury_10Y' in data.columns:
            treasury = data['Treasury_10Y'].iloc[idx]
            state_components.append(np.array([treasury if not np.isnan(treasury) else 2.5]))

        # Portfolio value (normalized)
        portfolio_value_norm = portfolio_value / initial_balance
        state_components.append(np.array([portfolio_value_norm]))

        # Historical allocation information
        if len(allocation_history) > 0:
            n_history = min(self.max_history_length, len(allocation_history))
            recent_allocations = np.concatenate(allocation_history[-n_history:])
            padded_allocations = np.zeros(self.n_assets * self.max_history_length)
            padded_allocations[:len(recent_allocations)] = recent_allocations
            state_components.append(padded_allocations)
        else:
            history_padding = np.tile(portfolio_weights, self.max_history_length)
            state_components.append(history_padding)

        state = np.concatenate([comp.flatten() for comp in state_components])
        state = np.nan_to_num(state, nan=0.0)

        return state.astype(np.float32)
