import numpy as np
from src.environments.feature_extractor import apply_deeppocket_normalization

def test_apply_deeppocket_normalization():
    """Should normalize close prices relative to open prices (or proxy)."""
    # [T, N]
    close_prices = np.array([[100.0, 50.0], [105.0, 48.0]])
    open_prices = np.array([[98.0, 51.0], [102.0, 49.0]])
    
    norm = apply_deeppocket_normalization(close_prices, open_prices)
    
    assert norm.shape == close_prices.shape
    # AAPL day 1: 100/98 approx 1.0204
    assert np.allclose(norm[0, 0], 100.0/98.0)
    # AMZN day 1: 50/51 approx 0.9804
    assert np.allclose(norm[0, 1], 50.0/51.0)

