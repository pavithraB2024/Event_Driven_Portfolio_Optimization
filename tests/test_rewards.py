import numpy as np
import sys
import os

# Add main repo path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.environments.reward_calculator import log_utility_reward

def test_log_utility_reward_positive_return():
    # p_t-1 = 100, p_t = 110 -> ln(110/100) = ~0.0953
    result = log_utility_reward(100.0, 110.0)
    assert np.isclose(result, np.log(110.0 / 100.0))

def test_log_utility_reward_negative_return():
    # p_t-1 = 100, p_t = 90 -> ln(90/100) = ~-0.1053
    result = log_utility_reward(100.0, 90.0)
    assert np.isclose(result, np.log(90.0 / 100.0))

def test_log_utility_reward_bankruptcy():
    # p_t <= 0 -> -10.0
    result = log_utility_reward(100.0, 0.0)
    assert result == -10.0
    
    result2 = log_utility_reward(100.0, -5.0)
    assert result2 == -10.0
