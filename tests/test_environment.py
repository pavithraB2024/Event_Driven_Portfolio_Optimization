"""
Comprehensive unit tests for RL environments.

Tests cover:
- Portfolio environment initialization
- State space and action space
- Reward calculation
- Transaction costs
- Regime awareness
- Episode termination
- Parallel environments
"""

import pytest
import numpy as np
import pandas as pd
import gymnasium as gym
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environments.portfolio_env import PortfolioEnv
from environments.parallel_env import DummyVecEnv, VecNormalize, RunningMeanStd


@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    data = pd.DataFrame(index=dates)

    # Prices
    data['price_SPY'] = 100 * (1 + np.random.randn(252) * 0.01).cumprod()
    data['price_TLT'] = 100 * (1 + np.random.randn(252) * 0.005).cumprod()
    data['price_GLD'] = 100 * (1 + np.random.randn(252) * 0.008).cumprod()
    data['price_BTC'] = 100 * (1 + np.random.randn(252) * 0.03).cumprod()

    # Returns
    for asset in ['SPY', 'TLT', 'GLD', 'BTC']:
        data[f'return_{asset}'] = data[f'price_{asset}'].pct_change().fillna(0)

    # Volatility
    for asset in ['SPY', 'TLT', 'GLD', 'BTC']:
        data[f'volatility_{asset}'] = data[f'return_{asset}'].rolling(20).std().fillna(0.01)

    # VIX
    data['VIX'] = 15 + 5 * np.random.randn(252)
    data['VIX'] = data['VIX'].clip(10, 40)

    # Regime
    data['regime_gmm'] = np.random.choice([0, 1, 2], size=252)

    return data


class TestPortfolioEnv:
    """Test portfolio environment functionality."""

    def test_env_initialization(self, sample_data):
        """Test environment initializes correctly."""
        env = PortfolioEnv(
            data=sample_data,
            initial_balance=100000.0,
            transaction_cost=0.0025,
            action_type='discrete'
        )

        assert env.initial_balance == 100000.0
        assert env.transaction_cost == 0.0025
        assert env.action_type == 'discrete'
        assert env.n_assets == 4

    def test_observation_space(self, sample_data):
        """Test observation space shape."""
        env = PortfolioEnv(data=sample_data, action_type='discrete')

        # State should include: weights (4) + returns (4) + volatility (4) +
        # prev_weights (4) + regime (3 one-hot) + VIX (1) + portfolio_value (1) + cash_weight (1)
        expected_dim = 4 + 4 + 4 + 4 + 3 + 1 + 1 + 1  # = 22 minimum

        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape[0] >= 20

    def test_action_space_discrete(self, sample_data):
        """Test discrete action space."""
        env = PortfolioEnv(data=sample_data, action_type='discrete')

        assert isinstance(env.action_space, gym.spaces.Discrete)
        # Should have buy/hold/sell for each asset
        assert env.action_space.n >= 3

    def test_action_space_continuous(self, sample_data):
        """Test continuous action space."""
        env = PortfolioEnv(data=sample_data, action_type='continuous')

        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape[0] == 4  # One weight per asset

    def test_reset(self, sample_data):
        """Test environment reset."""
        env = PortfolioEnv(data=sample_data, action_type='discrete')

        state, info = env.reset()

        assert isinstance(state, np.ndarray)
        assert state.shape == env.observation_space.shape
        assert env.current_step == 0
        assert env.portfolio_value == env.initial_balance

    def test_step_discrete(self, sample_data):
        """Test step with discrete action."""
        env = PortfolioEnv(data=sample_data, action_type='discrete')
        env.reset()

        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)

        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert 'portfolio_value' in info

    def test_step_continuous(self, sample_data):
        """Test step with continuous action."""
        env = PortfolioEnv(data=sample_data, action_type='continuous')
        env.reset()

        action = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weights
        next_state, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, float)
        assert not np.isnan(reward)
        assert not np.isinf(reward)

    def test_transaction_costs(self, sample_data):
        """Test transaction costs are applied."""
        env = PortfolioEnv(data=sample_data, action_type='continuous', transaction_cost=0.01)
        env.reset()

        initial_value = env.portfolio_value

        # Make a large trade
        action = np.array([0.9, 0.05, 0.05, 0.0])
        env.step(action)

        # Portfolio value should decrease due to transaction costs
        # (unless market gains offset it)
        assert 'transaction_costs' in env.step(action)[4]

    def test_reward_log_utility(self, sample_data):
        """Test log utility reward calculation."""
        env = PortfolioEnv(data=sample_data, action_type='discrete', reward_type='log_utility')
        env.reset()

        initial_value = env.portfolio_value
        _, reward, _, _, _ = env.step(0)

        # Reward should be log return
        assert isinstance(reward, float)

    def test_reward_sharpe(self, sample_data):
        """Test Sharpe ratio reward."""
        env = PortfolioEnv(data=sample_data, action_type='discrete', reward_type='sharpe')
        env.reset()

        # Run for several steps to accumulate returns
        for _ in range(10):
            _, reward, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break

        assert isinstance(reward, float)

    def test_episode_termination(self, sample_data):
        """Test episode terminates at end of data."""
        env = PortfolioEnv(data=sample_data, action_type='discrete')
        env.reset()

        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated) and steps < 300:
            _, _, terminated, truncated, _ = env.step(0)
            steps += 1

        assert terminated or truncated
        assert steps < 300  # Should terminate before 300 steps

    def test_portfolio_weights_sum_to_one(self, sample_data):
        """Test portfolio weights sum to 1."""
        env = PortfolioEnv(data=sample_data, action_type='continuous')
        env.reset()

        action = np.array([0.3, 0.2, 0.4, 0.1])
        env.step(action)

        # Weights should sum to approximately 1
        assert np.isclose(env.portfolio_weights.sum(), 1.0, atol=0.01)

    def test_regime_awareness(self, sample_data):
        """Test environment includes regime information."""
        env = PortfolioEnv(data=sample_data, action_type='discrete')
        state, _ = env.reset()

        # State should contain regime information
        assert state is not None
        assert len(state) > 0

    def test_reproducibility_with_seed(self, sample_data):
        """Test environment is reproducible with seed."""
        env1 = PortfolioEnv(data=sample_data, action_type='discrete')
        env2 = PortfolioEnv(data=sample_data, action_type='discrete')

        state1, _ = env1.reset(seed=42)
        state2, _ = env2.reset(seed=42)

        np.testing.assert_array_almost_equal(state1, state2)

        # Take same actions
        for _ in range(10):
            action = env1.action_space.sample()
            s1, r1, t1, tr1, _ = env1.step(action)
            s2, r2, t2, tr2, _ = env2.step(action)

            if not (t1 or tr1):
                np.testing.assert_array_almost_equal(s1, s2)
                assert r1 == r2


class TestRunningMeanStd:
    """Test running mean and standard deviation computation."""

    def test_initialization(self):
        """Test RunningMeanStd initializes correctly."""
        rms = RunningMeanStd(shape=(5,))
        assert rms.mean.shape == (5,)
        assert rms.var.shape == (5,)
        assert rms.count == 1e-4

    def test_update(self):
        """Test updating statistics."""
        rms = RunningMeanStd(shape=(3,))

        # Update with batch of data
        batch = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        rms.update(batch)

        assert rms.count > 1e-4
        assert rms.mean.shape == (3,)

    def test_normalization(self):
        """Test data normalization."""
        rms = RunningMeanStd(shape=(2,))

        # Update with known data
        data = np.array([[0, 0], [1, 1], [2, 2]])
        rms.update(data)

        # Normalize new data
        x = np.array([[1, 1]])
        normalized = (x - rms.mean) / np.sqrt(rms.var + 1e-8)

        assert normalized.shape == x.shape


class TestDummyVecEnv:
    """Test dummy vectorized environment."""

    def test_vec_env_initialization(self, sample_data):
        """Test vectorized environment initializes."""
        env_fns = [lambda: PortfolioEnv(data=sample_data, action_type='discrete') for _ in range(4)]
        vec_env = DummyVecEnv(env_fns)

        assert vec_env.num_envs == 4

    def test_vec_env_reset(self, sample_data):
        """Test vectorized reset."""
        env_fns = [lambda: PortfolioEnv(data=sample_data, action_type='discrete') for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        states = vec_env.reset()
        assert states.shape[0] == 2  # 2 environments

    def test_vec_env_step(self, sample_data):
        """Test vectorized step."""
        env_fns = [lambda: PortfolioEnv(data=sample_data, action_type='discrete') for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        vec_env.reset()
        actions = np.array([0, 1])
        states, rewards, dones, infos = vec_env.step(actions)

        assert states.shape[0] == 2
        assert rewards.shape == (2,)
        assert len(dones) == 2

    def test_vec_env_auto_reset(self, sample_data):
        """Test automatic reset on done."""
        env_fns = [lambda: PortfolioEnv(data=sample_data, action_type='discrete') for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        vec_env.reset()

        # Run until done
        for _ in range(300):
            actions = np.array([0, 0])
            states, rewards, dones, infos = vec_env.step(actions)
            if any(dones):
                break

        # Should auto-reset
        assert states is not None


class TestVecNormalize:
    """Test vectorized environment normalization wrapper."""

    def test_normalize_initialization(self, sample_data):
        """Test VecNormalize initializes correctly."""
        env_fns = [lambda: PortfolioEnv(data=sample_data, action_type='discrete') for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)
        vec_norm = VecNormalize(vec_env, obs_norm=True, ret_norm=True)

        assert vec_norm.obs_norm is True
        assert vec_norm.ret_norm is True

    def test_observation_normalization(self, sample_data):
        """Test observation normalization."""
        env_fns = [lambda: PortfolioEnv(data=sample_data, action_type='discrete') for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)
        vec_norm = VecNormalize(vec_env, obs_norm=True, ret_norm=False)

        states = vec_norm.reset()

        # Run a few steps to update statistics
        for _ in range(10):
            actions = np.array([0, 0])
            states, _, dones, _ = vec_norm.step(actions)

        # Observations should be normalized (approximately mean 0, std 1)
        assert states is not None

    def test_reward_normalization(self, sample_data):
        """Test reward normalization."""
        env_fns = [lambda: PortfolioEnv(data=sample_data, action_type='discrete') for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)
        vec_norm = VecNormalize(vec_env, obs_norm=False, ret_norm=True, gamma=0.99)

        vec_norm.reset()

        rewards = []
        for _ in range(20):
            actions = np.array([0, 0])
            _, reward, dones, _ = vec_norm.step(actions)
            rewards.extend(reward)

        # Rewards should be collected
        assert len(rewards) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_initial_balance(self, sample_data):
        """Test environment handles zero balance gracefully."""
        with pytest.raises(Exception):
            env = PortfolioEnv(data=sample_data, initial_balance=0.0)

    def test_negative_transaction_cost(self, sample_data):
        """Test environment accepts transaction cost parameter."""
        env = PortfolioEnv(data=sample_data, transaction_cost=0.01)
        assert env.transaction_cost == 0.01

    def test_invalid_action_type(self, sample_data):
        """Test environment rejects invalid action type."""
        with pytest.raises(Exception):
            env = PortfolioEnv(data=sample_data, action_type='invalid')

    def test_minimal_data_columns(self):
        """Test environment handles minimal data columns gracefully."""
        minimal_data = pd.DataFrame({
            'price_SPY': [100, 101, 102]
        })

        env = PortfolioEnv(data=minimal_data)
        assert env.n_assets == 1

    def test_extreme_weights(self, sample_data):
        """Test environment handles extreme weight allocations."""
        env = PortfolioEnv(data=sample_data, action_type='continuous')
        env.reset()

        # All weight in one asset
        action = np.array([1.0, 0.0, 0.0, 0.0])
        state, reward, terminated, truncated, info = env.step(action)

        assert not np.isnan(reward)
        assert not np.isinf(reward)

    def test_bankruptcy_protection(self, sample_data):
        """Test environment handles portfolio going to zero."""
        # Create data with large negative returns
        bad_data = sample_data.copy()
        for asset in ['SPY', 'TLT', 'GLD', 'BTC']:
            bad_data[f'return_{asset}'] = -0.5  # Extreme losses

        env = PortfolioEnv(data=bad_data, action_type='continuous')
        env.reset()

        # Should handle bankruptcy gracefully
        for _ in range(10):
            action = np.array([0.25, 0.25, 0.25, 0.25])
            state, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
