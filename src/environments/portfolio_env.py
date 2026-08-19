"""
Portfolio Allocation Gym Environment
MDP formulation for reinforcement learning
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any


from .reward_calculator import RewardConfig, get_reward_calculator

# ───────────────────────────────────────────────────────────────────
# Phase 2: Pure Functions — Noise Injection (FCIS Core)
# ───────────────────────────────────────────────────────────────────

from .feature_extractor import StandardFeatureExtractor


class PortfolioEnv(gym.Env):
    """
    Portfolio allocation environment for RL agents.

    State space includes:
    - Current portfolio weights
    - Asset prices and returns
    - Rolling volatility
    - Market regime
    - Macro indicators (VIX, Treasury rates)

    Action space:
    - Continuous: Target allocation weights for each asset [0, 1]
    - Discrete: Increase/Hold/Decrease risky allocation

    Reward:
    - Log utility or Sharpe ratio based
    - Transaction cost penalty
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.0025,
        risk_free_rate: float = 0.02,
        window_size: int = 60,
        action_type: str = 'continuous',
        reward_type: str = 'log_utility',
        oscillation_penalty: float = 0.001,
        drawdown_penalty: float = 0.002,
        reward_config: Optional[RewardConfig] = None,
        max_weight_per_asset: float = 1.0,
        # ── Phase 2: Noise Injection ──
        noise_augmentation: bool = False,
        noise_std: float = 0.001,
        # ── Phase 2: Random Episode Starts ──
        random_start_offset: bool = False,
        max_start_offset_pct: float = 0.1,
        # ── Phase 5.3: Diversification Bonus ──
        diversification_bonus: float = 0.0,
        # ── Phase 1.3: Multi-Horizon Reward Blending ──
        horizon_blend_lambda: float = 1.0,
    ):
        """
        Initialize portfolio environment.

        Args:
            data: DataFrame with asset prices, returns, regime, etc.
            initial_balance: Starting portfolio value
            transaction_cost: Proportional transaction cost (e.g., 0.0025 = 0.25%)
            risk_free_rate: Annual risk-free rate
            window_size: Look-back window for state features
            action_type: 'continuous' or 'discrete'
            reward_type: 'log_utility', 'sharpe', or 'return'
        """
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size
        self.action_type = action_type
        self.reward_type = reward_type
        self.max_weight_per_asset = max_weight_per_asset

        # Phase 2: Noise injection settings
        self.noise_augmentation = noise_augmentation
        self.noise_std = noise_std
        
        self.feature_extractor = StandardFeatureExtractor(
            noise_augmentation=noise_augmentation,
            noise_std=noise_std
        )

        # Phase 2: Random episode start settings
        self.random_start_offset = random_start_offset
        self.max_start_offset_pct = max_start_offset_pct

        # Phase 5.3: Diversification bonus coefficient
        self.diversification_coeff = diversification_bonus

        # Phase 1.3: Multi-Horizon Reward Blending
        self.horizon_blend_lambda = horizon_blend_lambda

        # Phase 1: DSR state variables
        self._dsr_ema_return = 0.0
        self._dsr_ema_return_sq = 0.0

        # Phase 3: Regime feature history
        self._return_history_window = []

        # Extract asset columns (prices and returns)
        self.price_cols = [col for col in data.columns if col.startswith('price_')]
        self.return_cols = [col for col in data.columns if col.startswith('return_')]
        self.n_assets = len(self.price_cols)

        # State initialization
        self.current_step = 0
        self.max_steps = len(data) - window_size - 1
        self.portfolio_value = initial_balance
        self.portfolio_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        self.cash = 0.0

        self.reward_calculator = get_reward_calculator(
            reward_type=reward_type,
            config=reward_config,
            oscillation_penalty=oscillation_penalty,
            drawdown_penalty=drawdown_penalty,
            diversification_coeff=diversification_bonus,
            horizon_blend_lambda=horizon_blend_lambda
        )
        self.reward_type = self.reward_calculator.config.base_type

        self.prev_weights = self.portfolio_weights.copy()
        self.prev_prev_weights = self.portfolio_weights.copy()
        self.peak_value = initial_balance

        # History tracking
        self.portfolio_history = []
        self.action_history = []
        self.reward_history = []

        # Define action space
        if action_type == 'continuous':
            # Target weights for each asset [0, 1]
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.n_assets,),
                dtype=np.float32
            )
        elif action_type == 'discrete':
            # Actions: 0=decrease, 1=hold, 2=increase for each asset
            # Simplified: just 3 actions for risky asset allocation
            self.action_space = spaces.Discrete(3)
        else:
            raise ValueError(f"Unknown action_type: {action_type}")

        # Define observation space
        # State includes: weights, returns, volatility, regime, macro
        state_dim = self._get_state().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

    def _get_state(self) -> np.ndarray:
        """
        Get current state observation via FeatureExtractor adapter.
        """
        idx = self.current_step + self.window_size
        return self.feature_extractor.extract(
            idx=idx,
            data=self.data,
            return_cols=self.return_cols,
            portfolio_weights=self.portfolio_weights,
            portfolio_value=self.portfolio_value,
            initial_balance=self.initial_balance,
            peak_value=self.peak_value,
            return_history_window=self._return_history_window
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Returns:
            Initial state and info dict
        """
        super().reset(seed=seed)

        # Phase 2: Random episode start offset
        if self.random_start_offset:
            max_offset = int(self.max_steps * self.max_start_offset_pct)
            if max_offset > 0:
                self.current_step = self.np_random.integers(0, max_offset)
            else:
                self.current_step = 0
        else:
            self.current_step = 0

        self.portfolio_value = self.initial_balance
        self.portfolio_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        self.cash = 0.0

        # Reset oscillation tracking
        self.prev_weights = self.portfolio_weights.copy()
        self.prev_prev_weights = self.portfolio_weights.copy()
        self.peak_value = self.initial_balance

        self.portfolio_history = [self.portfolio_value]
        self.action_history = []
        self.reward_history = []

        # Phase 1: Reset Reward state
        self.reward_calculator.reset()

        # Phase 3: Reset regime history
        self._return_history_window = []

        state = self._get_state()
        info = {}

        return state, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Portfolio allocation action

        Returns:
            next_state, reward, terminated, truncated, info
        """
        # Process action to get target weights
        target_weights = self._process_action(action)

        # --- Action Execution Threshold ("Lazy Trading") ---
        # Prevent micro-trades to reduce unnecessary transaction costs
        weight_diff = np.abs(target_weights - self.portfolio_weights)
        # Dynamic trading threshold: trade only if weight shift is significant
        # or if volatility is high (VIX proxy)
        vix = 20.0
        idx = self.current_step + self.window_size
        if 'VIX' in self.data.columns:
            vix = self.data['VIX'].iloc[idx]
        
        # Adaptive threshold: more lazy in stable markets, more active in high-vix
        # 0.005 base threshold, scales down with VIX (if VIX=40 -> 0.0025)
        dynamic_threshold = 0.005 * (20.0 / max(vix, 10.0))
        mask = weight_diff > dynamic_threshold
        lazy_weights = self.portfolio_weights.copy()
        lazy_weights[mask] = target_weights[mask]

        weight_sum = np.sum(lazy_weights)
        if weight_sum > 1e-6:
            target_weights = lazy_weights / weight_sum
        else:
            target_weights = np.ones(self.n_assets) / self.n_assets

        # Track weight history for oscillation penalty (before update)
        self.prev_prev_weights = self.prev_weights.copy()
        self.prev_weights = self.portfolio_weights.copy()

        # Calculate transaction costs (Base Fee + Dynamic VIX-Scaled Slippage)
        weight_change = np.abs(target_weights - self.portfolio_weights)
        base_fee = self.transaction_cost * weight_change.sum() * self.portfolio_value
        
        # R1 C4: Quadratic slippage penalty scaled by VIX
        slippage_factor = 0.0005
        quadratic_slippage = slippage_factor * max(vix, 10.0) * np.sum(weight_change ** 2) * self.portfolio_value
        
        transaction_costs = base_fee + quadratic_slippage

        # Update portfolio weights
        old_portfolio_value = self.portfolio_value
        self.portfolio_weights = target_weights

        # Move to next time step
        self.current_step += 1
        idx = self.current_step + self.window_size

        # Calculate portfolio return
        asset_returns = []
        for col in self.return_cols:
            ret = self.data[col].iloc[idx]
            # Guard: NaN returns (missing data) → treat as 0
            if np.isnan(ret) or np.isinf(ret):
                ret = 0.0
            # Guard: clip extreme single-asset returns (e.g. VIXY spikes) to ±50% per step
            ret = float(np.clip(ret, -0.5, 0.5))
            asset_returns.append(ret)

        asset_returns = np.array(asset_returns)
        portfolio_return = float(np.dot(self.portfolio_weights, asset_returns))
        # Guard: clip aggregate portfolio return to ±50% to prevent NaN cascade
        portfolio_return = float(np.clip(portfolio_return, -0.5, 0.5))

        # Guard: ensure transaction_costs is finite
        if np.isnan(transaction_costs) or np.isinf(transaction_costs):
            transaction_costs = 0.0

        # Update portfolio value; floor at 0 to prevent negative / NaN cascade
        new_value = old_portfolio_value * (1.0 + portfolio_return) - transaction_costs
        self.portfolio_value = max(float(new_value), 0.0)

        # Update peak value for drawdown penalty
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # Phase 3: Track return history for regime features
        self._return_history_window.append(portfolio_return)
        if len(self._return_history_window) > 60:
            self._return_history_window = self._return_history_window[-60:]

        # Calculate reward (with oscillation + drawdown penalties)
        reward = self._calculate_reward(
            old_portfolio_value,
            self.portfolio_value,
            portfolio_return,
            transaction_costs
        )

        # Track history
        self.portfolio_history.append(self.portfolio_value)
        self.action_history.append(action)
        self.reward_history.append(reward)

        # Check if episode is done
        terminated = self.current_step >= self.max_steps - 1
        truncated = self.portfolio_value <= 0  # Bankruptcy

        # Get next state
        next_state = self._get_state()

        # Info dict
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return,
            'transaction_costs': transaction_costs,
            'weights': self.portfolio_weights.copy(),
            'current_drawdown': (self.peak_value - self.portfolio_value) / self.peak_value
                                 if self.peak_value > 0 else 0.0,
        }

        return next_state, reward, terminated, truncated, info

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert action to target portfolio weights.

        Args:
            action: Raw action from agent

        Returns:
            Normalized target weights
        """
        if self.action_type == 'continuous':
            # Normalize to sum to 1
            weights = np.array(action).flatten()
            # Sanitize input: replace NaNs/Infs
            weights = np.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
            
            weights = np.maximum(weights, 0)  # No shorting
            
            max_w = getattr(self, 'max_weight_per_asset', 1.0)
            if max_w < 1.0:
                # Pre-normalize
                weight_sum = weights.sum()
                if weight_sum > 1e-6:
                    weights = weights / weight_sum
                else:
                    weights = np.ones_like(weights) / len(weights)
                
                # Iteratively cap and redistribute
                while True:
                    over_cap = weights > max_w + 1e-6
                    if not over_cap.any():
                        break
                        
                    excess = (weights[over_cap] - max_w).sum()
                    weights[over_cap] = max_w
                    
                    under_cap = weights < max_w - 1e-6
                    if not under_cap.any():
                        break
                        
                    under_cap_sum = weights[under_cap].sum()
                    if under_cap_sum > 1e-6:
                        weights[under_cap] += excess * (weights[under_cap] / under_cap_sum)
                    else:
                        weights[under_cap] += excess / under_cap.sum()
            else:
                weight_sum = weights.sum()
                if weight_sum > 1e-6: # Avoid division by zero or very small numbers
                    weights = weights / weight_sum
                else:
                    weights = np.ones(self.n_assets) / self.n_assets
            return weights

        elif self.action_type == 'discrete':
            # Simple discrete action: adjust first asset (risky), rest is risk-free
            current_risky_weight = self.portfolio_weights[0]

            if action == 0:  # Decrease
                new_risky_weight = max(0.0, current_risky_weight - 0.1)
            elif action == 1:  # Hold
                new_risky_weight = current_risky_weight
            else:  # action == 2, Increase
                new_risky_weight = min(1.0, current_risky_weight + 0.1)

            weights = np.zeros(self.n_assets)
            weights[0] = new_risky_weight
            weights[1:] = (1.0 - new_risky_weight) / (self.n_assets - 1)

            return weights

    def _calculate_reward(
        self,
        old_value: float,
        new_value: float,
        portfolio_return: float,
        transaction_costs: float
    ) -> float:
        """
        Calculate reward via injected RewardCalculator adapter.
        """
        return self.reward_calculator.calculate(
            old_value=old_value,
            new_value=new_value,
            portfolio_return=portfolio_return,
            transaction_costs=transaction_costs,
            weights=self.portfolio_weights,
            prev_weights=self.prev_weights,
            prev_prev_weights=self.prev_prev_weights,
            peak_value=self.peak_value,
            initial_balance=self.initial_balance,
            reward_history=self.reward_history,
            risk_free_rate=self.risk_free_rate
        )

    def render(self, mode: str = 'human'):
        """Render the environment (optional)."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Weights: {self.portfolio_weights}")
            print(f"Return: {self.portfolio_value / self.initial_balance - 1:.2%}")

    def get_portfolio_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        values = np.array(self.portfolio_history)
        returns = np.diff(values) / values[:-1]

        total_return = (values[-1] - values[0]) / values[0]
        sharpe = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(values)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': values[-1]
        }

    @staticmethod
    def _calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0:
            return 0.0

        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = returns - daily_rf

        if excess_returns.std() == 0:
            return 0.0

        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return sharpe

    @staticmethod
    def _calculate_max_drawdown(values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(values) == 0:
            return 0.0

        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax
        return abs(drawdown.min())


if __name__ == "__main__":
    # Test environment
    print("Testing Portfolio Environment...")

    # Create mock data
    n_days = 500
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    # Mock asset data
    data = pd.DataFrame(index=dates)

    # Prices (2 assets)
    data['price_SPY'] = 300 + np.cumsum(np.random.randn(n_days) * 2)
    data['price_TLT'] = 140 + np.cumsum(np.random.randn(n_days) * 1)

    # Returns
    data['return_SPY'] = data['price_SPY'].pct_change().fillna(0)
    data['return_TLT'] = data['price_TLT'].pct_change().fillna(0)

    # Regime
    data['regime'] = np.random.choice([0, 1, 2], size=n_days)

    # Macro
    data['VIX'] = 15 + 10 * np.abs(np.random.randn(n_days))
    data['Treasury_10Y'] = 2.5 + 0.5 * np.random.randn(n_days)

    # Create environment
    env = PortfolioEnv(
        data=data,
        action_type='continuous',
        reward_type='log_utility'
    )

    # Test reset
    state, info = env.reset()
    print(f"\nInitial state shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space.shape}")

    # Test random episode
    done = False
    total_reward = 0
    step_count = 0

    while not done and step_count < 100:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step_count += 1

    print("\nTest episode completed:")
    print(f"Steps: {step_count}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final portfolio value: ${info['portfolio_value']:,.2f}")

    # Get metrics
    metrics = env.get_portfolio_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
