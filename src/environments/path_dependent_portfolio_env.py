"""
Path-Dependent Transaction Cost Enhanced Portfolio Environment
MDP formulation with memory-augmented costs based on the patent innovations
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any


class PathDependentPortfolioEnv(gym.Env):
    """
    Portfolio allocation environment with path-dependent transaction costs
    based on the Memory-Augmented BSDE Control Loop patent innovations.

    Instead of simple proportional transaction costs, this environment models:
    - Costs based on historical trading patterns
    - Cumulative turnover penalties
    - Regime-dependent cost structures
    - Memory of past allocations
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000.0,
        base_transaction_cost: float = 0.0025,
        risk_free_rate: float = 0.02,
        window_size: int = 60,
        action_type: str = 'continuous',
        reward_type: str = 'log_utility',
        max_history_length: int = 20  # How far back to look for path-dependent costs
    ):
        """
        Initialize portfolio environment with path-dependent costs.

        Args:
            data: DataFrame with asset prices, returns, regime, etc.
            initial_balance: Starting portfolio value
            base_transaction_cost: Base proportional transaction cost (e.g., 0.0025 = 0.25%)
            risk_free_rate: Annual risk-free rate
            window_size: Look-back window for state features
            action_type: 'continuous' or 'discrete'
            reward_type: 'log_utility', 'sharpe', or 'return'
            max_history_length: Maximum length of allocation history to store
        """
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.base_transaction_cost = base_transaction_cost
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size
        self.action_type = action_type
        self.reward_type = reward_type
        self.max_history_length = max_history_length

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

        # Path-dependent cost tracking
        self.allocation_history = []  # Track past allocations
        self.rolling_turnover = 0.0    # For cumulative turnover cost
        self.recent_turnover = []      # Track recent turnover values

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
        # State includes: weights, returns, volatility, regime, macro, and history of recent allocations
        state_dim = self._get_state().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

    def _get_state(self) -> np.ndarray:
        """
        Get current state observation with historical allocation information.

        Returns:
            State vector including current features and allocation history
        """
        idx = self.current_step + self.window_size

        # Current portfolio weights
        state_components = [self.portfolio_weights]

        # Recent returns (last 5 days)
        recent_returns = []
        for col in self.return_cols:
            returns = self.data[col].iloc[idx-5:idx].values
            recent_returns.extend(returns)
        state_components.append(np.array(recent_returns))

        # Current volatility
        volatility = []
        for col in self.return_cols:
            vol = self.data[col].iloc[idx-20:idx].std()
            volatility.append(vol)
        state_components.append(np.array(volatility))

        # Market regime (if available - check both possible names)
        regime_col = None
        if 'regime' in self.data.columns:
            regime_col = 'regime'
        elif 'regime_gmm' in self.data.columns:
            regime_col = 'regime_gmm'
        elif 'regime_hmm' in self.data.columns:
            regime_col = 'regime_hmm'

        if regime_col:
            regime = self.data[regime_col].iloc[idx]
            # One-hot encode regime (assuming 3 regimes)
            regime_one_hot = np.zeros(3)
            if not np.isnan(regime):
                regime_one_hot[int(regime)] = 1.0
            state_components.append(regime_one_hot)

        # Macro indicators
        if 'VIX' in self.data.columns:
            vix = self.data['VIX'].iloc[idx]
            state_components.append(np.array([vix if not np.isnan(vix) else 20.0]))

        if 'Treasury_10Y' in self.data.columns:
            treasury = self.data['Treasury_10Y'].iloc[idx]
            state_components.append(np.array([treasury if not np.isnan(treasury) else 2.5]))

        # Portfolio value (normalized)
        portfolio_value_norm = self.portfolio_value / self.initial_balance
        state_components.append(np.array([portfolio_value_norm]))

        # Historical allocation information
        # Add recent allocation weights to the state (for memory-augmented networks)
        if len(self.allocation_history) > 0:
            # Take the last few allocation vectors to represent history
            n_history = min(self.max_history_length, len(self.allocation_history))
            recent_allocations = np.concatenate(self.allocation_history[-n_history:])
            # Pad with zeros if not enough history
            padded_allocations = np.zeros(self.n_assets * self.max_history_length)
            padded_allocations[:len(recent_allocations)] = recent_allocations
            state_components.append(padded_allocations)
        else:
            # No history yet, pad with current weights
            history_padding = np.tile(self.portfolio_weights, self.max_history_length)
            state_components.append(history_padding)

        # Concatenate all components
        state = np.concatenate([comp.flatten() for comp in state_components])

        # Replace any remaining NaN with 0
        state = np.nan_to_num(state, nan=0.0)

        return state.astype(np.float32)

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

        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.portfolio_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        self.cash = 0.0

        # Reset path-dependent tracking
        self.allocation_history = []
        self.rolling_turnover = 0.0
        self.recent_turnover = []

        self.portfolio_history = [self.portfolio_value]
        self.action_history = []
        self.reward_history = []

        state = self._get_state()
        info = {}

        return state, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment with path-dependent costs.

        Args:
            action: Portfolio allocation action

        Returns:
            next_state, reward, terminated, truncated, info
        """
        # Process action to get target weights
        target_weights = self._process_action(action)

        # Calculate base transaction costs
        weight_change = np.abs(target_weights - self.portfolio_weights)
        base_transaction_costs = self.base_transaction_cost * weight_change.sum() * self.portfolio_value

        # Calculate path-dependent transaction costs
        path_dependent_costs = self._calculate_path_dependent_costs(target_weights, weight_change)

        # Total transaction costs
        total_transaction_costs = base_transaction_costs + path_dependent_costs

        # Update portfolio weights
        old_portfolio_value = self.portfolio_value
        self.portfolio_weights = target_weights

        # Store allocation in history
        self.allocation_history.append(target_weights.copy())
        if len(self.allocation_history) > self.max_history_length:
            self.allocation_history.pop(0)  # Remove oldest

        # Move to next time step
        self.current_step += 1
        idx = self.current_step + self.window_size

        # Calculate portfolio return
        asset_returns = []
        for col in self.return_cols:
            ret = self.data[col].iloc[idx]
            asset_returns.append(ret if not np.isnan(ret) else 0.0)

        asset_returns = np.array(asset_returns)
        portfolio_return = np.dot(self.portfolio_weights, asset_returns)

        # Update portfolio value
        self.portfolio_value = old_portfolio_value * (1 + portfolio_return) - total_transaction_costs

        # Calculate reward
        reward = self._calculate_reward(
            old_portfolio_value,
            self.portfolio_value,
            portfolio_return,
            total_transaction_costs
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
            'total_transaction_costs': total_transaction_costs,
            'base_transaction_costs': base_transaction_costs,
            'path_dependent_costs': path_dependent_costs,
            'weights': self.portfolio_weights.copy()
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
            weights = np.maximum(weights, 0)  # No shorting
            weight_sum = weights.sum()
            if weight_sum > 0:
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

    def _calculate_path_dependent_costs(self, target_weights: np.ndarray, weight_change: np.ndarray) -> float:
        """
        Calculate path-dependent transaction costs based on historical trading patterns.
        This implements the patent innovation of path-dependent cost modeling.

        Args:
            target_weights: Proposed portfolio weights
            weight_change: Amount of weight change from previous allocation

        Returns:
            Path-dependent cost component
        """
        if len(self.allocation_history) < 2:
            # Not enough history yet
            return 0.0

        # Cost based on cumulative turnover over recent periods
        recent_turnover = weight_change.sum()  # Current turnover
        self.recent_turnover.append(recent_turnover)
        if len(self.recent_turnover) > 10:  # Keep last 10 periods
            self.recent_turnover.pop(0)

        # Penalty for high recent turnover
        avg_recent_turnover = np.mean(self.recent_turnover) if self.recent_turnover else 0.0
        turnover_penalty_cost = avg_recent_turnover * self.portfolio_value * 0.05  # 5% of avg turnover

        # Cost based on oscillation between allocations
        if len(self.allocation_history) >= 3:
            # Check if we're oscillating (frequently changing allocations)
            prev_alloc = self.allocation_history[-2]
            two_back_alloc = self.allocation_history[-3]
            
            # Measure volatility in allocation changes
            alloc_change = np.abs(target_weights - prev_alloc).sum()
            prev_change = np.abs(prev_alloc - two_back_alloc).sum()
            
            oscillation_factor = max(0, alloc_change - prev_change)  # How much more volatile this change is
            oscillation_penalty = oscillation_factor * self.portfolio_value * 0.02  # 2% of oscillation

            # Add regime-dependent cost multiplier
            idx = self.current_step + self.window_size
            regime_mult = 1.0  # Default
            if 'regime' in self.data.columns:
                regime = self.data['regime'].iloc[idx]
                if not np.isnan(regime) and int(regime) == 2:  # Crisis regime
                    regime_mult = 1.5  # Higher costs in crisis
            elif 'regime_gmm' in self.data.columns:
                regime = self.data['regime_gmm'].iloc[idx]
                if not np.isnan(regime) and int(regime) == 2:  # Crisis regime
                    regime_mult = 1.5
                    
            return (turnover_penalty_cost + oscillation_penalty) * regime_mult

        return turnover_penalty_cost * 1.2  # Default multiplier if not enough history

    def _calculate_reward(
        self,
        old_value: float,
        new_value: float,
        portfolio_return: float,
        transaction_costs: float
    ) -> float:
        """
        Calculate reward based on reward type.

        Args:
            old_value: Previous portfolio value
            new_value: Current portfolio value
            portfolio_return: Return for this period
            transaction_costs: Total transaction costs including path-dependent costs

        Returns:
            Reward value
        """
        if self.reward_type == 'log_utility':
            # Log utility of wealth change with transaction cost penalty
            if new_value > 0:
                reward = np.log(new_value) - np.log(old_value)
            else:
                reward = -10.0  # Penalty for bankruptcy

            # Subtract normalized transaction cost impact
            cost_penalty = transaction_costs / old_value
            reward = reward - cost_penalty

        elif self.reward_type == 'sharpe':
            # Sharpe ratio approximation with cost penalty
            daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
            excess_return = portfolio_return - daily_rf

            # Use recent volatility
            recent_returns = self.reward_history[-20:] if len(self.reward_history) >= 20 else self.reward_history
            volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.01

            # Add cost penalty to reduce reward if high costs
            cost_penalty = (transaction_costs / old_value) * 10  # Scale up for impact
            reward = (excess_return / (volatility + 1e-6)) - cost_penalty

        elif self.reward_type == 'return':
            # Simple return with full transaction cost penalty
            reward = portfolio_return - (transaction_costs / old_value)

        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        return reward

    def render(self, mode: str = 'human'):
        """Render the environment (optional)."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Weights: {self.portfolio_weights}")
            print(f"Return: {self.portfolio_value / self.initial_balance - 1:.2%}")
            print(f"Recent Turnover: {self.recent_turnover[-1] if self.recent_turnover else 0.0:.4f}")

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
    print("Testing Path-Dependent Portfolio Environment...")

    # Create mock data
    n_days = 500
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    # Mock asset data
    data = pd.DataFrame(index=dates)

    # Prices (2 assets)
    data['price_SPY'] = 300 + np.cumsum(np.random.randn(n_days) * 2)
    data['price_TLT'] = 140 + np.cumsum(np.random.randn(n_days) * 1)
    data['price_GLD'] = 180 + np.cumsum(np.random.randn(n_days) * 0.8)

    # Returns
    data['return_SPY'] = data['price_SPY'].pct_change().fillna(0)
    data['return_TLT'] = data['price_TLT'].pct_change().fillna(0)
    data['return_GLD'] = data['price_GLD'].pct_change().fillna(0)

    # Regime
    data['regime'] = np.random.choice([0, 1, 2], size=n_days)

    # Macro
    data['VIX'] = 15 + 10 * np.abs(np.random.randn(n_days))
    data['Treasury_10Y'] = 2.5 + 0.5 * np.random.randn(n_days)

    # Create environment
    env = PathDependentPortfolioEnv(
        data=data,
        action_type='continuous',
        reward_type='log_utility',
        max_history_length=10
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
    print(f"Total transaction costs: ${info['total_transaction_costs']:.2f}")
    print(f"Base transaction costs: ${info['base_transaction_costs']:.2f}")
    print(f"Path-dependent costs: ${info['path_dependent_costs']:.2f}")

    # Get metrics
    metrics = env.get_portfolio_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")