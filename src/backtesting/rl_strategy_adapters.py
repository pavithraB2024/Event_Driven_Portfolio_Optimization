"""
RL Strategy Adapters for Backtesting Engine

Connects trained RL agents (DQN, PPO, SAC) to the BacktestEngine
using the Strategy interface.
"""

import numpy as np
import torch
from pathlib import Path

from .backtest_engine import Strategy

from ..agents.sac_agent import SACAgent


class DQNStrategyAdapter(Strategy):
    """
    Adapter for trained DQN agent.

    DQN outputs discrete actions (0, 1, 2) which map to:
    - 0: Conservative (low stocks, high bonds)
    - 1: Balanced
    - 2: Aggressive (high stocks, low bonds)
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Args:
            model_path: Path to trained DQN model (.pth file)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model_path = model_path

        # Action mappings (will be customized based on asset count)
        # For 4 assets: SPY, TLT, GLD, BTC
        self.action_mappings = {
            0: np.array([0.20, 0.50, 0.20, 0.10]),  # Conservative: bonds-heavy
            1: np.array([0.40, 0.30, 0.20, 0.10]),  # Balanced
            2: np.array([0.60, 0.10, 0.20, 0.10]),  # Aggressive: stocks-heavy
        }

        # Load model
        self.agent = None
        self._load_model()

    def _load_model(self):
        """Load trained DQN agent from checkpoint."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"DQN model not found: {self.model_path}")

        # Initialize agent (state_dim and action_dim will be set from checkpoint)
        checkpoint = torch.load(self.model_path, map_location=self.device)

        state_dim = checkpoint.get('state_dim', 34)  # Default from our env
        action_dim = checkpoint.get('action_dim', 3)

        self.agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )

        self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.agent.policy_net.eval()

    def allocate(self, data, current_idx, current_weights, **kwargs) -> np.ndarray:
        """
        Generate portfolio weights using trained DQN agent.

        Args:
            data: Market data DataFrame
            current_idx: Current time index
            current_weights: Current portfolio weights (not used by DQN)
            **kwargs: Additional arguments

        Returns:
            weights: Portfolio weights (sums to 1)
        """
        # Extract state from current market data
        state = self._extract_state(data, current_idx)

        # Get action from agent
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.agent.select_action(state_tensor, epsilon=0.0)  # Greedy

        # Map action to weights
        weights = self.action_mappings.get(action, self.action_mappings[1])

        return weights

    def _extract_state(self, data, idx) -> np.ndarray:
        """
        Extract state features from market data.
        Matches the state construction in PortfolioEnv._get_state().

        State components:
        - Portfolio weights (n_assets)
        - Recent returns (5 days × n_assets)
        - Current volatility (n_assets)
        - Market regime one-hot (3)
        - VIX (1)
        - Treasury rate (1)
        - Portfolio value normalized (1)
        """
        # Get return columns
        return_cols = [col for col in data.columns if col.startswith('return_')]
        n_assets = len(return_cols)

        state_components = []

        # 1. Current portfolio weights (zeros for initial allocation)
        weights = np.zeros(n_assets)  # DQN doesn't use current weights
        state_components.append(weights)

        # 2. Recent returns (last 5 days)
        recent_returns = []
        for col in return_cols:
            returns = data[col].iloc[max(0, idx-5):idx].values
            if len(returns) < 5:
                returns = np.pad(returns, (5-len(returns), 0), constant_values=0)
            recent_returns.extend(returns)
        state_components.append(np.array(recent_returns))

        # 3. Current volatility (20-day rolling)
        volatility = []
        for col in return_cols:
            vol = data[col].iloc[max(0, idx-20):idx].std()
            volatility.append(vol if not np.isnan(vol) else 0.0)
        state_components.append(np.array(volatility))

        # 4. Market regime (one-hot)
        regime_col = None
        if 'regime' in data.columns:
            regime_col = 'regime'
        elif 'regime_gmm' in data.columns:
            regime_col = 'regime_gmm'
        elif 'regime_hmm' in data.columns:
            regime_col = 'regime_hmm'

        if regime_col and idx < len(data):
            regime = data[regime_col].iloc[idx]
            regime_one_hot = np.zeros(3)
            if not np.isnan(regime):
                regime_one_hot[int(regime)] = 1.0
            state_components.append(regime_one_hot)
        else:
            state_components.append(np.zeros(3))

        # 5. VIX
        if 'VIX' in data.columns and idx < len(data):
            vix = data['VIX'].iloc[idx]
            state_components.append(np.array([vix if not np.isnan(vix) else 20.0]))
        else:
            state_components.append(np.array([20.0]))

        # 6. Treasury rate
        if 'Treasury_10Y' in data.columns and idx < len(data):
            treasury = data['Treasury_10Y'].iloc[idx]
            state_components.append(np.array([treasury if not np.isnan(treasury) else 2.5]))
        else:
            state_components.append(np.array([2.5]))

        # 7. Portfolio value (normalized to 1.0 for backtest)
        state_components.append(np.array([1.0]))

        # Concatenate and clean
        state = np.concatenate([comp.flatten() for comp in state_components])
        state = np.nan_to_num(state, nan=0.0)

        return state.astype(np.float32)





class SACStrategyAdapter(Strategy):
    """
    Adapter for trained SAC agent.
    SAC outputs continuous weights directly.
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Args:
            model_path: Path to trained SAC model (.pth file)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model_path = model_path
        self.agent = None
        self._load_model()

    def _load_model(self):
        """Load trained SAC agent from checkpoint."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"SAC model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        state_dim = checkpoint.get('state_dim', 34)
        action_dim = checkpoint.get('action_dim', 4)

        self.agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )

        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.actor.eval()

    def allocate(self, data, current_idx, current_weights, **kwargs) -> np.ndarray:
        """
        Generate portfolio weights using trained SAC agent.

        Args:
            data: Market data DataFrame
            current_idx: Current time index
            current_weights: Current portfolio weights
            **kwargs: Additional arguments

        Returns:
            weights: Portfolio weights (sums to 1, each in [0, 1])
        """
        # Extract state
        state = self._extract_state(data, current_idx, current_weights)

        # Get action from agent (deterministic for evaluation)
        # SACAgent.select_action expects numpy array and returns numpy array
        action = self.agent.select_action(state, deterministic=True)

        # Convert action to weights (apply softmax)
        weights = np.exp(action) / np.sum(np.exp(action))
        weights = np.clip(weights, 0, 1)
        weights = weights / weights.sum()

        return weights

    def _extract_state(self, data, idx, current_weights) -> np.ndarray:
        """
        Extract state features from market data.
        Matches the state construction in PortfolioEnv._get_state().
        """
        # Get return columns
        return_cols = [col for col in data.columns if col.startswith('return_')]
        len(return_cols)

        state_components = []

        # 1. Current portfolio weights (use actual weights)
        state_components.append(current_weights)

        # 2. Recent returns (last 5 days)
        recent_returns = []
        for col in return_cols:
            returns = data[col].iloc[max(0, idx-5):idx].values
            if len(returns) < 5:
                returns = np.pad(returns, (5-len(returns), 0), constant_values=0)
            recent_returns.extend(returns)
        state_components.append(np.array(recent_returns))

        # 3. Current volatility (20-day rolling)
        volatility = []
        for col in return_cols:
            vol = data[col].iloc[max(0, idx-20):idx].std()
            volatility.append(vol if not np.isnan(vol) else 0.0)
        state_components.append(np.array(volatility))

        # 4. Market regime (one-hot)
        regime_col = None
        if 'regime' in data.columns:
            regime_col = 'regime'
        elif 'regime_gmm' in data.columns:
            regime_col = 'regime_gmm'
        elif 'regime_hmm' in data.columns:
            regime_col = 'regime_hmm'

        if regime_col and idx < len(data):
            regime = data[regime_col].iloc[idx]
            regime_one_hot = np.zeros(3)
            if not np.isnan(regime):
                regime_one_hot[int(regime)] = 1.0
            state_components.append(regime_one_hot)
        else:
            state_components.append(np.zeros(3))

        # 5. VIX
        if 'VIX' in data.columns and idx < len(data):
            vix = data['VIX'].iloc[idx]
            state_components.append(np.array([vix if not np.isnan(vix) else 20.0]))
        else:
            state_components.append(np.array([20.0]))

        # 6. Treasury rate
        if 'Treasury_10Y' in data.columns and idx < len(data):
            treasury = data['Treasury_10Y'].iloc[idx]
            state_components.append(np.array([treasury if not np.isnan(treasury) else 2.5]))
        else:
            state_components.append(np.array([2.5]))

        # 7. Portfolio value (normalized to 1.0 for backtest)
        state_components.append(np.array([1.0]))

        # Concatenate and clean
        state = np.concatenate([comp.flatten() for comp in state_components])
        state = np.nan_to_num(state, nan=0.0)

        return state.astype(np.float32)


from ..agents.memory_augmented_sac_agent import MemoryAugmentedSACAgent


class MemorySACStrategyAdapter(Strategy):
    """
    Adapter for trained Memory-Augmented SAC agent (LSTM).
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Args:
            model_path: Path to trained Memory SAC model (.pth file)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model_path = model_path
        self.agent = None
        self.hidden_state = None
        self._load_model()

    def _load_model(self):
        """Load trained Memory SAC agent from checkpoint."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Memory SAC model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        state_dim = checkpoint.get('state_dim', 34)
        action_dim = checkpoint.get('action_dim', 4)
        network_type = checkpoint.get('network_type', 'LSTM')

        self.agent = MemoryAugmentedSACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            network_type=network_type,
            device=self.device
        )

        self.agent.load(self.model_path)
        self.agent.policy_net.eval()

    def allocate(self, data, current_idx, current_weights, **kwargs) -> np.ndarray:
        """
        Generate portfolio weights using trained Memory SAC agent.
        """
        # Extract state
        state = self._extract_state(data, current_idx, current_weights)

        # Get action from agent
        with torch.no_grad():
            # Memory agent expects state as numpy array
            action, self.hidden_state = self.agent.select_action(
                state, 
                hidden_state=self.hidden_state, 
                deterministic=True
            )

        # Convert action to weights (apply softmax if needed, or if agent outputs weights directly)
        # MemoryAugmentedSACAgent usually outputs raw actions (tanh), so we need softmax
        # But wait, the agent's select_action returns the action. 
        # If the environment expects weights, we usually softmax them.
        # Let's assume standard softmax normalization for safety.
        weights = np.exp(action) / np.sum(np.exp(action))
        weights = np.clip(weights, 0, 1)
        weights = weights / weights.sum()

        return weights

    def _extract_state(self, data, idx, current_weights) -> np.ndarray:
        """
        Extract state features from market data.
        Matches the state construction in PathDependentPortfolioEnv._get_state().
        """
        # Get return columns
        return_cols = [col for col in data.columns if col.startswith('return_')]
        len(return_cols)

        state_components = []

        # 1. Current portfolio weights
        state_components.append(current_weights)

        # 2. Recent returns (last 5 days)
        recent_returns = []
        for col in return_cols:
            returns = data[col].iloc[max(0, idx-5):idx].values
            if len(returns) < 5:
                returns = np.pad(returns, (5-len(returns), 0), constant_values=0)
            recent_returns.extend(returns)
        state_components.append(np.array(recent_returns))

        # 3. Current volatility (20-day rolling)
        volatility = []
        for col in return_cols:
            vol = data[col].iloc[max(0, idx-20):idx].std()
            volatility.append(vol if not np.isnan(vol) else 0.0)
        state_components.append(np.array(volatility))

        # 4. Market regime (one-hot)
        regime_col = None
        if 'regime' in data.columns:
            regime_col = 'regime'
        elif 'regime_gmm' in data.columns:
            regime_col = 'regime_gmm'
        elif 'regime_hmm' in data.columns:
            regime_col = 'regime_hmm'

        if regime_col and idx < len(data):
            regime = data[regime_col].iloc[idx]
            regime_one_hot = np.zeros(3)
            if not np.isnan(regime):
                regime_one_hot[int(regime)] = 1.0
            state_components.append(regime_one_hot)
        else:
            state_components.append(np.zeros(3))

        # 5. VIX
        if 'VIX' in data.columns and idx < len(data):
            vix = data['VIX'].iloc[idx]
            state_components.append(np.array([vix if not np.isnan(vix) else 20.0]))
        else:
            state_components.append(np.array([20.0]))

        # 6. Treasury rate
        if 'Treasury_10Y' in data.columns and idx < len(data):
            treasury = data['Treasury_10Y'].iloc[idx]
            state_components.append(np.array([treasury if not np.isnan(treasury) else 2.5]))
        else:
            state_components.append(np.array([2.5]))

        # 7. Portfolio value (normalized)
        state_components.append(np.array([1.0]))

        # Concatenate and clean
        state = np.concatenate([comp.flatten() for comp in state_components])
        state = np.nan_to_num(state, nan=0.0)

        return state.astype(np.float32)


from ..agents.graphsage_ppo_agent import GraphSAGEPPOAgent


class GraphSAGEPPOStrategyAdapter(Strategy):
    """Adapter for the internally trained GraphSAGE-PPO DRL baseline .

    Wraps a trained :class:`GraphSAGEPPOAgent` so the backtesting engine can
    drive it via the same Strategy interface used by SACStrategyAdapter.

    Args:
        model_path: Path to a checkpoint produced by GraphSAGEPPOAgent.save.
        adj_matrix: Static graph topology (numpy array). Must match the
            ``num_nodes`` recorded in the checkpoint.
        node_features_dim: Per-node feature dim baked into the checkpoint.
        device: 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model_path: str,
        adj_matrix: np.ndarray,
        node_features_dim: int,
        device: str = 'cpu',
    ):
        self.device = device
        self.model_path = model_path
        self.adj_matrix = adj_matrix
        self.node_features_dim = node_features_dim
        self.agent: GraphSAGEPPOAgent | None = None
        self._load_model()

    def _load_model(self) -> None:
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"GraphSAGE-PPO model not found: {self.model_path}")
        ckpt = torch.load(self.model_path, map_location=self.device)
        num_nodes = ckpt.get('num_nodes', self.adj_matrix.shape[0])
        action_dim = ckpt.get('action_dim', num_nodes)
        state_dim = ckpt.get('state_dim', num_nodes * self.node_features_dim)
        self.agent = GraphSAGEPPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            node_features_dim=self.node_features_dim,
            num_nodes=num_nodes,
            adj_matrix=self.adj_matrix,
            device=self.device,
        )
        self.agent.load(self.model_path)
        self.agent.policy.eval()

    def allocate(self, data, current_idx, current_weights, **kwargs) -> np.ndarray:
        """Run the trained policy on the engine-supplied state vector.

        The backtest engine is expected to pass a pre-built flat state of
        shape ``[num_nodes * node_features_dim]`` via the ``state`` kwarg —
        the same layout produced by ``patch_env_for_graph`` in the
        evaluation pipeline. If no state is provided we fall back to an
        equal-weight allocation so the engine doesn't crash.
        """
        state = kwargs.get('state')
        if state is None:
            n = len(current_weights) if current_weights is not None else self.agent.action_dim
            return np.ones(n) / n
        adj = kwargs.get('adj')
        action, _ = self.agent.select_action(state, deterministic=True, adj=adj)
        action = np.maximum(action, 0)
        s = action.sum()
        if s <= 1e-8:
            return np.ones_like(action) / len(action)
        return action / s


def create_rl_strategy_adapter(agent_type: str, model_path: str, device: str = 'cpu', **kwargs) -> Strategy:
    """
    Factory function to create RL strategy adapters.

    Args:
        agent_type: 'dqn', 'sac', 'sac_memory', or 'graphsage_ppo'.
        model_path: Path to trained model.
        device: 'cpu' or 'cuda'.
        **kwargs: Extra arguments forwarded to the adapter constructor
            (e.g. ``adj_matrix`` and ``node_features_dim`` for GraphSAGE-PPO).

    Returns:
        Strategy adapter instance
    """
    adapters = {
        'dqn': DQNStrategyAdapter,

        'sac': SACStrategyAdapter,
        'sac_memory': MemorySACStrategyAdapter,
        'graphsage_ppo': GraphSAGEPPOStrategyAdapter,
    }

    agent_type = agent_type.lower()
    if agent_type not in adapters:
        raise ValueError(f"Unknown agent type: {agent_type}. Choose from {list(adapters.keys())}")

    if agent_type == 'graphsage_ppo':
        return adapters[agent_type](model_path=model_path, device=device, **kwargs)
    return adapters[agent_type](model_path=model_path, device=device)

