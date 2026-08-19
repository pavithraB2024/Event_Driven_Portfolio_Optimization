"""
Soft Actor-Critic (SAC) Agent for Continuous Portfolio Allocation

SAC is a state-of-the-art off-policy algorithm that:
- Maximizes both expected return and entropy (exploration)
- Uses twin Q-networks to reduce overestimation bias
- Learns a stochastic policy for better exploration
- Automatically tunes the temperature parameter

Reference: Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy
Deep Reinforcement Learning with a Stochastic Actor" (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
import logging
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for SAC.
    Outputs mean and log_std for a diagonal Gaussian distribution.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=[256, 256]):
        super(GaussianPolicy, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

        self.mean = nn.Linear(hidden_dims[1], action_dim)
        self.log_std = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor

        Returns:
            mean: Mean of Gaussian distribution
            log_std: Log standard deviation
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, state: torch.Tensor, epsilon: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: State tensor
            epsilon: Small constant for numerical stability

        Returns:
            action: Sampled action (squashed through tanh)
            log_prob: Log probability of action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Sample from Gaussian
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick

        # Apply tanh squashing
        action = torch.tanh(x_t)

        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get action for inference.

        Args:
            state: State tensor
            deterministic: If True, return mean action

        Returns:
            action: Action to take
        """
        mean, log_std = self.forward(state)

        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)

        return action


class QNetwork(nn.Module):
    """Q-Network (critic) for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=[256, 256]):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            q_value: Q-value estimate
        """
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int = 1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample random batch."""
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones)
        }

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    """
    Soft Actor-Critic agent for continuous control.

    Features:
    - Maximum entropy RL
    - Twin Q-networks (clipped double Q-learning)
    - Automatic temperature tuning
    - Replay buffer
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims=[256, 256],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        buffer_capacity: int = 1000000,
        batch_size: int = 256,
        device: str = 'cpu'
    ):
        """
        Initialize SAC agent.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Temperature parameter
            auto_tune_alpha: If True, automatically tune temperature
            buffer_capacity: Replay buffer size
            batch_size: Training batch size
            device: 'cpu' or 'cuda'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_tune_alpha = auto_tune_alpha
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Policy network
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Twin Q-networks
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)

        # Target Q-networks
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=learning_rate)

        # Automatic temperature tuning
        if self.auto_tune_alpha:
            self.target_entropy = -action_dim  # Heuristic: -dim(A)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.log_alpha = torch.tensor(np.log(alpha), device=self.device)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Training stats
        self.update_count = 0

        logger.info(f"SAC Agent initialized: state_dim={state_dim}, action_dim={action_dim}, "
                    f"auto_tune_alpha={auto_tune_alpha}")

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action using current policy.

        Args:
            state: Current state
            deterministic: If True, use mean action (for evaluation)

        Returns:
            action: Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.policy.get_action(state_tensor, deterministic=deterministic)

        return action.cpu().numpy()[0]

    def update(self) -> Dict[str, float]:
        """
        Update SAC networks.

        Returns:
            losses: Dictionary of loss values
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)

        # ========== Update Q-networks ==========

        with torch.no_grad():
            # Sample actions from current policy
            next_actions, next_log_probs = self.policy.sample(next_states)

            # Compute target Q-values using twin Q-networks
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next)

            # Add entropy term
            alpha = self.log_alpha.exp()
            target_q = rewards + (1 - dones) * self.gamma * (min_q_next - alpha * next_log_probs)

        # Update Q1
        q1_pred = self.q1(states, actions)
        q1_loss = F.mse_loss(q1_pred, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        # Update Q2
        q2_pred = self.q2(states, actions)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # ========== Update Policy ==========

        # Sample actions from current policy
        new_actions, log_probs = self.policy.sample(states)

        # Compute Q-values for new actions
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        # Policy loss: maximize Q - alpha * log_prob
        alpha = self.log_alpha.exp().detach()
        policy_loss = (alpha * log_probs - min_q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ========== Update Temperature ==========

        alpha_loss = 0.0
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            alpha_loss = alpha_loss.item()

        # ========== Soft update target networks ==========

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        self.update_count += 1

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss,
            'alpha': self.log_alpha.exp().item()
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """
        Soft update target network: θ_target = τ*θ_source + (1-τ)*θ_target

        Args:
            source: Source network
            target: Target network
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filepath: str):
        """Save model to file."""
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'update_count': self.update_count
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])

        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])

        self.log_alpha = checkpoint['log_alpha']
        self.update_count = checkpoint['update_count']

        logger.info(f"Model loaded from {filepath}")


if __name__ == '__main__':
    # Quick test
    agent = SACAgent(state_dim=34, action_dim=4)

    # Test action selection
    state = np.random.randn(34)
    action = agent.select_action(state)
    print(f"Sampled action: {action}")

    # Test training update
    for i in range(1000):
        agent.replay_buffer.push(
            state=np.random.randn(34),
            action=np.random.randn(4),
            reward=np.random.randn(),
            next_state=np.random.randn(34),
            done=False
        )

    losses = agent.update()
    print(f"Losses: {losses}")
