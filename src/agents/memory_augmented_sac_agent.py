"""
Memory-Augmented Soft Actor-Critic (SAC) Agent for Path-Dependent Portfolio Allocation

SAC with LSTM/GRU/Transformer for handling path-dependent transaction costs
based on the Memory-Augmented BSDE Control Loop patent innovations.
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
from typing import Tuple, Dict, Optional, List

import sys
import os

# Ensure models can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from models.graph_sage import GRL_Encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class LSTMGaussianPolicy(nn.Module):
    """
    LSTM-based Gaussian policy network for SAC with memory for path-dependent costs.
    Outputs mean and log_std for a diagonal Gaussian distribution.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 lstm_hidden_size: int = 64, num_lstm_layers: int = 1,
                 hidden_dims: List[int] = [256, 256],
                 use_graph: bool = False,
                 adj_matrix: Optional[torch.Tensor] = None,
                 node_features_dim: int = 32,
                 num_nodes: int = 20,
                 stock_mask: Optional[torch.Tensor] = None):
        super(LSTMGaussianPolicy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.use_graph = use_graph
        self.adj_matrix = adj_matrix
        self.num_nodes = num_nodes
        self.node_features_dim = node_features_dim

        # allocation-universe restriction for the tanh-squashed actor.
        # Forbidden indices are forced to 0 post-tanh; the env's positive
        # normalization then yields exactly zero weight on those assets.
        if stock_mask is None:
            stock_mask = torch.ones(action_dim, dtype=torch.bool)
        self.register_buffer('stock_mask', stock_mask.bool())
        self._action_mask_active = bool((~self.stock_mask).any().item())

        # Graph Encoder
        if self.use_graph:
            self.gnn = GRL_Encoder(in_dim=node_features_dim, hidden_dim=64, out_dim=64)
            # After GNN: We get (Batch, Num_Nodes, 64)
            # We assume we flatten it or pick specific nodes. 
            # For simplicity, we flatten all nodes: Num_Nodes * 64
            # Or we assume the LSTM takes the flattened graph embedding as input.
            self.lstm_input_dim = num_nodes * 64
        else:
            self.lstm_input_dim = state_dim

        # LSTM for processing historical state sequence
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # Network after LSTM
        self.fc1 = nn.Linear(lstm_hidden_size, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

        self.mean = nn.Linear(hidden_dims[1], action_dim)
        self.log_std = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, state: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with LSTM hidden state.

        Args:
            state: State tensor [batch_size, sequence_len, state_dim] or [batch_size, state_dim]
            hidden: LSTM hidden state (h, c)

        Returns:
            mean: Mean of Gaussian distribution
            log_std: Log standard deviation
            hidden: Updated hidden state
        """
        # Ensure state is 3D for LSTM [batch_size, seq_len, features]
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence dimension [batch_size, 1, state_dim]
        
        batch_size, seq_len, _ = state.shape

        if self.use_graph:
            # State reshaping: (Batch, Seq_Len, Flattened_Input) -> (Batch * Seq_Len, Num_Nodes, Node_Feats)
            # We assume state contains the raw node features flattened?
            # Or we assume state IS the raw node features [Batch, Seq_Len, Num_Nodes * Node_Feats]
            
            # Reshape for GNN
            # 1. Flatten Batch and Seq
            flat_state = state.view(-1, self.num_nodes, self.node_features_dim)
            
            # 2. Pass through GNN
            # Adj matrix needs to be on same device
            if self.adj_matrix is not None:
                adj = self.adj_matrix.to(state.device)
                
            # GNN out: (Batch*Seq, Num_Nodes, 64)
            gnn_out = self.gnn(flat_state, adj)
            
            # 3. Flatten back for LSTM input
            # (Batch*Seq, Num_Nodes * 64)
            gnn_flat = gnn_out.view(batch_size, seq_len, -1)
            
            lstm_input = gnn_flat
        else:
            lstm_input = state
        
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        
        # Use last time step
        x = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_size]
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std, hidden

    def sample(self, state: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, epsilon: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample action from policy with memory state.

        Args:
            state: State tensor
            hidden: LSTM hidden state
            epsilon: Small constant for numerical stability

        Returns:
            action: Sampled action (squashed through tanh)
            log_prob: Log probability of action
            hidden: Updated hidden state
        """
        mean, log_std, hidden = self.forward(state, hidden)
        std = log_std.exp()

        # Sample from Gaussian
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick

        # Apply tanh squashing
        action = torch.tanh(x_t)

        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)

        # enforce allocation-universe restriction. Forbidden indices are
        # zeroed in the action (env's positive normalization then yields zero
        # weight) and excluded from the entropy sum so masked dims don't inflate H.
        if self._action_mask_active:
            m = self.stock_mask.to(action.device).float()
            action = action * m
            log_prob = log_prob * m

        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, hidden

    def get_action(self, state: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                   deterministic: bool = False) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action for inference with memory state.

        Args:
            state: State tensor
            hidden: LSTM hidden state
            deterministic: If True, return mean action

        Returns:
            action: Action to take
            hidden: Updated hidden state
        """
        mean, log_std, hidden = self.forward(state, hidden)

        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)

        # apply universe restriction at inference too.
        if self._action_mask_active:
            action = action * self.stock_mask.to(action.device).float()

        return action, hidden

    def set_action_mask(self, stock_mask):
        """install or replace the allocation-universe restriction."""
        m = torch.as_tensor(stock_mask, dtype=torch.bool, device=self.stock_mask.device)
        assert m.shape == self.stock_mask.shape, "stock_mask shape mismatch"
        self.stock_mask.copy_(m)
        self._action_mask_active = bool((~self.stock_mask).any().item())


class GRUGaussianPolicy(nn.Module):
    """
    GRU-based Gaussian policy network for SAC with memory for path-dependent costs.
    Outputs mean and log_std for a diagonal Gaussian distribution.
    """

    def __init__(self, state_dim: int, action_dim: int, 
                 gru_hidden_size: int = 64, num_gru_layers: int = 1,
                 hidden_dims: List[int] = [256, 256]):
        super(GRUGaussianPolicy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gru_hidden_size = gru_hidden_size
        self.num_gru_layers = num_gru_layers

        # GRU for processing historical state sequence
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            batch_first=True
        )

        # Network after GRU
        self.fc1 = nn.Linear(gru_hidden_size, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

        self.mean = nn.Linear(hidden_dims[1], action_dim)
        self.log_std = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with GRU hidden state.

        Args:
            state: State tensor [batch_size, sequence_len, state_dim] or [batch_size, state_dim]
            hidden: GRU hidden state

        Returns:
            mean: Mean of Gaussian distribution
            log_std: Log standard deviation
            hidden: Updated hidden state
        """
        # Ensure state is 3D for GRU [batch_size, seq_len, features]
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence dimension [batch_size, 1, state_dim]
        
        gru_out, hidden = self.gru(state, hidden)
        
        # Use last time step
        x = gru_out[:, -1, :]  # [batch_size, gru_hidden_size]
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std, hidden

    def sample(self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None, epsilon: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy with memory state.

        Args:
            state: State tensor
            hidden: GRU hidden state
            epsilon: Small constant for numerical stability

        Returns:
            action: Sampled action (squashed through tanh)
            log_prob: Log probability of action
            hidden: Updated hidden state
        """
        mean, log_std, hidden = self.forward(state, hidden)
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

        return action, log_prob, hidden

    def get_action(self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None, 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action for inference with memory state.

        Args:
            state: State tensor
            hidden: GRU hidden state
            deterministic: If True, return mean action

        Returns:
            action: Action to take
            hidden: Updated hidden state
        """
        mean, log_std, hidden = self.forward(state, hidden)

        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)

        return action, hidden


class TransformerGaussianPolicy(nn.Module):
    """
    Transformer-based Gaussian policy network for SAC with attention for path-dependent costs.
    Outputs mean and log_std for a diagonal Gaussian distribution.
    """

    def __init__(self, state_dim: int, action_dim: int, 
                 transformer_hidden: int = 64, num_heads: int = 4, 
                 num_layers: int = 2, hidden_dims: List[int] = [256, 256],
                 max_seq_len: int = 100):
        super(TransformerGaussianPolicy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.transformer_hidden = transformer_hidden
        self.max_seq_len = max_seq_len

        # Linear layer to project state to transformer dimension
        self.input_projection = nn.Linear(state_dim, transformer_hidden)
        
        # Positional encoding
        self.positional_encoding = self._generate_positional_encoding(max_seq_len, transformer_hidden)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden,
            nhead=num_heads,
            dim_feedforward=transformer_hidden * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Network after transformer
        self.fc1 = nn.Linear(transformer_hidden, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

        self.mean = nn.Linear(hidden_dims[1], action_dim)
        self.log_std = nn.Linear(hidden_dims[1], action_dim)

    def _generate_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Generate positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: [1, max_len, d_model]

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with transformer attention.

        Args:
            state: State tensor [batch_size, sequence_len, state_dim] or [batch_size, state_dim]

        Returns:
            mean: Mean of Gaussian distribution
            log_std: Log standard deviation
        """
        # Ensure state is 3D for transformer [batch_size, seq_len, features]
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence dimension [batch_size, 1, state_dim]
        
        batch_size, seq_len = state.shape[0], state.shape[1]
        
        # Project state to transformer dimension
        x = self.input_projection(state)  # [batch_size, seq_len, transformer_hidden]
        
        # Add positional encoding
        if seq_len <= self.max_seq_len:
            pos_encoding = self.positional_encoding[:, :seq_len, :].expand(batch_size, -1, -1).to(x.device)
            x = x + pos_encoding
        else:
            # For sequences longer than max_seq_len, we'll use a sliding window approach
            # In practice, this should be handled by truncating or chunking
            raise ValueError(f"Sequence length {seq_len} exceeds max sequence length {self.max_seq_len}")
        
        # Apply transformer
        x = self.transformer(x)  # [batch_size, seq_len, transformer_hidden]
        
        # Use the last time step's output
        x = x[:, -1, :]  # [batch_size, transformer_hidden]
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, state: torch.Tensor, epsilon: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy with attention.

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
        Get action for inference with attention.

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


class LSTMQNetwork(nn.Module):
    """LSTM-based Q-Network (critic) for SAC with memory for path-dependent costs."""

    def __init__(self, state_dim: int, action_dim: int, 
                 lstm_hidden_size: int = 64, num_lstm_layers: int = 1,
                 hidden_dims: List[int] = [256, 256],
                 use_graph: bool = False,
                 adj_matrix: Optional[torch.Tensor] = None,
                 node_features_dim: int = 32,
                 num_nodes: int = 20):
        super(LSTMQNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.use_graph = use_graph
        self.adj_matrix = adj_matrix
        self.num_nodes = num_nodes
        self.node_features_dim = node_features_dim

        # Graph Encoder
        if self.use_graph:
            self.gnn = GRL_Encoder(in_dim=node_features_dim, hidden_dim=64, out_dim=64)
            self.lstm_input_dim = (num_nodes * 64) + action_dim
        else:
            self.lstm_input_dim = state_dim + action_dim

        # LSTM for processing state-action pairs
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # Network after LSTM
        layers = []
        input_dim = lstm_hidden_size

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM network.

        Args:
            state: State tensor [batch_size, seq_len, state_dim] or [batch_size, state_dim]
            action: Action tensor [batch_size, seq_len, action_dim] or [batch_size, action_dim]
            hidden: LSTM hidden state

        Returns:
            q_value: Q-value estimate
            hidden: Updated hidden state
        """
        # Ensure both state and action are 3D for LSTM
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [batch_size, 1, state_dim]
        if action.dim() == 2:
            action = action.unsqueeze(1)  # [batch_size, 1, action_dim]
        
        batch_size, seq_len, _ = state.shape

        if self.use_graph:
             # Reshape for GNN
            flat_state = state.view(-1, self.num_nodes, self.node_features_dim)
            
            adj = None
            if self.adj_matrix is not None:
                adj = self.adj_matrix.to(state.device)
            
            gnn_out = self.gnn(flat_state, adj)
            gnn_flat = gnn_out.view(batch_size, seq_len, -1)
            
            # Concatenate encoded state and action
            x = torch.cat([gnn_flat, action], dim=-1)
        else:
            # Concatenate state and action
            x = torch.cat([state, action], dim=-1)  # [batch_size, seq_len, state_dim + action_dim]
        
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Use last time step
        x = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_size]
        
        q_value = self.network(x)
        return q_value, hidden


class GRUQNetwork(nn.Module):
    """GRU-based Q-Network (critic) for SAC with memory for path-dependent costs."""

    def __init__(self, state_dim: int, action_dim: int, 
                 gru_hidden_size: int = 64, num_gru_layers: int = 1,
                 hidden_dims: List[int] = [256, 256]):
        super(GRUQNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gru_hidden_size = gru_hidden_size
        self.num_gru_layers = num_gru_layers

        # GRU for processing state-action pairs
        self.gru = nn.GRU(
            input_size=state_dim + action_dim,
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            batch_first=True
        )

        # Network after GRU
        layers = []
        input_dim = gru_hidden_size

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor, 
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GRU network.

        Args:
            state: State tensor [batch_size, seq_len, state_dim] or [batch_size, state_dim]
            action: Action tensor [batch_size, seq_len, action_dim] or [batch_size, action_dim]
            hidden: GRU hidden state

        Returns:
            q_value: Q-value estimate
            hidden: Updated hidden state
        """
        # Ensure both state and action are 3D for GRU
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [batch_size, 1, state_dim]
        if action.dim() == 2:
            action = action.unsqueeze(1)  # [batch_size, 1, action_dim]
        
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)  # [batch_size, seq_len, state_dim + action_dim]
        
        gru_out, hidden = self.gru(x, hidden)
        
        # Use last time step
        x = gru_out[:, -1, :]  # [batch_size, gru_hidden_size]
        
        q_value = self.network(x)
        return q_value, hidden


class TransformerQNetwork(nn.Module):
    """Transformer-based Q-Network (critic) for SAC with attention for path-dependent costs."""

    def __init__(self, state_dim: int, action_dim: int, 
                 transformer_hidden: int = 64, num_heads: int = 4, 
                 num_layers: int = 2, hidden_dims: List[int] = [256, 256],
                 max_seq_len: int = 100):
        super(TransformerQNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.transformer_hidden = transformer_hidden
        self.max_seq_len = max_seq_len

        # Linear layer to project state-action pair to transformer dimension
        self.input_projection = nn.Linear(state_dim + action_dim, transformer_hidden)
        
        # Positional encoding
        self.positional_encoding = self._generate_positional_encoding(max_seq_len, transformer_hidden)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden,
            nhead=num_heads,
            dim_feedforward=transformer_hidden * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Network after transformer
        layers = []
        input_dim = transformer_hidden

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def _generate_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Generate positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: [1, max_len, d_model]

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer network.

        Args:
            state: State tensor [batch_size, seq_len, state_dim] or [batch_size, state_dim]
            action: Action tensor [batch_size, seq_len, action_dim] or [batch_size, action_dim]

        Returns:
            q_value: Q-value estimate
        """
        # Ensure both state and action are 3D for transformer
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [batch_size, 1, state_dim]
        if action.dim() == 2:
            action = action.unsqueeze(1)  # [batch_size, 1, action_dim]
        
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)  # [batch_size, seq_len, state_dim + action_dim]
        
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Project to transformer dimension
        x = self.input_projection(x)  # [batch_size, seq_len, transformer_hidden]
        
        # Add positional encoding
        if seq_len <= self.max_seq_len:
            pos_encoding = self.positional_encoding[:, :seq_len, :].expand(batch_size, -1, -1).to(x.device)
            x = x + pos_encoding
        else:
            raise ValueError(f"Sequence length {seq_len} exceeds max sequence length {self.max_seq_len}")
        
        # Apply transformer
        x = self.transformer(x)  # [batch_size, seq_len, transformer_hidden]
        
        # Use the last time step's output
        x = x[:, -1, :]  # [batch_size, transformer_hidden]
        
        q_value = self.network(x)
        return q_value


class ReplayBuffer:
    """Experience replay buffer with path-dependent history."""

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


class SumTree:
    """
    Binary segment tree for O(log N) proportional-priority sampling.
    Used internally by PrioritizedReplayBuffer.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.size = 0
        self.write_idx = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data):
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(tree_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float):
        """Return (tree_idx, priority, data) for cumulative sum s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (Schaul et al., 2016).

    Samples transitions proportional to their TD-error priority.
    Uses importance-sampling weights to correct the bias.

    Drop-in replacement for ReplayBuffer — same push() / sample() interface
    with additional update_priorities() method.
    """

    def __init__(
        self,
        capacity: int = 1000000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 200000,
        epsilon: float = 1e-5,
    ):
        """
        Args:
            capacity: max buffer size
            alpha: how much prioritisation (0 = uniform, 1 = full priority)
            beta_start: initial importance-sampling exponent
            beta_frames: steps over which beta anneals to 1.0
            epsilon: small constant added to priorities to avoid zero-priority
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.max_priority = 1.0
        self.frame = 0

    def _beta(self) -> float:
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        """Add transition with max priority (ensures it gets sampled at least once)."""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample batch with proportional priorities.

        Returns dict with keys: states, actions, rewards, next_states, dones,
            weights (IS weights), tree_indices (for priority updates)
        """
        beta = self._beta()
        self.frame += 1

        indices = []
        priorities = []
        samples = []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            if data is None:
                # Safety: re-sample from a valid segment
                s = np.random.uniform(0, self.tree.total)
                idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            samples.append(data)

        priorities_np = np.array(priorities, dtype=np.float64)
        # Avoid division by zero
        total = max(self.tree.total, 1e-8)
        probs = priorities_np / total
        probs = np.clip(probs, 1e-8, None)

        # Importance-sampling weights
        N = self.tree.size
        weights = (N * probs) ** (-beta)
        weights /= weights.max()  # Normalise to [0, 1]

        states, actions, rewards, next_states, dones = zip(*samples)

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones),
            'weights': weights.astype(np.float32),
            'tree_indices': np.array(indices, dtype=np.int64),
        }

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities for sampled transitions based on new TD errors.

        Args:
            tree_indices: indices from sample() output
            td_errors: absolute TD-error for each transition
        """
        for idx, td in zip(tree_indices, td_errors):
            priority = (abs(td) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(idx), priority)

    def __len__(self):
        return self.tree.size


class MemoryAugmentedSACAgent:
    """
    Memory-Augmented Soft Actor-Critic agent for path-dependent portfolio allocation
    using LSTM/GRU/Transformer networks based on patent innovations.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        network_type: str = 'LSTM',  # 'LSTM', 'GRU', or 'TRANSFORMER'
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        buffer_capacity: int = 1000000,
        batch_size: int = 256,
        device: str = 'cpu',
        lstm_hidden_size: int = 64,
        num_lstm_layers: int = 1,
        max_seq_len: int = 100,
        hidden_dims: List[int] = [256, 256],
        use_graph: bool = False,
        adj_matrix: Optional[np.ndarray] = None,
        node_features_dim: int = 32,
        num_nodes: int = 20,
        use_per: bool = False,
        stock_mask: Optional[torch.Tensor] = None,
    ):
        """
        Initialize Memory-Augmented SAC agent.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            network_type: Type of memory network ('LSTM', 'GRU', 'TRANSFORMER')
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Temperature parameter
            auto_tune_alpha: If True, automatically tune temperature
            buffer_capacity: Replay buffer size
            batch_size: Training batch size
            device: 'cpu' or 'cuda'
            lstm_hidden_size: Hidden size for LSTM/GRU networks
            num_lstm_layers: Number of LSTM layers
            max_seq_len: Maximum sequence length for Transformer
            hidden_dims: List of hidden layer dimensions for the dense networks
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network_type = network_type
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_tune_alpha = auto_tune_alpha
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.max_seq_len = max_seq_len
        self.use_graph = use_graph
        self.adj_matrix = None
        self.node_features_dim = node_features_dim
        self.num_nodes = num_nodes
        
        if adj_matrix is not None:
            self.adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)

        # Initialize networks based on type
        if network_type == 'LSTM':
            # Policy network
            self.policy = LSTMGaussianPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                lstm_hidden_size=lstm_hidden_size,
                num_lstm_layers=num_lstm_layers,
                hidden_dims=hidden_dims,
                use_graph=use_graph,
                adj_matrix=self.adj_matrix,
                node_features_dim=node_features_dim,
                num_nodes=num_nodes,
                stock_mask=stock_mask,
            ).to(self.device)
            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

            # Twin Q-networks
            self.q1 = LSTMQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                lstm_hidden_size=lstm_hidden_size,
                num_lstm_layers=num_lstm_layers,
                hidden_dims=hidden_dims,
                use_graph=use_graph,
                adj_matrix=self.adj_matrix,
                node_features_dim=node_features_dim,
                num_nodes=num_nodes
            ).to(self.device)
            self.q2 = LSTMQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                lstm_hidden_size=lstm_hidden_size,
                num_lstm_layers=num_lstm_layers,
                hidden_dims=hidden_dims,
                use_graph=use_graph,
                adj_matrix=self.adj_matrix,
                node_features_dim=node_features_dim,
                num_nodes=num_nodes
            ).to(self.device)

            # Target Q-networks
            self.q1_target = LSTMQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                lstm_hidden_size=lstm_hidden_size,
                num_lstm_layers=num_lstm_layers,
                hidden_dims=hidden_dims,
                use_graph=use_graph,
                adj_matrix=self.adj_matrix,
                node_features_dim=node_features_dim,
                num_nodes=num_nodes
            ).to(self.device)
            self.q2_target = LSTMQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                lstm_hidden_size=lstm_hidden_size,
                num_lstm_layers=num_lstm_layers,
                hidden_dims=hidden_dims,
                use_graph=use_graph,
                adj_matrix=self.adj_matrix,
                node_features_dim=node_features_dim,
                num_nodes=num_nodes
            ).to(self.device)

        elif network_type == 'GRU':
            # Policy network
            self.policy = GRUGaussianPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                gru_hidden_size=lstm_hidden_size,
                num_gru_layers=num_lstm_layers,
                hidden_dims=hidden_dims
            ).to(self.device)
            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

            # Twin Q-networks
            self.q1 = GRUQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                gru_hidden_size=lstm_hidden_size,
                num_gru_layers=num_lstm_layers,
                hidden_dims=hidden_dims
            ).to(self.device)
            self.q2 = GRUQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                gru_hidden_size=lstm_hidden_size,
                num_gru_layers=num_lstm_layers,
                hidden_dims=hidden_dims
            ).to(self.device)

            # Target Q-networks
            self.q1_target = GRUQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                gru_hidden_size=lstm_hidden_size,
                num_gru_layers=num_lstm_layers,
                hidden_dims=hidden_dims
            ).to(self.device)
            self.q2_target = GRUQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                gru_hidden_size=lstm_hidden_size,
                num_gru_layers=num_lstm_layers,
                hidden_dims=hidden_dims
            ).to(self.device)

        elif network_type == 'TRANSFORMER':
            # Policy network
            self.policy = TransformerGaussianPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                transformer_hidden=lstm_hidden_size,
                max_seq_len=max_seq_len,
                hidden_dims=hidden_dims
            ).to(self.device)
            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

            # Twin Q-networks
            self.q1 = TransformerQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                transformer_hidden=lstm_hidden_size,
                max_seq_len=max_seq_len,
                hidden_dims=hidden_dims
            ).to(self.device)
            self.q2 = TransformerQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                transformer_hidden=lstm_hidden_size,
                max_seq_len=max_seq_len,
                hidden_dims=hidden_dims
            ).to(self.device)

            # Target Q-networks
            self.q1_target = TransformerQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                transformer_hidden=lstm_hidden_size,
                max_seq_len=max_seq_len,
                hidden_dims=hidden_dims
            ).to(self.device)
            self.q2_target = TransformerQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                transformer_hidden=lstm_hidden_size,
                max_seq_len=max_seq_len,
                hidden_dims=hidden_dims
            ).to(self.device)
        else:
            raise ValueError(f"Unknown network type: {network_type}")

        # Copy parameters to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_target.eval()
        self.q2_target.eval()

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
        self.use_per = use_per
        if self.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Training stats
        self.update_count = 0

        logger.info(f"Memory-Augmented SAC Agent initialized: state_dim={state_dim}, action_dim={action_dim}, "
                    f"network_type={network_type}, auto_tune_alpha={auto_tune_alpha}")

    def select_action(self, state: np.ndarray, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                     deterministic: bool = False) -> Tuple[np.ndarray, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Select action using current policy with memory state.

        Args:
            state: Current state
            hidden_state: Previous memory state for LSTM/GRU
            deterministic: If True, use mean action (for evaluation)

        Returns:
            action: Selected action
            hidden_state: Updated memory state
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, state_dim]

        with torch.no_grad():
            if self.network_type == 'LSTM':
                action, new_hidden_state = self.policy.get_action(state_tensor, hidden_state, deterministic=deterministic)
                return action.cpu().numpy()[0], new_hidden_state
            elif self.network_type == 'GRU':
                action, new_hidden_state = self.policy.get_action(state_tensor, hidden_state, deterministic=deterministic)
                return action.cpu().numpy()[0], new_hidden_state
            else:  # TRANSFORMER
                action = self.policy.get_action(state_tensor, deterministic=deterministic)
                return action.cpu().numpy()[0], hidden_state

    def update(self) -> Dict[str, float]:
        """
        Update SAC networks with memory states.

        Returns:
            losses: Dictionary of loss values
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(batch['states']).unsqueeze(1).to(self.device)  # [batch_size, 1, state_dim]
        actions = torch.FloatTensor(batch['actions']).unsqueeze(1).to(self.device)  # [batch_size, 1, action_dim]
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).unsqueeze(1).to(self.device)  # [batch_size, 1, state_dim]
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)

        # ========== Update Q-networks ==========

        # For LSTM/GRU: we'll process with zero hidden states for simplicity
        # In a more complex implementation, we might store and use actual hidden states
        if self.network_type in ['LSTM', 'GRU']:
            # Process next states with policy to get next actions and log probabilities
            with torch.no_grad():
                if self.network_type == 'LSTM':
                    next_actions, next_log_probs, _ = self.policy.sample(next_states, hidden=None)
                    
                    # Compute target Q-values using twin Q-networks
                    q1_next, _ = self.q1_target(next_states, next_actions, hidden=None)
                    q2_next, _ = self.q2_target(next_states, next_actions, hidden=None)
                else:  # GRU
                    next_actions, next_log_probs, _ = self.policy.sample(next_states, hidden=None)
                    
                    # Compute target Q-values using twin Q-networks
                    q1_next, _ = self.q1_target(next_states, next_actions, hidden=None)
                    q2_next, _ = self.q2_target(next_states, next_actions, hidden=None)
            # Add entropy term
            alpha = self.log_alpha.exp()
            target_q = rewards + (1 - dones) * self.gamma * (torch.min(q1_next, q2_next) - alpha * next_log_probs)
            target_q = target_q.detach()

            # Update Q1
            q1_pred, _ = self.q1(states, actions, hidden=None)
            q1_loss = F.mse_loss(q1_pred, target_q)

            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=1.0)
            self.q1_optimizer.step()

            # Update Q2
            q2_pred, _ = self.q2(states, actions, hidden=None)
            q2_loss = F.mse_loss(q2_pred, target_q)

            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=1.0)
            self.q2_optimizer.step()
        else:  # TRANSFORMER
            # Process next states with policy to get next actions and log probabilities
            with torch.no_grad():
                next_actions, next_log_probs = self.policy.sample(next_states)
                
                # Compute target Q-values using twin Q-networks
                q1_next = self.q1_target(next_states, next_actions)
                q2_next = self.q2_target(next_states, next_actions)
                
                # Add entropy term
                alpha = self.log_alpha.exp()
                target_q = rewards + (1 - dones) * self.gamma * (torch.min(q1_next, q2_next) - alpha * next_log_probs)

            # Update Q1
            q1_pred = self.q1(states, actions)
            q1_loss = F.mse_loss(q1_pred, target_q)

            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=1.0)
            self.q1_optimizer.step()

            # Update Q2
            q2_pred = self.q2(states, actions)
            q2_loss = F.mse_loss(q2_pred, target_q)

            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=1.0)
            self.q2_optimizer.step()

        # ========== Update Policy ==========

        # Sample actions from current policy
        if self.network_type in ['LSTM', 'GRU']:
            new_actions, log_probs, _ = self.policy.sample(states, hidden=None)
            
            # Compute Q-values for new actions
            if self.network_type == 'LSTM':
                q1_new, _ = self.q1(states, new_actions, hidden=None)
                q2_new, _ = self.q2(states, new_actions, hidden=None)
            else:  # GRU
                q1_new, _ = self.q1(states, new_actions, hidden=None)
                q2_new, _ = self.q2(states, new_actions, hidden=None)
        else:  # TRANSFORMER
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
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
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
            'network_type': self.network_type,
            'update_count': self.update_count
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy'], strict=False)
        if hasattr(self.policy, 'stock_mask'):
            self.policy._action_mask_active = bool(
                (~self.policy.stock_mask).any().item()
            )
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])

        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])

        self.log_alpha = checkpoint['log_alpha']
        self.network_type = checkpoint['network_type']
        self.update_count = checkpoint['update_count']

        logger.info(f"Model loaded from {filepath}")


if __name__ == '__main__':
    # Quick test
    agent = MemoryAugmentedSACAgent(
        state_dim=34,
        action_dim=4,
        network_type='LSTM'
    )

    # Test action selection
    state = np.random.randn(34)
    action, hidden = agent.select_action(state)
    print(f"LSTM Sampled action: {action}")
    
    agent_gru = MemoryAugmentedSACAgent(
        state_dim=34,
        action_dim=4,
        network_type='GRU'
    )
    
    # Test action selection
    action_gru, hidden_gru = agent_gru.select_action(state)
    print(f"GRU Sampled action: {action_gru}")
    
    agent_transformer = MemoryAugmentedSACAgent(
        state_dim=34,
        action_dim=4,
        network_type='TRANSFORMER'
    )
    
    # Test action selection
    action_transformer = agent_transformer.select_action(state)
    print(f"Transformer Sampled action: {action_transformer}")

    # Test training update - first populate buffer
    for i in range(1000):
        agent.replay_buffer.push(
            state=np.random.randn(34),
            action=np.random.randn(4),
            reward=np.random.randn(),
            next_state=np.random.randn(34),
            done=False
        )

    losses = agent.update()
    print(f"LSTM Losses: {losses}")
    
    for i in range(1000):
        agent_gru.replay_buffer.push(
            state=np.random.randn(34),
            action=np.random.randn(4),
            reward=np.random.randn(),
            next_state=np.random.randn(34),
            done=False
        )

    losses_gru = agent_gru.update()
    print(f"GRU Losses: {losses_gru}")
    
    for i in range(1000):
        agent_transformer.replay_buffer.push(
            state=np.random.randn(34),
            action=np.random.randn(4),
            reward=np.random.randn(),
            next_state=np.random.randn(34),
            done=False
        )

    losses_transformer = agent_transformer.update()
    print(f"Transformer Losses: {losses_transformer}")