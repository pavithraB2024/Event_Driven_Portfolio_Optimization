import pytest
import numpy as np
from src.agents.gnn_sac_agent import GNNSACAgent

def test_rc1_nextgen_respects_fixed_alpha():
    """
    RC-1: GNNSACAgent should not force auto_tune_alpha=True if use_nextgen=True.
    It should respect the auto_tune_alpha parameter passed in kwargs.
    """
    # Mock parameters
    kwargs = {
        'state_dim': 100,
        'action_dim': 10,
        'node_features_dim': 5,
        'num_nodes': 10,
        'use_nextgen': True,
        'auto_tune_alpha': False,
        'alpha': 0.05,
        'learning_rate': 3e-4,
        'device': 'cpu'
    }
    
    # We need an adj_matrix for GNNSACAgent init
    kwargs['adj_matrix'] = np.zeros((10, 10))
    
    agent = GNNSACAgent(**kwargs)
    
    # Assertions
    assert agent.auto_tune_alpha is False
    assert agent.log_alpha.exp().item() == pytest.approx(0.05, abs=1e-3)

def test_rc3_dsr_skip_normalization_logic():
    """
    RC-3: Check if a helper function can correctly identify reward types that skip normalization.
    """
    def should_skip_normalization(reward_type):
        return reward_type in ('aggressive', 'dsr')
    
    assert should_skip_normalization('dsr') is True
    assert should_skip_normalization('aggressive') is True
    assert should_skip_normalization('log_utility') is False
    assert should_skip_normalization('sharpe') is False
