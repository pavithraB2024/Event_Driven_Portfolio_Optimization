
import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from models.graph_sage import GRL_Encoder
from src.agents.memory_augmented_sac_agent import MemoryAugmentedSACAgent

class TestGraphSAGE(unittest.TestCase):
    def test_encoder_shape(self):
        print("\n--- Testing GRL_Encoder Shapes ---")
        batch_size = 4
        num_nodes = 5
        in_dim = 10
        out_dim = 64
        
        # Random Features: (Batch, Nodes, In)
        x = torch.randn(batch_size, num_nodes, in_dim)
        
        # Adjacency: (Nodes, Nodes)
        adj = torch.eye(num_nodes)
        
        encoder = GRL_Encoder(in_dim=in_dim, hidden_dim=32, out_dim=out_dim)
        output = encoder(x, adj)
        
        print(f"Input: {x.shape}")
        print(f"Output: {output.shape}")
        
        self.assertEqual(output.shape, (batch_size, num_nodes, out_dim))
        
    def test_agent_integration_lstm(self):
        print("\n--- Testing SAC Agent with Graph (LSTM) ---")
        state_dim = 20 * 32 # Nodes * Features
        action_dim = 20
        num_nodes = 20
        node_features = 32
        
        # Create Dummy Adjacency
        adj = np.eye(num_nodes)
        
        agent = MemoryAugmentedSACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            network_type='LSTM',
            use_graph=True,
            adj_matrix=adj,
            num_nodes=num_nodes,
            node_features_dim=node_features
        )
        
        # Fake State: (Nodes * Features)
        # The agent expects a FLATTENED state from the Env that it will reshape internally
        state = np.random.randn(state_dim)
        
        # Select Action
        action, hidden = agent.select_action(state)
        
        print(f"Action Shape: {action.shape}")
        self.assertEqual(action.shape[0], action_dim)
        
        # Forward pass verification
        print("Agent forward pass successful.")

if __name__ == '__main__':
    # Manual Run for Debugging
    t = TestGraphSAGE()
    try:
        t.test_encoder_shape()
        print("Encoder Test: PASS")
        t.test_agent_integration_lstm()
        print("Agent Test: PASS")
    except Exception as e:
        print(f"\n!!!! TEST FAILED !!!!\nError: {e}")
        import traceback
        traceback.print_exc()

