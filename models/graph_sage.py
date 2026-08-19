import torch
import torch.nn as nn
import torch.nn.functional as F

class GRL_Encoder(nn.Module):
    """
    Graph Representation Learning Encoder using GraphSAGE-like aggregation.
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GRL_Encoder, self).__init__()
        # Input projection
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        
        # Aggregation layer (Mean aggregator)
        # In standard GraphSAGE: New_h = W * Concat(Self, Mean(Neighbors))
        # Here we simplify to: New_h = W * (Self + Mean(Neighbors)) for GCN-style or similar
        self.lin_agg = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x, adj=None):
        """
        x: [Batch, Num_Nodes, In_Dim]
        adj: [Batch, Num_Nodes, Num_Nodes] (Adjacency Matrix)
        """
        # 1. Project Input
        h = F.relu(self.lin_in(x)) # [B, N, Hidden]
        
        if adj is not None:
             # Ensure adj is [B, N, N]
             if adj.dim() == 2:
                 adj = adj.unsqueeze(0)
             
             # 2. Aggregation: Mean of neighbors
             # A_norm = D^-1 * A. 
             # We assume adj passed in is already suitable or we just do simple sum/matmul.
             # h_neigh = Adj * h
             h_neigh = torch.matmul(adj, h)
             
             # 3. Combine (Self + Neighbor) - Simplified GCN/Sage
             # In full Sage we might concat.
             h_combined = h + h_neigh
             
             # 4. Output Projection
             out = self.lin_agg(h_combined)
        else:
             # Fallback if no graph (pure MLP)
             out = self.lin_agg(h)
             
        return F.relu(out)
