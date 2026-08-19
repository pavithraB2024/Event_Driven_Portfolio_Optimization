"""Temporal Graph Network components for sequential graph encoding."""
import torch
import torch.nn as nn

class TGNMemory(nn.Module):
    """
    Manages the hidden state (memory) for all nodes in the graph.
    """
    def __init__(self, num_nodes, memory_dim, device='cpu'):
        super(TGNMemory, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.device = device
        
        # We store memory as a buffer so it's part of state_dict but not a parameter
        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        
    def reset_memory(self):
        self.memory.fill_(0)
        
    def get_memory(self):
        return self.memory.clone()
    
    def set_memory(self, new_memory):
        # Detach to prevent infinite BPTT if not careful, 
        # but for BPTT training we might want to keep grad?
        # Typically in RL we detach between steps unless doing BPTT over trajectory.
        self.memory = new_memory
        
    def detach_memory(self):
        self.memory = self.memory.detach()

class TGNRecurrentLayer(nn.Module):
    """
    A Recurrent Graph Layer (Snapshot TGN).
    
    Logic:
    1. Aggregates neighborhood information using GAT.
    2. Updates node memory using a GRUCell: New_Mem = GRU(Agg_Msg, Old_Mem).
    3. Outputs the updated memory as the node embedding.
    """
    def __init__(self, in_dim, memory_dim, num_heads=4, dropout=0.1):
        super(TGNRecurrentLayer, self).__init__()
        
        self.in_dim = in_dim
        self.memory_dim = memory_dim
        
        # 1. Message Aggregator (GAT)
        # We project Input Features -> Memory Dim
        # This ensures divisible by beads
        
        self.lin_proj = nn.Linear(in_dim, memory_dim) 
        
        self.gat_agg = nn.MultiheadAttention(
            embed_dim=memory_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=False # Manual handling
        )
        
        # 2. Memory Updater (GRU)
        
        self.agg_to_mem = nn.Linear(memory_dim, memory_dim) 
        self.gru = nn.GRUCell(memory_dim, memory_dim)
        
        # Explicit Time Decay Parameter (Learnable)
        # Initializes to sigmoid(0) = 0.5 decay per step.
        # Can settle closer to 1 (slow decay) or 0 (fast decay).
        self.time_decay_param = nn.Parameter(torch.tensor(2.0)) # Init > 0 so sigmoid ~ 0.88
        
    def forward(self, x, memory, adj_mask=None):
        """
        x: [Batch, N, F] (Current Node Features)
        memory: [Batch, N, M] (Previous Node Memory)
        adj_mask: [Batch, N, N]
        """
        batch_size, num_nodes, _ = x.shape
        
        # --- 1. Aggregation (GAT) ---
        # Project features to memory dimension
        h = self.lin_proj(x) # [B, N, M]
        
        # Transpose for MHA: [N, B, M]
        h_t = h.transpose(0, 1)
        
        # Mask handling
        attn_bias = None
        if adj_mask is not None:
             # PyTorch MHA with batch_first=False expects [N, N] or [B*Headers, N, N]
             # adj_mask is [B, N, N] (Weighted)
             
             # Create Additive Bias: log(adj + eps)
             # Clamp to avoid NaN in log for negative correlations/events
             attn_bias = torch.log(torch.clamp(adj_mask, min=0) + 1e-9)
             
             # Expand to [B*Heads, N, N]
             # We repeat each batch element num_heads times contiguously
             attn_bias = attn_bias.repeat_interleave(self.gat_agg.num_heads, dim=0)

        # agg_msg: [N, B, M]
        # Pass attn_bias as mask
        agg_msg_t, _ = self.gat_agg(h_t, h_t, h_t, attn_mask=attn_bias, need_weights=False)
        
        # Transpose back: [B, N, M]
        agg_msg = agg_msg_t.transpose(0, 1)
        
        # --- 2. Memory Update (GRU) ---
        # Flatten for GRUCell: [B*N, M]
        
        # Project msg to memory aim (identity if sizes match)
        gru_input = self.agg_to_mem(agg_msg).reshape(-1, self.gru.input_size)
        
        # Flatten memory: [B*N, M]
        flat_memory = memory.reshape(-1, self.gru.hidden_size)
        
        # Apply Explicit Time Decay (Lingering Effect)
        # Decay the *old* memory before updating with *new* events
        decay_factor = torch.sigmoid(self.time_decay_param)
        flat_memory = flat_memory * decay_factor
        
        # Update
        new_memory_flat = self.gru(gru_input, flat_memory)
        
        # Reshape back: [B, N, M]
        new_memory = new_memory_flat.reshape(batch_size, num_nodes, -1)
        
        return new_memory
