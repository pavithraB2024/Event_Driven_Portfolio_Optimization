"""Event-Driven GNN-SAC agent with HGT encoder and hierarchical softmax actor."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Optional, Dict, Any
from .memory_augmented_sac_agent import MemoryAugmentedSACAgent, LSTMGaussianPolicy
from .tgn_components import TGNRecurrentLayer
from .nextgen_gnn_components import (
    NextGenEventGraphEncoder,
    IQNCritic,
    RegimeEntropyScheduler,
    DualHeadIQNCritic,
)

from .loss_calculator import StandardSACLossCalculator, NextGenCVaRLossCalculator


def apply_crisis_mask(
    logits: torch.Tensor,            # [B, N]
    panic_signal: torch.Tensor,      # [B, 1]
    equity_idx: list[int],
    haven_idx: list[int],
    threshold: float = 0.8,
    suppress: float = 0.9,
    boost: float = 5.0,
) -> torch.Tensor:
    """Suppress equity logits, boost safe-haven logits during panic. Pure function."""
    out = logits.clone()
    
    # Soft approximation of a step function to maintain gradient flow to panic_signal
    panic_mask = torch.sigmoid((panic_signal - threshold) * 100.0)
    
    # Apply suppression and boosting where panic condition is met
    for idx in equity_idx:
        out[:, idx] = out[:, idx] - (panic_mask.view(-1) * suppress)
    for idx in haven_idx:
        out[:, idx] = out[:, idx] + (panic_mask.view(-1) * boost)
        
    return out


def apply_momentum_mask(
    logits: torch.Tensor,        # [B, N]
    node_feats: torch.Tensor,     # [B, N, F]
    momentum_index: int = 0,
    threshold: float = -0.2,
) -> torch.Tensor:
    """
    Suppress logits for individual assets with high negative momentum.
    Pure function.
    """
    # Extract momentum: [B, N]
    momentum = node_feats[:, :, momentum_index]
    
    # Slice to match logits (only stocks have action logits)
    if momentum.size(1) > logits.size(1):
        momentum = momentum[:, :logits.size(1)]
    
    # Sigmoid mask centered at threshold
    mask = torch.sigmoid((momentum - threshold) * 10.0)
    
    return logits * mask


def apply_residual_connection(
    x_base: torch.Tensor,
    x_gnn: torch.Tensor,
    alpha: float = 0.5
) -> torch.Tensor:
    """
    Blends base features with GNN-transformed features to prevent over-smoothing.
    Pure function.
    """
    return alpha * x_base + (1.0 - alpha) * x_gnn



class FeatureGatingModule(nn.Module):
    """
    D6: Learnable Feature Gating (Learnable SHAP)
    Applies element-wise sigmoid gating to input features to suppress noise.
    
    Inspired by Sun et al. (2024) SHAP selection but learnable end-to-end.
    Allows the model to focus on the 32 (or K) most important features dynamically.
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )
        # Initialize gate slightly open
        nn.init.constant_(self.gate[0].bias, 0.5)

    def forward(self, x):
        """
        x: [B, N, F] - Batch of N nodes with F features.
        """
        g = self.gate(x) # [B, N, F]
        return x * g




def hierarchical_softmax(
    sector_logits: torch.Tensor,  # [B, num_sectors]
    stock_logits: torch.Tensor,   # [B, num_stocks]
    sector_map: torch.Tensor,     # [num_stocks] -> sector_id
    sector_mask: torch.Tensor = None,  # [num_sectors] bool; True = keep, False = forbid
    stock_mask: torch.Tensor = None,   # [num_stocks] bool; True = keep, False = forbid
) -> torch.Tensor:
    """
    Computes hierarchical softmax to ensure stock weights within a sector sum to the sector's weight,
    and all sector weights sum to 1.0.

    ``sector_mask`` and ``stock_mask`` enforce a hard logit-level restriction
    on the allowable allocation universe. Forbidden entries are set to -inf prior to
    softmax/normalization, so they receive exactly zero weight and contribute zero
    gradient to the policy.
    """
    # --- logit-level action masking ------------------------------------
    # Defaults preserve existing behavior (no mask -> no-op).
    NEG_INF = -1.0e9  # finite stand-in for -inf, gradient-safe under exp()
    if sector_mask is not None:
        sm = sector_mask.to(sector_logits.device).bool()
        sector_logits = sector_logits.masked_fill(~sm.unsqueeze(0), NEG_INF)
    if stock_mask is not None:
        km = stock_mask.to(stock_logits.device).bool()
        stock_logits = stock_logits.masked_fill(~km.unsqueeze(0), NEG_INF)

    # 1. Sector level: [B, num_sectors]
    sector_weights = torch.softmax(sector_logits, dim=-1)

    # 2. Intra-sector level:
    # stock_logits: [B, num_stocks]
    exp_logits = torch.exp(stock_logits)

    B, N = stock_logits.shape
    num_sectors = sector_logits.shape[-1]

    # Expand sector_map to batch: [B, N]
    batch_sector_map = sector_map.unsqueeze(0).expand(B, N)

    # Initialize sector sums: [B, num_sectors]
    sector_sums = torch.zeros((B, num_sectors), device=stock_logits.device)
    sector_sums.scatter_add_(1, batch_sector_map, exp_logits)

    # Gather sector sums back to [B, N]
    gathered_sums = torch.gather(sector_sums, 1, batch_sector_map)

    # Compute intra-weights: [B, N]
    intra_weights = exp_logits / (gathered_sums + 1e-8)

    # 3. Final weights: multiply sector weight by intra-sector weight
    gathered_sector_weights = torch.gather(sector_weights, 1, batch_sector_map)
    final_weights = gathered_sector_weights * intra_weights

    return final_weights


def build_action_masks_from_indices(
    allowed_indices,
    sector_map: torch.Tensor,
    num_sectors: int,
):
    """
    helper: convert a list of allowed stock indices into the
    (sector_mask, stock_mask) bool tensors expected by ``hierarchical_softmax``.

    Sectors with zero allowed members are masked out, so the sector-level
    softmax cannot place probability mass on empty sectors.
    """
    num_stocks = sector_map.shape[0]
    stock_mask = torch.zeros(num_stocks, dtype=torch.bool)
    if len(allowed_indices) > 0:
        idx_tensor = torch.as_tensor(list(allowed_indices), dtype=torch.long)
        stock_mask[idx_tensor] = True
    allowed_sectors = sector_map[stock_mask].unique().tolist() if stock_mask.any() else []
    sector_mask = torch.zeros(num_sectors, dtype=torch.bool)
    for s in allowed_sectors:
        sector_mask[int(s)] = True
    return sector_mask, stock_mask

# Pure PyTorch GAT Layer
class SimpleGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, concat=True):
        super(SimpleGATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        
        # MHA expects embed_dim = num_heads * head_dim
        self.head_dim = out_dim // num_heads if concat else out_dim
        self.embed_dim = self.head_dim * num_heads
        
        # Only support "Concat=True" logic where output = heads * head_dim usually?
        # PyG GAT logic:
        # out_dim usually refers to "per head output" if concat=True, so total = heads*out
        # OR "total output" split into heads.
        # We will follow standard attention: input -> Q, K, V
        
        # We start with linear projection to embedding dimension
        self.lin = nn.Linear(in_dim, self.embed_dim)
        
        # Standard MultiHeadAttention
        # We utilize batch_first=True
        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
    def forward(self, x, adj_mask=None):
        """
        x: [Batch, Num_Nodes, In_Dim]
        adj_mask: [Batch, Num_Nodes, Num_Nodes] (Optional weighted adjacency)
                  Ideally in range [0, 1].
        """
        # Project
        h = self.lin(x) # [B, N, Embed]
        
        # Attention
        # MHA returns (attn_output, attn_weights)
        # We use adj_mask as an ADDITIVE attention bias.
        
        attn_bias = None
        if adj_mask is not None:
             # Ensure broadcasting [B, N, N]
             if adj_mask.dim() == 2:
                 adj_mask = adj_mask.unsqueeze(0) # [1, N, N]
            
             # Create Additive Bias: log(adj + eps)
             # If adj=1 -> log(1)=0 (No penalty)
             # If adj=0.5 -> log(0.5)=-0.69 (Penalty)
             # If adj=0 -> log(eps)=-20 (Blocked)
             
             # If batch size > 1 and adj is [1, N, N], it broadcasts in torch.log? Yes.
             
             batch_size = x.shape[0]
             
             # Expand mask to [B * Num_Heads, N, N]
             # adj_mask: [B, N, N]
             if adj_mask.shape[0] == 1 and batch_size > 1:
                 adj_mask = adj_mask.repeat(batch_size, 1, 1)
                 
             # Now repeat for heads
             # [B, N, N] -> [B, 1, N, N] -> [B, Heads, N, N] -> [B*Heads, N, N]
             # Clamp to avoid NaN in log for negative correlations/events
             attn_bias = torch.log(torch.clamp(adj_mask, min=0) + 1e-9)
             
             attn_bias = attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
             attn_bias = attn_bias.reshape(batch_size * self.num_heads, x.shape[1], x.shape[1])
             
        attn_out, _ = self.mha(h, h, h, attn_mask=attn_bias, need_weights=False)
        
        if self.concat:
            return F.elu(attn_out)
        else:
            return attn_out

class HGTLayer(nn.Module):
    """
    Heterogeneous Graph Transformer Layer.
    Uses type-specific Linear projections for Q, K, V, and A.
    """
    def __init__(self, in_dim, out_dim, num_types=3, num_heads=4, dropout=0.1, residual_alpha=0.5):
        super(HGTLayer, self).__init__()
        self.num_types = num_types
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.residual_alpha = residual_alpha
        
        self.head_dim = out_dim // num_heads
        
        # Type-specific projections
        self.k_lins = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])
        self.q_lins = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])
        self.v_lins = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])
        self.a_lins = nn.ModuleList([nn.Linear(out_dim, out_dim) for _ in range(num_types)])
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, node_types, adj_mask=None):
        # x: [B, N, F]
        # node_types: [N] (Int)
        
        B, N, _ = x.shape
        
        k = torch.zeros(B, N, self.out_dim, device=x.device)
        q = torch.zeros(B, N, self.out_dim, device=x.device)
        v = torch.zeros(B, N, self.out_dim, device=x.device)
        
        # Project based on types
        for i in range(self.num_types):
             mask = (node_types == i)
             if mask.sum() > 0:
                 # x[:, mask, :] -> Linear[i] -> k[:, mask, :]
                 # Since mask is [N], we use index select on dim 1
                 indices = torch.nonzero(mask).squeeze(1)
                 if indices.dim() == 0: indices = indices.unsqueeze(0)
                 
                 # sub_x: [B, num_nodes_of_type_i, F]
                 sub_x = x.index_select(1, indices) 
                 
                 k.index_copy_(1, indices, self.k_lins[i](sub_x))
                 q.index_copy_(1, indices, self.q_lins[i](sub_x))
                 v.index_copy_(1, indices, self.v_lins[i](sub_x))
        
        # Reshape for Heads [B, Heads, N, HeadDim]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention: Q * K.T
        # [B, Heads, N, HeadDim] @ [B, Heads, HeadDim, N] -> [B, Heads, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply Mask/Bias
        if adj_mask is not None:
             # Clamp to avoid NaN in log for negative correlations/events
             attn_bias = torch.log(torch.clamp(adj_mask, min=0) + 1e-9)
             if attn_bias.dim() == 2: attn_bias = attn_bias.unsqueeze(0)
             if attn_bias.dim() == 3: attn_bias = attn_bias.unsqueeze(1) # [B, 1, N, N]
             scores = scores + attn_bias
             
        attn = torch.softmax(scores, dim=-1)
        
        # Aggregation
        out = torch.matmul(attn, v) # [B, Heads, N, HeadDim]
        out = out.transpose(1, 2).reshape(B, N, self.out_dim)
        
        # Output Projection (Type Specific)
        res = torch.zeros_like(out)
        for i in range(self.num_types):
             mask = (node_types == i)
             if mask.sum() > 0:
                 indices = torch.nonzero(mask).squeeze(1)
                 if indices.dim() == 0: indices = indices.unsqueeze(0)
                 sub_out = out.index_select(1, indices)
                 res.index_copy_(1, indices, self.a_lins[i](sub_out))
        
        # Apply Residual Connection to prevent over-smoothing
        if x.shape == res.shape:
             res = apply_residual_connection(x, res, alpha=self.residual_alpha)
                 
        return F.elu(res)

class EventAwareGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, dropout=0.1, use_hgt=True, num_types=3, residual_alpha=0.5):
        super(EventAwareGAT, self).__init__()
        self.use_hgt = use_hgt
        
        if use_hgt:
             self.gat1 = HGTLayer(in_dim, hidden_dim, num_types=num_types, num_heads=heads, dropout=dropout, residual_alpha=residual_alpha)
             self.gat2 = HGTLayer(hidden_dim, out_dim, num_types=num_types, num_heads=1, dropout=dropout, residual_alpha=residual_alpha)
        else:
             self.gat1 = SimpleGATLayer(in_dim, hidden_dim, num_heads=heads, dropout=dropout, concat=True)
             self.gat2 = SimpleGATLayer(hidden_dim, out_dim, num_heads=1, dropout=dropout, concat=False)
        
    def forward(self, x, adj, node_types=None):
        if self.use_hgt and node_types is not None:
             x = self.gat1(x, node_types, adj_mask=adj)
             # x logic for layer 2? HGT usually keeps type logic.
             # Ideally layer 2 also uses types.
             x = self.gat2(x, node_types, adj_mask=adj)
        else:
             x = self.gat1(x, adj_mask=adj)
             x = self.gat2(x, adj_mask=adj)
        return x

class GNNSACAgent(MemoryAugmentedSACAgent):
    def __init__(self, *args, **kwargs):
        # Intercept 'adj_matrix' in kwargs
        self.adj_matrix = kwargs.get('adj_matrix')
        if self.adj_matrix is not None:
             self.adj_matrix = torch.tensor(self.adj_matrix, dtype=torch.float32)

        # Whether to use Next-Gen architecture components
        self.use_nextgen = kwargs.get('use_nextgen', False)
        self.use_nlp_alpha = kwargs.get('use_nlp_alpha', False)
        self.nlp_alpha_multiplier = kwargs.get('nlp_alpha_multiplier', 0.1)
        self.use_attention_gating = kwargs.get('use_attention_gating', False)
        self.attn_gate_threshold = kwargs.get('attn_gate_threshold', 2.0)
        self.use_momentum_masking = kwargs.get('use_momentum_masking', False)
        self.momentum_threshold = kwargs.get('momentum_threshold', -0.2)
        self.use_edge_pruning = kwargs.get('use_edge_pruning', False)
        self.use_action_masking = kwargs.get('use_action_masking', False)
        self._device = kwargs.get('device', 'cpu')

        # Filter hidden_dims (not accepted by parent)
        _skip = ['hidden_dims', 'adj_matrix', 'num_stock_nodes', 'num_sector_nodes',
                 'num_event_nodes', 'use_tgn', 'use_nextgen', 'sector_map', 'target_update_interval',
                 'use_attention_gating', 'attn_gate_threshold', 'use_momentum_masking', 'momentum_threshold',
                 'use_nlp_alpha', 'nlp_alpha_multiplier', 'use_cvar_loss', 'cvar_multiplier', 'cvar_tau',
                 'use_dual_head_iqn', 'use_edge_pruning', 'use_action_masking',
                 'equity_indices', 'safe_haven_indices', 'residual_alpha', 'use_feature_gating',
                 # logit-level action-mask kwargs are consumed by the policy
                 'sector_mask', 'stock_mask']
        super_kwargs = {k: v for k, v in kwargs.items() if k not in _skip}
        super_kwargs.setdefault('use_per', True) # NextGen defaults to PER

        super().__init__(*args, **super_kwargs)

        # Overwrite Policy
        hidden_dims = kwargs.get('hidden_dims', [256, 256])

        self.policy = GNNGaussianPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            node_features_dim=kwargs['node_features_dim'],
            num_nodes=kwargs['num_nodes'],
            adj_matrix=self.adj_matrix,
            encoder_hidden=64,
            lstm_hidden_size=kwargs.get('lstm_hidden_size', 64),
            num_lstm_layers=kwargs.get('num_lstm_layers', 1),
            hidden_dims=hidden_dims,
            use_tgn=kwargs.get('use_tgn', True),
            use_nextgen=self.use_nextgen,
            num_stock_nodes=kwargs.get('num_stock_nodes'),
            num_sector_nodes=kwargs.get('num_sector_nodes'),
            num_event_nodes=kwargs.get('num_event_nodes'),
            sector_map=kwargs.get('sector_map'),
            use_attention_gating=self.use_attention_gating,
            attn_gate_threshold=self.attn_gate_threshold,
            use_momentum_masking=self.use_momentum_masking,
            momentum_threshold=self.momentum_threshold,
            residual_alpha=kwargs.get('residual_alpha', 0.5),
            use_feature_gating=kwargs.get('use_feature_gating', False),
            # optional logit-level allocation-universe restriction
            sector_mask=kwargs.get('sector_mask'),
            stock_mask=kwargs.get('stock_mask'),
        ).to(self._device)

        # RC5: Dual LR — encoder at 1/10 base rate to prevent gradient explosion
        _lr = kwargs.get('learning_rate', 3e-4)
        encoder_params = list(self.policy.nextgen_encoder.parameters()) if self.use_nextgen and hasattr(self.policy, 'nextgen_encoder') else []
        encoder_ids = set(id(p) for p in encoder_params)
        other_params = [p for p in self.policy.parameters() if id(p) not in encoder_ids]
        if encoder_params:
            self.policy_optimizer = optim.Adam([
                {'params': other_params, 'lr': _lr},
                {'params': encoder_params, 'lr': _lr}, # Use full LR for encoder
            ])
        else:
            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=_lr)

        # Phase 5.1: Removed per-element gradient hooks — using clip_grad_norm_ instead
        # Phase 5.2: Target update interval
        self.target_update_interval = kwargs.get('target_update_interval', 2)

        # RC5: NextGen used to force auto_tune_alpha, now it respects user kwarg.
        if self.use_nextgen and not hasattr(self, 'target_entropy'):
            if self.auto_tune_alpha:
                self.target_entropy = -self.action_dim
                # If base class initialized it as fixed tensor, recreate as trainable
                val = self.log_alpha.item() if hasattr(self, 'log_alpha') else 0.0
                self.log_alpha = torch.tensor([val], requires_grad=True, device=self._device)
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=_lr)

        # ── Next-Gen additions ─────────────────────────────────────────
        if self.use_nextgen:
            self.use_dual_head_iqn = kwargs.get('use_dual_head_iqn', False)
            self.use_cvar_loss = kwargs.get('use_cvar_loss', False)
            self.cvar_tau = kwargs.get('cvar_tau', 0.1)
            self.cvar_multiplier = kwargs.get('cvar_multiplier', 5.0)
            if self.use_dual_head_iqn:
                self.iqn_critic1 = DualHeadIQNCritic(
                    state_dim=self.state_dim, action_dim=self.action_dim,
                    equity_indices=kwargs.get('equity_indices', list(range(28))),
                    safe_haven_indices=kwargs.get('safe_haven_indices', [28, 29]),
                    hidden_dims=hidden_dims, n_quantile_samples=32,
                ).to(self._device)
                self.iqn_critic2 = DualHeadIQNCritic(
                    state_dim=self.state_dim, action_dim=self.action_dim,
                    equity_indices=kwargs.get('equity_indices', list(range(28))),
                    safe_haven_indices=kwargs.get('safe_haven_indices', [28, 29]),
                    hidden_dims=hidden_dims, n_quantile_samples=32,
                ).to(self._device)
                self.iqn_critic1_target = DualHeadIQNCritic(
                    state_dim=self.state_dim, action_dim=self.action_dim,
                    equity_indices=kwargs.get('equity_indices', list(range(28))),
                    safe_haven_indices=kwargs.get('safe_haven_indices', [28, 29]),
                    hidden_dims=hidden_dims, n_quantile_samples=32,
                ).to(self._device)
                self.iqn_critic2_target = DualHeadIQNCritic(
                    state_dim=self.state_dim, action_dim=self.action_dim,
                    equity_indices=kwargs.get('equity_indices', list(range(28))),
                    safe_haven_indices=kwargs.get('safe_haven_indices', [28, 29]),
                    hidden_dims=hidden_dims, n_quantile_samples=32,
                ).to(self._device)
            else:
                self.iqn_critic1 = IQNCritic(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    hidden_dims=hidden_dims,
                    n_quantile_samples=32,
                ).to(self._device)
                self.iqn_critic2 = IQNCritic(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    hidden_dims=hidden_dims,
                    n_quantile_samples=32,
                ).to(self._device)
                self.iqn_critic1_target = IQNCritic(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    hidden_dims=hidden_dims,
                    n_quantile_samples=32,
                ).to(self._device)
                self.iqn_critic2_target = IQNCritic(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    hidden_dims=hidden_dims,
                    n_quantile_samples=32,
                ).to(self._device)
            
            self.iqn_critic1_target.load_state_dict(self.iqn_critic1.state_dict())
            self.iqn_critic2_target.load_state_dict(self.iqn_critic2.state_dict())
            
            self.iqn_optimizer = optim.Adam(
                list(self.iqn_critic1.parameters()) +
                list(self.iqn_critic2.parameters()),
                lr=kwargs.get('learning_rate', 3e-4),
            )
            # Regime-adaptive entropy temperature
            self.entropy_scheduler = RegimeEntropyScheduler(
                alpha_base=kwargs.get('alpha', 0.2),
            ).to(self._device)

        # ── Setup Loss Calculator Adapter ──────────────────────────────
        if self.use_nextgen:
            self.loss_calculator = NextGenCVaRLossCalculator(
                cvar_tau=self.cvar_tau,
                cvar_multiplier=self.cvar_multiplier,
                use_nlp_alpha=self.use_nlp_alpha,
                nlp_alpha_multiplier=self.nlp_alpha_multiplier,
                device=self._device
            )
        else:
            self.loss_calculator = StandardSACLossCalculator()

    def select_action(
        self, state, hidden=None, deterministic=False, adj=None,
        nextgen_ctx: Optional[Dict[str, Any]] = None,
    ):
        """Select action. nextgen_ctx carries optional vol_vec, regime_feats,
        event_emb, dd_context, delta_t for the NextGen encoder."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, new_hidden = self.policy.get_action(
            state, hidden, deterministic=deterministic, adj=adj, nextgen_ctx=nextgen_ctx
        )
        return action.cpu().detach().numpy()[0], new_hidden

    def update(self, nextgen_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        NextGen SAC Update loop with IQN Distributional critics.
        If use_nextgen is False, falls back to standard MemoryAugmentedSACAgent update.
        """
        # Phase 2 Fix: Allow explicit fallback to parent update for temporal stability
        if not self.use_nextgen or (nextgen_ctx is not None and nextgen_ctx.get('use_legacy_update', False)):
            return super().update()
            
        if len(self.replay_buffer) < self.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(batch['states']).to(self._device)
        actions = torch.FloatTensor(batch['actions']).to(self._device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self._device)
        next_states = torch.FloatTensor(batch['next_states']).to(self._device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self._device)
        
        # PER weights
        weights = batch.get('weights', None)
        if weights is not None:
            weights = torch.FloatTensor(weights).to(self._device)

        with torch.no_grad():
            next_states_seq = next_states.unsqueeze(1)
            next_actions, next_log_probs, _ = self.policy.sample(next_states_seq, hidden=None)
            
            if getattr(self, 'use_dual_head_iqn', False):
                eq1, sh1, _ = self.iqn_critic1_target(next_states, next_actions)
                eq2, sh2, _ = self.iqn_critic2_target(next_states, next_actions)
                z1_next = eq1 + sh1
                z2_next = eq2 + sh2
            else:
                z1_next, _ = self.iqn_critic1_target(next_states, next_actions)
                z2_next, _ = self.iqn_critic2_target(next_states, next_actions)
            z_next = torch.min(z1_next, z2_next)
            
            # Entropy processing (with fallback if no ctx)
            if nextgen_ctx and 'vix' in nextgen_ctx and 'vix_mean' in nextgen_ctx:
                vix = nextgen_ctx['vix']
                vix_mean = nextgen_ctx['vix_mean']
                alpha = self.entropy_scheduler(vix, vix_mean).unsqueeze(1)
            else:
                alpha = self.log_alpha.exp()
            
            # --- Z-Target Computation via Adapter ---
            z_target, alpha = self.loss_calculator.compute_z_target(
                rewards, dones, z_next, next_log_probs, alpha, self.gamma, nextgen_ctx
            )

        if getattr(self, 'use_dual_head_iqn', False):
            eq1, sh1, tau1 = self.iqn_critic1(states, actions)
            eq2, sh2, tau2 = self.iqn_critic2(states, actions)
            z1_pred = eq1 + sh1
            z2_pred = eq2 + sh2
        else:
            z1_pred, tau1 = self.iqn_critic1(states, actions)
            z2_pred, tau2 = self.iqn_critic2(states, actions)

        # --- Critic Loss Computation via Adapter ---
        iqn_loss, td_err1, td_err2 = self.loss_calculator.compute_critic_loss(
            z1_pred, z2_pred, z_target, tau1, tau2, weights, IQNCritic.quantile_huber_loss
        )
        
        self.iqn_optimizer.zero_grad()
        iqn_loss.backward()
        # Two-level gradient defense: value clip → norm clip
        torch.nn.utils.clip_grad_value_(self.iqn_critic1.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(self.iqn_critic2.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.iqn_critic1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.iqn_critic2.parameters(), 1.0)
        self.iqn_optimizer.step()
        
        # Update PER priorities
        if 'tree_indices' in batch:
            td_errors = (td_err1 + td_err2) / 2.0
            self.replay_buffer.update_priorities(
                batch['tree_indices'],
                td_errors.detach().cpu().numpy()
            )

        states_seq = states.unsqueeze(1)
        new_actions, log_probs, _ = self.policy.sample(states_seq, hidden=None)
        
        if getattr(self, 'use_dual_head_iqn', False):
            eq1, sh1, _ = self.iqn_critic1(states, new_actions)
            eq2, sh2, _ = self.iqn_critic2(states, new_actions)
            z1_new = eq1 + sh1
            z2_new = eq2 + sh2
        else:
            z1_new, _ = self.iqn_critic1(states, new_actions)
            z2_new, _ = self.iqn_critic2(states, new_actions)
        
        q1_new = z1_new.mean(dim=1, keepdim=True)
        q2_new = z2_new.mean(dim=1, keepdim=True)
        min_q_new = torch.min(q1_new, q2_new)
        
        # Get stock_log_std for volatility penalty
        stock_log_std = None
        if self.use_nextgen and nextgen_ctx and 'vix' in nextgen_ctx and 'vix_mean' in nextgen_ctx:
            _, _, _, stock_log_std, _ = self.policy.forward(states_seq, nextgen_ctx=nextgen_ctx)

        # --- Policy Loss Computation via Adapter ---
        policy_loss = self.loss_calculator.compute_policy_loss(
            alpha, log_probs, min_q_new, weights, nextgen_ctx, stock_log_std
        )

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # Phase 5.1: Two-level gradient defense (value clip → norm clip)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()

        alpha_loss_val = 0.0
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = alpha_loss.item()
            
        # Phase 5.2: Only update targets every N steps
        if self.update_count % self.target_update_interval == 0:
            self._soft_update_tensor(self.iqn_critic1, self.iqn_critic1_target)
            self._soft_update_tensor(self.iqn_critic2, self.iqn_critic2_target)
        
        self.update_count += 1
        
        return {
            'iqn_loss': iqn_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss_val,
            'alpha': self.log_alpha.exp().item()
        }

    def _soft_update_tensor(self, source: torch.nn.Module, target: torch.nn.Module):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

class GNNGaussianPolicy(LSTMGaussianPolicy):
    def __init__(self, state_dim, action_dim,
                 node_features_dim, num_nodes,
                 adj_matrix,
                 encoder_hidden=64,
                 use_tgn=True,
                 use_nextgen=False,
                 num_stock_nodes=None,
                 num_sector_nodes=None,
                 num_event_nodes=None,
                 sector_map=None,
                 use_action_masking=False,
                 equity_indices=None,
                 safe_haven_indices=None,
                 use_attention_gating=False,
                 attn_gate_threshold=2.0,
                 use_momentum_masking=False,
                 momentum_threshold=-0.2,
                 residual_alpha=0.5,
                 use_feature_gating=False,
                 **kwargs):
                  
        # Filter kwargs for parent (mask kwargs are consumed locally)
        parent_kwargs = {k: v for k, v in kwargs.items()
                         if k not in ['hidden_dims', 'sector_mask', 'stock_mask']}

        super(GNNGaussianPolicy, self).__init__(
            state_dim=state_dim, action_dim=action_dim, use_graph=False, **parent_kwargs
        )
        
        self.use_action_masking = use_action_masking
        self.equity_indices = equity_indices if equity_indices is not None else list(range(action_dim))
        self.safe_haven_indices = safe_haven_indices if safe_haven_indices is not None else []
        self.use_nextgen = use_nextgen
        self.use_attention_gating = use_attention_gating
        self.attn_gate_threshold = attn_gate_threshold
        self.use_momentum_masking = use_momentum_masking
        self.momentum_threshold = momentum_threshold
        self.residual_alpha = residual_alpha

        self.register_buffer('sector_map', sector_map if sector_map is not None else torch.zeros(action_dim, dtype=torch.long))
        self.num_sectors = len(torch.unique(self.sector_map)) if sector_map is not None else 1

        # optional allocation-universe restriction. Set via
        # ``set_action_mask`` (or constructor kwargs ``sector_mask`` / ``stock_mask``).
        # Defaults of ``None`` preserve the unmasked baseline behavior.
        _smask = kwargs.get('sector_mask', None)
        _kmask = kwargs.get('stock_mask', None)
        if _smask is None:
            _smask = torch.ones(self.num_sectors, dtype=torch.bool)
        if _kmask is None:
            _kmask = torch.ones(action_dim, dtype=torch.bool)
        self.register_buffer('sector_mask', _smask.bool())
        self.register_buffer('stock_mask', _kmask.bool())
        self._action_mask_active = bool(
            (~self.sector_mask).any().item() or (~self.stock_mask).any().item()
        )

        self.node_features_dim = node_features_dim
        self.num_nodes = num_nodes
        self.encoder_hidden = encoder_hidden
        self.use_tgn = use_tgn
        self.use_nextgen = use_nextgen

        # Build Node Types for HGT
        self.num_types = 3 # Default
        if num_stock_nodes is not None and num_sector_nodes is not None:
             self.use_hgt = True
             types_stock = torch.zeros(num_stock_nodes, dtype=torch.long)
             types_sector = torch.ones(num_sector_nodes, dtype=torch.long)
             types_macro = torch.full((1,), 2, dtype=torch.long)
             
             parts = [types_stock, types_sector, types_macro]
             
             if num_event_nodes is not None:
                 types_event = torch.full((num_event_nodes,), 3, dtype=torch.long)
                 parts.append(types_event)
                 self.num_types = 4
             
             _nt = torch.cat(parts)
             
             # Safety Check / Pad
             if len(_nt) < num_nodes:
                  pad = torch.full((num_nodes - len(_nt),), 1, dtype=torch.long)
                  _nt = torch.cat([_nt, pad])
             elif len(_nt) > num_nodes:
                  _nt = _nt[:num_nodes]
                  
             self.register_buffer('node_types', _nt)
        else:
             self.use_hgt = False
             self.register_buffer('node_types', None)
        
        if hasattr(self, 'adj_matrix'):
            del self.adj_matrix
        self.register_buffer('adj_matrix', adj_matrix)

        # ── Next-Gen Encoder (C1-C7) ────────────────────────────────────
        if self.use_nextgen:
            self.nextgen_encoder = NextGenEventGraphEncoder(
                in_dim=node_features_dim,
                memory_dim=encoder_hidden,
                num_nodes=num_nodes,
                num_heads=4,
                num_node_types=self.num_types,
            )
            self.lstm_input_dim = num_nodes * encoder_hidden
        elif self.use_tgn:
            # TGN Layer (Recurrent GAT w/ Memory) — legacy path
            self.tgn = TGNRecurrentLayer(
                in_dim=node_features_dim,
                memory_dim=encoder_hidden,
                num_heads=4
            )
            self.lstm_input_dim = num_nodes * encoder_hidden
        else:
            # Snapshot E-GAT / HGT — legacy path
            self.egat = EventAwareGAT(
                in_dim=node_features_dim,
                hidden_dim=encoder_hidden,
                out_dim=encoder_hidden,
                heads=4,
                use_hgt=self.use_hgt,
                num_types=self.num_types,
                residual_alpha=self.residual_alpha
            )
            self.lstm_input_dim = num_nodes * encoder_hidden
        
        # Phase 4: Feature Gating Bottleneck
        self.use_feature_gating = use_feature_gating
        if self.use_feature_gating:
            self.feature_gate = FeatureGatingModule(node_features_dim)

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True
        )
        _hd = kwargs.get('hidden_dims', [256, 256])
        self.ln = nn.LayerNorm(_hd[0])
        self.fc1 = nn.Linear(self.lstm_hidden_size, _hd[0])
        # Override parent's fc2 so it matches user-supplied hidden_dims
        # (parent LSTMGaussianPolicy builds fc2 with default [256,256])
        self.fc2 = nn.Linear(_hd[0], _hd[1])

        hidden_dim_2 = _hd[1]
        self.sector_mean = nn.Linear(hidden_dim_2, self.num_sectors)
        self.sector_log_std = nn.Linear(hidden_dim_2, self.num_sectors)
        
        self.stock_mean = nn.Linear(hidden_dim_2, action_dim)
        self.stock_log_std = nn.Linear(hidden_dim_2, action_dim)

    def forward(self, state, hidden=None, adj=None,
                nextgen_ctx: Optional[Dict[str, Any]] = None):
        """
        hidden: Tuple of (LSTM_State, TGN_Memory)
            LSTM_State: (h, c)
            TGN_Memory: [batch_size, num_nodes, memory_dim]
        nextgen_ctx: optional dict with keys:
            vol_vec, regime_feats, event_emb, dd_context, delta_t, adj_event
        """
        # --- Robust Sanitization ---
        if torch.isnan(state).any() or torch.isinf(state).any():
            state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        if adj is not None:
            if torch.isnan(adj).any() or torch.isinf(adj).any():
                adj = torch.nan_to_num(adj, nan=0.0, posinf=1.0, neginf=-1.0)

        if state.dim() == 2:
            state = state.unsqueeze(1)

        batch_size, seq_len = state.shape[:2]

        # Unpack Hidden
        lstm_hidden = None
        tgn_memory  = None

        if hidden is not None:
            lstm_hidden = hidden[0]
            if len(hidden) > 1:
                tgn_memory = hidden[1]

        # Initialise TGN memory if not provided
        if tgn_memory is None:
            tgn_memory = torch.zeros(
                batch_size, self.num_nodes, self.encoder_hidden,
                device=state.device
            )

        # 1. Reshape [B, S, N*F] -> [B, S, N, F]
        flat_state_seq = state.reshape(
            batch_size, seq_len, self.num_nodes, self.node_features_dim
        )

        outputs = []
        current_adj = adj if adj is not None else self.adj_matrix

        # Extract nextgen_ctx tensors
        ctx = nextgen_ctx or {}

        for t in range(seq_len):
            xt = flat_state_seq[:, t, :, :]  # [B, N, F]

            # Phase 4: Feature Gating
            if self.use_feature_gating:
                xt = self.feature_gate(xt)

            if self.use_nextgen:
                # --- Next-Gen path (C1-C7) ---
                tgn_memory = self.nextgen_encoder(
                    node_feats=xt,
                    memory=tgn_memory,
                    adj_base=current_adj,
                    adj_event=ctx.get('adj_event'),
                    node_types=self.node_types,
                    vol_vec=ctx.get('vol_vec'),
                    regime_feats=ctx.get('regime_feats'),
                    event_emb=ctx.get('event_emb'),
                    dd_context=ctx.get('dd_context'),
                    delta_t=ctx.get('delta_t'),
                )
                outputs.append(tgn_memory)

            elif self.use_tgn:
                # --- Legacy TGN path ---
                tgn_memory = self.tgn(xt, tgn_memory, adj_mask=current_adj)
                outputs.append(tgn_memory)

            else:
                # --- Legacy EGAT/HGT snapshot path ---
                gat_out = self.egat(xt, current_adj, node_types=self.node_types)
                outputs.append(gat_out)

        # Cache last node features for D3: Local Momentum Masking
        # xt is [B, N, F] from the last iteration of the loop
        if ctx is not None:
             ctx['node_feats'] = xt

        # Stack: [B, S, N, M]
        enc_out = torch.stack(outputs, dim=1)
        # Gradient-safe clamp (NOT nan_to_num which kills gradients)
        enc_out = enc_out.clamp(-10.0, 10.0)

        # 2. Flatten for LSTM: [B, S, N*M]
        lstm_input = enc_out.reshape(batch_size, seq_len, -1)

        # 3. LSTM
        lstm_out, lstm_hidden = self.lstm(lstm_input, lstm_hidden)
        x = lstm_out[:, -1, :]

        # 4. MLP head with LayerNorm
        x = self.fc1(x)
        x = self.ln(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))

        sector_mean = self.sector_mean(x)
        sector_log_std = self.sector_log_std(x)
        stock_mean = self.stock_mean(x)
        stock_log_std = self.stock_log_std(x)

        # --- B2: Macro-Conditioned Action Masking ---
        if self.use_action_masking and ctx and ctx.get('macro_panic_signal') is not None:
            stock_mean = apply_crisis_mask(
                stock_mean, ctx['macro_panic_signal'],
                self.equity_indices, self.safe_haven_indices
            )
            
        # --- D3: Local Momentum Action Masking ---
        if self.use_momentum_masking:
            # node_feats are in enc_out or ctx? 
            # In GNNGaussianPolicy, forward gets state, hidden, ctx.
            # ctx['node_feats'] should contain the last step's features
            if ctx and ctx.get('node_feats') is not None:
                stock_mean = apply_momentum_mask(
                    stock_mean, ctx['node_feats'], 
                    momentum_index=0, # Assumption
                    threshold=self.momentum_threshold
                )

        # Safety clamp on outputs (gradient-safe)
        sector_mean = sector_mean.clamp(-5.0, 5.0)
        stock_mean = stock_mean.clamp(-5.0, 5.0)
        sector_log_std = sector_log_std.clamp(-5.0, 2.0)
        stock_log_std = stock_log_std.clamp(-5.0, 2.0)

        new_hidden = (lstm_hidden, tgn_memory)
        return sector_mean, sector_log_std, stock_mean, stock_log_std, new_hidden

    def set_action_mask(self, sector_mask=None, stock_mask=None):
        """install or update the allocation-universe restriction.

        Pass ``None`` to either argument to leave that level unchanged. Pass
        all-True masks to disable masking entirely.
        """
        if sector_mask is not None:
            sm = torch.as_tensor(sector_mask, dtype=torch.bool, device=self.sector_mask.device)
            assert sm.shape == self.sector_mask.shape, "sector_mask shape mismatch"
            self.sector_mask.copy_(sm)
        if stock_mask is not None:
            km = torch.as_tensor(stock_mask, dtype=torch.bool, device=self.stock_mask.device)
            assert km.shape == self.stock_mask.shape, "stock_mask shape mismatch"
            self.stock_mask.copy_(km)
        self._action_mask_active = bool(
            (~self.sector_mask).any().item() or (~self.stock_mask).any().item()
        )

    def sample(self, state, hidden=None, adj=None, nextgen_ctx=None):
        sector_mean, sector_log_std, stock_mean, stock_log_std, hidden = self.forward(
            state, hidden, adj, nextgen_ctx
        )
        
        sector_std = sector_log_std.exp()
        stock_std = stock_log_std.exp()
        
        sector_normal = Normal(sector_mean, sector_std)
        stock_normal = Normal(stock_mean, stock_std)
        
        sector_z = sector_normal.rsample()
        stock_z = stock_normal.rsample()
        
        action = hierarchical_softmax(
            sector_z, stock_z, self.sector_map,
            sector_mask=self.sector_mask if self._action_mask_active else None,
            stock_mask=self.stock_mask if self._action_mask_active else None,
        )
        
        log_prob = sector_normal.log_prob(sector_z).sum(dim=-1, keepdim=True) + \
                   stock_normal.log_prob(stock_z).sum(dim=-1, keepdim=True)
                   
        return action, log_prob, hidden
        
    def get_action(self, state, hidden=None, deterministic=False, adj=None, nextgen_ctx=None):
        sector_mean, sector_log_std, stock_mean, stock_log_std, hidden = self.forward(
            state, hidden, adj, nextgen_ctx
        )
        
        if deterministic:
            sector_z = sector_mean
            stock_z = stock_mean
        else:
            sector_z = Normal(sector_mean, sector_log_std.exp()).rsample()
            stock_z = Normal(stock_mean, stock_log_std.exp()).rsample()
            
        action = hierarchical_softmax(
            sector_z, stock_z, self.sector_map,
            sector_mask=self.sector_mask if self._action_mask_active else None,
            stock_mask=self.stock_mask if self._action_mask_active else None,
        )
        return action, hidden

    def save(self, filepath: str):
        """Save NextGen GNN-SAC model."""
        state = {
            'policy': self.policy.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'update_count': self.update_count,
            'use_nextgen': getattr(self, 'use_nextgen', False),
        }
        if getattr(self, 'use_nextgen', False):
            state.update({
                'iqn_critic1': self.iqn_critic1.state_dict(),
                'iqn_critic2': self.iqn_critic2.state_dict(),
                'iqn_critic1_target': self.iqn_critic1_target.state_dict(),
                'iqn_critic2_target': self.iqn_critic2_target.state_dict(),
                'iqn_optimizer': self.iqn_optimizer.state_dict(),
            })
        else:
            state.update({
                'q1': self.q1.state_dict(),
                'q2': self.q2.state_dict(),
                'q1_target': self.q1_target.state_dict(),
                'q2_target': self.q2_target.state_dict(),
                'q1_optimizer': self.q1_optimizer.state_dict(),
                'q2_optimizer': self.q2_optimizer.state_dict(),
            })
        torch.save(state, filepath)

    def load(self, filepath: str):
        """Load NextGen GNN-SAC model."""
        checkpoint = torch.load(filepath, map_location=self._device)
        
        # Policy is universal; strict=False tolerates buffers
        # (stock_mask, sector_mask) absent from pre-checkpoints.
        if 'policy' in checkpoint:
            self.policy.load_state_dict(checkpoint['policy'], strict=False)
        elif 'policy_state_dict' in checkpoint:
            self.policy.load_state_dict(checkpoint['policy_state_dict'], strict=False)
        
        # Critics depend on mode
        if getattr(self, 'use_nextgen', False) and 'iqn_critic1' in checkpoint:
            self.iqn_critic1.load_state_dict(checkpoint['iqn_critic1'])
            self.iqn_critic2.load_state_dict(checkpoint['iqn_critic2'])
            self.iqn_critic1_target.load_state_dict(checkpoint['iqn_critic1_target'])
            self.iqn_critic2_target.load_state_dict(checkpoint['iqn_critic2_target'])
            if 'iqn_optimizer' in checkpoint and hasattr(self, 'iqn_optimizer'):
                self.iqn_optimizer.load_state_dict(checkpoint['iqn_optimizer'])
        elif 'q1' in checkpoint:
            self.q1.load_state_dict(checkpoint['q1'])
            self.q2.load_state_dict(checkpoint['q2'])
            self.q1_target.load_state_dict(checkpoint['q1_target'])
            self.q2_target.load_state_dict(checkpoint['q2_target'])
            
        self.log_alpha = checkpoint.get('log_alpha', self.log_alpha)
        self.update_count = checkpoint.get('update_count', self.update_count)


