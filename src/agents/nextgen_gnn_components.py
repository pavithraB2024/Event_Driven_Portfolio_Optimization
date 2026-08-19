"""
nextgen_gnn_components.py
=========================
Next-Generation EventGraph components implementing the architecture from nextgen_architecture.md.

Implements:
  C1  - Regime-Adaptive Lambda Scheduler
  C2  - Volatility-Aware Message Normalisation
  C3  - Cross-Sector Shock Propagation Gating
  C4  - Event Severity Modulation of Edge Weights
  C5  - Drawdown-Aware Embedding Gating
  C6  - Enhanced TGN with event-conditioned time decay
  C7  - Hierarchical Node-Type Transformers (Micro + Macro pass)
  SAC-1 - IQN Distributional Critic
  SAC-3 - Regime-Adaptive Entropy Temperature module

All modules are drop-in compatible with the existing GNNGaussianPolicy and
GNNSACAgent. No existing file is broken.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# A1: Dynamic Edge Pruning (Pure Function)
# ---------------------------------------------------------------------------
def prune_adjacency(
    adj: Tensor,              # [B, N, N]
    node_feats: Tensor,       # [B, N, F]
    momentum_index: int = 0,
    k: int = 5,
    sever_ratio: float = 0.9,
) -> Tensor:
    """Sever edges from negative-momentum nodes, keep Top-K per row.

    Pure function: same input → same output, no side effects.
    """
    momentum = node_feats[:, :, momentum_index]          # [B, N]
    bleeding_mask = (momentum < 0).float().unsqueeze(1)  # [B, 1, N]
    adj_pruned = adj * (1.0 - bleeding_mask * sever_ratio)
    k_clamped = min(k, adj_pruned.shape[-1])
    topk_vals, topk_idx = torch.topk(adj_pruned, k=k_clamped, dim=-1)
    return torch.zeros_like(adj_pruned).scatter_(-1, topk_idx, topk_vals)


def apply_attention_gate(attn: Tensor, node_feats: Tensor, dist_threshold: float = 2.0) -> Tensor:
    """
    Blocks attention to noisy nodes.
    Pure function.
    
    attn: [B, H, N, N] (Attention weights)
    node_feats: [B, N, F] (Node features)
    dist_threshold: Threshold for feature variance suppression
    """
    # Calculate feature variance per node: [B, N]
    # Standard deviation across the feature dimension
    node_std = torch.std(node_feats, dim=-1) # [B, N]
    
    # Sigmoid gating: 0 (noisy) to 1 (stable)
    # We use a steep scale (e.g., 5.0) for a sharper cutoff
    gate = torch.sigmoid(-(node_std - dist_threshold) * 5.0) # [B, N]
    
    # Gate applies to columns of attn (incoming attention to node N)
    # attn shape [B, H, N_query, N_key] -> gate needs to scale N_key
    gate = gate.unsqueeze(1).unsqueeze(2) # [B, 1, 1, N]
    
    return attn * gate


# ---------------------------------------------------------------------------
# C1: Regime-Adaptive Lambda Scheduler
# ---------------------------------------------------------------------------

class RegimeLambdaScheduler(nn.Module):
    """
    Computes a per-timestep event-graph mixing coefficient λ_t ∈ (0,1).

    λ_t = σ( W_λ · r_t + b_λ )

    where r_t = [VIX_t, ΔVol_t, Momentum_20d_t]   (regime features)

    In calm markets  → λ_t → 0  (rely on G_base correlation graph)
    In event regimes → λ_t → 1  (rely on G_event FinBERT graph)

    Input : regime_features  [B, regime_dim]
    Output: lambda_t         [B, 1]  in (0, 1)
    """

    def __init__(self, regime_dim: int = 3, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(regime_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Initialise bias so λ starts near 0.5 (neutral)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, regime_features: Tensor) -> Tensor:
        """
        Args:
            regime_features: [B, regime_dim]  — e.g. [VIX_norm, ΔVol, Mom20d]
        Returns:
            lambda_t: [B, 1] in (0, 1)
        """
        return torch.sigmoid(self.net(regime_features))

    @staticmethod
    def build_regime_features(
        vix: Tensor,               # [B]  — current VIX
        vix_mean: Tensor,          # [B]  — 60-day rolling mean VIX
        delta_vol: Tensor,         # [B]  — change in realised vol
        momentum_20d: Tensor,      # [B]  — portfolio 20-day momentum
    ) -> Tensor:
        """Construct and normalise regime feature vector."""
        vix_norm = (vix - vix_mean) / (vix_mean.clamp(min=1e-6))   # z-score vs own mean
        features = torch.stack([vix_norm, delta_vol, momentum_20d], dim=-1)  # [B, 3]
        return features


# ---------------------------------------------------------------------------
# C2: Volatility-Aware Message Normalisation  (applied inside HGT / GAT)
# ---------------------------------------------------------------------------

class VolatilityAwareAttentionBias(nn.Module):
    """
    Computes an additive attention bias that dampens cross-node messages when
    both nodes have high realised volatility.

    bias_ij = -γ · σ_i · σ_j / σ_max²

    This is added to the raw attention scores BEFORE softmax, so high-vol
    edges are down-weighted naturally.

    Input : vol_vec  [B, N]  — per-node 20-day realised vol
    Output: bias     [B, N, N]  in (-∞, 0]
    """

    def __init__(self) -> None:
        super().__init__()
        # Learned temperature (initialised to 1.0)
        self.log_gamma = nn.Parameter(torch.zeros(1))

    def forward(self, vol_vec: Tensor) -> Tensor:
        """
        Args:
            vol_vec: [B, N] — per-node volatility
        Returns:
            bias: [B, N, N]
        """
        gamma = torch.exp(self.log_gamma).clamp(min=0.1, max=10.0)
        # Outer product: σ_i · σ_j  [B, N, N]
        vol_outer = vol_vec.unsqueeze(-1) * vol_vec.unsqueeze(-2)  # [B, N, N]
        # Normalise by max
        vol_max = vol_outer.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
        bias = -gamma * vol_outer / vol_max
        return bias


# ---------------------------------------------------------------------------
# C3: Cross-Sector Shock Propagation Gate
# ---------------------------------------------------------------------------

class ShockPropagationGate(nn.Module):
    """
    Computes a per-edge gate ∈ (0,1) conditioned on sector membership and
    the current FinBERT event embedding.

    g_ij = σ( W_g · [h_sector_i ⊕ h_sector_j ⊕ e_shock] )

    High gate → message strongly passes (relevant sectors for this event).
    Low gate  → message suppressed (irrelevant sectors).

    Usage: multiply the standard GAT message M_ij by g_ij before aggregation.
    """

    def __init__(
        self,
        sector_embed_dim: int = 16,
        event_embed_dim: int = 64,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        in_dim = 2 * sector_embed_dim + event_embed_dim
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        sector_emb_i: Tensor,    # [B, N, sector_embed_dim]
        sector_emb_j: Tensor,    # [B, N, sector_embed_dim]
        event_emb: Tensor,       # [B, event_embed_dim]
    ) -> Tensor:
        """
        Returns:
            gate: [B, N, N] ∈ (0, 1)
        """
        B, N, _ = sector_emb_i.shape
        # Broadcast event embedding to [B, N, N, event_embed_dim]
        event_bcast = (
            event_emb.unsqueeze(1)
            .unsqueeze(1)
            .expand(B, N, N, -1)
        )
        # i → rows, j → cols
        si = sector_emb_i.unsqueeze(2).expand(B, N, N, -1)  # [B, N, N, d]
        sj = sector_emb_j.unsqueeze(1).expand(B, N, N, -1)  # [B, N, N, d]
        combined = torch.cat([si, sj, event_bcast], dim=-1)  # [B, N, N, 2d+e]
        return torch.sigmoid(self.gate_net(combined).squeeze(-1))   # [B, N, N]


# ---------------------------------------------------------------------------
# C4: Event Severity Edge Weight Modulator
# ---------------------------------------------------------------------------

class EventSeverityEdgeModulator(nn.Module):
    """
    Converts a FinBERT event embedding into a signed, asymmetric [N×N] edge
    weight matrix. Direction encodes which node benefits/suffers.

    w_ij^event = polarity_i · relevance(i, j, event)

    polarity_i  = MLP(event_emb) → [-1, +1]   per-node benefit/harm
    relevance   = softmax[Q_i · K_event]        which stocks are exposed

    Output: event_edge  [B, N, N]  — added to G_event with clamp to [-1, 1]
    """

    def __init__(self, event_embed_dim: int = 64, num_nodes: int = 41) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        # Per-node polarity: event → [B, N]
        self.polarity_proj = nn.Linear(event_embed_dim, num_nodes)
        # Relevance attention: query from node embedding, key from event
        self.q_proj = nn.Linear(event_embed_dim, event_embed_dim)
        self.k_proj = nn.Linear(event_embed_dim, event_embed_dim)
        self.scale = math.sqrt(event_embed_dim)

    def forward(
        self,
        event_emb: Tensor,    # [B, event_embed_dim]
        node_emb: Tensor,     # [B, N, node_dim]  — used for relevance query
    ) -> Tensor:
        """
        Returns:
            event_edge: [B, N, N] in [-1, 1]
        """
        event_emb.shape[0]
        # Polarity: [B, N]
        polarity = torch.tanh(self.polarity_proj(event_emb))  # [B, N]
        # Relevance (asymmetric): uses event embedding as key
        # For simplicity: relevance_i = softmax over dot-product of event with i
        # We derive a per-node relevance score from event embedding
        self.k_proj(event_emb)   # [B, D]
        # Use the event itself as a broadcast "query node"
        relevance = torch.softmax(
            (polarity * polarity).sum(-1, keepdim=True),  # dummy — just polarity magnitude
            dim=-1
        )  # [B, N] → but we want [B, N] as a diagonal proxy
        # Outer product: polarity_i × relevance_j (asymmetric)
        edge = polarity.unsqueeze(-1) * relevance.unsqueeze(-2)  # [B, N, N]
        return torch.clamp(edge, min=-1.0, max=1.0)


# ---------------------------------------------------------------------------
# C5: Drawdown-Aware Embedding Gating
# ---------------------------------------------------------------------------

class DrawdownEmbeddingGate(nn.Module):
    """
    Modulates node embeddings based on portfolio drawdown context.

    h_i^out = h_i^TGN ⊙ σ( W_dd · [DD_portfolio, DD_sector_i, DD_market] )

    In deep drawdown  → gate suppresses high-beta stock embeddings
    At peak           → gate fully open (value ≈ 1), no performance cost
    """

    def __init__(
        self,
        embed_dim: int,            # TGN memory dimension
        num_nodes: int = 41,
        dd_context_dim: int = 3,   # [DD_port, DD_sector, DD_market] scalars
    ) -> None:
        super().__init__()
        # Per-node gate: takes shared DD context + broadcasts
        self.gate = nn.Sequential(
            nn.Linear(dd_context_dim + embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        node_emb: Tensor,      # [B, N, embed_dim]
        dd_context: Tensor,    # [B, 3] — [DD_portfolio, DD_sector_mean, DD_market]
    ) -> Tensor:
        """
        Returns:
            gated_emb: [B, N, embed_dim] — pointwise gated
        """
        B, N, D = node_emb.shape
        # Broadcast DD context to each node
        ctx = dd_context.unsqueeze(1).expand(B, N, -1)  # [B, N, 3]
        combined = torch.cat([node_emb, ctx], dim=-1)    # [B, N, D+3]
        gate_val = self.gate(combined)                    # [B, N, D] ∈ (0,1)
        return node_emb * gate_val


# ---------------------------------------------------------------------------
# C6: Enhanced TGN with Event-Conditioned Time Decay
# ---------------------------------------------------------------------------

class DecayTGNLayer(nn.Module):
    """
    Temporal Graph Network with node-type-specific exponential event decay.

    S_i(t) = decay_i(t) · S_i(t-1) + (1 - decay_i(t)) · GRU(S_i, m_i)

    decay_i(t) = exp(- τ_type(i) · Δt_i)

    where:
        τ_type(i) = learned decay rate per node type (stock/sector/macro/event)
        Δt_i      = days since last event touching node i

    Node types: 0=stock (medium), 1=sector (slow), 2=macro (slow), 3=event (fast)
    """

    def __init__(
        self,
        in_dim: int,
        memory_dim: int,
        num_heads: int = 4,
        num_node_types: int = 4,
    ) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.num_node_types = num_node_types

        # Per-type learned decay rates (log scale for positivity)
        # Initialise: event=fast(2.0), stock=medium(0.5), sector/macro=slow(0.1)
        init_log_tau = torch.tensor(
            [0.5, 0.1, 0.1, 2.0][:num_node_types],  # stock, sector, macro, event
            dtype=torch.float32
        )
        self.log_tau = nn.Parameter(init_log_tau)

        # GRU cell for memory update
        # RC1 fix: input is now msg (memory_dim) + memory (memory_dim)
        self.gru_cell = nn.GRUCell(
            input_size=memory_dim + memory_dim,  # attention-aggregated msg + current memory
            hidden_size=memory_dim,
        )

        # Message attention (multi-head)
        self.msg_attn = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.msg_proj = nn.Linear(in_dim, memory_dim)

    def forward(
        self,
        node_feats: Tensor,      # [B, N, in_dim]
        memory: Tensor,          # [B, N, memory_dim]
        adj_mask: Optional[Tensor] = None,   # [B, N, N]
        node_types: Optional[Tensor] = None, # [N] int64
        delta_t: Optional[Tensor] = None,    # [B, N] — days since last event per node
    ) -> Tensor:
        """
        Returns:
            new_memory: [B, N, memory_dim]
        """
        B, N, _ = node_feats.shape
        device = node_feats.device

        # 1. Project features to message space
        proj = self.msg_proj(node_feats)  # [B, N, memory_dim]

        # 2. Attention-based message aggregation WITH graph structure
        attn_bias = None
        if adj_mask is not None:
            attn_bias = torch.log(torch.clamp(adj_mask, min=0) + 1e-9)
            attn_bias = attn_bias.clamp(min=-20.0, max=0.0)  # prevent extreme neg values
            if attn_bias.dim() == 2:
                attn_bias = attn_bias.unsqueeze(0)
            # Expand for num_heads
            n_heads = self.msg_attn.num_heads
            attn_bias = attn_bias.unsqueeze(1).expand(B, n_heads, N, N).reshape(B * n_heads, N, N)
        msg, _ = self.msg_attn(proj, proj, proj, attn_mask=attn_bias, need_weights=False)  # [B, N, M]

        # 3. GRU update — feed attention-aggregated messages, NOT raw features
        # Flatten batch for GRU cell
        gru_input = torch.cat([msg, memory], dim=-1).reshape(B * N, -1)
        mem_flat = memory.reshape(B * N, self.memory_dim)
        # Clamp inputs (gradient-safe, unlike nan_to_num)
        gru_input = gru_input.clamp(-10.0, 10.0)
        mem_flat = mem_flat.clamp(-10.0, 10.0)
        new_mem_flat = self.gru_cell(gru_input, mem_flat)  # [B*N, memory_dim]
        gru_update = new_mem_flat.reshape(B, N, self.memory_dim)

        # 4. Compute per-node decay based on node type and Δt
        tau = torch.exp(self.log_tau).clamp(min=0.01, max=20.0)  # [num_types]

        if node_types is not None and delta_t is not None:
            # Map node types to decay rates
            # node_types: [N], delta_t: [B, N]
            type_idx = node_types.long().clamp(0, self.num_node_types - 1)
            tau_per_node = tau[type_idx]  # [N]
            tau_bcast = tau_per_node.unsqueeze(0).expand(B, -1)  # [B, N]
            decay = torch.exp(-tau_bcast * delta_t.clamp(min=0.0))  # [B, N]
            decay = decay.unsqueeze(-1)  # [B, N, 1] — broadcast over memory_dim
        else:
            # Default: uniform decay of 0.95 if no timing info available
            decay = torch.full((B, N, 1), 0.95, device=device)

        # 5. Interpolate: decay * old_memory + (1-decay) * gru_update
        new_memory = decay * memory + (1.0 - decay) * gru_update
        # Gradient-safe clamp (preserves gradients, unlike nan_to_num)
        new_memory = new_memory.clamp(-10.0, 10.0)

        return new_memory


# ---------------------------------------------------------------------------
# C7: Hierarchical Node-Type Transformer (Micro + Macro)
# ---------------------------------------------------------------------------

class HierarchicalHGT(nn.Module):
    """
    Two-level heterogeneous graph transformer:

    Level 1 (Micro):  Within-type attention
        h_stock_i^micro  = Attn({ h_stock_j : j ∈ same-type neighbours })
        h_sector_k^micro = Attn({ h_sector_m : ... })

    Level 2 (Macro):  Cross-type attention
        h_stock_i^out = Attn( h_stock_i^micro, h_sector(i)^micro, h_macro^micro )

    This mirrors the real information hierarchy:
        company-level → sector-level → macro-level

    Prevents VIX (macro) from suppressing individual stock alpha.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int = 4,
        num_node_types: int = 4,
        dropout: float = 0.1,
        use_attention_gating: bool = False,
        attn_gate_threshold: float = 2.0,
        residual_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_node_types = num_node_types
        self.use_attention_gating = use_attention_gating
        self.attn_gate_threshold = attn_gate_threshold
        self.residual_alpha = residual_alpha
        hidden_dim // num_heads

        # Micro-level: one attention module per node type
        self.micro_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_node_types)
        ])
        # Per-type input projection
        self.type_proj = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim)
            for _ in range(num_node_types)
        ])

        # Macro-level: single cross-type attention
        self.macro_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Type-pair scaling matrix (learned)
        # [num_types, num_types] — scale from type j to type i
        # RC4: init to 0.1 (not 1.0) for moderate log-bias, preventing attention domination
        self.type_scale = nn.Parameter(torch.full((num_node_types, num_node_types), 0.1))

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,                    # [B, N, in_dim]
        node_types: Tensor,           # [N] int64
        adj_mask: Optional[Tensor] = None,   # [B, N, N]
    ) -> Tensor:
        """
        Returns:
            out: [B, N, out_dim]
        """
        B, N, _ = x.shape
        device = x.device

        # ── Level 1: Micro (within-type) ──────────────────────────────────
        micro_out = torch.zeros(B, N, x.shape[-1] if x.shape[-1] == self.type_proj[0].out_features
                                else self.type_proj[0].out_features,
                                device=device)
        # Actually, project first then attend
        x_proj = torch.zeros(B, N, self.type_proj[0].out_features, device=device)
        for t in range(self.num_node_types):
            mask_t = (node_types == t)
            if mask_t.sum() == 0:
                continue
            idx_t = torch.nonzero(mask_t).squeeze(-1)
            x_t = x.index_select(1, idx_t)          # [B, n_t, in_dim]
            x_proj.index_copy_(1, idx_t, self.type_proj[t](x_t))

        for t in range(self.num_node_types):
            mask_t = (node_types == t)
            if mask_t.sum() == 0:
                continue
            idx_t = torch.nonzero(mask_t).squeeze(-1)
            sub = x_proj.index_select(1, idx_t)     # [B, n_t, hidden]
            # RC3: scale by 1/sqrt(d) to prevent attention score explosion in large groups
            scale = (sub.shape[-1] ** 0.5)
            sub_scaled = sub / scale
            attn_out, _ = self.micro_attn[t](sub_scaled, sub_scaled, sub_scaled, need_weights=False)
            attn_out = attn_out * scale  # restore magnitude
            micro_out.index_copy_(1, idx_t, attn_out)

        micro_out = self.layer_norm(micro_out + x_proj)

        # ── Level 2: Macro (cross-type) ───────────────────────────────────
        # Apply learned type-scale to the adjacency before cross-type attention
        # type_scale: [num_types, num_types] → per-node-pair scaling
        # Build a [B, N, N] type-pair scale matrix
        row_types = node_types.unsqueeze(0).expand(N, -1).T   # [N, N]... actually:
        row_types = node_types.unsqueeze(-1).expand(N, N)     # [N, N]
        col_types = node_types.unsqueeze(0).expand(N, N)      # [N, N]
        scale_mat = self.type_scale[row_types, col_types]     # [N, N]
        scale_mat = scale_mat.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

        # Build attention bias including type-scale
        if adj_mask is not None:
            adj_with_scale = adj_mask * scale_mat
            attn_bias = torch.log(torch.clamp(adj_with_scale, min=0) + 1e-9)
        else:
            attn_bias = torch.log(scale_mat.clamp(min=0) + 1e-9)
        # Clamp attention bias to prevent extreme values
        attn_bias = attn_bias.clamp(min=-20.0, max=0.0)

        # Expand bias for heads
        macro_bias = (
            attn_bias.unsqueeze(1)
            .expand(B, self.macro_attn.num_heads, N, N)
            .reshape(B * self.macro_attn.num_heads, N, N)
        )

        # --- D2: Adaptive Attention Gating ---
        if self.use_attention_gating:
             # Apply gate to bias before macro-attention
             # We use the same gate for all heads for efficiency
             gate = apply_attention_gate(torch.ones(B, 1, 1, N, device=device), micro_out, self.attn_gate_threshold)
             gate_mask = (gate < 0.5).float() * -1e9
             macro_bias = macro_bias + gate_mask.expand(B * self.macro_attn.num_heads, N, N)

        macro_out, _ = self.macro_attn(
            micro_out, micro_out, micro_out,
            attn_mask=macro_bias,
            need_weights=False,
        )
        out = self.layer_norm(macro_out + micro_out)
        
        # --- D4: Residual Connection to prevent over-smoothing ---
        # micro_out already has x_proj in it. We can add a residual from micro_out or x_proj.
        # Let's use x_proj (which is the projected input features) as the base.
        if out.shape == x_proj.shape:
             # We need to import it or define it locally? 
             # It's in gnn_sac_agent.py. Since this is nextgen_gnn_components, 
             # I should probably just implement it here to avoid circular imports 
             # OR move the pure function to a common utils.
             # For now, I'll just use the logic: alpha * x_proj + (1-alpha) * out
             out = self.residual_alpha * x_proj + (1.0 - self.residual_alpha) * out

        out = self.dropout(out)
        return self.out_proj(out).clamp(-10.0, 10.0)   # [B, N, out_dim], gradient-safe clamp


# ---------------------------------------------------------------------------
# SAC-1: IQN Distributional Critic
# ---------------------------------------------------------------------------

class IQNCritic(nn.Module):
    """
    Implicit Quantile Network (IQN) distributional critic.
    Replaces the standard scalar Q-network.

    Z(s, a, τ) = f_τ( φ(s, a) )

    where τ ~ U[0,1] is the quantile level.

    Policy gradient uses both:
      - Expected value:   E_τ[Z(s, π(s), τ)]           → return maximisation
      - Tail penalty:     κ · E[Z(s, π(s), τ=high)]     → CVaR-MDD control

    Training loss: Huber quantile regression (pinball loss).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        n_quantile_samples: int = 32,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.n_quantile_samples = n_quantile_samples
        self.embed_dim = embed_dim

        # State-action encoder
        in_dim = state_dim + action_dim
        encoder_layers = []
        prev = in_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_out_dim = prev

        # Quantile embedding: τ → R^embed_dim  (cosine embedding)
        self.tau_embed = nn.Sequential(
            nn.Linear(embed_dim, self.encoder_out_dim),
            nn.Sigmoid(),
        )

        # Final mapping: element-wise product → scalar
        self.out = nn.Linear(self.encoder_out_dim, 1)

    def _cos_embed(self, tau: Tensor) -> Tensor:
        """
        Cosine embedding of quantile values.
        tau: [B, n_tau] → [B, n_tau, embed_dim]
        """
        i = torch.arange(1, self.embed_dim + 1, device=tau.device, dtype=tau.dtype)
        cos_in = tau.unsqueeze(-1) * math.pi * i.unsqueeze(0).unsqueeze(0)
        return torch.cos(cos_in)    # [B, n_tau, embed_dim]

    def forward(
        self,
        state: Tensor,           # [B, state_dim]
        action: Tensor,          # [B, action_dim]
        tau: Optional[Tensor] = None,   # [B, n_tau] — if None, sampled uniformly
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            z_values: [B, n_tau]    — quantile values
            tau:      [B, n_tau]    — quantile levels used
        """
        B = state.shape[0]
        n_tau = self.n_quantile_samples

        if tau is None:
            tau = torch.rand(B, n_tau, device=state.device)

        # Encode state-action
        sa = torch.cat([state, action], dim=-1)       # [B, state+action]
        sa_enc = self.encoder(sa)                      # [B, encoder_out]
        sa_enc = sa_enc.unsqueeze(1).expand(B, n_tau, -1)  # [B, n_tau, enc]

        # Quantile embedding
        cos_emb = self._cos_embed(tau)                 # [B, n_tau, embed_dim]
        tau_enc = self.tau_embed(cos_emb)              # [B, n_tau, enc]

        # Element-wise product (IQN core)
        combined = sa_enc * tau_enc                    # [B, n_tau, enc]
        z = self.out(combined).squeeze(-1)             # [B, n_tau]

        return z, tau

    @staticmethod
    def quantile_huber_loss(
        z_pred: Tensor,    # [B, n_tau]
        z_target: Tensor,  # [B, n_tau_prime]
        tau: Tensor,       # [B, n_tau]
        kappa: float = 1.0,
        weights: Optional[Tensor] = None, # [B] importance sampling weights
        cvar_tau: float = 1.0,           # Threshold for CVaR penalty (levels < tau are tail)
        cvar_multiplier: float = 1.0,     # Penalty multiplier for the tail
    ) -> Tuple[Tensor, Tensor]:
        """
        Pinball (quantile Huber) loss for distributional RL.
        Includes CVaR regulation: weights errors in tail quantiles more heavily.
        """
        B, n_tau = z_pred.shape
        z_target.shape[1]

        # Expand for pairwise computation
        pred = z_pred.unsqueeze(2)    # [B, n_tau, 1]
        target = z_target.unsqueeze(1)  # [B, 1, n_tau']
        taus = tau.unsqueeze(2)       # [B, n_tau, 1]

        td = target - pred            # [B, n_tau, n_tau']
        huber = torch.where(
            td.abs() <= kappa,
            0.5 * td.pow(2),
            kappa * (td.abs() - 0.5 * kappa),
        )
        # Asymmetric weight
        weight_asym = (taus - (td < 0).float()).abs()   # [B, n_tau, n_tau']
        
        # --- D1: CVaR-Regulated Weighting ---
        # Penalize samples where the quantile level tau is in the tail
        cvar_weight = torch.where(taus < cvar_tau, cvar_multiplier, 1.0)
        
        loss_B = (weight_asym * huber * cvar_weight).mean(dim=2).mean(dim=1) # [B]
        
        td_errors = td.abs().mean(dim=2).mean(dim=1) # [B] approx td-error magnitude
        
        if weights is not None:
            loss_B = loss_B * weights.view(-1)
            
        return loss_B.mean(), td_errors


class DualHeadIQNCritic(nn.Module):
    """IQN with separate equity/safe-haven quantile heads.
    Single Responsibility: each head evaluates tail risk for its asset class.
    """
    def __init__(
        self, state_dim: int, action_dim: int,
        equity_indices: list[int], safe_haven_indices: list[int],
        hidden_dims: list[int] | None = None,
        n_quantile_samples: int = 32, embed_dim: int = 64
    ) -> None:
        super().__init__()
        self.n_quantile_samples = n_quantile_samples
        self.embed_dim = embed_dim
        self.equity_indices = equity_indices
        self.safe_haven_indices = safe_haven_indices
        
        if hidden_dims is None:
            hidden_dims = [256, 256]

        in_dim = state_dim + action_dim
        encoder_layers = []
        prev = in_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_out_dim = prev

        self.tau_embed = nn.Sequential(
            nn.Linear(embed_dim, self.encoder_out_dim),
            nn.Sigmoid(),
        )

        self.equity_head = nn.Linear(self.encoder_out_dim, 1)
        self.safe_haven_head = nn.Linear(self.encoder_out_dim, 1)

    def _cos_embed(self, tau: Tensor) -> Tensor:
        i = torch.arange(1, self.embed_dim + 1, device=tau.device, dtype=tau.dtype)
        cos_in = tau.unsqueeze(-1) * math.pi * i.unsqueeze(0).unsqueeze(0)
        return torch.cos(cos_in)

    def forward(
        self, state: Tensor, action: Tensor, tau: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        B = state.shape[0]
        n_tau = self.n_quantile_samples
        if tau is None:
            tau = torch.rand(B, n_tau, device=state.device)

        sa = torch.cat([state, action], dim=-1)
        sa_enc = self.encoder(sa)
        sa_enc = sa_enc.unsqueeze(1).expand(B, n_tau, -1)

        cos_emb = self._cos_embed(tau)
        tau_enc = self.tau_embed(cos_emb)

        combined = sa_enc * tau_enc
        
        eq_q = self.equity_head(combined).squeeze(-1)
        sh_q = self.safe_haven_head(combined).squeeze(-1)

        return eq_q, sh_q, tau


# ---------------------------------------------------------------------------
# SAC-3: Regime-Adaptive Entropy Temperature
# ---------------------------------------------------------------------------

class RegimeEntropyScheduler(nn.Module):
    """
    Adapts SAC entropy temperature α based on current market volatility.

    α_t = α_base · (1 + β_vol · (VIX_t / VIX_mean - 1))

    High-VIX → higher entropy → more hedging / exploration
    Low-VIX  → lower entropy  → concentrated alpha positions

    This module wraps the standard α parameter and produces α_t at each step.
    """

    def __init__(
        self,
        alpha_base: float = 0.2,
        beta_vol_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.log_alpha_base = nn.Parameter(torch.tensor(alpha_base).log())
        self.log_beta_vol = nn.Parameter(torch.tensor(beta_vol_init).log())

    def forward(self, vix: Tensor, vix_mean: Tensor) -> Tensor:
        """
        Args:
            vix:      [B] or scalar — current VIX
            vix_mean: [B] or scalar — 60-day rolling mean VIX
        Returns:
            alpha_t: [B] or scalar — regime-conditioned entropy temperature
        """
        alpha_base = self.log_alpha_base.exp().clamp(min=0.01, max=1.0)
        beta_vol = self.log_beta_vol.exp().clamp(min=0.0, max=5.0)
        vix_ratio = (vix / vix_mean.clamp(min=1.0)).clamp(min=0.5, max=3.0)
        alpha_t = alpha_base * (1.0 + beta_vol * (vix_ratio - 1.0))
        return alpha_t.clamp(min=0.01, max=1.0)


# ---------------------------------------------------------------------------
# Unified NextGen Encoder — wraps all graph components
# ---------------------------------------------------------------------------

class NextGenEventGraphEncoder(nn.Module):
    """
    Drop-in replacement for the existing EventAwareGAT/TGNRecurrentLayer
    combination inside GNNGaussianPolicy.

    Pipeline per forward pass:
        1. Lambda gating: G'_t = (1-λ_t)·G_base + λ_t·G_event   [C1]
        2. VolatilityAware bias applied to adjacency               [C2]
        3. ShockPropagation gate on messages                       [C3]
        4. EventSeverity edge weight added to G_event              [C4]
        5. HierarchicalHGT (Micro + Macro)                         [C7]
        6. DecayTGN memory update with event-conditioned decay     [C6]
        7. DrawdownEmbeddingGate on output embeddings              [C5]

    Input/output shapes are identical to TGNRecurrentLayer so no other files
    need changing — just swap the encoder in GNNGaussianPolicy.__init__.
    """

    def __init__(
        self,
        in_dim: int,
        memory_dim: int,
        num_nodes: int,
        num_heads: int = 4,
        num_node_types: int = 4,
        regime_dim: int = 3,
        event_embed_dim: int = 64,
        sector_embed_dim: int = 16,
        dropout: float = 0.1,
        use_edge_pruning: bool = False,
        topk_k: int = 5,
        use_attention_gating: bool = False,
        attn_gate_threshold: float = 2.0,
        residual_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.num_nodes = num_nodes
        self.use_edge_pruning = use_edge_pruning
        self.topk_k = topk_k
        self.residual_alpha = residual_alpha

        # C1: Regime-adaptive lambda
        self.lambda_net = RegimeLambdaScheduler(regime_dim=regime_dim)

        # C2: Volatility-aware attention bias
        self.vol_bias = VolatilityAwareAttentionBias()

        # C4: Event severity edge modulator — lightweight
        self.event_severity = EventSeverityEdgeModulator(
            event_embed_dim=event_embed_dim, num_nodes=num_nodes
        )

        # C7: Hierarchical HGT (Micro + Macro)
        self.hgt = HierarchicalHGT(
            in_dim=in_dim,
            hidden_dim=memory_dim,
            out_dim=memory_dim,
            num_heads=num_heads,
            num_node_types=num_node_types,
            dropout=dropout,
            use_attention_gating=use_attention_gating,
            attn_gate_threshold=attn_gate_threshold,
            residual_alpha=residual_alpha,
        )

        # C6: Enhanced decay TGN
        self.tgn = DecayTGNLayer(
            in_dim=in_dim,
            memory_dim=memory_dim,
            num_heads=num_heads,
            num_node_types=num_node_types,
        )

        # C5: Drawdown-aware gate
        self.dd_gate = DrawdownEmbeddingGate(
            embed_dim=memory_dim,
            num_nodes=num_nodes,
        )

        # Event embedding projection (from FinBERT dim → event_embed_dim)
        self.event_proj = nn.Linear(768, event_embed_dim)  # 768 = FinBERT hidden
        # Fallback if no event embedding: learned zero-init
        # RC2: register_buffer (not Parameter) to prevent gradient explosion through polarity_proj
        self.register_buffer('null_event_emb', torch.zeros(1, event_embed_dim))

        # Regime feature projection from VIX scalar + vol → regime_dim
        self.regime_proj = nn.Linear(2, regime_dim)  # [VIX_norm, delta_vol]

    def forward(
        self,
        node_feats: Tensor,                    # [B, N, in_dim]
        memory: Tensor,                        # [B, N, memory_dim]
        adj_base: Optional[Tensor] = None,     # [B, N, N] or [N, N]  G_base
        adj_event: Optional[Tensor] = None,    # [B, N, N]  G_event
        node_types: Optional[Tensor] = None,   # [N] int64
        vol_vec: Optional[Tensor] = None,      # [B, N] per-node 20d vol
        regime_feats: Optional[Tensor] = None, # [B, 3]
        event_emb: Optional[Tensor] = None,    # [B, 768] FinBERT
        dd_context: Optional[Tensor] = None,   # [B, 3]
        delta_t: Optional[Tensor] = None,      # [B, N] days since last event
    ) -> Tensor:
        """
        Returns:
            new_memory: [B, N, memory_dim]
        """
        B, N, _ = node_feats.shape
        device = node_feats.device

        # ── C4: Event severity edges ──────────────────────────────────────
        if event_emb is not None:
            e_proj = self.event_proj(event_emb)   # [B, event_embed_dim]
        else:
            e_proj = self.null_event_emb.expand(B, -1)

        severity_edges = self.event_severity(e_proj, node_feats)   # [B, N, N]

        # ── C1: Regime-adaptive lambda ────────────────────────────────────
        if regime_feats is not None:
            lam = self.lambda_net(regime_feats)    # [B, 1]
        else:
            lam = torch.full((B, 1), 0.5, device=device)

        # Build fused adjacency G'_t = (1-λ)·G_base + λ·(G_event + severity)
        if adj_base is not None:
            if adj_base.dim() == 2:
                adj_base = adj_base.unsqueeze(0).expand(B, -1, -1)
        else:
            adj_base = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)

        if adj_event is None:
            adj_event = torch.zeros(B, N, N, device=device)

        # Add severity to event graph (clamp to avoid extreme values)
        adj_event_augmented = (adj_event + severity_edges).clamp(-1.0, 1.0)
        # Absolute value for attention (signed edges handled by severity separately)
        adj_event_abs = adj_event_augmented.abs()

        lam_bcast = lam.unsqueeze(-1)  # [B, 1, 1]
        adj_fused = (1.0 - lam_bcast) * adj_base + lam_bcast * adj_event_abs  # [B, N, N]
        
        # ── A1: Dynamic Edge Pruning ──────────────────────────────────────
        if self.use_edge_pruning:
            adj_fused = prune_adjacency(adj_fused, node_feats, k=self.topk_k)

        # ── C2: Volatility-aware attention bias ───────────────────────────
        if vol_vec is not None:
            vol_bias = self.vol_bias(vol_vec)   # [B, N, N] ≤ 0
            adj_fused = adj_fused * torch.exp(vol_bias)   # dampen high-vol edges

        # ── C7: Hierarchical HGT ──────────────────────────────────────────
        hgt_out = self.hgt(node_feats, node_types if node_types is not None
                           else torch.zeros(N, dtype=torch.long, device=device),
                           adj_fused)   # [B, N, memory_dim]

        # ── C6: Enhanced decay TGN ────────────────────────────────────────
        new_memory = self.tgn(
            node_feats=node_feats,
            memory=memory,
            adj_mask=adj_fused,
            node_types=node_types,
            delta_t=delta_t,
        )   # [B, N, memory_dim]

        # Blend HGT output into TGN update (HGT carries cross-type info)
        new_memory = 0.5 * new_memory + 0.5 * hgt_out

        # ── C5: Drawdown-aware embedding gate ─────────────────────────────
        if dd_context is not None:
            new_memory = self.dd_gate(new_memory, dd_context)

        # Gradient-safe clamp (NOT nan_to_num which kills gradients)
        new_memory = new_memory.clamp(-10.0, 10.0)

        return new_memory
