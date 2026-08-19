import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.agents.gnn_sac_agent import (
    GNNSACAgent, apply_crisis_mask, 
    apply_momentum_mask, apply_residual_connection
)
from src.agents.loss_calculator import (
    compute_nlp_dynamic_alpha, compute_cvar_loss, compute_volatility_exploration_penalty
)
from src.agents.nextgen_gnn_components import IQNCritic, prune_adjacency, DualHeadIQNCritic, apply_attention_gate

# ...

# ==========================================
# D3: Local Momentum Action Masking Tests
# ==========================================

def test_apply_momentum_mask_suppresses_crashing_stocks():
    """Stock with momentum < threshold -> logit should be significantly reduced."""
    # B=1, N=4
    logits = torch.ones(1, 4)
    # Feature 0 is momentum. Node 1 is crashing (-0.5). Others stable.
    node_feats = torch.tensor([[[0.1], [-0.5], [0.2], [0.0]]])
    
    # Threshold -0.2
    result = apply_momentum_mask(logits, node_feats, momentum_index=0, threshold=-0.2)
    
    # Node 1 logit should be smaller than Node 0 logit
    assert result[0, 1] < result[0, 0]
    assert result[0, 1] < 0.2 # suppressed
    assert result[0, 0] > 0.9 # kept

def test_apply_momentum_mask_is_differentiable():
    """Gradients flow through momentum mask."""
    logits = torch.ones(1, 4, requires_grad=True)
    node_feats = torch.tensor([[[0.1], [-0.5], [0.2], [0.0]]])
    result = apply_momentum_mask(logits, node_feats, momentum_index=0)
    result.sum().backward()
    assert logits.grad is not None


# ...

# ==========================================
# D2: Adaptive Attention Gating Tests
# ==========================================

def test_apply_attention_gate_suppresses_noisy_nodes():
    """Attention to nodes with high volatility (feature variance) should be gated toward zero."""
     # B=1, H=1, N=3
    attn = torch.ones(1, 1, 3, 3)
    # Node 2 is noisy (high values across features), others stable
    # Feature variance: Node 0 (0), Node 1 (0), Node 2 (std=5.0 roughly)
    node_feats = torch.tensor([[[1., 1.], [1., 1.], [10., 0.]]]) 
    
    # Gate with threshold low enough to target node 2 (variance of [10, 0] is ~50?)
    # Actually, simplistic gate: check std across features
    result = apply_attention_gate(attn, node_feats, dist_threshold=2.0)
    
    # Attention to node 2 (column 2) should be suppressed
    assert result[0, 0, 0, 2] < 0.1
    assert result[0, 0, 0, 0] > 0.9 # stable node kept

def test_apply_attention_gate_keeps_stable_graph():
    """All stable nodes -> attention preserved."""
    attn = torch.ones(1, 1, 2, 2)
    node_feats = torch.ones(1, 2, 4)
    result = apply_attention_gate(attn, node_feats, dist_threshold=5.0)
    assert torch.allclose(result, attn)


# ... (rest of imports remains same)

# ==========================================
# D1: CVaR-Regulated Critic Loss Tests
# ==========================================

def test_compute_cvar_loss_penalizes_lower_quantiles():
    """Errors in lower quantiles yield significantly higher loss."""
    # Dummy quantiles and targets
    # B=1, N=10
    q = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]], requires_grad=True)
    t = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]) # uniform -0.1 error
    
    # Standard loss (all quantiles equal)
    loss_std = compute_cvar_loss(q, t, tau=1.0, multiplier=1.0)
    # CVaR loss (only bottom 20% penalized with 5x)
    loss_cvar = compute_cvar_loss(q, t, tau=0.2, multiplier=5.0)
    
    # Since all errors are -0.1, std loss is mean(|-0.1|^2) = 0.01
    # For CVaR(0.2), 2/10 nodes are in the tail. Tail loss = 0.01 * 5 = 0.05.
    # Total loss = (TailLoss * 2 + NormalLoss * 8) / 10 = (0.1 + 0.08) / 10 = 0.018?
    # Regardless, loss_cvar should be strictly greater than loss_std
    assert loss_cvar > loss_std

def test_compute_cvar_loss_is_differentiable():
    """Gradients flow through CVaR loss."""
    q = torch.randn(2, 32, requires_grad=True)
    t = torch.randn(2, 32)
    loss = compute_cvar_loss(q, t, tau=0.1)
    loss.backward()
    assert q.grad is not None


def test_gnn_sac_nextgen_init_has_targets():
    """Ensure that NextGen initialization creates target networks for IQN critics."""
    agent = GNNSACAgent(
        state_dim=20,
        action_dim=2,
        node_features_dim=5,
        num_nodes=4,
        adj_matrix=np.eye(4),
        use_nextgen=True
    )
    
    assert hasattr(agent, 'iqn_critic1_target'), "Missing target critic 1"
    assert hasattr(agent, 'iqn_critic2_target'), "Missing target critic 2"
    assert isinstance(agent.iqn_critic1_target, IQNCritic), "Target 1 is not IQNCritic"

def test_gnn_sac_nextgen_update_uses_iqn():
    """Ensure update() doesn't crash and returns expected loss dict using IQN."""
    agent = GNNSACAgent(
        state_dim=20,
        action_dim=2,
        node_features_dim=5,
        num_nodes=4,
        adj_matrix=np.eye(4),
        use_nextgen=True,
        batch_size=2  # Very small batch for testing
    )
    
    # Populate buffer
    for _ in range(5):
        state = np.random.randn(20)
        action = np.random.randn(2)
        next_state = np.random.randn(20)
        agent.replay_buffer.push(state, action, 1.0, next_state, False)
        
    losses = agent.update()
    assert 'iqn_loss' in losses, "iqn_loss missing from returned losses"
    assert 'policy_loss' in losses, "policy_loss missing from returned losses"

# ==========================================
# A1: Dynamic Edge Pruning Tests (FCIS Core)
# ==========================================

def test_prune_adjacency_severs_negative_momentum_nodes():
    """Nodes with negative momentum have ~90% of incoming edges severed."""
    adj = torch.ones(1, 4, 4)
    # Node 0 has negative momentum (-0.5), others positive
    feats = torch.tensor([[[-0.5, 0.1], [0.3, 0.2], [0.1, 0.4], [0.2, 0.3]]])
    result = prune_adjacency(adj, feats, momentum_index=0, k=2, sever_ratio=0.9)
    # Edge from node 0 should be ~10% of original (1.0 * 0.1 = 0.1)
    assert result[0, 1, 0] < 0.15

def test_prune_adjacency_keeps_at_most_k_neighbors():
    """Each row retains at most k non-zero entries."""
    adj = torch.rand(1, 6, 6)
    feats = torch.rand(1, 6, 4)
    result = prune_adjacency(adj, feats, momentum_index=0, k=3)
    nonzero_per_row = (result[0] > 0).sum(dim=-1)
    assert (nonzero_per_row <= 3).all()

def test_prune_adjacency_preserves_positive_momentum_edges():
    """All-positive momentum → no severing, only Top-K filtering."""
    adj = torch.ones(1, 3, 3)
    feats = torch.tensor([[[0.5, 0.1], [0.3, 0.2], [0.1, 0.4]]])
    result = prune_adjacency(adj, feats, momentum_index=0, k=3)
    assert (result[0] > 0.99).all()  # No severing happened


# ==========================================
# A2: NLP-Conditioned Alpha Tests (FCIS Core)
# ==========================================

def test_nlp_alpha_increases_with_event_severity():
    """Higher NLP severity → higher alpha."""
    base = torch.tensor([[0.1]])
    calm = torch.zeros(1, 768)
    hot = torch.ones(1, 768) * 5.0
    alpha_calm = compute_nlp_dynamic_alpha(base, calm, multiplier=0.1)
    alpha_hot = compute_nlp_dynamic_alpha(base, hot, multiplier=0.1)
    assert alpha_hot.item() > alpha_calm.item()

def test_nlp_alpha_clamps_within_bounds():
    """Extreme embeddings don't produce alpha > 1.0 or < 0.01."""
    base = torch.tensor([[0.9]])
    extreme = torch.ones(1, 768) * 100.0
    alpha = compute_nlp_dynamic_alpha(base, extreme, multiplier=1.0)
    assert 0.01 <= alpha.item() <= 1.0

def test_nlp_alpha_zero_embedding_returns_base():
    """Zero event embedding → alpha equals base_alpha."""
    base = torch.tensor([[0.2]])
    zero = torch.zeros(1, 768)
    alpha = compute_nlp_dynamic_alpha(base, zero, multiplier=0.5)
    assert torch.allclose(alpha, base)


# ==========================================
# B1: Dual-Head IQN Critic Tests
# ==========================================

def test_dual_head_iqn_returns_separate_quantiles():
    """Forward returns (equity_q, safe_haven_q, tau) with correct shapes."""
    critic = DualHeadIQNCritic(
        state_dim=20, action_dim=4,
        equity_indices=[0, 1], safe_haven_indices=[2, 3],
    )
    s, a = torch.randn(2, 20), torch.randn(2, 4)
    eq_q, sh_q, tau = critic(s, a)
    assert eq_q.shape == (2, 32)
    assert sh_q.shape == (2, 32)
    assert tau.shape == (2, 32)

def test_dual_head_iqn_equity_head_independent():
    """Changing safe-haven action dims doesn't affect equity quantiles shape."""
    critic = DualHeadIQNCritic(
        state_dim=10, action_dim=4,
        equity_indices=[0, 1], safe_haven_indices=[2, 3],
    )
    s = torch.randn(1, 10)
    a1 = torch.tensor([[0.3, 0.3, 0.2, 0.2]])
    a2 = torch.tensor([[0.3, 0.3, 0.0, 1.0]])  # different safe-haven split
    eq1, _, _ = critic(s, a1)
    eq2, _, _ = critic(s, a2)
    assert eq1.shape == eq2.shape

# ==========================================
# B2: Macro-Conditioned Action Masking Tests
# ==========================================

def test_crisis_mask_boosts_safe_havens_during_panic():
    """Panic signal > threshold → safe-haven logits increase, equities strictly suppressed."""
    logits = torch.zeros(1, 6)
    panic = torch.tensor([[0.95]])
    result = apply_crisis_mask(logits, panic, equity_idx=[0,1,2],
                               haven_idx=[3,4,5], threshold=0.8)
    assert result[0, 3] > logits[0, 3]  # haven boosted
    assert result[0, 0] < logits[0, 0]  # equity suppressed

def test_crisis_mask_no_effect_below_threshold():
    """Panic signal < threshold → logits unchanged."""
    logits = torch.ones(1, 4)
    panic = torch.tensor([[0.3]])
    result = apply_crisis_mask(logits, panic, [0,1], [2,3], threshold=0.8)
    assert torch.allclose(result, logits)

def test_crisis_mask_is_differentiable():
    """Gradients flow through the mask for end-to-end training."""
    logits = torch.randn(1, 4, requires_grad=True)
    panic = torch.tensor([[0.95]], requires_grad=True)
    result = apply_crisis_mask(logits, panic, [0,1], [2,3], threshold=0.8)
    loss = result.sum()
    loss.backward()
    assert logits.grad is not None
    assert panic.grad is not None




# ==========================================
# D4: Graph Residual Connection Tests
# ==========================================

def test_apply_residual_connection_blends_features():
    """Residual blend should satisfy: out = x_base * alpha + x_gnn * (1-alpha)."""
    x_base = torch.ones(1, 4, 10) * 2.0
    x_gnn = torch.ones(1, 4, 10) * 5.0
    
    # 50/50 blend -> 3.5
    res = apply_residual_connection(x_base, x_gnn, alpha=0.5)
    assert torch.allclose(res, torch.ones(1, 4, 10) * 3.5)
    
    # 100% base -> 2.0
    res_base = apply_residual_connection(x_base, x_gnn, alpha=1.0)
    assert torch.allclose(res_base, x_base)

def test_apply_residual_connection_preserves_gradients():
    """Gradients should flow back to both inputs."""
    x_base = torch.ones(1, 4, 10, requires_grad=True)
    x_gnn = torch.ones(1, 4, 10, requires_grad=True)
    res = apply_residual_connection(x_base, x_gnn, alpha=0.5)
    res.sum().backward()
    assert x_base.grad is not None
    assert x_gnn.grad is not None

# ==========================================
# D5: Volatility-Penalized Exploration Tests
# ==========================================

def test_compute_volatility_exploration_penalty_increases_with_vix():
    """Higher VIX and higher log_std should yield higher penalty."""
    log_std = torch.tensor([0.5], requires_grad=True)
    vix_low = torch.tensor([0.1])
    vix_high = torch.tensor([0.9])
    
    p_low = compute_volatility_exploration_penalty(log_std, vix_low, penalty_coef=1.0)
    p_high = compute_volatility_exploration_penalty(log_std, vix_high, penalty_coef=1.0)
    
    assert p_high > p_low
    assert p_high > 0

def test_volatility_penalty_is_differentiable():
    """Policy should be able to minimize the penalty through gradients."""
    log_std = torch.tensor([0.0], requires_grad=True)
    vix = torch.tensor([0.5])
    penalty = compute_volatility_exploration_penalty(log_std, vix)
    penalty.backward()
    assert log_std.grad is not None

def test_feature_gating_module_suppresses_noise():
    """FeatureGatingModule should multiply features by a learned sigmoid gate."""
    from src.agents.gnn_sac_agent import FeatureGatingModule
    feat_dim = 10
    module = FeatureGatingModule(feat_dim)
    x = torch.randn(2, 5, feat_dim) # [B, N, F]
    out = module(x)
    assert out.shape == x.shape
    # Check that it's between 0 and x (for positive x) or between 0 and x (for negative x)
    # Since sigmoid is (0, 1), the magnitude should be reduced or stay same.
    assert torch.all(torch.abs(out) <= torch.abs(x) + 1e-6)

def test_feature_gating_is_differentiable():
    from src.agents.gnn_sac_agent import FeatureGatingModule
    feat_dim = 5
    module = FeatureGatingModule(feat_dim)
    x = torch.randn(1, 1, feat_dim, requires_grad=True)
    out = module(x)
    loss = out.pow(2).sum()
    loss.backward()
    assert x.grad is not None
