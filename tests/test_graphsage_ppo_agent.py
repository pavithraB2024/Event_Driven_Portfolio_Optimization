"""Tests for the GraphSAGE-PPO baseline (replication).

Covers the shape/contract guarantees that the evaluation pipeline depends on
when it drops GraphSAGEPPOAgent into ``run_evaluation_loop`` next to
GNN-SAC/LSTM-SAC: encoder output shape, Dirichlet actor validity, scalar
critic, rollout buffer push/clear, deterministic vs stochastic action,
PPO update returning sensible metrics, and save/load round-trip.
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch
from torch.distributions import Dirichlet

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agents.graphsage_ppo_agent import (
    GraphSAGEEncoder,
    GraphSAGELayer,
    GraphSAGEPPOActorCritic,
    GraphSAGEPPOAgent,
    PPOConfig,
    RolloutBuffer,
)


NUM_NODES = 6
FEAT_DIM = 4
ACTION_DIM = NUM_NODES
STATE_DIM = NUM_NODES * FEAT_DIM


def _make_adj(num_nodes=NUM_NODES):
    # Symmetric non-negative adjacency with no self-loops zero diag.
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 1.0, size=(num_nodes, num_nodes)).astype(np.float32)
    a = 0.5 * (a + a.T)
    np.fill_diagonal(a, 1.0)
    return a


def _make_agent(num_nodes=NUM_NODES, feat_dim=FEAT_DIM, action_dim=ACTION_DIM):
    return GraphSAGEPPOAgent(
        state_dim=num_nodes * feat_dim,
        action_dim=action_dim,
        node_features_dim=feat_dim,
        num_nodes=num_nodes,
        adj_matrix=_make_adj(num_nodes),
        config=PPOConfig(update_epochs=2, minibatch_size=8),
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Encoder / layer shape tests
# ---------------------------------------------------------------------------

def test_graphsage_layer_output_shape():
    layer = GraphSAGELayer(in_dim=FEAT_DIM, out_dim=16)
    x = torch.randn(3, NUM_NODES, FEAT_DIM)
    adj = torch.tensor(_make_adj())
    out = layer(x, adj)
    assert out.shape == (3, NUM_NODES, 16)


def test_graphsage_encoder_two_layer_output_shape():
    enc = GraphSAGEEncoder(in_dim=FEAT_DIM, hidden_dim=8, out_dim=12)
    x = torch.randn(2, NUM_NODES, FEAT_DIM)
    adj = torch.tensor(_make_adj())
    out = enc(x, adj)
    assert out.shape == (2, NUM_NODES, 12)


def test_actor_critic_forward_dirichlet_and_value():
    adj_t = torch.tensor(_make_adj())
    ac = GraphSAGEPPOActorCritic(
        node_features_dim=FEAT_DIM,
        num_nodes=NUM_NODES,
        action_dim=ACTION_DIM,
        adj_matrix=adj_t,
    )
    state = torch.randn(4, STATE_DIM)
    dist = ac.actor_distribution(state)
    assert isinstance(dist, Dirichlet)
    assert dist.concentration.shape == (4, ACTION_DIM)
    # Strictly positive concentrations — Dirichlet requires alpha > 0.
    assert torch.all(dist.concentration > 0)
    value = ac.value(state)
    assert value.shape == (4,)


def test_actor_distribution_samples_lie_on_simplex():
    adj_t = torch.tensor(_make_adj())
    ac = GraphSAGEPPOActorCritic(FEAT_DIM, NUM_NODES, ACTION_DIM, adj_t)
    state = torch.randn(8, STATE_DIM)
    samples = ac.actor_distribution(state).sample()
    assert samples.shape == (8, ACTION_DIM)
    sums = samples.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert torch.all(samples >= 0)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

def test_rollout_buffer_push_and_clear():
    buf = RolloutBuffer()
    for i in range(5):
        buf.push(
            state=np.zeros(STATE_DIM, dtype=np.float32),
            action=np.full(ACTION_DIM, 1.0 / ACTION_DIM, dtype=np.float32),
            log_prob=0.1 * i,
            value=0.2 * i,
            reward=float(i),
            done=0.0,
            adj=None,
        )
    assert len(buf) == 5
    buf.clear()
    assert len(buf) == 0
    assert buf.states == []


# ---------------------------------------------------------------------------
# select_action contract
# ---------------------------------------------------------------------------

def test_select_action_deterministic_is_normalised():
    agent = _make_agent()
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action, hidden = agent.select_action(state, deterministic=True)
    assert hidden is None
    assert action.shape == (ACTION_DIM,)
    assert np.isclose(action.sum(), 1.0, atol=1e-5)
    assert np.all(action >= 0)


def test_select_action_stochastic_differs_across_calls():
    agent = _make_agent()
    state = np.random.randn(STATE_DIM).astype(np.float32)
    torch.manual_seed(0)
    a1, _ = agent.select_action(state, deterministic=False)
    torch.manual_seed(1)
    a2, _ = agent.select_action(state, deterministic=False)
    assert a1.shape == a2.shape == (ACTION_DIM,)
    # With different seeds the Dirichlet samples must not be identical.
    assert not np.allclose(a1, a2)


def test_select_action_accepts_nextgen_ctx_kwarg():
    """Evaluation loop passes nextgen_ctx — agent must accept and ignore it."""
    agent = _make_agent()
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action, _ = agent.select_action(state, hidden=None, deterministic=True, nextgen_ctx={"vix": torch.tensor([20.0])})
    assert action.shape == (ACTION_DIM,)


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def test_update_runs_and_returns_metrics():
    agent = _make_agent()
    rng = np.random.default_rng(0)
    state = rng.standard_normal(STATE_DIM).astype(np.float32)
    for _ in range(16):
        action, _ = agent.select_action(state, deterministic=False)
        agent.store_transition(
            state=state,
            action=action,
            reward=float(rng.standard_normal()),
            done=0.0,
            adj=None,
        )
        state = rng.standard_normal(STATE_DIM).astype(np.float32)
    metrics = agent.update(last_state=state)
    assert isinstance(metrics, dict)
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        assert key in metrics
        assert np.isfinite(metrics[key])
    # Buffer must be cleared after the update.
    assert len(agent.buffer) == 0


def test_update_no_op_on_empty_buffer():
    agent = _make_agent()
    metrics = agent.update()
    assert metrics == {}


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip_preserves_policy():
    agent = _make_agent()
    # eval() disables dropout so deterministic actions are reproducible — the
    # evaluation pipeline does this too via ``policy.eval()`` after load.
    agent.policy.eval()
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action_before, _ = agent.select_action(state, deterministic=True)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pth")
        agent.save(path)
        assert os.path.exists(path)

        fresh = _make_agent()
        fresh.policy.eval()
        action_fresh_before, _ = fresh.select_action(state, deterministic=True)
        fresh.load(path)
        fresh.policy.eval()
        action_after, _ = fresh.select_action(state, deterministic=True)

    assert np.allclose(action_before, action_after, atol=1e-6)
    # Sanity: fresh-init action genuinely differed before load.
    assert not np.allclose(action_fresh_before, action_after, atol=1e-6)


# ---------------------------------------------------------------------------
# Adjacency override
# ---------------------------------------------------------------------------

def test_select_action_accepts_external_adjacency():
    agent = _make_agent()
    state = np.random.randn(STATE_DIM).astype(np.float32)
    dyn_adj = _make_adj()  # ndarray
    action, _ = agent.select_action(state, deterministic=True, adj=dyn_adj)
    assert action.shape == (ACTION_DIM,)
    assert np.isclose(action.sum(), 1.0, atol=1e-5)
