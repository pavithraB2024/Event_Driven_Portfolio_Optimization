"""GraphSAGE-PPO baseline.

Direct internal replication of the GraphSAGE-PPO DRL baseline reported by
Sun et al. (2024) for Reviewer 2 Comment 8. The reviewer asked us to retrain
a contemporary GNN-based DRL baseline inside our own environment, split,
and transaction-cost model so the comparison in Table~\\ref{tab:performance}
is not contaminated by hardware/software differences.

Design choices:
  * Encoder: 2-layer GraphSAGE with mean aggregation (matches the original
    GraphSAGE formulation: ``h_v = sigma(W . concat(h_v, mean(h_N(v))))``).
    Adjacency is the static mixed sector/correlation template already used
    by GNN-SAC, so both baselines see the same graph topology.
  * Actor head: Dirichlet over the 28-asset simplex (positive concentrations).
    This keeps PPO's entropy bonus well-defined while producing simplex
    allocations directly, removing the softmax-of-tanh ambiguity that
    affects continuous SAC baselines on portfolio MDPs.
  * Critic head: scalar value V(s) from a 2-layer MLP on the same encoder.
  * Optimizer: clipped surrogate PPO with GAE (Schulman et al., 2017).

The agent's public interface mirrors ``GNNSACAgent`` (``select_action``,
``save``, ``load``) so that the evaluation pipeline in
``scripts/evaluation/evaluate_gnn_sac.py`` can drop it into the same
``run_evaluation_loop`` next to LSTM-SAC without special-casing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Dirichlet

logger = logging.getLogger(__name__)


class GraphSAGELayer(nn.Module):
    """Mean-aggregator GraphSAGE layer.

    Implements the original Hamilton et al. (2017) update:
        h_v <- sigma(W [h_v || mean_{u in N(v)} h_u]).
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: [B, N, F]; adj: [B, N, N] or [N, N] (row-normalized internally)."""
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
        adj = torch.clamp(adj, min=0.0)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        adj_norm = adj / deg
        h_neigh = torch.matmul(adj_norm, x)
        h = self.lin_self(x) + self.lin_neigh(h_neigh)
        h = F.relu(h)
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GraphSAGEEncoder(nn.Module):
    """Two-layer GraphSAGE encoder, returns per-node embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.layer1 = GraphSAGELayer(in_dim, hidden_dim, dropout=dropout)
        self.layer2 = GraphSAGELayer(hidden_dim, out_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x, adj)
        h = self.layer2(h, adj)
        return h


class GraphSAGEPPOActorCritic(nn.Module):
    """Shared GraphSAGE trunk feeding a Dirichlet actor and a scalar critic."""

    def __init__(
        self,
        node_features_dim: int,
        num_nodes: int,
        action_dim: int,
        adj_matrix: torch.Tensor,
        encoder_hidden: int = 64,
        encoder_out: int = 64,
        head_hidden: int = 128,
        log_alpha_init: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_features_dim = node_features_dim
        self.action_dim = action_dim

        self.encoder = GraphSAGEEncoder(
            in_dim=node_features_dim,
            hidden_dim=encoder_hidden,
            out_dim=encoder_out,
            dropout=dropout,
        )
        self.register_buffer("adj_matrix", adj_matrix)

        trunk_in = num_nodes * encoder_out
        self.actor_trunk = nn.Sequential(
            nn.Linear(trunk_in, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, head_hidden),
            nn.Tanh(),
        )
        # Output log-concentrations so the Dirichlet alpha stays positive.
        self.actor_head = nn.Linear(head_hidden, action_dim)
        nn.init.constant_(self.actor_head.bias, log_alpha_init)

        self.critic_trunk = nn.Sequential(
            nn.Linear(trunk_in, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, head_hidden),
            nn.Tanh(),
        )
        self.critic_head = nn.Linear(head_hidden, 1)

    def _encode(self, state: torch.Tensor, adj: Optional[torch.Tensor]) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch_size = state.shape[0]
        x = state.view(batch_size, self.num_nodes, self.node_features_dim)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        if adj is None:
            adj = self.adj_matrix
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        elif adj.shape[0] == 1 and batch_size > 1:
            adj = adj.expand(batch_size, -1, -1)
        h = self.encoder(x, adj)
        return h.reshape(batch_size, -1)

    def actor_distribution(self, state: torch.Tensor, adj: Optional[torch.Tensor] = None) -> Dirichlet:
        h = self._encode(state, adj)
        log_concentration = self.actor_head(self.actor_trunk(h))
        # Clamp before exp to keep concentrations in a numerically safe range.
        log_concentration = torch.clamp(log_concentration, -5.0, 5.0)
        concentration = torch.exp(log_concentration) + 1e-3
        return Dirichlet(concentration)

    def value(self, state: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self._encode(state, adj)
        return self.critic_head(self.critic_trunk(h)).squeeze(-1)

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.actor_distribution(state, adj)
        # Stabilise against tiny / degenerate actions before scoring them.
        action = torch.clamp(action, min=1e-6)
        action = action / action.sum(dim=-1, keepdim=True)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value(state, adj)
        return log_prob, entropy, value


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.05
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 64


class RolloutBuffer:
    """Fixed-length rollout buffer for synchronous PPO."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.adjs: List[np.ndarray] = []

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: float,
        adj: Optional[np.ndarray] = None,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.adjs.append(adj if adj is not None else np.zeros((0,), dtype=np.float32))

    def __len__(self) -> int:
        return len(self.states)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.adjs.clear()


class GraphSAGEPPOAgent:
    """PPO agent built on a GraphSAGE state encoder."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        node_features_dim: int,
        num_nodes: int,
        adj_matrix,
        config: Optional[PPOConfig] = None,
        device: str = "cpu",
        encoder_hidden: int = 64,
        encoder_out: int = 64,
        head_hidden: int = 128,
        dropout: float = 0.1,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.node_features_dim = node_features_dim
        self.num_nodes = num_nodes
        self.config = config or PPOConfig()
        self.device = torch.device(device)

        if isinstance(adj_matrix, np.ndarray):
            adj_t = torch.tensor(adj_matrix, dtype=torch.float32)
        elif isinstance(adj_matrix, torch.Tensor):
            adj_t = adj_matrix.float()
        else:
            raise TypeError("adj_matrix must be a numpy array or torch.Tensor")

        # Match the encoder topology to the loaded checkpoint / data slice.
        adj_t = adj_t[:num_nodes, :num_nodes]

        self.policy = GraphSAGEPPOActorCritic(
            node_features_dim=node_features_dim,
            num_nodes=num_nodes,
            action_dim=action_dim,
            adj_matrix=adj_t,
            encoder_hidden=encoder_hidden,
            encoder_out=encoder_out,
            head_hidden=head_hidden,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.buffer = RolloutBuffer()
        self.update_count = 0

        logger.info(
            "GraphSAGE-PPO initialized: state_dim=%d action_dim=%d num_nodes=%d feat=%d",
            state_dim,
            action_dim,
            num_nodes,
            node_features_dim,
        )

    # ------------------------------------------------------------------
    # Interaction interface (kept signature-compatible with GNNSACAgent)
    # ------------------------------------------------------------------
    def select_action(
        self,
        state: np.ndarray,
        hidden=None,
        deterministic: bool = False,
        adj: Optional[torch.Tensor] = None,
        nextgen_ctx=None,
    ):
        """Return (action, hidden) for parity with GNNSACAgent.select_action.

        PPO is stateless, so ``hidden`` is ignored and the second return
        value is ``None``. ``nextgen_ctx`` is accepted and ignored so the
        evaluation loop can pass it without special-casing.
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        adj_t = self._prepare_adj(adj)
        with torch.no_grad():
            dist = self.policy.actor_distribution(state_t, adj_t)
            value = self.policy.value(state_t, adj_t)
            if deterministic:
                concentration = dist.concentration
                action = concentration / concentration.sum(dim=-1, keepdim=True)
                log_prob = dist.log_prob(action.clamp(min=1e-6))
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action)

        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        # Cache last-step diagnostics so train loops can read them without
        # re-running the policy.
        self._last_log_prob = float(log_prob.item())
        self._last_value = float(value.item())
        return action_np, None

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: float,
        adj: Optional[np.ndarray] = None,
    ) -> None:
        self.buffer.push(
            state=np.asarray(state, dtype=np.float32),
            action=np.asarray(action, dtype=np.float32),
            log_prob=getattr(self, "_last_log_prob", 0.0),
            value=getattr(self, "_last_value", 0.0),
            reward=float(reward),
            done=float(done),
            adj=np.asarray(adj, dtype=np.float32) if adj is not None else None,
        )

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def update(self, last_state: Optional[np.ndarray] = None, last_adj: Optional[np.ndarray] = None) -> Dict[str, float]:
        if len(self.buffer) == 0:
            return {}

        states = torch.tensor(np.stack(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.stack(self.buffer.actions), dtype=torch.float32, device=self.device)
        # Re-normalize stored actions to the simplex before scoring them.
        actions = actions.clamp(min=1e-6)
        actions = actions / actions.sum(dim=-1, keepdim=True)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        values = np.array(self.buffer.values, dtype=np.float32)

        # Bootstrap value for the final step.
        if last_state is not None:
            with torch.no_grad():
                last_state_t = torch.tensor(last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                last_adj_t = self._prepare_adj(last_adj)
                last_value = self.policy.value(last_state_t, last_adj_t).item()
        else:
            last_value = 0.0

        advantages, returns = self._compute_gae(rewards, values, dones, last_value)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Batched optional adjacency stack (only used when shape matches).
        adjs = self.buffer.adjs
        has_dyn_adj = all(a.size == self.num_nodes * self.num_nodes for a in adjs)
        if has_dyn_adj:
            adj_batch = torch.tensor(
                np.stack(adjs).reshape(-1, self.num_nodes, self.num_nodes),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            adj_batch = None

        cfg = self.config
        n_samples = states.shape[0]
        idx = np.arange(n_samples)
        last_metrics: Dict[str, float] = {}

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(idx)
            policy_losses, value_losses, entropies, kls = [], [], [], []
            for start in range(0, n_samples, cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]
                mb_idx = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_adv = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_adj = adj_batch[mb_idx] if adj_batch is not None else None

                log_prob, entropy, value = self.policy.evaluate(mb_states, mb_actions, mb_adj)
                ratio = torch.exp(log_prob - mb_old_log_probs)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(value, mb_returns)
                entropy_loss = -entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - log_prob).mean().item()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                kls.append(approx_kl)

            mean_kl = float(np.mean(kls)) if kls else 0.0
            last_metrics = {
                "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
                "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
                "entropy": float(np.mean(entropies)) if entropies else 0.0,
                "approx_kl": mean_kl,
                "epoch": epoch + 1,
            }
            if mean_kl > 1.5 * cfg.target_kl:
                # Early-stop epoch loop on KL blow-up — standard PPO practice.
                break

        self.buffer.clear()
        self.update_count += 1
        return last_metrics

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------
    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_value = last_value
        next_non_terminal = 1.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + cfg.gamma * next_value * next_non_terminal - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            next_value = values[t]
            next_non_terminal = 1.0 - dones[t]
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # Adjacency helper
    # ------------------------------------------------------------------
    def _prepare_adj(self, adj) -> Optional[torch.Tensor]:
        if adj is None:
            return None
        if isinstance(adj, np.ndarray):
            adj_t = torch.tensor(adj, dtype=torch.float32, device=self.device)
        elif isinstance(adj, torch.Tensor):
            adj_t = adj.float().to(self.device)
        else:
            raise TypeError("adj must be ndarray or torch.Tensor")
        if adj_t.dim() == 2:
            adj_t = adj_t.unsqueeze(0)
        return adj_t[:, : self.num_nodes, : self.num_nodes]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "node_features_dim": self.node_features_dim,
                "num_nodes": self.num_nodes,
                "update_count": self.update_count,
            },
            filepath,
        )
        logger.info("GraphSAGE-PPO model saved to %s", filepath)

    def load(self, filepath: str) -> None:
        ckpt = torch.load(filepath, map_location=self.device)
        if "policy" in ckpt:
            self.policy.load_state_dict(ckpt["policy"])
        elif "policy_state_dict" in ckpt:
            self.policy.load_state_dict(ckpt["policy_state_dict"])
        else:
            self.policy.load_state_dict(ckpt)
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                logger.warning("Optimizer state shape mismatch — skipping optimizer load.")
        self.update_count = ckpt.get("update_count", self.update_count)
        logger.info("GraphSAGE-PPO model loaded from %s", filepath)
