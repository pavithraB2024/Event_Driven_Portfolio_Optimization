"""
xai_explainer.py
=================
Node-Feature Masking explainer for the Event-Driven GNN-SAC agent.

Strategy: Node Feature Masking (Occlusion)
  For each node in the graph:
    1. Zero out ALL features of that node in the state tensor
    2. Compute the agent's new action under the masked state
    3. Measure L2 deviation from the original (unmasked) action
    4. Higher deviation = that node was more important for the decision

Why node masking vs edge perturbation?
  - Edge perturbation on dense graphs: removing ONE edge from 100+ edges has
    near-zero effect (information re-routes through other paths). Impact = 0.0.
  - Node feature masking: removes ALL information from that node completely.
    Much larger, cleanly-interpretable impact score.
  - Directly answers the paper's research question: "Which stocks/events/sectors
    does the agent attend to most when making allocation decisions?"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


class SimpleTGNExplainer:
    """
    Node-Feature-Masking explainer for GNNSACAgent.

    Measures how much each node's features influence the portfolio allocation
    by selectively zeroing out one node at a time and measuring the output shift.
    """

    def __init__(self, agent):
        self.agent     = agent
        self.device    = agent.device

    def explain(self, state, hidden, original_adj, target_node_idx=None,
                node_feat_dim=None, num_nodes=None):
        """
        Run node-feature-masking analysis.

        Args:
            state            : State tensor [state_dim] or [1, 1, state_dim]
            hidden           : LSTM hidden state (or None)
            original_adj     : Adjacency [1, N, N] or [N, N]
            target_node_idx  : Not used (kept for API compatibility)
            node_feat_dim    : Feature dimension per node (auto-detected if None)
            num_nodes        : Number of nodes (auto-detected if None)

        Returns:
            node_scores: List of (node_idx, impact_score) sorted descending.
        """
        self.agent.policy.eval()

        # ── Prepare state ────────────────────────────────────────────
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        state = state.to(self.device).float()
        if state.dim() == 1:
            state_3d = state.unsqueeze(0).unsqueeze(0)   # [1, 1, D]
        elif state.dim() == 2:
            state_3d = state.unsqueeze(0)
        else:
            state_3d = state
        state_dim = state_3d.shape[-1]

        # ── Prepare adjacency ────────────────────────────────────────
        if not isinstance(original_adj, torch.Tensor):
            original_adj = torch.FloatTensor(original_adj)
        original_adj = original_adj.to(self.device)
        if original_adj.dim() == 2:
            original_adj = original_adj.unsqueeze(0)

        # ── Auto-detect node count and feature dim ───────────────────
        if num_nodes is None:
            num_nodes = original_adj.shape[-1]
        if node_feat_dim is None:
            if state_dim % num_nodes == 0:
                node_feat_dim = state_dim // num_nodes
            else:
                # Fallback: single-node treatment
                print(f"  Warning: state_dim={state_dim} not divisible by num_nodes={num_nodes}. "
                      f"Using single-node fallback.")
                num_nodes     = 1
                node_feat_dim = state_dim

        # ── Baseline prediction ──────────────────────────────────────
        with torch.no_grad():
            policy_out = self.agent.policy(state_3d, hidden, adj=original_adj)
            # Handle both old 3-return and new 5-return (sector_mean, sector_log_std, stock_mean, stock_log_std, hidden)
            if len(policy_out) == 5:
                base_mean = policy_out[2]  # stock_mean as action proxy
            else:
                base_mean = policy_out[0]

        base_std = base_mean.std().item()
        print(f"  Policy output std (baseline): {base_std:.6f}")
        if base_std < 1e-6:
            print("  WARNING: Model outputs near-constant actions. "
                  "Did the model load correctly? Results will be meaningless.")

        # ── Node feature masking loop ────────────────────────────────
        print(f"  Running node masking: {num_nodes} nodes × {node_feat_dim} features = state_dim {state_dim}")
        node_importances = []

        state_flat = state_3d.view(-1).clone()

        for n in range(num_nodes):
            start = n * node_feat_dim
            end   = start + node_feat_dim

            # Zero out this node's features
            masked_flat = state_flat.clone()
            masked_flat[start:end] = 0.0
            masked_3d = masked_flat.view(1, 1, -1)

            with torch.no_grad():
                pert_out = self.agent.policy(masked_3d, hidden, adj=original_adj)
                if len(pert_out) == 5:
                    pert_mean = pert_out[2]
                else:
                    pert_mean = pert_out[0]

            impact = torch.norm(base_mean - pert_mean).item()
            node_importances.append((n, impact))

        # Sort by impact descending
        node_importances.sort(key=lambda x: x[1], reverse=True)

        top_impact = node_importances[0][1] if node_importances else 0.0
        mean_impact = np.mean([s for _, s in node_importances])
        print(f"  Node masking complete. Top impact={top_impact:.6f}, Mean={mean_impact:.6f}")

        return node_importances

    def explain_edges(self, state, hidden, original_adj):
        """
        Edge perturbation (original approach).
        Only useful on sparse graphs. On dense graphs all scores will be ~0.
        Included for completeness / sparse-graph scenarios.
        """
        self.agent.policy.eval()

        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)

        if not isinstance(original_adj, torch.Tensor):
            original_adj = torch.FloatTensor(original_adj)
        if original_adj.dim() == 2:
            original_adj = original_adj.unsqueeze(0)
        original_adj = original_adj.to(self.device)

        with torch.no_grad():
            policy_out = self.agent.policy(state, hidden, adj=original_adj)
            if len(policy_out) == 5:
                base_mean = policy_out[2]
            else:
                base_mean = policy_out[0]

        adj_sq = original_adj.squeeze(0)
        rows, cols = torch.nonzero(adj_sq, as_tuple=True)
        n_edges = len(rows)
        print(f"  Found {n_edges} active edges. Running edge perturbation...")

        edge_importances = []
        for i in range(n_edges):
            r, c = rows[i].item(), cols[i].item()
            if r == c:
                continue
            pert_adj = original_adj.clone()
            pert_adj[0, r, c] = 0.0
            with torch.no_grad():
                pert_mean, _, _ = self.agent.policy(state, hidden, adj=pert_adj)
            impact = torch.norm(base_mean - pert_mean).item()
            edge_importances.append(((r, c), impact))

        edge_importances.sort(key=lambda x: x[1], reverse=True)
        return edge_importances

    def visualize_subgraph(self, node_importances, node_names=None, top_k=10,
                           save_path="explanation.png"):
        """
        Visualize top-k most important nodes as a bar chart + network highlight.
        """
        if node_names is None:
            node_names = {}

        scores  = [s for _, s in node_importances[:top_k]]
        labels  = [node_names.get(n, f"Node-{n}") for n, _ in node_importances[:top_k]]
        max_s   = max(scores) if scores else 1.0
        colors  = [plt.cm.Reds(0.3 + 0.7 * (s / max_s)) for s in scores]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(len(labels))[::-1], scores[::-1], color=colors[::-1])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels[::-1], fontsize=9)
        ax.set_xlabel("Node Importance (L2 action shift on masking)", fontsize=10)
        ax.set_title(f"Top-{top_k} Most Influential Nodes — GNN-SAC XAI", fontsize=12, fontweight='bold')

        print(f"\n  Top-{top_k} Most Influential Nodes:")
        for rank, (n, s) in enumerate(node_importances[:top_k], 1):
            name = node_names.get(n, f"Node-{n}")
            print(f"  [{rank:2d}] {name:<22} | importance={s:.6f}"
                  f"{'  * TOP' if rank == 1 else ''}")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
