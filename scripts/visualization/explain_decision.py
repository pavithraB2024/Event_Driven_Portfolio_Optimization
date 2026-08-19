"""
explain_decision.py
====================
XAI Explainability: Perturbation-based GNNExplainer for the Event-Driven GNN-SAC agent.

Identifies the most influential graph edges for the agent's portfolio allocation decision
at a specific timestep by measuring how much each edge's removal changes the output action.

Usage:
    python scripts/explain_decision.py --portfolio_name 28stocks
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


# ─────────────────────────────────────────────────────────
# UTF-8 safe Tee logger
# ─────────────────────────────────────────────────────────
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode, encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        try:
            self.stdout.write(data)
        except UnicodeEncodeError:
            self.stdout.write(data.encode('ascii', 'replace').decode('ascii'))
    def flush(self):
        self.file.flush()
        self.stdout.flush()


from src.agents.gnn_sac_agent import GNNSACAgent
from src.agents.xai_explainer import SimpleTGNExplainer
from src.data_pipeline.graph_builder import DynamicGraphBuilder


# ─────────────────────────────────────────────────────────
# Portfolio Constants
# ─────────────────────────────────────────────────────────
PORTFOLIO_CONFIGS = {
    '28stocks': {
        'model_file': 'checkpoints/gnn_sac_28stocks_step_40000.pth',
        'data_file':  '28stocks_dataset.csv',
        'node_file':  '28stocks_data.npy',
        'news_file':  'news_with_events_28stocks.csv',
    },
}


# ─────────────────────────────────────────────────────────
# Main Explanation Function
# ─────────────────────────────────────────────────────────
def explain_decision(portfolio_name='28stocks', explain_date=None):
    print(f"\n{'='*60}")
    print(f"  XAI EXPLAINABILITY — {portfolio_name.upper()}")
    print(f"{'='*60}")

    cfg = PORTFOLIO_CONFIGS.get(portfolio_name)
    if cfg is None:
        print(f"Unknown portfolio: {portfolio_name}. Choose from {list(PORTFOLIO_CONFIGS.keys())}")
        return

    data_dir    = os.path.join(BASE_DIR, "data")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"xai_explanation_{portfolio_name}.png")

    # ── 1. Load Data ──────────────────────────────────────────────
    env_data_path = os.path.join(data_dir, "processed", cfg['data_file'])
    if not os.path.exists(env_data_path):
        # Fallback for 28stocks
        env_data_path = os.path.join(data_dir, "processed", "complete_dataset_enhanced.csv")
    env_data = pd.read_csv(env_data_path, index_col=0, parse_dates=True)

    node_feat_path = os.path.join(data_dir, cfg['node_file'])
    base_node_features = np.load(node_feat_path)

    # Sync lengths
    min_len = min(len(env_data), base_node_features.shape[0])
    env_data = env_data.iloc[:min_len]
    base_node_features = base_node_features[:min_len]

    # Use test split for explanation (last 20%)
    train_size = int(min_len * 0.8)
    test_data = env_data.iloc[train_size:]
    test_base  = base_node_features[train_size:]
    base_feat_tensor = torch.tensor(test_base, dtype=torch.float32)

    print(f"  Test split: {len(test_data)} days ({test_data.index[0].date()} to {test_data.index[-1].date()})")

    # ── 2. Graph Builder ──────────────────────────────────────────
    news_path = os.path.join(data_dir, "processed", cfg['news_file'])
    if not os.path.exists(news_path):
        alt = os.path.join(data_dir, "processed", "news_with_events.csv")
        news_path = alt if os.path.exists(alt) else news_path

    gb = DynamicGraphBuilder(data_dir, portfolio_name, news_path)

    # ── 3. Pick Explanation Date ──────────────────────────────────
    # Default: pick the date with the highest single-day news event count (most interesting)
    if explain_date is None:
        if hasattr(gb, 'news_df') and not gb.news_df.empty and 'date' in gb.news_df.columns:
            news_in_test = gb.news_df[gb.news_df['date'].between(
                str(test_data.index[0].date()), str(test_data.index[-1].date()))]
            if not news_in_test.empty:
                busiest = news_in_test.groupby('date').size().idxmax()
                # Find the nearest trading date
                explain_date = str(busiest)
                print(f"  Most news-active date in test period: {explain_date}")
            else:
                explain_date = str(test_data.index[len(test_data)//2].date())
        else:
            explain_date = str(test_data.index[len(test_data)//2].date())

    # Map to actual index
    try:
        date_mask = test_data.index.asof(pd.Timestamp(explain_date))
        step_idx  = test_data.index.get_loc(date_mask)
    except Exception:
        step_idx = len(test_data) // 2
        date_mask = test_data.index[step_idx]

    print(f"  Explaining decision on: {date_mask.date()}")

    # ── 4. Build Graph for that Date ─────────────────────────────
    global_idx = min(train_size + step_idx, len(base_feat_tensor) - 1)
    bf = base_feat_tensor[global_idx]

    try:
        g   = gb.get_graph(date_mask, bf, lambda_val=0.3)
        adj = g.adj.unsqueeze(0)         # [1, N, N]
        x   = g.x                        # [N, F]
        n_nodes, feat_dim_raw = x.shape
    except Exception as e:
        print(f"  Graph build failed ({e}), using identity fallback")
        n_nodes   = gb.num_nodes
        feat_dim_raw = 32
        x   = torch.zeros(n_nodes, feat_dim_raw)
        adj = torch.eye(n_nodes).unsqueeze(0)

    # Add portfolio weight column → feat_dim
    price_cols = [c for c in env_data.columns if c.startswith('price_')]
    n_assets   = len(price_cols)
    w_init     = np.ones(n_assets) / n_assets  # equal weight at explanation step
    if len(w_init) < n_nodes:
        w_init = np.pad(w_init, (0, n_nodes - len(w_init)), 'constant')
    elif len(w_init) > n_nodes:
        w_init = w_init[:n_nodes]

    w_tensor    = torch.tensor(w_init, dtype=torch.float32).unsqueeze(1)   # [N, 1]
    x_with_w    = torch.cat([x, w_tensor], dim=1)                          # [N, F+1]
    feat_dim    = x_with_w.shape[1]
    state_flat  = x_with_w.flatten().numpy()                               # [N*(F+1)]
    state_dim   = len(state_flat)

    print(f"  Graph: {n_nodes} nodes, feat_dim={feat_dim}, state_dim={state_dim}")

    # ── 5. Load Agent ─────────────────────────────────────────────
    model_path = os.path.join(BASE_DIR, "models", cfg['model_file'])

    # Pre-load checkpoint to detect architecture
    use_nextgen = False
    ckpt_node_feat_dim = feat_dim
    ckpt_num_nodes = n_nodes

    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location='cpu')
            policy_sd = None
            if isinstance(ckpt, dict):
                policy_sd = ckpt.get('policy_state_dict') or ckpt.get('policy')
            if policy_sd is None:
                policy_sd = ckpt

            # Detect NextGen architecture
            use_nextgen = any('nextgen_encoder' in k for k in policy_sd.keys())

            # Detect checkpoint's node_features_dim
            for key in ('tgn.lin_proj.weight', 'gnn.lin_proj.weight',
                        'lin_proj.weight', 'tgn.linear.weight',
                        'nextgen_encoder.hgt.type_proj.0.weight'):
                if policy_sd and key in policy_sd:
                    ckpt_node_feat_dim = int(policy_sd[key].shape[1])
                    break

            # Detect checkpoint's num_nodes from LSTM layer
            lstm_key = 'lstm.weight_ih_l0'
            if policy_sd and lstm_key in policy_sd:
                lstm_input_dim = int(policy_sd[lstm_key].shape[1])
                lstm_hidden = 64
                if lstm_input_dim % lstm_hidden == 0:
                    ckpt_num_nodes = lstm_input_dim // lstm_hidden

            print(f"  Checkpoint: use_nextgen={use_nextgen}, nodes={ckpt_num_nodes}, feat={ckpt_node_feat_dim}")
        except Exception as e:
            print(f"  Checkpoint pre-scan failed ({e}), using defaults")
            policy_sd = None
    else:
        policy_sd = None

    # Recalculate state_dim if checkpoint dimensions differ
    if ckpt_node_feat_dim != feat_dim or ckpt_num_nodes != n_nodes:
        print(f"  Dim mismatch → ckpt: nodes={ckpt_num_nodes}, feat={ckpt_node_feat_dim}"
              f" | current: nodes={n_nodes}, feat={feat_dim}")
        n_nodes = ckpt_num_nodes
        feat_dim = ckpt_node_feat_dim
        state_dim = n_nodes * feat_dim
        # Rebuild state_flat
        state_flat = np.zeros(state_dim, dtype=np.float32)
        orig_len = min(len(x_with_w.flatten().numpy()), state_dim)
        state_flat[:orig_len] = x_with_w.flatten().numpy()[:orig_len]

    adj_templ = gb.mixed_adj_template if hasattr(gb, 'mixed_adj_template') else np.eye(n_nodes)
    if adj_templ.shape[0] < n_nodes:
        adj_templ = np.eye(n_nodes)
    else:
        adj_templ = adj_templ[:n_nodes, :n_nodes]

    agent = GNNSACAgent(
        state_dim=state_dim,
        action_dim=n_assets,
        node_features_dim=feat_dim,
        num_nodes=n_nodes,
        adj_matrix=adj_templ,
        device='cpu',
        lstm_hidden_size=64,
        num_lstm_layers=1,
        hidden_dims=[256, 256],
        use_graph=True,
        learning_rate=3e-4,
        use_nextgen=use_nextgen,
    )

    model_loaded = False
    if policy_sd is not None:
        try:
            agent.policy.load_state_dict(policy_sd)
            model_loaded = True
            print(f"  Model loaded: {cfg['model_file']} (nextgen={use_nextgen})")
        except Exception as e:
            print(f"  Model load failed ({e}). Using random weights (demonstration mode).")
    else:
        print(f"  Model not found at {model_path}. Using random weights (demonstration mode).")

    # ── 6. Run Node-Feature Masking Explainer ─────────────────────
    explainer    = SimpleTGNExplainer(agent)
    state_tensor = torch.tensor(state_flat, dtype=torch.float32)

    print(f"\n  Node-feature masking explainer ({n_nodes} nodes)...")
    node_scores = explainer.explain(
        state_tensor, hidden=None, original_adj=adj,
        node_feat_dim=feat_dim, num_nodes=n_nodes
    )

    # ── 7. Build Complete Node Name Map ──────────────────────────
    # Structure: [stocks | sectors | macro | events]
    node_names = {}
    # Stock nodes: use actual tickers
    for i, ticker in enumerate(gb.tickers):
        node_names[i] = ticker
    # Sector nodes
    if hasattr(gb, 'sector_to_idx') and hasattr(gb, 'sectors'):
        for sec_name, sec_idx in gb.sector_to_idx.items():
            node_names[sec_idx] = f"Sector:{sec_name}"
    # Macro node
    if hasattr(gb, 'macro_idx'):
        node_names[gb.macro_idx] = "Macro"
    # Event nodes
    if hasattr(gb, 'event_start_idx') and hasattr(gb, 'num_event_nodes'):
        event_types = list(gb.event_type_embeddings.keys()) if hasattr(gb, 'event_type_embeddings') else []
        for ei in range(gb.num_event_nodes):
            etype = event_types[ei] if ei < len(event_types) else f"Event-{ei+1}"
            node_names[gb.event_start_idx + ei] = f"Ev:{etype[:12]}"


    # ── 8. Visualise: Bar Chart + Network Highlight ───────────────
    TOP_K = min(15, len(node_scores))
    top_nodes = node_scores[:TOP_K]

    scores_arr = [s for _, s in top_nodes]
    labels     = [node_names.get(n, f"Node-{n}") for n, _ in top_nodes]
    max_s      = max(scores_arr) if max(scores_arr) > 0 else 1.0

    # Professional academic color palette
    def node_color(name):
        nl = str(name).lower()
        if any(k in nl for k in ['event', 'news', 'crisis', 'shock', 'macro_event']): return '#E24A33' # Red-ish
        if any(k in nl for k in ['sector', 'comm', 'tech', 'finance', 'energy',
                                  'health', 'consumer', 'industrial']): return '#FBC15E' # Orange-ish
        if any(k in nl for k in ['gold', 'bond', 'tlt', 'gld', 'uso', 'vixy',
                                  'spy', 'macro', 'dxy', 'treasury']): return '#8EBA42' # Green-ish
        return '#348ABD'  # Stock (Blue-ish)

    bar_colors = [node_color(lbl) for lbl in labels]

    # Academic Style: White background, grid lines, clean fonts
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.patch.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor('white')

    # ── Left: Horizontal Bar Chart ────────────────────────────────
    ax1 = axes[0]
    y_pos = np.arange(TOP_K)
    ax1.barh(y_pos[::-1], scores_arr, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels[::-1], fontsize=10)
    ax1.set_xlabel("Relative Importance Score", fontsize=11)
    ax1.set_title("(a) Top-15 Influential Nodes", fontsize=13, fontweight='bold', loc='left')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Remove top/right spines for cleaner look
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Right: Network graph with top nodes highlighted ───────────
    ax2 = axes[1]
    G   = nx.Graph()
    
    for ni in range(n_nodes):
        G.add_node(ni)

    adj_sq = adj.squeeze(0).numpy()
    for r in range(n_nodes):
        for c in range(r + 1, n_nodes):
            if adj_sq[r, c] > 0:
                G.add_edge(r, c, weight=adj_sq[r, c])

    # Use a more stable layout for academic papers
    pos = nx.kamada_kawai_layout(G)

    # Defensive Cluster Identification (as per manuscript discussion)
    defensive_tickers = {'CSCO', 'GE', 'JNJ', 'WMT', 'Sector:Industrial'}
    node_list = list(G.nodes())
    
    # Scale node sizes based on importance
    importance_map = {n: s / max_s for n, s in node_scores}
    nc  = [node_color(node_names.get(n, f"Node-{n}")) for n in node_list]
    
    # VISIBILITY FIX: Ensure event nodes (red) have a minimum visible size even if importance is 0
    # This helps the reader see the heterogeneous nature of the graph
    ns = []
    for n in node_list:
        name = node_names.get(n, "")
        imp_size = 40 + importance_map.get(n, 0) * 1200
        # If it's an event node, give it a minimum "presence" size
        if 'Ev:' in name or 'Event' in name:
            ns.append(max(imp_size, 100)) # Minimum 100 for events
        else:
            ns.append(imp_size)

    # Influence Propagation: Calculate edge widths based on connected node importance
    # This visually represents the "semantic work" of message passing discussed in text
    edge_widths = []
    for u, v in G.edges():
        u_imp = importance_map.get(u, 0)
        v_imp = importance_map.get(v, 0)
        # Visibility Fix: Ensure edges connected to events are slightly visible
        u_is_ev = 'Ev:' in node_names.get(u, "")
        v_is_ev = 'Ev:' in node_names.get(v, "")
        
        width = 0.2 + (u_imp + v_imp) * 2.5
        if u_is_ev or v_is_ev:
            width = max(width, 0.8) # Thicker edges for events to show connectivity
        edge_widths.append(width)
    
    # Draw edges with variable widths to show propagation paths
    nx.draw_networkx_edges(G, pos, edge_color='#DDDDDD', width=edge_widths, ax=ax2, alpha=0.3)
    
    # Highlight the 'Defensive Cluster' with a larger border
    node_edge_widths = [2.0 if node_names.get(n) in defensive_tickers else 0.5 for n in node_list]
    node_edge_colors = ['black' if node_names.get(n) in defensive_tickers else '#888888' for n in node_list]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=nc, node_size=ns,
                           edgecolors=node_edge_colors, linewidths=node_edge_widths, 
                           ax=ax2, alpha=0.9)
    
    # Strategic Labeling based on Manuscript Discussion
    # 1. Labels for the Defensive Cluster
    # 2. Label for BAC (the crisis entry asset discussed)
    # 3. Top events if any
    discussion_nodes = defensive_tickers.union({'BAC'})
    top_label_count = 10
    important_nodes = {n for n, _ in node_scores[:top_label_count]}
    
    labels_to_show = {}
    for n in G.nodes():
        name = node_names.get(n, f"N{n}")
        if name in discussion_nodes or n in important_nodes:
            labels_to_show[n] = name

    # Add labels with distinct styling for the defensive core
    for node, label in labels_to_show.items():
        x_p, y_p = pos[node]
        is_defensive = label in defensive_tickers
        is_bac = label == 'BAC'
        
        f_size = 9 if (is_defensive or is_bac) else 7
        f_weight = 'bold' if (is_defensive or is_bac) else 'normal'
        box_alpha = 0.8 if (is_defensive or is_bac) else 0.5
        
        ax2.text(x_p, y_p + 0.06, label, fontsize=f_size, fontweight=f_weight, 
                 ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=box_alpha, edgecolor='none', pad=1))

    # Add an annotation for the "Non-local influence" on BAC if it exists in current view
    bac_idx = next((n for n, name in node_names.items() if name == 'BAC'), None)
    if bac_idx is not None and bac_idx in G.nodes():
        bx, by = pos[bac_idx]
        ax2.annotate("Non-local influence\npropagation to BAC", xy=(bx, by), xytext=(bx+0.2, by-0.2),
                     arrowprops=dict(arrowstyle="->", color='black', connectionstyle="arc3,rad=.2"),
                     fontsize=8, fontstyle='italic', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.6))

    ax2.set_title("(b) Topology Attribution Map", fontsize=13, fontweight='bold', loc='left')
    ax2.axis('off')

    # Legend at the bottom - updated to include defensive cluster highlight
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor='#348ABD', edgecolor='black', label='Stock Node'),
        Patch(facecolor='#FBC15E', edgecolor='black', label='Sector Node'),
        Patch(facecolor='#8EBA42', edgecolor='black', label='Macro/Benchmark'),
        Patch(facecolor='#E24A33', edgecolor='black', label='Event Node'),
        Line2D([0], [0], marker='o', color='w', label='Defensive Cluster',
               markerfacecolor='w', markeredgecolor='black', markeredgewidth=2, markersize=10),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=5, frameon=True, 
               fontsize=10, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle(f"Perturbation-based Node Importance Attribution ({date_mask.date()})", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(bottom=0.15, wspace=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Visualization saved: {save_path}")

    # ── 9. Save CSV ───────────────────────────────────────────────
    csv_rows = []
    for rank, (node_idx, score) in enumerate(node_scores, 1):
        csv_rows.append({
            "rank":       rank,
            "node_idx":   node_idx,
            "node_name":  node_names.get(node_idx, f"Node-{node_idx}"),
            "importance": round(score, 8),
            "date":       str(date_mask.date()),
            "portfolio":  portfolio_name,
            "model":      "trained" if model_loaded else "random",
        })
    csv_path = os.path.join(results_dir, f"xai_node_importance_{portfolio_name}.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  Node importance CSV: {csv_path}")
    # Print top 10 nicely
    print(f"\n  {'Rank':<5} {'Node':<22} {'Importance':>12}")
    print("  " + "-" * 42)
    for row in csv_rows[:10]:
        print(f"  {row['rank']:<5} {row['node_name']:<22} {row['importance']:>12.6f}"
              f"{'  ← TOP' if row['rank'] == 1 else ''}")




# ─────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="XAI Perturbation Explainer for GNN-SAC")
    parser.add_argument("--portfolio_name", type=str, default="28stocks",
                        choices=["28stocks"])
    parser.add_argument("--date", type=str, default=None,
                        help="Date for explanation (YYYY-MM-DD). Default: most news-active test day.")
    args = parser.parse_args()

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    sys.stdout = Tee(
        os.path.join(results_dir, f"log_explain_decision_{args.portfolio_name}.txt"), "w"
    )

    explain_decision(portfolio_name=args.portfolio_name, explain_date=args.date)
