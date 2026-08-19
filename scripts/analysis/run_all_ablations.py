"""
run_all_ablations.py
====================
Unified ablation study producing all ablation outputs required for the paper.

STUDY 1 — Architecture Ladder (Table 2A)
  Proves C1: Dynamic FinBERT graph > static correlation > no graph
  EXP-0-MLP       : Standard SAC (MLP, no memory, no graph)
  EXP-1-VECTOR    : Memory-SAC (LSTM, no graph)
  EXP-2-STATIC    : GNN-SAC + Static correlation graph (lambda=0, no news)
  EXP-5-NEXTGEN   : Full NextGen (FinBERT + IQN + CVaR + Gating)

STUDY 2 — Component Isolation (Table 2B)
  Proves C2 + C3: which NextGen components help vs hurt
  baseline        : GNN-SAC, plain SAC critics (use_nextgen=False)
  iqn_only        : NextGen IQN critic only
  momask          : NextGen + Momentum Masking
  attgate         : NextGen + Attention Gating
  full_stack      : All components enabled

STUDY 3 — FinBERT Trajectory (Section 4.3 text)
  Proves C1 at trajectory level: keyword policy collapses, FinBERT stabilises
  EXP-3-KEYWORDS  : binary sentiment (sentiment_scalar=1.0)
  EXP-4-FULL      : real signed FinBERT scores

Outputs:
  results/ablation_architecture_28stocks.csv   <- Table 2A
  results/ablation_components_28stocks.csv     <- Table 2B
  results/finbert_impact_results_28stocks.csv  <- Section 4.3
  results/ablation_run_log.txt                 <- Full log

Usage:
  python scripts/analysis/run_all_ablations.py                        # all studies, 1000 steps
  python scripts/analysis/run_all_ablations.py --steps 2000           # quick test
  python scripts/analysis/run_all_ablations.py --steps 30000          # paper-quality
  python scripts/analysis/run_all_ablations.py --study 1              # run only study 1
"""

import sys
import os
import time
import types
import torch
import numpy as np
import pandas as pd
import gymnasium as gym
from tqdm import tqdm
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.data_pipeline.graph_builder import DynamicGraphBuilder
from src.agents.gnn_sac_agent import GNNSACAgent
from src.agents.memory_augmented_sac_agent import MemoryAugmentedSACAgent
from src.agents.sac_agent import SACAgent
from src.environments.portfolio_env import PortfolioEnv

# Force CPU — GPU (1.95 GiB) is too small for sequential multi-agent ablation runs
DEVICE = "cpu"
print(f"[Device] Using: {DEVICE.upper()}")

torch.manual_seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, path, mode="w"):
        self.file = open(path, mode, encoding="utf-8")
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
            self.stdout.write(data.encode("ascii", "replace").decode("ascii"))

    def flush(self):
        self.file.flush()
        self.stdout.flush()


class RunningMeanStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x):
        self.count += 1
        d = float(x) - self.mean
        self.mean += d / self.count
        self.var = self.var * (1 - 1 / self.count) + (d ** 2) / self.count

    def normalize(self, x):
        return (x - self.mean) / max(np.sqrt(self.var), 1e-8)

    def reset(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4


def reset_seed():
    torch.manual_seed(42)
    np.random.seed(42)


def patch_env(env, features_np):
    """Patch PortfolioEnv to serve flattened [node_feats | weight] as observation."""
    env.graph_features = features_np

    def get_graph_state(self_env):
        idx = min(self_env.current_step + self_env.window_size, len(self_env.graph_features) - 1)
        feats = self_env.graph_features[idx]
        w = self_env.portfolio_weights
        N = feats.shape[0]
        if len(w) < N:
            w = np.pad(w, (0, N - len(w)), "constant")
        elif len(w) > N:
            w = w[:N]
        return np.hstack([feats, w.reshape(-1, 1)]).flatten().astype(np.float32)

    env._get_state = types.MethodType(get_graph_state, env)
    feat_dim = features_np.shape[2] + 1
    n_nodes  = features_np.shape[1]
    env.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_nodes * feat_dim,), np.float32)
    return feat_dim, n_nodes


def compute_metrics(portfolio_values):
    pv   = np.array(portfolio_values)
    roi  = (pv[-1] - pv[0]) / pv[0] * 100.0
    rets = pd.Series(pv).pct_change().dropna()
    sharpe = float((rets.mean() / rets.std()) * np.sqrt(252)) if rets.std() > 1e-9 else 0.0
    cummax = np.maximum.accumulate(pv)
    mdd    = float(((pv / cummax) - 1).min() * 100.0)
    return {"sharpe": round(sharpe, 4), "roi": round(roi, 2), "mdd": round(mdd, 2)}


def save_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nSaved: {path}")


def print_table(title, rows, cols):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")
    header = " | ".join(f"{c:<30}" if i == 0 else f"{c:>9}" for i, c in enumerate(cols))
    print(header)
    print("-" * 75)
    for r in rows:
        line = " | ".join(f"{str(r[c]):<30}" if i == 0 else f"{r[c]:>9}" for i, c in enumerate(cols))
        print(line)
    print("=" * 75)


def build_graph_features(gb, base_feats, dates, label, device="cpu"):
    """
    Rebuild graph features from scratch using current gb.news_df state.
    Must be called AFTER any news_df mutation so the mutation is reflected.
    Graph tensors are kept on CPU (numpy arrays returned).
    """
    feats_list, adjs_list = [], []
    # Keep on CPU — .numpy() requires CPU tensors
    bf_t = torch.tensor(base_feats, dtype=torch.float32)
    for t in tqdm(range(len(dates)), desc=f"  Graphs ({label})", leave=False):
        try:
            g = gb.get_graph(dates[t], bf_t[t])
            # Ensure tensors are on CPU before converting to numpy
            feats_list.append(g.x.cpu().numpy())
            adjs_list.append(g.adj.cpu().numpy())
        except Exception:
            feats_list.append(np.zeros((gb.num_nodes, base_feats.shape[2])))
            adjs_list.append(np.eye(gb.num_nodes))
    return np.array(feats_list), np.array(adjs_list)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading (shared once across all studies)
# ─────────────────────────────────────────────────────────────────────────────

def load_shared_data(portfolio_name, data_dir):
    print(f"\nLoading shared data for {portfolio_name}...")

    csv_path = os.path.join(data_dir, "processed", f"{portfolio_name}_dataset.csv")
    env_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    npy_path = os.path.join(data_dir, f"{portfolio_name}_data.npy")
    base_feats = np.load(npy_path)

    # Normalise features
    mu = np.nanmean(base_feats, axis=(0, 1), keepdims=True)
    sd = np.nanstd(base_feats,  axis=(0, 1), keepdims=True) + 1e-6
    base_feats = np.clip((base_feats - mu) / sd, -5, 5)

    min_len    = min(len(env_data), base_feats.shape[0])
    env_data   = env_data.iloc[:min_len]
    base_feats = base_feats[:min_len]

    train_size = int(min_len * 0.8)
    train_data = env_data.iloc[:train_size]
    test_data  = env_data.iloc[train_size:]
    dates      = env_data.index

    train_base = base_feats[:train_size]
    test_base  = base_feats[train_size:]

    # Cached graph features (FinBERT-enriched, used for EXP-5-NEXTGEN only)
    cache_path = os.path.join(data_dir, f"processed/{portfolio_name}_precomputed_graphs.npz")
    cache = np.load(cache_path)
    all_feats = cache["features"][:min_len]
    all_adjs  = cache["adjs"][:min_len]
    cached_train_feats = all_feats[:train_size]
    cached_train_adjs  = all_adjs[:train_size]
    cached_test_feats  = all_feats[train_size:]
    cached_test_adjs   = all_adjs[train_size:]

    # Graph builder
    news_path = os.path.join(data_dir, "processed", f"news_with_events_{portfolio_name}.csv")
    gb = DynamicGraphBuilder(data_dir, portfolio_name, news_path)
    n_assets = len([c for c in env_data.columns if c.startswith("price_")])
    if gb.num_stock_nodes > n_assets:
        gb.resize_to_match_portfolio(n_assets)

    print(f"  Train: {len(train_data)} | Test: {len(test_data)} | Nodes: {gb.num_nodes} | Assets: {n_assets}")
    return {
        "train_data": train_data, "test_data": test_data,
        # Cached FinBERT-enriched features — only valid for EXP-5-NEXTGEN
        "cached_train_feats": cached_train_feats, "cached_train_adjs": cached_train_adjs,
        "cached_test_feats":  cached_test_feats,  "cached_test_adjs":  cached_test_adjs,
        # Raw base features — used to rebuild graphs with mutations
        "train_base": train_base, "test_base": test_base,
        "dates": dates, "train_size": train_size,
        "graph_builder": gb, "n_assets": n_assets,
        "data_dir": data_dir,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STUDY 1 — Architecture Ladder
# ─────────────────────────────────────────────────────────────────────────────

ARCH_EXPERIMENTS = {
    "EXP-0-MLP":     {"desc": "Standard SAC (MLP, no memory)", "use_graph": False, "use_lstm": False, "static": False, "finbert": False},
    "EXP-1-VECTOR":  {"desc": "Memory-SAC (LSTM, no graph)",    "use_graph": False, "use_lstm": True,  "static": False, "finbert": False},
    "EXP-2-STATIC":  {"desc": "GNN-SAC + Static Graph (λ=0)",  "use_graph": True,  "use_lstm": True,  "static": True,  "finbert": False},
    "EXP-5-NEXTGEN": {"desc": "GNN-SAC + FinBERT + NextGen",   "use_graph": True,  "use_lstm": True,  "static": False, "finbert": True,  "use_nextgen": True},
}


def run_arch_experiment(exp_id, shared, train_steps):
    reset_seed()

    cfg      = ARCH_EXPERIMENTS[exp_id]
    gb       = shared["graph_builder"]
    # Study 1 always runs on CPU.
    device = "cpu"

    print(f"\n{'='*60}")
    print(f"  [Study 1] {exp_id}: {cfg['desc']}")
    print(f"  Device: {device.upper()}")
    print(f"{'='*60}")

    train_data = shared["train_data"]
    test_data  = shared["test_data"]
    n_assets   = shared["n_assets"]

    if cfg["use_graph"]:
        real_news    = gb.news_df.copy()
        train_dates  = shared["dates"][:shared["train_size"]]
        test_dates   = shared["dates"][shared["train_size"]:]

        if cfg.get("use_nextgen", False) and cfg["finbert"]:
            # EXP-5-NEXTGEN: use pre-cached FinBERT features (same as full training)
            train_feats = shared["cached_train_feats"]
            train_adjs  = shared["cached_train_adjs"]
            test_feats  = shared["cached_test_feats"]
            test_adjs   = shared["cached_test_adjs"]
        else:
            # Reset temporal memory before rebuilding so Study 1 experiments
            # do not contaminate each other or Study 3 experiments.
            gb.event_memory = np.zeros(gb.num_stock_nodes)
            gb.sector_memory = np.zeros(gb.num_sector_nodes)
            # EXP-2-STATIC: empty news → purely static correlation graph
            # Mutate THEN rebuild so mutation actually takes effect
            if cfg["static"]:
                gb.news_df = pd.DataFrame(columns=real_news.columns)
            elif not cfg["finbert"]:
                # Only sentiment_scalar is read by graph_builder.get_graph()
                if "sentiment_scalar" in gb.news_df.columns:
                    gb.news_df["sentiment_scalar"] = 1.0

            train_feats, train_adjs = build_graph_features(gb, shared["train_base"], train_dates, "train")
            test_feats,  test_adjs  = build_graph_features(gb, shared["test_base"],  test_dates,  "test")
            gb.news_df = real_news  # Restore AFTER rebuilding

        env_train = PortfolioEnv(train_data, initial_balance=100000.0, transaction_cost=0.0025)
        feat_dim, n_nodes = patch_env(env_train, train_feats)
        env_train.dynamic_adjs = train_adjs

        env_test = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
        patch_env(env_test, test_feats)
        env_test.dynamic_adjs = test_adjs
    else:
        # Flat return features for non-graph variants (MLP, LSTM)
        ret_cols    = [c for c in train_data.columns if c.startswith("return_")]
        tr_flat     = train_data[ret_cols].fillna(0).values[:, np.newaxis, :]
        te_flat     = test_data[ret_cols].fillna(0).values[:, np.newaxis, :]
        train_feats = tr_flat
        test_feats  = te_flat
        train_adjs  = np.eye(1)[np.newaxis].repeat(len(tr_flat), axis=0)
        test_adjs   = np.eye(1)[np.newaxis].repeat(len(te_flat), axis=0)

        env_train = PortfolioEnv(train_data, initial_balance=100000.0, transaction_cost=0.0025)
        feat_dim, n_nodes = patch_env(env_train, train_feats)
        env_train.dynamic_adjs = train_adjs

        env_test = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
        patch_env(env_test, test_feats)
        env_test.dynamic_adjs = test_adjs

    state_dim = env_train.observation_space.shape[0]

    try:
        if cfg["use_graph"]:
            agent = GNNSACAgent(
                state_dim=state_dim, action_dim=n_assets,
                node_features_dim=feat_dim, num_nodes=n_nodes,
                adj_matrix=gb.mixed_adj_template, device=device,
                lstm_hidden_size=32, num_lstm_layers=1,
                hidden_dims=[128, 128], use_graph=True,
                use_nextgen=cfg.get("use_nextgen", False),
                learning_rate=1e-4, batch_size=4,
            )
        elif cfg.get("use_lstm", True):
            agent = MemoryAugmentedSACAgent(
                state_dim=state_dim, action_dim=n_assets,
                lstm_hidden_size=32, num_lstm_layers=1,
                hidden_dims=[128, 128], device=device,
                learning_rate=1e-4, batch_size=4,
            )
        else:
            agent = SACAgent(
                state_dim=state_dim, action_dim=n_assets,
                hidden_dims=[128, 128], device=device,
                learning_rate=1e-4, batch_size=4,
            )
    except Exception as e:
        print(f"  Agent init failed: {e}")
        return {"sharpe": 0.0, "roi": 0.0, "mdd": 0.0}

    # Train
    n_random = min(200, train_steps // 5)
    rms      = RunningMeanStd()
    state, _ = env_train.reset(seed=42)
    hidden   = None

    for step in tqdm(range(train_steps), desc=f"  Train {exp_id}"):
        idx   = min(env_train.current_step + env_train.window_size, len(train_adjs) - 1)
        adj_t = torch.tensor(train_adjs[idx], dtype=torch.float32).unsqueeze(0).to(device)

        if step < n_random:
            action, hidden = env_train.action_space.sample(), None
        elif cfg["use_graph"]:
            action, hidden = agent.select_action(state, hidden, adj=adj_t)
        elif cfg.get("use_lstm", True):
            action, hidden = agent.select_action(state, hidden)
        else:
            action = agent.select_action(state, deterministic=False)
            hidden = None

        next_state, reward, term, trunc, _ = env_train.step(action)
        rms.update(reward)
        # Clip + normalize reward (consistent with full training pipeline)
        norm_reward = np.clip(rms.normalize(reward), -10, 10)
        agent.replay_buffer.push(state, action, norm_reward, next_state, float(term or trunc))
        state = next_state

        if step >= n_random and len(agent.replay_buffer) >= agent.batch_size:
            # nextgen_ctx only accepted by GNNSACAgent
            if cfg["use_graph"]:
                agent.update(nextgen_ctx={})
            else:
                agent.update()
        if term or trunc:
            state, _ = env_train.reset()
            hidden = None
            rms.reset()  # Reset normalizer at episode boundary

    # Evaluate
    state, _ = env_test.reset(seed=42)
    hidden, done = None, False
    vals = [100000.0]
    while not done:
        idx   = min(env_test.current_step + env_test.window_size, len(test_adjs) - 1)
        adj_t = torch.tensor(test_adjs[idx], dtype=torch.float32).unsqueeze(0).to(device)
        if cfg["use_graph"]:
            action, hidden = agent.select_action(state, hidden, deterministic=True, adj=adj_t)
        elif cfg.get("use_lstm", True):
            action, hidden = agent.select_action(state, hidden, deterministic=True)
        else:
            action = agent.select_action(state, deterministic=True)
            hidden = None
        state, _, term, trunc, info = env_test.step(action)
        done = term or trunc
        vals.append(info.get("portfolio_value", vals[-1]))

    m = compute_metrics(vals)
    print(f"  Result: Sharpe={m['sharpe']:.4f}  ROI={m['roi']:.2f}%  MDD={m['mdd']:.2f}%")
    del agent
    return m


def run_study1(shared, steps, portfolio_name, results_dir):
    print("\n" + "="*75)
    print("  STUDY 1 — Architecture Ladder")
    print("="*75)
    results, rows = {}, []

    for exp_id, cfg in ARCH_EXPERIMENTS.items():
        try:
            m = run_arch_experiment(exp_id, shared, steps)
            results[exp_id] = m
        except Exception as e:
            print(f"  FAILED {exp_id}: {e}")
            import traceback; traceback.print_exc()
            results[exp_id] = m = {"sharpe": 0.0, "roi": 0.0, "mdd": 0.0}
        rows.append({"Configuration": cfg["desc"], "Sharpe": m["sharpe"],
                     "ROI (%)": m["roi"], "MDD (%)": m["mdd"]})

    print_table("STUDY 1 — Architecture Ladder Results", rows,
                ["Configuration", "Sharpe", "ROI (%)", "MDD (%)"])

    print("\n  Delta Sharpe (vs previous ablation step):")
    exp_list    = list(ARCH_EXPERIMENTS.keys())
    prev_sharpe = results[exp_list[0]]["sharpe"]
    delta_labels = {
        "EXP-1-VECTOR":  "+ LSTM Temporal Memory",
        "EXP-2-STATIC":  "+ Static Correlation Graph",
        "EXP-5-NEXTGEN": "+ FinBERT + NextGen IQN + Gating",
    }
    for exp_id in exp_list[1:]:
        d = results[exp_id]["sharpe"] - prev_sharpe
        print(f"  {delta_labels.get(exp_id, exp_id):<45} {d:>+.4f}")
        prev_sharpe = results[exp_id]["sharpe"]

    print("\n[LaTeX — Table 2A: Architecture Ladder]")
    print(r"\begin{table}[h]\centering")
    print(r"\caption{Architecture Ablation: Progressive Component Contribution}")
    print(r"\label{tab:arch_ladder}")
    print(r"\begin{tabular}{lcccc}\hline")
    print(r"\textbf{Variant} & \textbf{Sharpe} & \textbf{ROI (\%)} & \textbf{MDD (\%)} & \textbf{$\Delta$Sharpe} \\")
    print(r"\hline")
    prev = None
    for exp_id in exp_list:
        m = results[exp_id]
        delta_str = f"{m['sharpe'] - prev:+.4f}" if prev is not None else "---"
        bold = exp_id == "EXP-5-NEXTGEN"
        w = (r"\textbf{", "}") if bold else ("", "")
        print(f"{w[0]}{ARCH_EXPERIMENTS[exp_id]['desc']}{w[1]} & "
              f"{w[0]}{m['sharpe']:.4f}{w[1]} & "
              f"{w[0]}{m['roi']:.2f}{w[1]} & "
              f"{w[0]}{m['mdd']:.2f}{w[1]} & "
              f"{w[0]}{delta_str}{w[1]} \\\\")
        prev = m["sharpe"]
    print(r"\hline\end{tabular}\end{table}")

    csv_rows = [{"exp_id": eid, "description": ARCH_EXPERIMENTS[eid]["desc"],
                 **results[eid]} for eid in exp_list]
    save_csv(csv_rows, os.path.join(results_dir, f"ablation_architecture_{portfolio_name}.csv"))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STUDY 2 — Component Isolation
# ─────────────────────────────────────────────────────────────────────────────

COMPONENT_CONFIGS = {
    # use_nextgen=False → falls back to super().update() (plain SAC critics, no IQN)
    "baseline":    {"desc": "Baseline (Plain SAC)",             "use_nextgen": False, "use_attention_gating": False, "use_momentum_masking": False, "use_cvar_loss": False, "use_dual_head_iqn": False},
    "iqn_only":    {"desc": "NextGen (IQN Only)",               "use_nextgen": True,  "use_attention_gating": False, "use_momentum_masking": False, "use_cvar_loss": True,  "use_dual_head_iqn": True},
    "momask":      {"desc": "NextGen + Momentum Masking",       "use_nextgen": True,  "use_attention_gating": False, "use_momentum_masking": True,  "use_cvar_loss": False, "use_dual_head_iqn": False},
    "iqn_momask":  {"desc": "NextGen + IQN + Momentum Masking", "use_nextgen": True,  "use_attention_gating": False, "use_momentum_masking": True,  "use_cvar_loss": True,  "use_dual_head_iqn": True},
    "attgate":     {"desc": "NextGen + Attention Gating",       "use_nextgen": True,  "use_attention_gating": True,  "use_momentum_masking": False, "use_cvar_loss": False, "use_dual_head_iqn": False},
    "full_stack":  {"desc": "Full Stack (All Enabled)",         "use_nextgen": True,  "use_attention_gating": True,  "use_momentum_masking": True,  "use_cvar_loss": True,  "use_dual_head_iqn": True},
}


def run_component_experiment(cfg_key, shared, train_steps):
    reset_seed()

    cfg    = COMPONENT_CONFIGS[cfg_key]
    gb     = shared["graph_builder"]
    # Study 2 always runs on CPU.
    device = "cpu"

    print(f"\n{'='*60}")
    print(f"  [Study 2] {cfg_key}: {cfg['desc']}")
    print(f"  Device: {device.upper()}")
    print(f"{'='*60}")

    train_feats = shared["cached_train_feats"]
    train_adjs  = shared["cached_train_adjs"]
    val_feats   = shared["cached_test_feats"]
    val_adjs    = shared["cached_test_adjs"]

    env_train = PortfolioEnv(shared["train_data"], initial_balance=100000.0, transaction_cost=0.0025,
                             reward_type="log_utility")
    feat_dim, _ = patch_env(env_train, train_feats)

    env_val = PortfolioEnv(shared["test_data"], initial_balance=100000.0, transaction_cost=0.0025)
    patch_env(env_val, val_feats)

    state_dim = env_train.observation_space.shape[0]
    n_assets  = shared["n_assets"]

    try:
        agent = GNNSACAgent(
            state_dim=state_dim, action_dim=n_assets,
            node_features_dim=feat_dim, num_nodes=gb.num_nodes,
            adj_matrix=gb.mixed_adj_template, device=device,
            lstm_hidden_size=32, num_lstm_layers=1,
            hidden_dims=[128, 128], use_graph=True, use_tgn=False,
            use_nextgen=cfg["use_nextgen"],
            use_attention_gating=cfg["use_attention_gating"],
            use_momentum_masking=cfg["use_momentum_masking"],
            use_cvar_loss=cfg["use_cvar_loss"],
            use_dual_head_iqn=cfg["use_dual_head_iqn"],
            alpha=0.02, auto_tune_alpha=False,
            learning_rate=1e-4, batch_size=4,
        )
    except Exception as e:
        print(f"  Agent init failed: {e}")
        return {"sharpe": 0.0, "roi": 0.0, "mdd": 0.0}

    state, _ = env_train.reset(seed=42)
    hidden    = None
    n_random  = min(200, train_steps // 5)
    rms       = RunningMeanStd()

    for step in tqdm(range(train_steps), desc=f"  Train {cfg_key}"):
        idx   = min(env_train.current_step + env_train.window_size, len(train_adjs) - 1)
        adj_t = torch.tensor(train_adjs[idx], dtype=torch.float32).unsqueeze(0).to(device)

        if step < n_random:
            action, hidden = env_train.action_space.sample(), None
        else:
            action, hidden = agent.select_action(state, hidden, adj=adj_t)

        next_state, reward, term, trunc, _ = env_train.step(action)
        rms.update(reward)
        # Consistent reward normalization + clipping across all studies
        norm_reward = np.clip(rms.normalize(reward), -10, 10)
        agent.replay_buffer.push(state, action, norm_reward, next_state, float(term or trunc))
        state = next_state

        if step >= n_random and len(agent.replay_buffer) >= agent.batch_size:
            # nextgen_ctx only valid when use_nextgen=True; baseline uses plain SAC update
            if cfg["use_nextgen"]:
                agent.update(nextgen_ctx={})
            else:
                agent.update()
        if term or trunc:
            state, _ = env_train.reset()
            hidden = None
            rms.reset()

    obs, _ = env_val.reset(seed=42)
    h, done = None, False
    vals = [100000.0]
    while not done:
        vi    = min(env_val.current_step + env_val.window_size, len(val_adjs) - 1)
        va    = torch.tensor(val_adjs[vi], dtype=torch.float32).unsqueeze(0).to(device)
        act, h = agent.select_action(obs, h, deterministic=True, adj=va)
        obs, _, t, tr, info = env_val.step(act)
        done = t or tr
        vals.append(info.get("portfolio_value", vals[-1]))

    m = compute_metrics(vals)
    print(f"  Result: Sharpe={m['sharpe']:.4f}  ROI={m['roi']:.2f}%  MDD={m['mdd']:.2f}%")
    del agent
    return m


def run_study2(shared, steps, portfolio_name, results_dir):
    print("\n" + "="*75)
    print("  STUDY 2 — Component Isolation")
    print("="*75)
    results, rows = {}, []

    for cfg_key, cfg in COMPONENT_CONFIGS.items():
        try:
            m = run_component_experiment(cfg_key, shared, steps)
            results[cfg_key] = m
        except Exception as e:
            print(f"  FAILED {cfg_key}: {e}")
            import traceback; traceback.print_exc()
            results[cfg_key] = m = {"sharpe": 0.0, "roi": 0.0, "mdd": 0.0}
        rows.append({"Configuration": cfg["desc"], "Sharpe": m["sharpe"],
                     "ROI (%)": m["roi"], "MDD (%)": m["mdd"]})

    print_table("STUDY 2 — Component Isolation Results", rows,
                ["Configuration", "Sharpe", "ROI (%)", "MDD (%)"])

    baseline_sharpe = results.get("baseline", {}).get("sharpe", 0.0)
    print(f"\n  Delta Sharpe vs Baseline ({baseline_sharpe:.4f}):")
    for k, m in results.items():
        if k == "baseline":
            continue
        d   = m["sharpe"] - baseline_sharpe
        tag = "BENEFICIAL" if d > 0 else "HARMFUL"
        print(f"  {COMPONENT_CONFIGS[k]['desc']:<45} {d:>+.4f}  [{tag}]")

    ranked = sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    print("\n[LaTeX — Table 2B: Component Ablation]")
    print(r"\begin{table}[h]\centering")
    print(r"\caption{Architectural Component Ablation}")
    print(r"\label{tab:ablation}")
    print(r"\begin{tabular}{lcc}\hline")
    print(r"\textbf{Configuration} & \textbf{Sharpe Ratio} & \textbf{ROI (\%)} \\ \hline")
    for k, m in ranked:
        bold = k == "attgate"
        w = (r"\textbf{", "}") if bold else ("", "")
        print(f"{w[0]}{COMPONENT_CONFIGS[k]['desc']}{w[1]} & "
              f"{w[0]}{m['sharpe']:.4f}{w[1]} & "
              f"{w[0]}{m['roi']:.2f}{w[1]} \\\\")
    print(r"\hline\end{tabular}\end{table}")

    csv_rows = [{"config": k, "description": COMPONENT_CONFIGS[k]["desc"],
                 **results[k]} for k in COMPONENT_CONFIGS]
    save_csv(csv_rows, os.path.join(results_dir, f"ablation_components_{portfolio_name}.csv"))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STUDY 3 — FinBERT Trajectory
# ─────────────────────────────────────────────────────────────────────────────

FINBERT_EXPERIMENTS = {
    "EXP-3-KEYWORDS": {"desc": "GNN-SAC + Keyword Events (sentiment_scalar=1.0)", "finbert": False},
    "EXP-4-FULL":     {"desc": "GNN-SAC + FinBERT-MRC Causal Sentiment",          "finbert": True},
}


def run_finbert_experiment(exp_id, shared, train_steps, eval_every, device="cpu"):
    reset_seed()

    cfg = FINBERT_EXPERIMENTS[exp_id]
    gb  = shared["graph_builder"]

    print(f"\n{'='*60}")
    print(f"  [Study 3] {exp_id}: {cfg['desc']}")
    print(f"  FinBERT: {'ENABLED' if cfg['finbert'] else 'DISABLED (sentiment_scalar=1.0)'}")
    print(f"{'='*60}")

    train_data, test_data = shared["train_data"], shared["test_data"]
    train_base, test_base = shared["train_base"], shared["test_base"]
    train_dates = shared["dates"][:shared["train_size"]]
    test_dates  = shared["dates"][shared["train_size"]:]

    real_news = gb.news_df.copy()
    # Reset temporal memory so EXP-3-KEYWORDS cannot contaminate EXP-4-FULL.
    # gb.event_memory accumulates with every get_graph() call; without reset,
    # the second experiment inherits the first experiment's keyword-only state.
    gb.event_memory = np.zeros(gb.num_stock_nodes)
    gb.sector_memory = np.zeros(gb.num_sector_nodes)
    if not cfg["finbert"]:
        # Only sentiment_scalar is read by graph_builder.get_graph() — sentiment_score is not used
        if "sentiment_scalar" in gb.news_df.columns:
            gb.news_df["sentiment_scalar"] = 1.0

    # Rebuild features AFTER mutation so the mutation is reflected in graphs
    train_feats, train_adjs = build_graph_features(gb, train_base, train_dates, "train", device)
    test_feats,  test_adjs  = build_graph_features(gb, test_base,  test_dates,  "test",  device)
    gb.news_df = real_news  # Restore AFTER rebuilding

    env_train = PortfolioEnv(train_data, initial_balance=100000.0, transaction_cost=0.0025,
                             reward_type="log_utility")
    feat_dim, n_nodes = patch_env(env_train, train_feats)
    env_train.dynamic_adjs = train_adjs

    env_test = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
    patch_env(env_test, test_feats)
    env_test.dynamic_adjs = test_adjs

    state_dim = env_train.observation_space.shape[0]
    n_assets  = shared["n_assets"]

    try:
        agent = GNNSACAgent(
            state_dim=state_dim, action_dim=n_assets,
            node_features_dim=feat_dim, num_nodes=n_nodes,
            adj_matrix=gb.mixed_adj_template, device=device,
            lstm_hidden_size=64, num_lstm_layers=1,
            hidden_dims=[256, 256], use_graph=True, use_nextgen=True,
            use_attention_gating=True, use_momentum_masking=True,
            auto_tune_alpha=False, alpha=0.02,
            learning_rate=5e-5, batch_size=8,
        )
    except Exception as e:
        print(f"  Agent init failed: {e}")
        return []

    n_random = min(int(train_steps * 0.4), 400)
    state, _ = env_train.reset(seed=42)
    hidden    = None
    rms       = RunningMeanStd()
    checkpoints = []

    pbar = tqdm(range(train_steps), desc=f"  {exp_id}")
    for step in pbar:
        idx   = min(env_train.current_step + env_train.window_size, len(train_adjs) - 1)
        adj_t = torch.tensor(train_adjs[idx], dtype=torch.float32).unsqueeze(0).to(device)

        if step < n_random:
            action, hidden = env_train.action_space.sample(), None
        else:
            action, hidden = agent.select_action(state, hidden, adj=adj_t)

        next_state, reward, term, trunc, _ = env_train.step(action)
        rms.update(reward)
        norm_reward = np.clip(rms.normalize(reward), -10, 10)
        agent.replay_buffer.push(state, action, norm_reward, next_state, float(term or trunc))
        state = next_state

        if step >= n_random and len(agent.replay_buffer) >= agent.batch_size:
            agent.update(nextgen_ctx={})
        if term or trunc:
            state, _ = env_train.reset()
            hidden = None
            rms.reset()

        if (step + 1) % eval_every == 0:
            agent.policy.eval()
            obs, _ = env_test.reset(seed=42)
            h, done = None, False
            vals = [100000.0]
            while not done:
                vi = min(env_test.current_step + env_test.window_size, len(test_adjs) - 1)
                va = torch.tensor(test_adjs[vi], dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    act, h = agent.select_action(obs, h, deterministic=True, adj=va)
                obs, _, t, tr, info = env_test.step(act)
                done = t or tr
                vals.append(info.get("portfolio_value", vals[-1]))
            m = compute_metrics(vals)
            m["step"]    = step + 1
            m["exp_id"]  = exp_id
            m["finbert"] = cfg["finbert"]
            checkpoints.append(m)
            tqdm.write(f"  [{exp_id}] Step {step+1:>5} | Sharpe {m['sharpe']:+.4f} | ROI {m['roi']:+.2f}% | MDD {m['mdd']:+.2f}%")
            agent.policy.train()

    pbar.close()
    del agent
    return checkpoints


def run_study3(shared, steps, portfolio_name, results_dir):
    eval_every = max(steps // 5, 1)
    print("\n" + "="*75)
    print(f"  STUDY 3 — FinBERT Trajectory  ({steps} steps, eval every {eval_every})")
    print("="*75)

    all_results = {}
    for exp_id in FINBERT_EXPERIMENTS:
        try:
            all_results[exp_id] = run_finbert_experiment(exp_id, shared, steps, eval_every, device="cpu")
        except Exception as e:
            print(f"  FAILED {exp_id}: {e}")
            import traceback; traceback.print_exc()
            all_results[exp_id] = []

    steps_seen = sorted({r["step"] for v in all_results.values() for r in v})
    lookup     = {eid: {r["step"]: r for r in chk} for eid, chk in all_results.items()}

    print(f"\n  {'Step':>6} | {'Keywords Sharpe':>16} | {'FinBERT Sharpe':>15} | {'ΔSharpe':>9} | {'KW MDD':>8} | {'FB MDD':>8}")
    print("  " + "-" * 75)
    for s in steps_seen:
        kw = lookup.get("EXP-3-KEYWORDS", {}).get(s, {})
        fb = lookup.get("EXP-4-FULL",     {}).get(s, {})
        sk = kw.get("sharpe", float("nan"))
        sf = fb.get("sharpe", float("nan"))
        mk = kw.get("mdd", float("nan"))
        mf = fb.get("mdd", float("nan"))
        d  = sf - sk if not any(np.isnan([sk, sf])) else float("nan")
        print(f"  {s:>6} | {sk:>+16.4f} | {sf:>+15.4f} | {d:>+9.4f} | {mk:>8.2f} | {mf:>8.2f}")

    kw_f = lookup.get("EXP-3-KEYWORDS", {}).get(steps, {})
    fb_f = lookup.get("EXP-4-FULL",     {}).get(steps, {})
    if kw_f and fb_f:
        d = fb_f["sharpe"] - kw_f["sharpe"]
        verdict = "BENEFICIAL" if d > 0.05 else "MARGINAL" if d > 0.01 else "NO LIFT"
        print(f"\n  FinBERT vs Keywords final ΔSharpe: {d:+.4f}  [{verdict}]")
        dm = fb_f["mdd"] - kw_f["mdd"]
        verdict_mdd = "SAFER" if dm > 0 else "RISKIER"
        print(f"  FinBERT vs Keywords final ΔMDD:   {dm:+.2f}%  [{verdict_mdd}]")

    print("\n[LaTeX — FinBERT Trajectory]")
    print(r"\begin{table}[h]\centering")
    print(r"\caption{FinBERT-MRC Impact: Learning Trajectory}")
    print(r"\label{tab:finbert_impact}")
    print(r"\begin{tabular}{rcccc}\hline")
    print(r"\textbf{Step} & \multicolumn{2}{c}{\textbf{Keywords}} & \multicolumn{2}{c}{\textbf{FinBERT}} \\")
    print(r" & Sharpe & ROI(\%) & Sharpe & ROI(\%) \\ \hline")
    for s in steps_seen:
        kw = lookup.get("EXP-3-KEYWORDS", {}).get(s, {})
        fb = lookup.get("EXP-4-FULL",     {}).get(s, {})
        print(f"{s} & {kw.get('sharpe',0):.4f} & {kw.get('roi',0):.2f} & "
              f"{fb.get('sharpe',0):.4f} & {fb.get('roi',0):.2f} \\\\")
    print(r"\hline\end{tabular}\end{table}")

    csv_rows = [r for v in all_results.values() for r in v]
    save_csv(csv_rows, os.path.join(results_dir, f"finbert_impact_results_{portfolio_name}.csv"))
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Ablation Study for GNN-SAC Paper")
    parser.add_argument("--portfolio_name", type=str, default="28stocks")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Training steps for Study 1 & 2 (default 1000)")
    parser.add_argument("--finbert_steps", type=int, default=None,
                        help="Training steps for Study 3. Defaults to --steps value.")
    parser.add_argument("--study", type=int, default=0,
                        help="Run only one study: 1, 2, or 3. Default 0 = run all.")
    args = parser.parse_args()

    finbert_steps = args.finbert_steps if args.finbert_steps else args.steps

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "ablation_run_log.txt")
    tee = Tee(log_path)

    print("=" * 75)
    print(f"  UNIFIED ABLATION STUDY — {args.portfolio_name.upper()}")
    print(f"  Device: {DEVICE.upper()}")
    print(f"  Study 1 & 2: {args.steps} steps | Study 3: {finbert_steps} steps")
    print("=" * 75)

    data_dir = os.path.join(BASE_DIR, "data")
    shared   = load_shared_data(args.portfolio_name, data_dir)

    t0 = time.time()

    if args.study in (0, 1):
        run_study1(shared, args.steps, args.portfolio_name, results_dir)

    if args.study in (0, 2):
        run_study2(shared, args.steps, args.portfolio_name, results_dir)

    if args.study in (0, 3):
        run_study3(shared, finbert_steps, args.portfolio_name, results_dir)

    elapsed = time.time() - t0
    print(f"\n{'='*75}")
    print(f"  All studies complete in {elapsed/60:.1f} min")
    print(f"  Outputs in: {results_dir}")
    print(f"  Log:        {log_path}")
    print("=" * 75)
