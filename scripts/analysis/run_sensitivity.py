"""
Hyperparameter Sensitivity Analysis for Event-Driven GNN-SAC
=============================================================

Sweeps two critical hyperparameters:
  Sweep 1 — δ (memory decay rate):      [0.01, 0.05, 0.1*, 0.2, 0.5, 1.0]
  Sweep 2 — ξ (CVaR tail multiplier):   [1.0, 2.0, 3.0, 5.0*, 7.0, 10.0]

* = paper/code baseline value

Two-phase execution:
  Phase 1 (SMOKE=True):  1,000 steps  — validates pipeline, prints result table
  Phase 2 (SMOKE=False): 10,000 steps — full run, saves CSVs + publication figure

Usage:
  Phase 1: python run_sensitivity.py
  Phase 2: python run_sensitivity.py --final
"""

import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
# scripts/analysis/run_sensitivity.py → go up 2 levels to reach repo root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.data_pipeline.graph_builder import DynamicGraphBuilder
from src.agents.gnn_sac_agent import GNNSACAgent
from src.environments.portfolio_env import PortfolioEnv
from src.utils.news_path_resolver import get_news_path

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

# ── Configuration ─────────────────────────────────────────────────────────────
PORTFOLIO_NAME = "28stocks"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

def reset_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Benchmark window (matches paper evaluation protocol)
BENCHMARK_START = "2013-11-04"
BENCHMARK_END   = "2014-03-14"
WARMUP_DAYS     = 60

# Agent config (matches eval log: use_tgn=False, att_gate=True, mom_mask=True)
AGENT_KWARGS = dict(
    hidden_dims       = [256, 256],
    learning_rate     = 5e-5,
    use_graph         = True,
    use_tgn           = False,
    use_nextgen       = True,
    use_attention_gating = True,
    use_momentum_masking = True,
    use_feature_gating = False,
    alpha             = 0.02,
    auto_tune_alpha   = False,
    batch_size        = 8,
    device            = DEVICE,
)

# Sweep values
DELTA_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
XI_VALUES    = [1.0,  2.0,  3.0, 5.0, 7.0, 10.0]

DELTA_BASELINE = 0.1   # paper value
XI_BASELINE    = 5.0   # code default (cvar_multiplier)

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR          = os.path.join(BASE_DIR, "data")
ENV_DATA_PATH      = os.path.join(DATA_DIR, "processed", "28stocks_dataset.csv")
PRECOMPUTED_GRAPHS = os.path.join(DATA_DIR, "processed", "28stocks_precomputed_graphs.npz")
NODE_FEAT_PATH    = os.path.join(DATA_DIR, f"{PORTFOLIO_NAME}_data.npy")
RESULTS_DIR       = os.path.join(BASE_DIR, "results")

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def calc_metrics(portfolio_values):
    vals = np.array(portfolio_values, dtype=float)
    if len(vals) < 2:
        return 0.0, 0.0, 0.0
    rets = pd.Series(vals).pct_change().dropna()
    if rets.std() < 1e-9:
        return 0.0, 0.0, 0.0
    roi    = (vals[-1] - vals[0]) / vals[0] * 100.0
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252)
    cummax = np.maximum.accumulate(vals)
    mdd    = ((vals / cummax) - 1.0).min() * 100.0
    return round(roi, 4), round(sharpe, 4), round(mdd, 4)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_env_data():
    """Load and return the full environment dataframe."""
    if not os.path.exists(ENV_DATA_PATH):
        raise FileNotFoundError(f"Environment data not found: {ENV_DATA_PATH}")
    df = pd.read_csv(ENV_DATA_PATH, index_col=0, parse_dates=True)
    return df


def split_data(df):
    """Return (train_df, benchmark_df) using paper benchmark window."""
    bench_start = pd.Timestamp(BENCHMARK_START)
    bench_end   = pd.Timestamp(BENCHMARK_END)
    warmup_start = bench_start - pd.tseries.offsets.BDay(WARMUP_DAYS)

    train_df = df[df.index < bench_start]
    # Benchmark slice includes warm-up days
    bench_df  = df[(df.index >= warmup_start) & (df.index <= bench_end)]
    return train_df, bench_df


def load_precomputed_graphs():
    """Load cached .npz graphs (used for ξ sweep where δ is not varied)."""
    if not os.path.exists(PRECOMPUTED_GRAPHS):
        raise FileNotFoundError(f"Precomputed graphs not found: {PRECOMPUTED_GRAPHS}")
    npz = np.load(PRECOMPUTED_GRAPHS)
    feats = npz["features"]   # (T, N, F)
    adjs  = npz["adjs"]       # (T, N, N)
    return feats, adjs


def compute_graphs_for_delta(decay_rate, env_data):
    """
    Recompute graphs on-the-fly for a given decay rate δ.
    Required because the cached .npz has δ=0.1 baked into event-adjacency
    weights and node features — it cannot be safely reused for other δ values.
    """
    news_path = get_news_path(DATA_DIR, PORTFOLIO_NAME) or ""
    gb = DynamicGraphBuilder(DATA_DIR, PORTFOLIO_NAME, news_path,
                             decay_rate=decay_rate, inject_sentiment=True)

    base_node_features = np.load(NODE_FEAT_PATH)  # (T_raw, N_raw, F)

    # !! Critical: normalize exactly as run_all_ablations.py does !!
    # Without this, raw features can be very large, causing GNN to output NaN.
    mu = np.nanmean(base_node_features, axis=(0, 1), keepdims=True)
    sd = np.nanstd(base_node_features,  axis=(0, 1), keepdims=True) + 1e-6
    base_node_features = np.clip((base_node_features - mu) / sd, -5, 5)
    # Replace any residual NaNs/Infs with zero (safety net)
    base_node_features = np.nan_to_num(base_node_features, nan=0.0, posinf=5.0, neginf=-5.0)

    dates = env_data.index

    # Align lengths
    T = min(len(dates), len(base_node_features))
    dates = dates[:T]
    base_feat_tensor = torch.tensor(
        base_node_features[:T], dtype=torch.float32
    )

    all_features, all_adjs = [], []
    for t in tqdm(range(T), desc=f"  Graphs δ={decay_rate}", leave=False):
        try:
            g = gb.get_graph(dates[t], base_feat_tensor[t])
            feats = g.x.numpy()
            adjs  = g.adj.numpy()
        except Exception:
            feats = np.zeros((gb.num_nodes, base_node_features.shape[2]))
            adjs  = np.eye(gb.num_nodes)
        # Guard: replace any NaN/Inf introduced by graph construction
        feats = np.nan_to_num(feats, nan=0.0, posinf=5.0, neginf=-5.0)
        all_features.append(feats)
        all_adjs.append(adjs)

    return np.array(all_features), np.array(all_adjs), gb.num_nodes


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(train_df, bench_df, all_features, all_adjs,
                   num_nodes, xi_val, train_steps, train_offset=0):
    """
    Train for `train_steps` on train_df, then evaluate on bench_df.
    Returns (roi, sharpe, mdd).
    """
    N = all_features.shape[1]
    F = all_features.shape[2]

    # ── Train environment ─────────────────────────────────────────────────
    env_train = PortfolioEnv(train_df, initial_balance=100_000.0)

    # Patch env to use precomputed graph features
    feat_dim = F
    state_dim = N * (feat_dim + 1)

    env_train.graph_features = all_features
    env_train.dynamic_adjs   = all_adjs
    env_train._graph_offset   = train_offset

    def _get_state_train(self):
        idx = self._graph_offset + self.current_step + getattr(self, 'window_size', 60)
        idx = min(idx, len(self.graph_features) - 1)
        f = self.graph_features[idx]           # (N, F)
        w = self.portfolio_weights              # (action_dim,)
        # Pad/trim weights to match N
        if len(w) < N:
            w = np.pad(w, (0, N - len(w)), 'constant')
        else:
            w = w[:N]
        return np.hstack([f, w.reshape(-1, 1)]).flatten().astype(np.float32)

    import types
    env_train._get_state = types.MethodType(_get_state_train, env_train)
    import gymnasium as gym
    env_train.observation_space = gym.spaces.Box(
        -np.inf, np.inf, (state_dim,), np.float32
    )

    # ── Agent ─────────────────────────────────────────────────────────────
    reset_seed(42)
    dummy_adj = torch.tensor(all_adjs[0], dtype=torch.float32)

    agent = GNNSACAgent(
        state_dim        = state_dim,
        action_dim       = len([c for c in train_df.columns if c.startswith("price_")]),
        node_features_dim= feat_dim + 1,
        num_nodes        = num_nodes,
        adj_matrix       = dummy_adj.numpy(),
        cvar_multiplier  = xi_val,
        cvar_tau         = 0.25,
        use_cvar_loss    = True,
        **AGENT_KWARGS,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    state, _ = env_train.reset()
    hidden    = None
    rms       = RunningMeanStd()

    for step in tqdm(range(train_steps), desc=f"    Train (ξ={xi_val})", leave=False):
        idx = train_offset + env_train.current_step + getattr(env_train, 'window_size', 60)
        idx = min(idx, len(all_adjs) - 1)
        adj_t = torch.tensor(all_adjs[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        action, hidden = agent.select_action(state, hidden, adj=adj_t)
        next_state, reward, term, trunc, _ = env_train.step(action)
        done = term or trunc

        rms.update(reward)
        norm_reward = np.clip(rms.normalize(reward), -10, 10)

        agent.replay_buffer.push(state, action, norm_reward, next_state, float(done))
        state = next_state

        if len(agent.replay_buffer) > 256:
            agent.update(nextgen_ctx={})

        if done:
            state, _ = env_train.reset()
            hidden    = None
            rms.reset()

    # ── Evaluation ────────────────────────────────────────────────────────
    # Find benchmark offset in the full dataset
    bench_offset = max(0, len(all_features) - len(bench_df))

    env_bench = PortfolioEnv(bench_df, initial_balance=100_000.0)
    env_bench.graph_features = all_features
    env_bench.dynamic_adjs   = all_adjs
    env_bench._graph_offset   = bench_offset

    def _get_state_bench(self):
        idx = self._graph_offset + self.current_step + getattr(self, 'window_size', 60)
        idx = min(idx, len(self.graph_features) - 1)
        f = self.graph_features[idx]
        w = self.portfolio_weights
        if len(w) < N:
            w = np.pad(w, (0, N - len(w)), 'constant')
        else:
            w = w[:N]
        return np.hstack([f, w.reshape(-1, 1)]).flatten().astype(np.float32)

    env_bench._get_state = types.MethodType(_get_state_bench, env_bench)
    env_bench.observation_space = gym.spaces.Box(
        -np.inf, np.inf, (state_dim,), np.float32
    )

    obs, _  = env_bench.reset()
    hidden   = None
    vals     = [100_000.0]
    done     = False

    while not done:
        idx = bench_offset + env_bench.current_step + getattr(env_bench, 'window_size', 60)
        idx = min(idx, len(all_adjs) - 1)
        adj_t = torch.tensor(all_adjs[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        action, hidden = agent.select_action(obs, hidden,
                                             deterministic=True, adj=adj_t)
        obs, _, term, trunc, info = env_bench.step(action)
        done = term or trunc
        vals.append(info.get('portfolio_value', vals[-1]))

    return calc_metrics(vals)


# ─────────────────────────────────────────────────────────────────────────────
# Sweep runners
# ─────────────────────────────────────────────────────────────────────────────
def sweep_delta(train_df, bench_df, train_steps):
    """Sweep δ — recomputes graphs on-the-fly for each value."""
    print("\n" + "=" * 65)
    print(f"  SWEEP 1 — δ (Memory Decay Rate)  [{train_steps} steps/run]")
    print("=" * 65)

    env_data = pd.concat([train_df, bench_df]).drop_duplicates().sort_index()
    results  = []

    for delta in DELTA_VALUES:
        half_life = int(np.log(2) / delta) if delta > 0 else 9999
        label     = f"δ={delta:.2f} ({'*' if delta == DELTA_BASELINE else ' '})"
        print(f"\n  [{label}] half-life ≈ {half_life} days")

        try:
            feats, adjs, num_nodes = compute_graphs_for_delta(delta, env_data)
            roi, sharpe, mdd       = run_experiment(
                train_df, bench_df, feats, adjs,
                num_nodes, xi_val=XI_BASELINE,
                train_steps=train_steps
            )
        except Exception as e:
            print(f"    WARNING: Failed: {e}")
            roi, sharpe, mdd = float("nan"), float("nan"), float("nan")

        results.append({"delta": delta, "half_life_days": half_life,
                        "sharpe": sharpe, "roi": roi, "mdd": mdd})
        print(f"    → Sharpe={sharpe:.3f}  ROI={roi:.2f}%  MDD={mdd:.2f}%")

    return pd.DataFrame(results)


def sweep_xi(train_df, bench_df, train_steps):
    """Sweep ξ — reuses precomputed graphs (ξ only affects IQN loss)."""
    print("\n" + "=" * 65)
    print(f"  SWEEP 2 — ξ (CVaR Tail Multiplier)  [{train_steps} steps/run]")
    print("=" * 65)

    # Load precomputed graphs (δ=0.1 baseline)
    feats, adjs = load_precomputed_graphs()
    num_nodes   = feats.shape[1]
    results     = []

    for xi in XI_VALUES:
        label = f"ξ={xi:.1f} ({'*' if xi == XI_BASELINE else ' '})"
        print(f"\n  [{label}]")

        try:
            roi, sharpe, mdd = run_experiment(
                train_df, bench_df, feats, adjs,
                num_nodes, xi_val=xi,
                train_steps=train_steps
            )
        except Exception as e:
            print(f"    WARNING: Failed: {e}")
            roi, sharpe, mdd = float("nan"), float("nan"), float("nan")

        results.append({"xi": xi, "sharpe": sharpe, "roi": roi, "mdd": mdd})
        print(f"    → Sharpe={sharpe:.3f}  ROI={roi:.2f}%  MDD={mdd:.2f}%")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────
def print_results_table(df_delta, df_xi):
    """Pretty-print result tables for smoke-test analysis."""
    sep  = "-" * 65
    star = "*"

    print("\n\n" + "=" * 65)
    print("  SENSITIVITY SWEEP RESULTS")
    print("=" * 65)

    print("\n[Sweep 1: δ — Memory Decay Rate]")
    print(f"  {'δ':<8} {'Half-life':<12} {'Sharpe':<10} {'ROI (%)':<10} {'MDD (%)'}")
    print(f"  {sep}")
    for _, row in df_delta.iterrows():
        marker = star if row["delta"] == DELTA_BASELINE else " "
        hl     = f"~{int(row['half_life_days'])} days"
        print(f"  {row['delta']:<7.2f}{marker}  {hl:<12} "
              f"{row['sharpe']:<10.3f} {row['roi']:<10.2f} {row['mdd']:.2f}")

    print(f"\n[Sweep 2: ξ — CVaR Tail-Amplification Factor]")
    print(f"  {'ξ':<8} {'Sharpe':<10} {'ROI (%)':<10} {'MDD (%)'}")
    print(f"  {sep}")
    for _, row in df_xi.iterrows():
        marker = star if row["xi"] == XI_BASELINE else " "
        print(f"  {row['xi']:<7.1f}{marker}  {row['sharpe']:<10.3f} "
              f"{row['roi']:<10.2f} {row['mdd']:.2f}")
    print()


def save_results(df_delta, df_xi):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    delta_path = os.path.join(RESULTS_DIR, "sensitivity_delta_28stocks.csv")
    xi_path    = os.path.join(RESULTS_DIR, "sensitivity_xi_28stocks.csv")
    df_delta.to_csv(delta_path, index=False)
    df_xi.to_csv(xi_path, index=False)
    print(f"  Saved: {delta_path}")
    print(f"  Saved: {xi_path}")


def plot_figure(df_delta, df_xi):
    """Publication-quality 2-panel sensitivity figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Hyperparameter Sensitivity Analysis — Event-Driven GNN-SAC (28 Stocks)",
        fontsize=13, fontweight="bold", y=1.01
    )

    BLUE   = "#2563EB"
    RED    = "#DC2626"
    GREY   = "#6B7280"
    STAR_C = "#D97706"

    # ── Panel 1: δ sweep ──────────────────────────────────────────────────
    ax1 = axes[0]
    x1  = df_delta["delta"].values
    y1  = df_delta["sharpe"].values

    ax1.plot(x1, y1, marker="o", color=BLUE, linewidth=2, markersize=6,
             label="Sharpe Ratio")
    ax1.axvline(DELTA_BASELINE, color=GREY, linestyle="--", linewidth=1.2,
                label=f"Paper value δ={DELTA_BASELINE}")

    # Mark baseline point
    idx_base = list(x1).index(DELTA_BASELINE) if DELTA_BASELINE in x1 else -1
    if idx_base >= 0:
        ax1.scatter([x1[idx_base]], [y1[idx_base]], color=STAR_C, zorder=5,
                    s=100, marker="*", label="Baseline *")

    ax1.set_xscale("log")
    ax1.set_xlabel("δ  (Memory Decay Rate)", fontsize=11)
    ax1.set_ylabel("Sharpe Ratio", fontsize=11)
    ax1.set_title("Sweep 1: δ — Sentiment Memory Decay", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.set_xticks(x1)

    # ── Panel 2: ξ sweep ──────────────────────────────────────────────────
    ax2   = axes[1]
    ax2r  = ax2.twinx()
    x2    = df_xi["xi"].values
    y2s   = df_xi["sharpe"].values
    y2m   = df_xi["mdd"].values

    l1, = ax2.plot(x2, y2s, marker="o", color=BLUE, linewidth=2, markersize=6,
                   label="Sharpe Ratio")
    l2, = ax2r.plot(x2, y2m, marker="s", color=RED, linewidth=2, markersize=6,
                    linestyle="--", label="Max Drawdown (%)")
    ax2.axvline(XI_BASELINE, color=GREY, linestyle="--", linewidth=1.2,
                label=f"Code baseline ξ={XI_BASELINE}")

    idx_xi = list(x2).index(XI_BASELINE) if XI_BASELINE in x2 else -1
    if idx_xi >= 0:
        ax2.scatter([x2[idx_xi]], [y2s[idx_xi]], color=STAR_C, zorder=5,
                    s=100, marker="*")

    ax2.set_xlabel("ξ  (CVaR Tail-Amplification Factor)", fontsize=11)
    ax2.set_ylabel("Sharpe Ratio", fontsize=11, color=BLUE)
    ax2r.set_ylabel("Max Drawdown (%)", fontsize=11, color=RED)
    ax2.set_title("Sweep 2: ξ — CVaR Tail Multiplier", fontsize=11)
    ax2.legend(handles=[l1, l2,
        plt.Line2D([0], [0], color=GREY, linestyle="--", label=f"Baseline *")],
        fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "Figure_sensitivity_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved figure: {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Sensitivity Analysis for Event-Driven GNN-SAC",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test — both sweeps, 200 steps each (default):
  python run_sensitivity.py

  # Run ONLY the delta sweep, 5000 steps:
  python run_sensitivity.py --sweep delta --steps 5000

  # Run ONLY the xi sweep, 10000 steps:
  python run_sensitivity.py --sweep xi --steps 10000

  # Full final run — both sweeps, 10000 steps, save CSVs + figure:
  python run_sensitivity.py --sweep both --steps 10000 --save

  # Legacy shortcut (equivalent to --sweep both --steps 10000 --save):
  python run_sensitivity.py --final
        """
    )
    parser.add_argument(
        "--sweep",
        choices=["delta", "xi", "both"],
        default="both",
        help="Which sweep to run:\n"
             "  delta  — Sweep 1 only: memory decay rate (delta)\n"
             "  xi     — Sweep 2 only: CVaR tail multiplier (xi)\n"
             "  both   — Run both sweeps (default)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Training steps per configuration (default: 200).\n"
             "Use 200 for smoke test, 10000 for final publication run."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save CSVs and figure to results/ directory.\n"
             "Omit for a quick smoke test (no files written)."
    )
    # Legacy alias kept for backward compatibility
    parser.add_argument(
        "--final",
        action="store_true",
        help="(Legacy) Equivalent to: --sweep both --steps 10000"
    )
    args = parser.parse_args()

    # Legacy --final override
    if args.final:
        args.sweep = "both"
        args.steps = 10_000

    TRAIN_STEPS = args.steps

    print("\n" + "=" * 65)
    print(f"  SENSITIVITY ANALYSIS")
    print(f"  Portfolio : {PORTFOLIO_NAME}  |  Device : {DEVICE}")
    print(f"  Sweep     : {args.sweep.upper()}")
    print(f"  Steps/run : {TRAIN_STEPS:,}")
    print(f"  Outputs   : auto-saved to results/")
    print("=" * 65)

    env_data           = load_env_data()
    train_df, bench_df = split_data(env_data)
    print(f"  Train rows: {len(train_df)} | Benchmark rows: {len(bench_df)}")

    df_delta = df_xi = None
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.sweep in ("delta", "both"):
        df_delta = sweep_delta(train_df, bench_df, TRAIN_STEPS)
        # Auto-save delta CSV immediately
        delta_path = os.path.join(RESULTS_DIR, f"sensitivity_delta_28stocks_{TRAIN_STEPS}steps.csv")
        df_delta.to_csv(delta_path, index=False)
        print(f"  Saved: {delta_path}")

    if args.sweep in ("xi", "both"):
        df_xi = sweep_xi(train_df, bench_df, TRAIN_STEPS)
        # Auto-save xi CSV immediately
        xi_path = os.path.join(RESULTS_DIR, f"sensitivity_xi_28stocks_{TRAIN_STEPS}steps.csv")
        df_xi.to_csv(xi_path, index=False)
        print(f"  Saved: {xi_path}")

    # Print result table for whichever sweeps ran
    _df_delta = df_delta if df_delta is not None else pd.DataFrame()
    _df_xi    = df_xi    if df_xi    is not None else pd.DataFrame()
    print_results_table(_df_delta, _df_xi)

    # Generate figure only when both sweeps have results
    if df_delta is not None and df_xi is not None:
        plot_figure(df_delta, df_xi)
        print("\nComplete. All outputs saved to results/")
    else:
        print("\nSweep complete. CSV saved to results/")
        print("   Run with --sweep both to also generate the combined figure.")


if __name__ == "__main__":
    main()

