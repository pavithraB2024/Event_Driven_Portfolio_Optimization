"""
evaluate_imputed_portfolio_v2.py
================================
(revised): Isolated sub-portfolio benchmark using *retrained* agents
with training-time logit-level action masking. Replaces the v1 post-hoc
mask experiment which produced numerically degenerate Static vs Event
results (differences only at the 1e-12 magnitude).

Five rows compared on the 9 news-starved (imputed) assets over the
canonical Nov 4, 2013 -> Mar 14, 2014 benchmark window:

  1. Buy-and-Hold (equal weight over the 9 imputed assets)
  2. Mean-Variance (Markowitz) over the 9 imputed assets
  3. LSTM-SAC, 9-asset action mask, no graph
  4. Static-Graph GNN-SAC, 9-asset action mask, lambda_mix=0
  5. Event-Driven GNN-SAC, 9-asset action mask, lambda_mix>0 (full pipeline)

Each DRL agent must have been trained with --restrict_to_assets so its
allocation universe matches; the canonical step-40k checkpoint is loaded
(matching the headline GNN-SAC step-40k canonical choice).

Usage:
  PYTHONPATH=. python scripts/analysis/evaluate_imputed_portfolio_v2.py \\
      --portfolio_name 28stocks --checkpoint_step 40000
"""
import argparse
import os
import sys
import types

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.agents.gnn_sac_agent import GNNSACAgent, build_action_masks_from_indices
from src.agents.memory_augmented_sac_agent import MemoryAugmentedSACAgent
from src.baselines.mean_variance import MeanVarianceStrategy
from src.data_pipeline.graph_builder import DynamicGraphBuilder
from src.environments.portfolio_env import PortfolioEnv

IMPUTED = ['BA', 'INTC', 'JPM', 'WMT', 'CVX', 'GE', 'HON', 'MSFT', 'SHEL']


def patch_env_for_graph(env, features_tensor):
    env.graph_features = features_tensor

    def get_graph_state(self_env):
        idx = self_env.current_step + self_env.window_size
        if idx >= len(self_env.graph_features):
            idx = len(self_env.graph_features) - 1
        feats = self_env.graph_features[idx]
        w = self_env.portfolio_weights
        if len(w) == feats.shape[0]:
            w = w.reshape(-1, 1)
        elif len(w) < feats.shape[0]:
            pad_len = feats.shape[0] - len(w)
            w = np.pad(w, (0, pad_len), 'constant').reshape(-1, 1)
        return np.hstack([feats, w]).flatten().astype(np.float32)

    env._get_state = types.MethodType(get_graph_state, env)
    n_nodes = features_tensor.shape[1]
    n_raw_feats = features_tensor.shape[2]
    feat_dim = n_raw_feats + 1
    env.observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(n_nodes * feat_dim,), dtype=np.float32,
    )
    return feat_dim


def compute_metrics(values, weights=None):
    values = np.array(values, dtype=float)
    if len(values) < 2:
        return dict(ARR=0, ROI=0, Sharpe=0, Sortino=0, Calmar=0, MDD=0, Turnover=0)
    returns = pd.Series(values).pct_change().dropna()
    cum_return = (values[-1] - values[0]) / values[0]
    n_days = len(values) - 1
    arr = (1 + cum_return) ** (252 / n_days) - 1 if n_days > 0 else 0
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    cummax = np.maximum.accumulate(values)
    drawdown = (values / cummax) - 1
    mdd = float(drawdown.min())
    down = returns.copy()
    down[down > 0] = 0
    sortino = (returns.mean() / down.std()) * np.sqrt(252) if down.std() > 0 else 0
    calmar = arr / abs(mdd) if abs(mdd) > 0 else 0
    turnover = 0.0
    if weights is not None and len(weights) > 1:
        w = np.array(weights)
        turnover = float(np.mean(np.sum(np.abs(w[1:] - w[:-1]), axis=1)))
    return dict(ARR=arr, ROI=cum_return, Sharpe=sharpe, Sortino=sortino,
                Calmar=calmar, MDD=mdd, Turnover=turnover)


def run_drl_eval(env, agent, agent_name, nextgen_data=None, use_graph=True):
    obs, _ = env.reset(seed=42)
    done = False
    hidden = None
    values = [env.initial_balance]
    weights = []
    vix_vals = nextgen_data.get('vix') if nextgen_data else None
    vix_means = nextgen_data.get('vix_mean') if nextgen_data else None
    while not done:
        ctx = None
        if vix_vals is not None and use_graph:
            idx = min(env.current_step + env.window_size, len(vix_vals) - 1)
            dev = getattr(agent, 'device', 'cpu')
            ctx = {
                'vix': torch.tensor([vix_vals[idx]], dtype=torch.float32).to(dev),
                'vix_mean': torch.tensor([vix_means[idx]], dtype=torch.float32).to(dev),
            }
        try:
            action, hidden = agent.select_action(obs, hidden, deterministic=True, nextgen_ctx=ctx)
        except TypeError:
            action, hidden = agent.select_action(obs, hidden, deterministic=True)
        if isinstance(action, torch.Tensor):
            action = action.cpu().detach().numpy().flatten()
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        values.append(info['portfolio_value'])
        weights.append(action)
    return values, np.array(weights)


def run_equal_weight(env, mask):
    obs, _ = env.reset(seed=42)
    done = False
    values = [env.initial_balance]
    weights = []
    w = (mask / mask.sum()).astype(np.float32)
    while not done:
        obs, _, term, trunc, info = env.step(w)
        done = term or trunc
        values.append(info['portfolio_value'])
        weights.append(w.copy())
    return values, np.array(weights)


def run_mean_variance(env, returns_imputed_only, n_assets, tickers, actual_warmup):
    mv = MeanVarianceStrategy(estimation_window=60, rebalance_freq=1, risk_aversion=2.0)
    obs, _ = env.reset(seed=42)
    done = False
    values = [env.initial_balance]
    weights = []
    t = 0
    while not done:
        idx = actual_warmup + t
        w_imputed = mv.allocate(returns_imputed_only, idx)
        w_full = np.zeros(n_assets)
        for i, tk in enumerate(IMPUTED):
            if tk in tickers:
                w_full[tickers.index(tk)] = w_imputed[i]
        obs, _, term, trunc, info = env.step(w_full)
        done = term or trunc
        values.append(info['portfolio_value'])
        weights.append(w_full)
        t += 1
    return values, np.array(weights)


def _load_checkpoint_paths(model_root, base, step):
    """Prefer the explicit step checkpoint; fall back to *_best.pth then final."""
    checkpoints = os.path.join(model_root, "checkpoints")
    candidates = [
        os.path.join(checkpoints, f"{base}_step_{step}.pth"),
        os.path.join(model_root, f"{base}_best.pth"),
        os.path.join(model_root, f"{base}.pth"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _build_gnn_agent(state_dim, n_assets, feat_dim, graph_builder, device, sector_map,
                    sector_mask, stock_mask, ckpt_sd):
    use_nextgen = any('nextgen_encoder' in k for k in ckpt_sd.keys())
    use_tgn = any('tgn.' in k for k in ckpt_sd.keys())
    use_attention_gating = any(('attn_gate' in k) or ('gate_mask' in k) for k in ckpt_sd.keys())
    use_momentum_masking = any('momentum_mask' in k for k in ckpt_sd.keys())
    use_feature_gating = any('feature_gate' in k for k in ckpt_sd.keys())
    use_dual_head_iqn = any('alpha_head' in k for k in ckpt_sd.keys())
    return GNNSACAgent(
        state_dim=state_dim,
        action_dim=n_assets,
        node_features_dim=feat_dim,
        num_nodes=graph_builder.num_nodes,
        adj_matrix=graph_builder.mixed_adj_template,
        device=device,
        hidden_dims=[256, 256],
        lstm_hidden_size=64, num_lstm_layers=1, use_graph=True,
        use_nextgen=use_nextgen,
        use_tgn=use_tgn,
        learning_rate=3e-4,
        use_attention_gating=use_attention_gating,
        use_momentum_masking=use_momentum_masking,
        use_feature_gating=use_feature_gating,
        use_dual_head_iqn=use_dual_head_iqn,
        residual_alpha=0.5,
        sector_map=sector_map.to(device),
        sector_mask=sector_mask.to(device),
        stock_mask=stock_mask.to(device),
    )


def main():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True, warn_only=True)

    parser = argparse.ArgumentParser(description="v2: isolated sub-portfolio with retrained restricted agents")
    parser.add_argument("--portfolio_name", type=str, default="28stocks")
    parser.add_argument("--checkpoint_step", type=int, default=40000,
                        help="Canonical step-N checkpoint to load (matches headline)")
    parser.add_argument("--event_suffix", type=str, default="_imputed9_event")
    parser.add_argument("--static_suffix", type=str, default="_imputed9_static")
    parser.add_argument("--lstm_suffix", type=str, default="_imputed9_lstm")
    args = parser.parse_args()

    portfolio_name = args.portfolio_name
    data_dir = os.path.join(BASE_DIR, "data")
    csv_path = os.path.join(data_dir, "processed", f"{portfolio_name}_dataset.csv")
    env_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    with open(os.path.join(data_dir, f"{portfolio_name}_tickers.txt")) as f:
        tickers = [t.strip() for t in f.read().strip().split(",") if t.strip()]
    n_assets = len([c for c in env_data.columns if c.startswith("price_")])

    imputed_indices = [tickers.index(t) for t in IMPUTED if t in tickers]
    if len(imputed_indices) != len(IMPUTED):
        missing = [t for t in IMPUTED if t not in tickers]
        print(f"WARNING: missing imputed tickers from portfolio: {missing}")
    mask = np.zeros(n_assets, dtype=np.float32)
    mask[imputed_indices] = 1.0

    # ---- Graph features / static template ----
    base_node_features = np.load(os.path.join(data_dir, f"{portfolio_name}_data.npy"))
    dates = env_data.index
    min_len = min(len(dates), base_node_features.shape[0])
    dates = dates[:min_len]
    base_node_features = base_node_features[:min_len]
    env_data = env_data.iloc[:min_len]
    if base_node_features.shape[1] > n_assets:
        base_node_features = base_node_features[:, :n_assets, :]

    processed_news_path = os.path.join(data_dir, "processed", f"news_with_events_{portfolio_name}.csv")
    graph_builder = DynamicGraphBuilder(data_dir, portfolio_name, processed_news_path)
    if graph_builder.num_stock_nodes > n_assets:
        graph_builder.resize_to_match_portfolio(n_assets)

    cache_path = os.path.join(data_dir, "processed", f"{portfolio_name}_precomputed_graphs.npz")
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        event_features = cache['features']
    else:
        event_features = []
        base_t = torch.tensor(base_node_features, dtype=torch.float32)
        for t in tqdm(range(min_len), desc="Building event-aware features"):
            g = graph_builder.get_graph(dates[t], base_t[t])
            event_features.append(g.x.numpy())
        event_features = np.array(event_features)

    benchmark_start = "2013-11-04"
    benchmark_end = "2014-03-14"
    bmask = (dates >= benchmark_start) & (dates <= benchmark_end)
    bench_start_idx = int(np.argmax(bmask))
    warmup_size = 60
    data_start_idx = max(0, bench_start_idx - warmup_size)
    actual_warmup = bench_start_idx - data_start_idx
    end_mask_idx = len(dates) - 1 - int(np.argmax(bmask[::-1]))
    test_data = env_data.iloc[data_start_idx: end_mask_idx + 1]
    test_event_feats = event_features[data_start_idx: end_mask_idx + 1]

    # Static features for static-graph eval: rebuild for the test window with news disabled.
    print("Building static-graph features for the benchmark window...")
    real_news = graph_builder.news_df.copy()
    graph_builder.news_df = pd.DataFrame(columns=real_news.columns)
    graph_builder.event_memory = np.zeros(graph_builder.num_stock_nodes)
    graph_builder.sector_memory = np.zeros(graph_builder.num_sector_nodes)
    test_static_feats = []
    base_t = torch.tensor(base_node_features, dtype=torch.float32)
    for idx in range(data_start_idx, end_mask_idx + 1):
        g = graph_builder.get_graph(dates[idx], base_t[idx])
        test_static_feats.append(g.x.numpy())
    test_static_feats = np.array(test_static_feats)
    graph_builder.news_df = real_news

    # VIX context
    try:
        vix_df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "vix.csv"), index_col=0, parse_dates=True)
        vix_series = vix_df.reindex(test_data.index).ffill().fillna(20.0).iloc[:, 0]
        vix_mean = vix_series.rolling(60, min_periods=1).mean()
        nextgen_data = {'vix': vix_series.values, 'vix_mean': vix_mean.values}
    except Exception as e:
        print(f"VIX missing: {e} -> defaulting to 20.0 constant")
        nextgen_data = {
            'vix': np.full(len(test_data), 20.0),
            'vix_mean': np.full(len(test_data), 20.0),
        }

    # Sector map + masks (must mirror training-side construction)
    sector_names_unique = sorted(set(graph_builder.sector_map.get(t, "Other") for t in graph_builder.tickers))
    sector_name_to_id = {name: i for i, name in enumerate(sector_names_unique)}
    sector_map = torch.tensor(
        [sector_name_to_id[graph_builder.sector_map.get(t, "Other")] for t in graph_builder.tickers],
        dtype=torch.long,
    )
    sector_mask, stock_mask_full = build_action_masks_from_indices(
        imputed_indices, sector_map, len(sector_names_unique)
    )
    if stock_mask_full.shape[0] != n_assets:
        buf = torch.zeros(n_assets, dtype=torch.bool)
        n_copy = min(stock_mask_full.shape[0], n_assets)
        buf[:n_copy] = stock_mask_full[:n_copy]
        stock_mask_full = buf
    stock_mask = stock_mask_full

    device = torch.device("cpu")
    model_root = os.path.join(BASE_DIR, "models")
    step = args.checkpoint_step

    event_base = f"gnn_sac_event_driven_{portfolio_name}{args.event_suffix}"
    static_base = f"gnn_sac_event_driven_{portfolio_name}{args.static_suffix}"
    lstm_base = f"lstm_sac_{portfolio_name}{args.lstm_suffix}"

    event_ckpt = _load_checkpoint_paths(model_root, event_base, step)
    static_ckpt = _load_checkpoint_paths(model_root, static_base, step)
    lstm_ckpt = _load_checkpoint_paths(model_root, lstm_base, step)
    if not (event_ckpt and static_ckpt and lstm_ckpt):
        print("Missing one or more retrained checkpoints:")
        print(f"  event:  {event_ckpt}")
        print(f"  static: {static_ckpt}")
        print(f"  lstm:   {lstm_ckpt}")
        print("Train them first via train_gnn_sac.py / train_lstm_sac_imputed.py "
              "with --restrict_to_assets and the matching model_suffix.")
        sys.exit(2)
    print(f"Event-Driven GNN-SAC ckpt:  {event_ckpt}")
    print(f"Static-Graph GNN-SAC ckpt:  {static_ckpt}")
    print(f"LSTM-SAC ckpt:              {lstm_ckpt}")

    # ---- 1. Event-Driven GNN-SAC ----
    env_e = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
    feat_dim = patch_env_for_graph(env_e, test_event_feats)
    sd = torch.load(event_ckpt, map_location='cpu')
    sd_policy = sd.get('policy', sd.get('policy_state_dict', sd))
    agent_e = _build_gnn_agent(
        state_dim=env_e.observation_space.shape[0],
        n_assets=n_assets, feat_dim=feat_dim, graph_builder=graph_builder,
        device=device, sector_map=sector_map,
        sector_mask=sector_mask, stock_mask=stock_mask, ckpt_sd=sd_policy,
    )
    agent_e.load(event_ckpt)
    vals_e, w_e = run_drl_eval(env_e, agent_e, "Event-Driven GNN-SAC", nextgen_data)

    # ---- 2. Static-Graph GNN-SAC ----
    env_s = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
    patch_env_for_graph(env_s, test_static_feats)
    sds = torch.load(static_ckpt, map_location='cpu')
    sds_policy = sds.get('policy', sds.get('policy_state_dict', sds))
    agent_s = _build_gnn_agent(
        state_dim=env_s.observation_space.shape[0],
        n_assets=n_assets, feat_dim=feat_dim, graph_builder=graph_builder,
        device=device, sector_map=sector_map,
        sector_mask=sector_mask, stock_mask=stock_mask, ckpt_sd=sds_policy,
    )
    agent_s.load(static_ckpt)
    vals_s, w_s = run_drl_eval(env_s, agent_s, "Static-Graph GNN-SAC", nextgen_data=None)

    # ---- 3. LSTM-SAC ----
    env_l = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
    agent_l = MemoryAugmentedSACAgent(
        state_dim=env_l.observation_space.shape[0],
        action_dim=env_l.action_space.shape[0],
        use_graph=False, device=device,
        hidden_dims=[256, 256], lstm_hidden_size=64, num_lstm_layers=1,
        learning_rate=3e-4,
        stock_mask=stock_mask.to(device),
    )
    agent_l.load(lstm_ckpt)
    vals_l, w_l = run_drl_eval(env_l, agent_l, "LSTM-SAC", use_graph=False)

    # ---- 4. Buy-and-Hold (equal-weight over the 9 imputed) ----
    env_bh = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
    vals_bh, w_bh = run_equal_weight(env_bh, mask)

    # ---- 5. Mean-Variance over the 9 imputed ----
    returns_df = test_data[[f'return_{t}' for t in tickers]].copy()
    returns_df.columns = [c.replace('return_', '') for c in returns_df.columns]
    returns_imputed_only = returns_df[IMPUTED].copy()
    env_mv = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
    vals_mv, w_mv = run_mean_variance(env_mv, returns_imputed_only, n_assets, tickers, actual_warmup)

    # ---- Metrics + table ----
    rows = [
        ("Buy-and-Hold (9 imputed)",         compute_metrics(vals_bh, w_bh)),
        ("Mean-Variance (9 imputed)",        compute_metrics(vals_mv, w_mv)),
        ("LSTM-SAC (no graph, mask)",        compute_metrics(vals_l, w_l)),
        ("Static-Graph GNN-SAC (mask)",      compute_metrics(vals_s, w_s)),
        ("Event-Driven GNN-SAC (Ours, mask)",compute_metrics(vals_e, w_e)),
    ]
    df = pd.DataFrame([
        dict(Model=n, ROI=m["ROI"], Sharpe=m["Sharpe"], Sortino=m["Sortino"],
             Calmar=m["Calmar"], MDD=m["MDD"], Turnover=m["Turnover"])
        for n, m in rows
    ])

    print("\n" + "=" * 90)
    print("v2: Isolated 9-asset sub-portfolio (Nov 4 2013 -> Mar 14 2014)")
    print("=" * 90)
    fmt = "{:<36} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>9}"
    print(fmt.format("Model", "ROI %", "Sharpe", "Sortino", "Calmar", "MDD %", "Turnover"))
    print("-" * 96)
    for n, m in rows:
        print(fmt.format(
            n, f"{m['ROI']*100:.2f}", f"{m['Sharpe']:.2f}", f"{m['Sortino']:.2f}",
            f"{m['Calmar']:.2f}", f"{m['MDD']*100:.2f}", f"{m['Turnover']:.4f}",
        ))
    print("=" * 90 + "\n")

    out_csv = os.path.join(BASE_DIR, "results", f"imputed_portfolio_comparison_v2_{portfolio_name}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")

    # Save wealth curves for restyle script
    curves_df = pd.DataFrame({
        'Event_Driven': vals_e, 'Static_Graph': vals_s, 'LSTM_SAC': vals_l,
        'Mean_Variance': vals_mv, 'Buy_and_Hold': vals_bh,
    })
    curves_csv = os.path.join(BASE_DIR, "results", f"imputed_wealth_curves_{portfolio_name}.csv")
    curves_df.to_csv(curves_csv, index=False)
    print(f"Saved wealth curves -> {curves_csv}")

    plt.figure(figsize=(11, 6))
    plt.plot(vals_e, label='Event-Driven GNN-SAC', linewidth=2.5)
    plt.plot(vals_s, label='Static-Graph GNN-SAC', linestyle='--', linewidth=1.6)
    plt.plot(vals_l, label='LSTM-SAC', linestyle='-.', linewidth=1.4)
    plt.plot(vals_mv, label='Mean-Variance', linestyle=':', linewidth=1.4)
    plt.plot(vals_bh, label='Buy-and-Hold', linestyle=':', linewidth=1.4, color='grey')
    plt.xlabel('Trading Days', fontsize=14)
    plt.ylabel('Portfolio Value ($)', fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.tick_params(labelsize=14)
    fig_path = os.path.join(BASE_DIR, "results", f"Figure_Imputed_Portfolio_Wealth_v2_{portfolio_name}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    eps_path = os.path.join(BASE_DIR, "results", f"Figure_Imputed_Portfolio_Wealth_v2_{portfolio_name}.eps")
    plt.savefig(eps_path, format='eps', bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"Saved -> {fig_path}")
    print(f"Saved Vector EPS -> {eps_path}")


if __name__ == "__main__":
    main()
