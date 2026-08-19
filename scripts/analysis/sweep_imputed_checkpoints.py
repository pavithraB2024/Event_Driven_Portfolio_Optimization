"""
sweep_imputed_checkpoints.py
============================
diagnostic: sweep every retrained 9-asset checkpoint across the
Nov 4 2013 -> Mar 14 2014 benchmark window so we can see whether the
best-by-validation-Sharpe checkpoint is actually the best on the
held-out benchmark window. Investigates whether val->bench transfer
is breaking down for the imputed-9 retrains.

Reuses the setup from evaluate_imputed_portfolio_v2.py, but loops
through every `*_step_*.pth` checkpoint per agent rather than loading
a single canonical one.

Usage:
  PYTHONPATH=. python scripts/analysis/sweep_imputed_checkpoints.py
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from scripts.analysis.evaluate_imputed_portfolio_v2 import (  # noqa: E402
    IMPUTED, _build_gnn_agent, compute_metrics, patch_env_for_graph,
    run_drl_eval,
)
from src.agents.gnn_sac_agent import build_action_masks_from_indices  # noqa: E402
from src.agents.memory_augmented_sac_agent import MemoryAugmentedSACAgent  # noqa: E402
from src.data_pipeline.graph_builder import DynamicGraphBuilder  # noqa: E402
from src.environments.portfolio_env import PortfolioEnv  # noqa: E402


def _discover_step_checkpoints(model_root, agent_kind, portfolio_name, suffix):
    """Return ordered list of (step:int, path:str) for an agent."""
    ckpt_dir = os.path.join(model_root, "checkpoints")
    # Training script saves as gnn_sac_{portfolio_name}{suffix}_step_X.pth
    # or lstm_sac_{portfolio_name}{suffix}_step_X.pth (no "event_driven" prefix).
    prefix = ("lstm_sac_" if agent_kind == "lstm" else "gnn_sac_") + f"{portfolio_name}{suffix}_step_"
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)\.pth$")
    out = []
    for fn in os.listdir(ckpt_dir):
        m = pat.match(fn)
        if m:
            out.append((int(m.group(1)), os.path.join(ckpt_dir, fn)))
    out.sort(key=lambda t: t[0])
    return out


def main():
    parser = argparse.ArgumentParser(description="Sweep all imputed-9 checkpoints on benchmark window")
    parser.add_argument("--portfolio_name", type=str, default="28stocks")
    parser.add_argument("--event_suffix", type=str, default="_imputed9_event")
    parser.add_argument("--static_suffix", type=str, default="_imputed9_static")
    parser.add_argument("--lstm_suffix", type=str, default="_imputed9_lstm")
    parser.add_argument("--out_csv", type=str,
                        default=os.path.join(BASE_DIR, "results", "imputed_checkpoint_sweep.csv"))
    args = parser.parse_args()

    portfolio_name = args.portfolio_name
    data_dir = os.path.join(BASE_DIR, "data")
    csv_path = os.path.join(data_dir, "processed", f"{portfolio_name}_dataset.csv")
    env_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    with open(os.path.join(data_dir, f"{portfolio_name}_tickers.txt")) as f:
        tickers = [t.strip() for t in f.read().strip().split(",") if t.strip()]
    n_assets = len([c for c in env_data.columns if c.startswith("price_")])
    imputed_indices = [tickers.index(t) for t in IMPUTED if t in tickers]

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
    if not os.path.exists(cache_path):
        raise SystemExit(f"Missing precomputed graphs cache at {cache_path}. "
                         "Run train_gnn_sac.py once to build it.")
    event_features = np.load(cache_path)['features']

    benchmark_start = "2013-11-04"
    benchmark_end = "2014-03-14"
    bmask = (dates >= benchmark_start) & (dates <= benchmark_end)
    bench_start_idx = int(np.argmax(bmask))
    warmup_size = 60
    data_start_idx = max(0, bench_start_idx - warmup_size)
    end_mask_idx = len(dates) - 1 - int(np.argmax(bmask[::-1]))
    test_data = env_data.iloc[data_start_idx: end_mask_idx + 1]
    test_event_feats = event_features[data_start_idx: end_mask_idx + 1]

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

    try:
        vix_df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "vix.csv"),
                             index_col=0, parse_dates=True)
        vix_series = vix_df.reindex(test_data.index).ffill().fillna(20.0).iloc[:, 0]
        vix_mean = vix_series.rolling(60, min_periods=1).mean()
        nextgen_data = {'vix': vix_series.values, 'vix_mean': vix_mean.values}
    except Exception as e:
        print(f"VIX missing: {e} -> defaulting to 20.0 constant")
        nextgen_data = {'vix': np.full(len(test_data), 20.0),
                        'vix_mean': np.full(len(test_data), 20.0)}

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

    # Build a fresh env + feat_dim once per model kind (event/static/lstm share env shape)
    env_e = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
    feat_dim = patch_env_for_graph(env_e, test_event_feats)
    env_s = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
    patch_env_for_graph(env_s, test_static_feats)
    env_l = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)

    sweeps = [
        ("event", args.event_suffix, env_e, test_event_feats, True,  nextgen_data),
        ("static", args.static_suffix, env_s, test_static_feats, True, None),
        ("lstm",  args.lstm_suffix,  env_l, None,             False, None),
    ]

    all_rows = []
    for kind, suffix, env, feats, is_gnn, ctx in sweeps:
        ckpts = _discover_step_checkpoints(model_root, kind, portfolio_name, suffix)
        print(f"\n=== Sweeping {kind} ({len(ckpts)} checkpoints) ===")
        if not ckpts:
            print("  (no checkpoints discovered)")
            continue
        for step, path in ckpts:
            sd = torch.load(path, map_location='cpu')
            sd_policy = sd.get('policy', sd.get('policy_state_dict', sd))
            if is_gnn:
                agent = _build_gnn_agent(
                    state_dim=env.observation_space.shape[0],
                    n_assets=n_assets, feat_dim=feat_dim, graph_builder=graph_builder,
                    device=device, sector_map=sector_map,
                    sector_mask=sector_mask, stock_mask=stock_mask, ckpt_sd=sd_policy,
                )
            else:
                agent = MemoryAugmentedSACAgent(
                    state_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.shape[0],
                    use_graph=False, device=device,
                    hidden_dims=[256, 256], lstm_hidden_size=64, num_lstm_layers=1,
                    learning_rate=3e-4, stock_mask=stock_mask.to(device),
                )
            agent.load(path)
            vals, weights = run_drl_eval(env, agent, kind, nextgen_data=ctx, use_graph=is_gnn)
            m = compute_metrics(vals, weights)
            row = dict(agent=kind, step=step, ROI=m['ROI'], Sharpe=m['Sharpe'],
                       Sortino=m['Sortino'], Calmar=m['Calmar'], MDD=m['MDD'],
                       Turnover=m['Turnover'], path=os.path.basename(path))
            all_rows.append(row)
            print(f"  step {step:>6}: ROI={m['ROI']*100:+6.2f}%  Sharpe={m['Sharpe']:+5.2f}  "
                  f"MDD={m['MDD']*100:+6.2f}%  Turnover={m['Turnover']:.4f}")

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved sweep -> {args.out_csv}")

    print("\n" + "=" * 80)
    print("Best benchmark Sharpe per agent (across all step checkpoints)")
    print("=" * 80)
    for kind in ("event", "static", "lstm"):
        sub = df[df.agent == kind]
        if sub.empty:
            continue
        best = sub.loc[sub.Sharpe.idxmax()]
        print(f"  {kind:7s} step {int(best.step):>6}: "
              f"Sharpe={best.Sharpe:+.2f}  ROI={best.ROI*100:+.2f}%  "
              f"MDD={best.MDD*100:+.2f}%  ({best.path})")


if __name__ == "__main__":
    main()
