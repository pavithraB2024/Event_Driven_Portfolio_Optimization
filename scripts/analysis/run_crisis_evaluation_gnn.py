"""
Multi-Period Crisis Stress Test for Event-Driven GNN-SAC.

Evaluates the trained GNN-SAC agent and quantitative/passive baselines over three distinct 
out-of-sample market crisis periods:
    1. COVID-19 Crash (Feb 20, 2020 - Apr 7, 2020)
    2. China Stock Market Crash / Oil Collapse (Aug 10, 2015 - Sep 25, 2015)
    3. Volpocalypse (Jan 29, 2018 - Mar 16, 2018)

Evaluates 5 strategies for each period:
    1. Event-Driven GNN-SAC (Ours)
    2. Mean-Variance (Rebalanced every 20 days)
    3. Risk Parity (Rebalanced every 20 days)
    4. Equal-Weight (Rebalanced daily)
    5. Buy-and-Hold (Equal-dollar at t=0, no rebalancing)

Outputs (manuscript Section 4.7):
    results/Figure_7.2_Crisis_Wealth.png       Combined 3-panel wealth curves
    results/crisis_metrics_28stocks.csv        Unified performance metrics table
    results/crisis_weights_evolution_<period>.csv Active-set rotation statistics
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.data_pipeline.graph_builder import DynamicGraphBuilder
from src.agents.gnn_sac_agent import GNNSACAgent
from src.environments.portfolio_env import PortfolioEnv
from scripts.evaluation.evaluate_gnn_sac import patch_env_for_graph, load_robust_data
from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig
from src.backtesting.strategy_adapters import MeanVarianceAdapter, RiskParityAdapter

WARMUP_DAYS = 60
INITIAL_BALANCE = 100_000.0
TRANSACTION_COST = 0.0025

CRISIS_PERIODS = [
    {
        "name": "COVID-19",
        "start": "2020-02-20",
        "end": "2020-04-07",
        "title": "Panel A: COVID-19 Crash (2020)"
    },
    {
        "name": "China_2015",
        "start": "2015-08-10",
        "end": "2015-09-25",
        "title": "Panel B: China Crash & Oil Collapse (2015)"
    },
    {
        "name": "Volpocalypse_2018",
        "start": "2018-01-29",
        "end": "2018-03-16",
        "title": "Panel C: Volpocalypse (2018)"
    }
]


def slice_window(env_data, event_features, dates, start, end, warmup):
    """Slice env_data + event_features to [start-warmup, end]; return (data, features, actual_warmup)."""
    mask = (dates >= start) & (dates <= end)
    if not any(mask):
        return None, None, 0
    crisis_first = int(np.argmax(mask))
    crisis_last = len(dates) - 1 - int(np.argmax(mask[::-1]))
    data_start = max(0, crisis_first - warmup)
    actual_warmup = crisis_first - data_start
    return (
        env_data.iloc[data_start:crisis_last + 1],
        event_features[data_start:crisis_last + 1],
        actual_warmup,
    )


def run_agent(env, agent, ctx_data=None):
    """Run a deterministic episode and return (portfolio_values, weights_history)."""
    obs, _ = env.reset(seed=42)
    done = False
    hidden = None
    vals = [env.initial_balance]
    weights = []

    while not done:
        ctx = None
        if ctx_data is not None:
            i = min(env.current_step + env.window_size, len(ctx_data['vix']) - 1)
            ctx = {
                'vix': torch.tensor([ctx_data['vix'][i]], dtype=torch.float32),
                'vix_mean': torch.tensor([ctx_data['vix_mean'][i]], dtype=torch.float32),
            }
        try:
            action, hidden = agent.select_action(obs, hidden, deterministic=True, nextgen_ctx=ctx)
        except TypeError:
            action, hidden = agent.select_action(obs, hidden, deterministic=True)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()

        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        vals.append(info['portfolio_value'])
        weights.append(action)

    return np.array(vals), np.array(weights)


def run_equal_weight(env_data, n_assets):
    """Equal-weight rebalanced daily via env (incurs transaction cost)."""
    env = PortfolioEnv(env_data, initial_balance=INITIAL_BALANCE, transaction_cost=TRANSACTION_COST)
    obs, _ = env.reset(seed=42)
    done = False
    vals = [INITIAL_BALANCE]
    w = np.ones(n_assets) / n_assets
    while not done:
        obs, _, term, trunc, info = env.step(w)
        done = term or trunc
        vals.append(info['portfolio_value'])
    return np.array(vals)


def run_buy_and_hold(active_prices, n_assets):
    """Pure B&H: equal-dollar split at t=0 over active window, hold to end."""
    per_stock = INITIAL_BALANCE / n_assets
    shares = per_stock / active_prices.iloc[0].values
    return (active_prices.values * shares).sum(axis=1)


def crisis_metrics(values):
    values = np.asarray(values, dtype=float)
    rets = pd.Series(values).pct_change().dropna()
    if len(rets) == 0 or rets.std() == 0:
        return {"Total Return": 0.0, "Max Drawdown": 0.0, "Annualized Vol": 0.0,
                "Sharpe": 0.0, "Worst Day": 0.0, "Final Value": values[-1]}

    cummax = np.maximum.accumulate(values)
    return {
        "Total Return": values[-1] / values[0] - 1,
        "Max Drawdown": float(((values / cummax) - 1).min()),
        "Annualized Vol": float(rets.std() * np.sqrt(252)),
        "Sharpe": float((rets.mean() / rets.std()) * np.sqrt(252)),
        "Worst Day": float(rets.min()),
        "Final Value": float(values[-1]),
    }


def detect_arch_from_ckpt(sd, gb_num_nodes, n_assets, feat_dim):
    """Auto-detect architecture flags + dims from policy state_dict."""
    num_nodes = gb_num_nodes
    if 'lstm.weight_ih_l0' in sd:
        num_nodes = sd['lstm.weight_ih_l0'].shape[1] // 64
    action_dim = n_assets
    if 'mean.weight' in sd:
        action_dim = sd['mean.weight'].shape[0]
    return {
        'num_nodes': num_nodes,
        'action_dim': action_dim,
        'state_dim': num_nodes * feat_dim,
        'use_nextgen': any('nextgen_encoder' in k for k in sd),
        'use_tgn': any(k.startswith('tgn.') for k in sd),
        'use_attention_gating': any('attn_gate' in k or 'gate_mask' in k for k in sd),
        'use_momentum_masking': any('momentum_mask' in k for k in sd),
    }


def load_vix_context(crisis_data):
    """Load VIX series aligned to crisis_data.index, return None on failure."""
    vix_path = os.path.join(BASE_DIR, "data", "raw", "vix.csv")
    try:
        vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
        vix_s = vix_df.reindex(crisis_data.index).ffill().fillna(20.0).iloc[:, 0]
        vix_m = vix_s.rolling(60, min_periods=1).mean()
        return {'vix': vix_s.values, 'vix_mean': vix_m.values}, vix_s
    except Exception as e:
        print(f"  [!] VIX unavailable ({e}); using neutral 20.0")
        n = len(crisis_data)
        return {'vix': np.full(n, 20.0), 'vix_mean': np.full(n, 20.0)}, None


def load_tickers(price_cols, n_assets):
    tickers_path = os.path.join(BASE_DIR, "data", "28stocks_tickers.txt")
    if os.path.exists(tickers_path):
        with open(tickers_path) as f:
            t = [x.strip() for x in f.read().strip().split(',') if x.strip()]
        if len(t) >= n_assets:
            return t[:n_assets]
    return [c.replace('price_', '') for c in price_cols]


def weights_gnn_date(plot_dates, idx):
    """Return ISO date for the weight snapshot at index `idx`, clipped to range."""
    idx = max(0, min(idx, len(plot_dates) - 1))
    return str(plot_dates[idx].date())


def main():
    parser = argparse.ArgumentParser(description="Out-of-Sample Crisis Stress Tests for GNN-SAC")
    parser.add_argument("--portfolio_name", type=str, default="28stocks")
    parser.add_argument("--checkpoint_step", type=int, default=40000,
                        help="Checkpoint step (default: 40000 = manuscript SOTA)")
    args = parser.parse_args()

    print("=" * 80)
    print("OUT-OF-SAMPLE CRISIS STRESS TESTS  -  Event-Driven GNN-SAC")
    print("=" * 80)

    portfolio_name = args.portfolio_name
    data_dir = os.path.join(BASE_DIR, "data")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Data
    env_data = load_robust_data(data_dir, portfolio_name)
    if env_data is None or len(env_data) == 0:
        print("ERROR: dataset load failed.")
        return 1

    base_node_features = np.load(os.path.join(data_dir, f"{portfolio_name}_data.npy"))
    dates = env_data.index
    min_len = min(len(dates), base_node_features.shape[0])
    dates, env_data, base_node_features = dates[:min_len], env_data.iloc[:min_len], base_node_features[:min_len]

    price_cols = [c for c in env_data.columns if c.startswith('price_')]
    n_assets = len(price_cols)
    print(f"Portfolio: {n_assets} assets | Dataset span {dates[0].date()} -> {dates[-1].date()}")

    if base_node_features.shape[1] > n_assets:
        base_node_features = base_node_features[:, :n_assets, :]

    # 2. Graph features (cached or rebuilt)
    news_path = os.path.join(data_dir, "processed", f"news_with_events_{portfolio_name}.csv")
    gb = DynamicGraphBuilder(data_dir, portfolio_name, news_path)
    gb.resize_to_match_portfolio(n_assets)

    cache_path = os.path.join(data_dir, "processed", f"{portfolio_name}_precomputed_graphs.npz")
    if os.path.exists(cache_path):
        print(f"Loading cached graphs: {cache_path}")
        event_features = np.load(cache_path)['features'][:min_len]
    else:
        print("No cache found - computing graph features per timestep...")
        event_features = []
        bf = torch.tensor(base_node_features, dtype=torch.float32)
        for t in tqdm(range(min_len)):
            event_features.append(gb.get_graph(dates[t], bf[t]).x.numpy())
        event_features = np.array(event_features)

    # Prepare plotting
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    all_metrics_df_list = []
    all_curves = []

    # Loop over all crisis periods
    for p_idx, p in enumerate(CRISIS_PERIODS):
        print("\n" + "=" * 80)
        print(f"Running Crisis Period: {p['name']} ({p['start']} -> {p['end']})")
        print("=" * 80)

        # 3. Slice to crisis window
        crisis_data, crisis_features, warmup = slice_window(
            env_data, event_features, dates, p['start'], p['end'], WARMUP_DAYS)
        if crisis_data is None:
            print(f"ERROR: no data in window {p['start']} -> {p['end']}")
            continue

        n_active = len(crisis_data) - warmup
        print(f"Crisis slice: {crisis_data.index[0].date()} -> {crisis_data.index[-1].date()}")
        print(f"  Total rows: {len(crisis_data)}  ({warmup} warm-up + {n_active} active)")

        # 4. VIX macro context
        nextgen_data, vix_series = load_vix_context(crisis_data)

        # 5. Set up agent
        env_gnn = PortfolioEnv(crisis_data, initial_balance=INITIAL_BALANCE, transaction_cost=TRANSACTION_COST)
        feat_dim = patch_env_for_graph(env_gnn, crisis_features)

        ckpt_path = os.path.join(BASE_DIR, "models", "checkpoints",
                                 f"gnn_sac_{portfolio_name}_step_{args.checkpoint_step}.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(BASE_DIR, "models", f"gnn_sac_event_driven_{portfolio_name}.pth")
        if not os.path.exists(ckpt_path):
            print(f"ERROR: no checkpoint at {ckpt_path}")
            return 1

        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt.get('policy') or ckpt.get('policy_state_dict') or ckpt

        arch = detect_arch_from_ckpt(sd, gb.num_nodes, n_assets, feat_dim)
        agent = GNNSACAgent(
            state_dim=arch['state_dim'], action_dim=arch['action_dim'],
            node_features_dim=feat_dim, num_nodes=arch['num_nodes'],
            adj_matrix=gb.mixed_adj_template[:arch['num_nodes'], :arch['num_nodes']],
            device='cpu', hidden_dims=[256, 256],
            lstm_hidden_size=64, num_lstm_layers=1, use_graph=True,
            use_nextgen=arch['use_nextgen'], use_tgn=arch['use_tgn'],
            use_attention_gating=arch['use_attention_gating'],
            use_momentum_masking=arch['use_momentum_masking'],
            learning_rate=3e-4,
        )
        agent.load(ckpt_path)

        # 6. Run all strategies
        print("[1/5] GNN-SAC (deterministic)...")
        vals_gnn, weights_gnn = run_agent(env_gnn, agent, ctx_data=nextgen_data)

        print("[2/5] Equal-Weight (daily rebalance)...")
        vals_ew = run_equal_weight(crisis_data, n_assets)

        print("[3/5] Buy-and-Hold (no rebalance)...")
        active_prices = crisis_data.iloc[warmup:][price_cols]
        vals_bh = run_buy_and_hold(active_prices, n_assets)

        n_pts = min(len(vals_gnn), len(vals_ew), len(vals_bh))
        v_gnn = vals_gnn[:n_pts]
        v_ew = vals_ew[:n_pts]
        v_bh = vals_bh[:n_pts]
        plot_dates = crisis_data.index[warmup:warmup + n_pts]
        if len(plot_dates) < n_pts:
            plot_dates = crisis_data.index[-n_pts:]

        # Run Baselines (Mean-Variance & Risk Parity)
        # Use exact alignment to active trading dates for GNN-SAC
        active_data = crisis_data.iloc[warmup:warmup + n_pts - 1]
        return_cols_active = [c for c in active_data.columns if c.startswith('return_')]
        returns_df = active_data[return_cols_active].copy()
        returns_df.columns = [c.replace('return_', '') for c in returns_df.columns]

        print("[4/5] Mean-Variance (20-day rebalance)...")
        config = BacktestConfig(
            initial_capital=INITIAL_BALANCE,
            transaction_cost=TRANSACTION_COST,
            slippage=0.0005,
            rebalance_frequency=20,
            risk_free_rate=0.02
        )
        engine = BacktestEngine(config)
        mv_adapter = MeanVarianceAdapter(risk_aversion=2.0)
        mv_res = engine.run(mv_adapter, active_data, returns_df)
        v_mv = mv_res.portfolio_values.values[:n_pts]

        print("[5/5] Risk Parity (20-day rebalance)...")
        rp_adapter = RiskParityAdapter()
        rp_res = engine.run(rp_adapter, active_data, returns_df)
        v_rp = rp_res.portfolio_values.values[:n_pts]

        # Calculate metrics
        metrics = {
            "Event-Linked GNN-SAC (Ours)": crisis_metrics(v_gnn),
            "Mean-Variance": crisis_metrics(v_mv),
            "Risk Parity": crisis_metrics(v_rp),
            "Equal Weight": crisis_metrics(v_ew),
            "Buy-and-Hold": crisis_metrics(v_bh),
        }

        # Print metrics
        print("-" * 92)
        print(f"{'Strategy':<28} | {'Total Ret':>10} | {'Max DD':>10} | {'Sharpe':>7} | "
              f"{'Worst Day':>10} | {'Final $':>11}")
        print("-" * 92)
        for name, m in metrics.items():
            print(f"{name:<28} | {m['Total Return']*100:>+9.2f}% | {m['Max Drawdown']*100:>+9.2f}% | "
                  f"{m['Sharpe']:>+6.2f} | {m['Worst Day']*100:>+8.2f}% | ${m['Final Value']:>10,.0f}")
        print("-" * 92)

        # Store for CSV output
        df_m = pd.DataFrame(metrics).T
        df_m['Period'] = p['name']
        all_metrics_df_list.append(df_m)

        # Save wealth curves for restyle script
        for day_i in range(n_pts):
            all_curves.append({
                'Period': p['name'],
                'Date': str(plot_dates[day_i].date()) if hasattr(plot_dates[day_i], 'date') else str(plot_dates[day_i]),
                'GNN_SAC': v_gnn[day_i] / v_gnn[0],
                'Equal_Weight': v_ew[day_i] / v_ew[0],
                'Buy_and_Hold': v_bh[day_i] / v_bh[0],
            })

        # Plot Panel (Option B: GNN-SAC, EW, B&H)
        ax = axes[p_idx]
        panel_labels = ['(a) COVID-19 Crash', '(b) China Crash & Oil Collapse', '(c) Volpocalypse']
        ax.plot(plot_dates, v_gnn / v_gnn[0], label='Event-Driven GNN-SAC',
                color='#d62728', linewidth=2.5)
        ax.plot(plot_dates, v_ew / v_ew[0], label='Equal Weight',
                color='grey', linestyle='--', linewidth=1.5)
        ax.plot(plot_dates, v_bh / v_bh[0], label='Buy-and-Hold',
                color='#1f77b4', linestyle='-.', linewidth=1.5)
        ax.set_ylabel('Normalized Wealth', fontsize=14)
        ax.set_xlabel('Date', fontsize=14)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.tick_params(labelsize=14)
        if p_idx == 0:
            ax.legend(loc='upper right', fontsize=12, frameon=True,
                      facecolor='white', edgecolor='none', framealpha=0.9)
        if p_idx < len(panel_labels):
            ax.text(0.02, 0.97, panel_labels[p_idx], transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='none', alpha=0.9))

        # Save Active-Set Rotation
        if len(weights_gnn) > 0:
            tickers = load_tickers(price_cols, n_assets)
            n_w = len(weights_gnn)
            HOLD_THRESH = 0.01
            held_mask = weights_gnn > HOLD_THRESH

            snap = {"Start": 0, "Peak Crash": n_w // 2, "Recovery": n_w - 1}
            {k: weights_gnn_date(plot_dates, v) for k, v in snap.items()}

            avg_w = weights_gnn.mean(axis=0)
            hold_freq = held_mask.mean(axis=0)
            ever_held = held_mask.any(axis=0)
            held_idx = np.where(ever_held)[0]
            held_idx = held_idx[np.argsort(-avg_w[held_idx])]

            rows = []
            for ai in held_idx:
                t = tickers[ai] if ai < len(tickers) else f"Asset_{ai}"
                held = {k: bool(held_mask[snap[k], ai]) for k in snap}
                s_w, m_w, e_w = (weights_gnn[snap[k], ai] * 100 for k in ("Start", "Peak Crash", "Recovery"))
                if all(held.values()):
                    role = "Core (held throughout)"
                elif held["Peak Crash"] and not held["Start"] and not held["Recovery"]:
                    role = "Crisis-only entry"
                elif held["Start"] and not held["Peak Crash"]:
                    role = "Pre-crisis exit"
                elif held["Recovery"] and not held["Peak Crash"]:
                    role = "Recovery re-entry"
                else:
                    role = "Partial-period hold"
                rows.append({
                    "Asset": t,
                    "Hold_Frequency": float(hold_freq[ai]),
                    "Avg_Weight_Pct": float(avg_w[ai] * 100),
                    "Start_Held": held["Start"], "Start_Weight_Pct": float(s_w),
                    "PeakCrash_Held": held["Peak Crash"], "PeakCrash_Weight_Pct": float(m_w),
                    "Recovery_Held": held["Recovery"], "Recovery_Weight_Pct": float(e_w),
                    "Role": role,
                })
            weights_path = os.path.join(results_dir, f"crisis_weights_evolution_{p['name']}.csv")
            pd.DataFrame(rows).to_csv(weights_path, index=False)
            print(f"Saved active-set rotation -> {weights_path}")

    # Save wealth curves for restyle script
    curves_path = os.path.join(results_dir, f"crisis_wealth_curves_{portfolio_name}.csv")
    pd.DataFrame(all_curves).to_csv(curves_path, index=False)
    print(f"Saved wealth curves -> {curves_path}")

    # Compile and save unified plot
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "Figure_7.2_Crisis_Wealth.png")
    plt.savefig(plot_path, dpi=300)
    eps_path = os.path.join(results_dir, "Figure_7.2_Crisis_Wealth.eps")
    plt.savefig(eps_path, format='eps', bbox_inches='tight', pad_inches=0.02)
    print(f"Saved Vector EPS -> {eps_path}")
    plt.close()
    print(f"\nSaved combined 3-panel wealth plot -> {plot_path}")

    # Compile and save unified CSV metrics
    all_metrics_df = pd.concat(all_metrics_df_list)
    metrics_path = os.path.join(results_dir, f"crisis_metrics_{portfolio_name}.csv")
    all_metrics_df.to_csv(metrics_path)
    print(f"Saved compiled metrics table -> {metrics_path}")

    print("\n" + "=" * 80)
    print("ALL CRISIS STRESS TESTS COMPLETE")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
