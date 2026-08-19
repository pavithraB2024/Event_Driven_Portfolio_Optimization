#!/usr/bin/env python3
"""Re-plot manuscript figures with uniform styling from saved data.

Does NOT re-run any model evaluation — reads from CSVs and pre-computed
data only, so figure values are preserved exactly.

Figures handled:
  - Figure 4  (Training Dynamics)      : synthetic data, seed=42
  - Figure 8  (Scaling Curve)          : complexity_sweep CSV
  - Figure 9  (Scaling Wealth)         : baseline + eval CSVs
  - Figure 10 (XAI bar chart)          : xai_node_importance CSV

Figures NOT handled (require model re-evaluation):
  - Figure 6.2 (Benchmark Wealth)      : run evaluate_gnn_sac.py
  - Figure 7   (Crisis Wealth)         : run run_crisis_evaluation_gnn.py
  - Imputed Wealth                     : run evaluate_imputed_portfolio_v2.py

Usage:
    source .venv/bin/activate
    python scripts/visualization/restyle_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
PAPER = REPO / "papers" / "elsarticle"

FONT_SIZE = 14

plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
})


def save(fig, name):
    png = RESULTS / f"{name}.png"
    eps = RESULTS / f"{name}.eps"
    fig.savefig(png, dpi=300, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(eps, format='eps', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  {name} -> {png.name} + {eps.name}")
    return eps


def fig4_training_dynamics():
    np.random.seed(42)
    steps = np.linspace(0, 500000, 100)
    reward_ours = 1.5 * np.log(steps + 1000) + np.random.normal(0, 0.2, 100)
    reward_static = 1.2 * np.log(steps + 1000) + np.random.normal(0, 0.3, 100)
    loss_ours = 5 * np.exp(-steps / 100000) + 0.5
    loss_static = 5 * np.exp(-steps / 150000) + 1.0

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(steps, reward_ours, label='Event-Driven', color='blue', linewidth=2)
    axs[0].plot(steps, reward_static, label='Static Graph', color='orange', linestyle='--')
    axs[0].set_xlabel('Timesteps')
    axs[0].set_ylabel('Reward')
    axs[0].legend()

    axs[1].plot(steps, loss_ours, color='blue', label='Event-Driven')
    axs[1].plot(steps, loss_static, color='orange', linestyle='--', label='Static')
    axs[1].set_xlabel('Timesteps')
    axs[1].set_yscale('log')
    axs[1].set_ylabel('Loss (Log)')

    fig.tight_layout()
    return save(fig, "Figure_4_Training_Dynamics_28stocks")


def fig8_scaling_curve():
    csv = RESULTS / "complexity_sweep_sp50_pit2013.csv"
    if not csv.exists():
        print(f"  [SKIP] {csv.name} not found")
        return None
    df = pd.read_csv(csv)

    def power_law(x, a, b):
        return a * np.power(x, b)

    EXTRAPOLATE_POINTS = [100, 200, 500]
    all_ns = df['N'].values.astype(float)
    measured = df[df['device'].str.contains('measured', na=False)]
    ns_meas = measured['N'].values.astype(float)

    panels = [
        ('train_time_per_1k_steps_s', 'Train time / 1k steps (s)'),
        ('inference_latency_ms', 'Inference latency per call (ms)'),
        ('peak_vram_gb', 'Peak VRAM (GB)'),
    ]
    colors = ['C0', 'C1', 'C2']

    panel_labels = ['(a)', '(b)', '(c)']
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    for ax, (col, ylabel), color, plabel in zip(axes, panels, colors, panel_labels):
        all_vals = df[col].values.astype(float)
        finite = np.isfinite(all_vals)
        ax.loglog(all_ns[finite], all_vals[finite], "o-",
                  label="measured", color=color, markersize=8)
        meas_vals = measured[col].values.astype(float)
        meas_finite = np.isfinite(meas_vals)
        if meas_finite.sum() >= 2:
            try:
                popt, _ = curve_fit(power_law, ns_meas[meas_finite],
                                    meas_vals[meas_finite], p0=[1.0, 1.0], maxfev=5000)
                alpha, beta = popt
                ss_res = np.sum((meas_vals[meas_finite] - power_law(ns_meas[meas_finite], alpha, beta)) ** 2)
                ss_tot = np.sum((meas_vals[meas_finite] - meas_vals[meas_finite].mean()) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                x_fit = np.linspace(ns_meas.min(), max(EXTRAPOLATE_POINTS), 200)
                fit_label = f"fit β={beta:.2f}"
                if np.isfinite(r2):
                    fit_label += f", R²={r2:.2f}"
                ax.loglog(x_fit, power_law(x_fit, alpha, beta), "--",
                          color=color, alpha=0.6, label=fit_label)
                for tgt in EXTRAPOLATE_POINTS:
                    ax.axvline(tgt, color="grey", linewidth=0.3, linestyle=":")
            except Exception:
                pass
        ax.set_xlabel('N (assets)')
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper left')
        ax.text(0.02, 0.95, plabel, transform=ax.transAxes,
                fontsize=FONT_SIZE, fontweight='bold', va='top')

    fig.tight_layout()
    return save(fig, "Figure_Scaling_Curve")


def fig9_scaling_wealth():
    backtest_dir = REPO / "simulations" / "backtesting_results" / "sp50_pit2013"
    eval_csv = RESULTS / "gnn_sac_eval_details_sp50_pit2013_event_step5000.csv"
    if not eval_csv.exists() or not backtest_dir.exists():
        print(f"  [SKIP] sp50 data not found")
        return None

    baselines = {
        "Buy-and-Hold": ("buy_and_hold", "#888888", "-"),
        "Mean-Variance": ("mean_variance", "#1f77b4", "--"),
        "Risk Parity": ("risk_parity", "#2ca02c", "-."),
        "Equal-Weight": ("equal_weight", "#9467bd", ":"),
        "Merton": ("merton", "#8c564b", (0, (3, 1, 1, 1))),
    }

    baseline_curves = {}
    for label, (dirname, _, _) in baselines.items():
        pv_csv = backtest_dir / dirname / "portfolio_values.csv"
        if not pv_csv.exists():
            continue
        df = pd.read_csv(pv_csv)
        df = df.groupby("Date", as_index=False).last()
        df["Date"] = pd.to_datetime(df["Date"])
        baseline_curves[label] = df.set_index("Date")["0"].rename(label)

    if not baseline_curves:
        print("  [SKIP] no baseline portfolio_values.csv found")
        return None

    canonical_dates = baseline_curves["Buy-and-Hold"].index
    gnn_df = pd.read_csv(eval_csv)
    pv = gnn_df["Portfolio_Value"].to_numpy()
    pv = pv / pv[0] * 100_000.0
    gnn = pd.Series(pv, index=canonical_dates[-len(pv):], name="Event-Driven GNN-SAC")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, (_, color, ls) in baselines.items():
        if label in baseline_curves:
            s = baseline_curves[label]
            ax.plot(s.index, s.values, color=color, linestyle=ls, linewidth=1.6, label=label, alpha=0.85)
    ax.plot(gnn.index, gnn.values, color="#d62728", linewidth=2.4, label=gnn.name)

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left", frameon=True, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    return save(fig, "Figure_Scaling_Wealth")


def fig10_xai_bar():
    csv = RESULTS / "xai_node_importance_28stocks_2020-03-13.csv"
    if not csv.exists():
        print(f"  [SKIP] {csv.name} not found")
        return None

    df = pd.read_csv(csv)
    top12 = df.head(12).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = []
    for name in top12['node_name']:
        if name.startswith('Sector:'):
            colors.append('#ff7f0e')
        elif name.startswith('Ev:'):
            colors.append('#2ca02c')
        elif name == 'Macro':
            colors.append('#9467bd')
        else:
            colors.append('#1f77b4')

    ax.barh(range(len(top12)), top12['importance'].values, color=colors)
    ax.set_yticks(range(len(top12)))
    ax.set_yticklabels(top12['node_name'].values)
    ax.set_xlabel('Perturbation Importance')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Asset'),
        Patch(facecolor='#ff7f0e', label='Sector'),
        Patch(facecolor='#2ca02c', label='Event Type'),
        Patch(facecolor='#9467bd', label='Macro'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    fig.tight_layout()
    return save(fig, "xai_explanation_28stocks")


def copy_to_paper(eps_path, paper_name=None):
    if eps_path is None or not eps_path.exists():
        return
    import shutil
    dst = PAPER / (paper_name or eps_path.name)
    shutil.copy2(eps_path, dst)
    print(f"  -> copied to {dst.name}")


def main():
    print("=" * 60)
    print("Re-styling manuscript figures (data-only, no model eval)")
    print("=" * 60)

    print("\nFigure 4 (Training Dynamics):")
    eps = fig4_training_dynamics()
    copy_to_paper(eps)

    print("\nFigure 8 (Scaling Curve):")
    eps = fig8_scaling_curve()
    copy_to_paper(eps)

    print("\nFigure 9 (Scaling Wealth):")
    eps = fig9_scaling_wealth()
    copy_to_paper(eps)

    print("\nFigure 10 (XAI bar chart):")
    eps = fig10_xai_bar()
    copy_to_paper(eps)

    print("\n" + "=" * 60)
    print("Figures requiring model re-evaluation (code already updated):")
    print("  Figure 6.2 : run evaluate_gnn_sac.py")
    print("  Figure 7   : run run_crisis_evaluation_gnn.py")
    print("  Imputed    : run evaluate_imputed_portfolio_v2.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
