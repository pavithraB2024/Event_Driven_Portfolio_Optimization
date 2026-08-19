"""Generate Figure_Scaling_Wealth.png for §5.4 of the manuscript.

Plots cumulative wealth on the sp50_pit2013 50-asset benchmark window for the
event-trained GNN-SAC (best validation checkpoint) against classical baselines.
The GNN eval CSV's `Date` column is mislabeled; we remap to the canonical
Nov 4, 2013 -- Mar 14, 2014 window using the trading-day index taken from the
baseline portfolio_values series.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BACKTEST = REPO / "simulations" / "backtesting_results" / "sp50_pit2013"
EVAL_CSV = REPO / "results" / "gnn_sac_eval_details_sp50_pit2013_event_step5000.csv"
OUT_PNG = REPO / "results" / "Figure_Scaling_Wealth.png"
OUT_EPS = REPO / "results" / "Figure_Scaling_Wealth.eps"

BASELINES = {
    "Buy-and-Hold": ("buy_and_hold", "#888888", "-"),
    "Mean-Variance": ("mean_variance", "#1f77b4", "--"),
    "Risk Parity": ("risk_parity", "#2ca02c", "-."),
    "Equal-Weight": ("equal_weight", "#9467bd", ":"),
    "Merton": ("merton", "#8c564b", (0, (3, 1, 1, 1))),
}


def load_baseline(name: str) -> pd.Series:
    df = pd.read_csv(BACKTEST / name / "portfolio_values.csv")
    df = df.groupby("Date", as_index=False).last()
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")["0"].rename(name)


def load_gnn(canonical_dates: pd.DatetimeIndex) -> pd.Series:
    df = pd.read_csv(EVAL_CSV)
    pv = df["Portfolio_Value"].to_numpy()
    pv = pv / pv[0] * 100_000.0
    return pd.Series(pv, index=canonical_dates[-len(pv):], name="Event-Driven GNN-SAC")


def main() -> None:
    baseline_curves = {label: load_baseline(d[0]) for label, d in BASELINES.items()}
    canonical_dates = baseline_curves["Buy-and-Hold"].index
    gnn = load_gnn(canonical_dates)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for label, (_, color, ls) in BASELINES.items():
        s = baseline_curves[label]
        ax.plot(s.index, s.values, color=color, linestyle=ls, linewidth=1.6, label=label, alpha=0.85)
    ax.plot(gnn.index, gnn.values, color="#d62728", linewidth=2.4, label=gnn.name)

    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Portfolio Value ($)", fontsize=14)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left", frameon=True, fontsize=14)
    ax.tick_params(labelsize=14)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    fig.savefig(OUT_EPS, format="eps")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_EPS}")


if __name__ == "__main__":
    main()
