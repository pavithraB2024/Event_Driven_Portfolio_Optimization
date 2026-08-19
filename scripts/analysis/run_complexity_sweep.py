"""
— Computational Complexity Sweep.

For each N in {10, 28, 50} drawn deterministically from the top of the
liquidity-ranked sp50_pit2013 universe, instantiate a NextGen GNN-SAC agent
on the real DynamicGraphBuilder (sectoral + market + event edges) and measure:

  * single-sample inference latency (ms)
  * per-1k SAC update-step training time (s)
  * peak GPU memory (GB, if CUDA available; CPU run records NaN)

The script fits a power law T(N) = alpha * N^beta (and a separate fit for
peak VRAM) to the *measured* points only, reports R^2, and extrapolates
to N in {100, 200, 500}. The CSV and figure both distinguish measured rows
from extrapolated rows so the manuscript can cite each regime cleanly.

Outputs:
  results/complexity_sweep_sp50_pit2013.csv
  results/Figure_Scaling_Curve.{png,eps}

Use --smoke_test to verify shape / control-flow correctness locally
before running on a GPU instance.

NOTE: a previous revision of this script extended the sweep to N=100/200 via
a synthetic identity-adjacency MockGB. Those points produced an artificially
flat / negative-slope power-law because the GNN had no edges to propagate
through, and have been removed. To measure beyond N=50, build a real
sp100/sp200_pit2013 universe via prepare_sp50_pit2013.py's pattern.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uuid
from src.utils.remote_logger import RemoteLogger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.data_pipeline.graph_builder import DynamicGraphBuilder  # noqa: E402
from src.agents.gnn_sac_agent import GNNSACAgent  # noqa: E402


SWEEP_POINTS = [10, 28, 50]
EXTRAPOLATE_POINTS = [100, 200, 500]
BASE_PORTFOLIO = "sp50_pit2013"
NODE_RAW_FEATS = 7  # base OHLCV-style features per node (typical for this codebase)
SENTIMENT_CHANNELS = 2  # polarity + scalar (graph_builder appends both)
FEAT_DIM_NODE = NODE_RAW_FEATS + SENTIMENT_CHANNELS + 1  # +1 = portfolio weight column


def resolve_device(force_cpu=False):
    """Pick a usable device. CUDA reports as available even on sub-sm_75 cards
    where cuDNN-backed ops (LSTM flatten_parameters) will crash; detect that
    case and fall back to CPU so local smoke tests stay green."""
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    cc = cc_major * 10 + cc_minor
    if cc < 75:
        print(f"WARNING: CUDA device sm_{cc} is below sm_75; falling back to CPU. "
              f"Manuscript-grade numbers must be regenerated on a Kaggle P100/T4 session.")
        return torch.device("cpu")
    return torch.device("cuda")


def build_sub_universe(n_assets, data_dir):
    """Build a graph builder for the top-n_assets slice of sp50_pit2013."""
    # processed_news_path is optional for our purposes (we only read graph shapes,
    # not actual event-time computation). Point at the real file when present;
    # the builder tolerates an empty/missing path for the resize use case.
    news_path = os.path.join(
        data_dir, "processed", f"news_with_events_{BASE_PORTFOLIO}.csv"
    )
    if not os.path.exists(news_path):
        # Fall back to the 28-stock news file just to get the builder to construct;
        # we never actually call get_graph(), so the content is irrelevant.
        news_path = os.path.join(data_dir, "processed", "news_with_events_28stocks.csv")

    gb = DynamicGraphBuilder(data_dir, BASE_PORTFOLIO, news_path)
    if gb.num_stock_nodes > n_assets:
        gb.resize_to_match_portfolio(n_assets)
    return gb


def make_agent(gb, device):
    """Instantiate a NextGen GNN-SAC agent matching the architecture flags.

    Returns (agent, state_dim, action_dim, num_nodes, adj_tensor).
    NOTE: agent.adj_matrix is set to None by the parent's __init__ (the policy
    holds the adjacency internally); we return the tensor here so callers can
    pass it to select_action via the `adj` argument.
    """
    num_nodes = gb.num_nodes
    state_dim = num_nodes * FEAT_DIM_NODE
    action_dim = gb.num_stock_nodes

    sectors_unique = sorted(set(gb.sector_map.get(t, "Other") for t in gb.tickers))
    sector_name_to_id = {s: i for i, s in enumerate(sectors_unique)}
    sector_map_array = [sector_name_to_id[gb.sector_map.get(t, "Other")] for t in gb.tickers]
    sector_map_tensor = torch.tensor(sector_map_array, dtype=torch.long, device=device)

    agent = GNNSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        node_features_dim=FEAT_DIM_NODE,
        num_nodes=num_nodes,
        adj_matrix=gb.mixed_adj_template,
        device=device,
        learning_rate=3e-4,
        batch_size=256,
        buffer_capacity=4096,
        lstm_hidden_size=64,
        num_lstm_layers=1,
        hidden_dims=[256, 256],
        use_graph=True,
        use_tgn=True,
        use_nextgen=True,
        alpha=0.2,
        auto_tune_alpha=True,
        gamma=0.995,
        tau=0.005,
        target_update_interval=2,
        num_stock_nodes=gb.num_stock_nodes,
        num_sector_nodes=gb.num_sector_nodes,
        num_event_nodes=gb.num_event_nodes,
        sector_map=sector_map_tensor,
        use_attention_gating=True,
        attn_gate_threshold=2.0,
        use_momentum_masking=True,
        momentum_threshold=-0.2,
        residual_alpha=0.5,
        cvar_multiplier=5.0,
        cvar_tau=0.1,
        use_cvar_loss=True,
    )
    adj_tensor = torch.tensor(gb.mixed_adj_template, dtype=torch.float32, device=device).unsqueeze(0)
    return agent, state_dim, action_dim, num_nodes, adj_tensor


def synth_ctx(device, vix=20.0):
    return {
        "vix": torch.tensor([vix], dtype=torch.float32, device=device),
        "vix_mean": torch.tensor([vix], dtype=torch.float32, device=device),
    }


def time_inference(agent, state_dim, num_nodes, device, adj, n_warmup, n_timed):
    """Time agent.select_action over n_timed iterations after n_warmup."""
    rng = np.random.default_rng(42)
    state = rng.standard_normal(state_dim).astype(np.float32) * 0.1
    ctx = synth_ctx(device)
    hidden = None

    # Warmup
    for _ in range(n_warmup):
        _, hidden = agent.select_action(state, hidden, deterministic=True, adj=adj, nextgen_ctx=ctx)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_timed):
        _, hidden = agent.select_action(state, hidden, deterministic=True, adj=adj, nextgen_ctx=ctx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (elapsed / n_timed) * 1000.0  # ms per call


def populate_buffer(agent, state_dim, action_dim, n_transitions):
    rng = np.random.default_rng(7)
    for _ in range(n_transitions):
        s = rng.standard_normal(state_dim).astype(np.float32) * 0.1
        a = rng.standard_normal(action_dim).astype(np.float32) * 0.1
        # Hierarchical-softmax action lives on the simplex; project softmaxed
        # for plausibility — gradient updates only care about shape, not validity.
        a = np.exp(a - a.max())
        a = a / a.sum()
        r = float(rng.standard_normal())
        ns = rng.standard_normal(state_dim).astype(np.float32) * 0.1
        d = 0.0
        agent.replay_buffer.push(s, a, r, ns, d)


def time_training(agent, state_dim, action_dim, device, n_steps):
    """Pre-populate the replay buffer, then time n_steps update() calls."""
    populate_buffer(agent, state_dim, action_dim, max(2 * agent.batch_size, 64))
    ctx = synth_ctx(device)

    # Warmup once to JIT any lazy buffers
    _ = agent.update(nextgen_ctx=ctx)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        agent.update(nextgen_ctx=ctx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda"
        else float("nan")
    )
    return elapsed, peak_vram_gb


def power_law(n, alpha, beta):
    return alpha * np.power(n, beta)


def fit_and_extrapolate(ns, ys, targets):
    """Fit T(N) = alpha N^beta on log-log; return (alpha, beta, r_squared, preds).

    R^2 is computed on log10(y) vs log10(N) so the user can spot a degenerate
    fit (R^2 < 0.5 means the measured points don't follow a power-law and the
    extrapolation should not be trusted).
    """
    ns_arr = np.asarray(ns, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    mask = np.isfinite(ns_arr) & np.isfinite(ys_arr) & (ys_arr > 0)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan"), [float("nan")] * len(targets)
    try:
        popt, _ = curve_fit(power_law, ns_arr[mask], ys_arr[mask],
                            p0=[1e-3, 2.0], maxfev=5000)
        alpha, beta = popt
        log_y = np.log10(ys_arr[mask])
        log_y_hat = np.log10(power_law(ns_arr[mask], alpha, beta))
        ss_res = float(np.sum((log_y - log_y_hat) ** 2))
        ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        preds = power_law(np.array(targets, dtype=float), alpha, beta)
        return float(alpha), float(beta), float(r2), preds.tolist()
    except Exception as e:
        print(f"Power-law fit failed ({e}); using NaN extrapolation.")
        return float("nan"), float("nan"), float("nan"), [float("nan")] * len(targets)


def _panel(ax, ns, ys, alpha, beta, r2, color, ylabel, title, loc="upper left"):
    """Helper: log-log scatter + optional power-law fit overlay for one panel."""
    ns_arr = np.asarray(ns, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    finite = np.isfinite(ys_arr)
    ax.loglog(ns_arr[finite], ys_arr[finite], "o-", label="measured (real graph)", color=color)
    if np.isfinite(beta):
        x_fit = np.linspace(ns_arr.min(), max(EXTRAPOLATE_POINTS), 200)
        fit_label = f"fit α·Nᵝ, β={beta:.2f}"
        if np.isfinite(r2):
            fit_label += f", R²={r2:.2f}"
        ax.loglog(x_fit, power_law(x_fit, alpha, beta), "--",
                  color=color, alpha=0.6, label=fit_label)
        for tgt in EXTRAPOLATE_POINTS:
            ax.axvline(tgt, color="grey", linewidth=0.3, linestyle=":")
    ax.set_xlabel("N (assets)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14, loc=loc)


def plot_scaling_curve(rows,
                       alpha_train, beta_train, r2_train,
                       alpha_inf, beta_inf, r2_inf,
                       alpha_vram, beta_vram, r2_vram,
                       out_path):
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    ns = [r["N"] for r in rows]

    _panel(axes[0], ns,
           [r["train_time_per_1k_steps_s"] for r in rows],
           alpha_train, beta_train, r2_train,
           "C0", "Train time / 1k steps (s)", "Training Time Scaling", loc="upper left")

    _panel(axes[1], ns,
           [r["inference_latency_ms"] for r in rows],
           alpha_inf, beta_inf, r2_inf,
           "C1", "Inference latency per call (ms)", "Inference Latency Scaling", loc="lower left")

    vram_vals = [(r["peak_vram_gb"] if r["peak_vram_gb"] is not None else float("nan"))
                 for r in rows]
    _panel(axes[2], ns, vram_vals,
           alpha_vram, beta_vram, r2_vram,
           "C2", "Peak VRAM (GB)", "Peak Memory Scaling", loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    eps_path = out_path.replace(".png", ".eps")
    fig.savefig(eps_path, format="eps", bbox_inches="tight")

    plt.close(fig)
    return eps_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run with tiny iteration counts for control-flow / shape verification.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU (useful on sub-sm_75 cards where cuDNN crashes).")
    parser.add_argument("--data_dir", type=str, default=os.path.join(BASE_DIR, "data"))
    parser.add_argument("--results_dir", type=str, default=os.path.join(BASE_DIR, "results"))
    parser.add_argument("--sweep_points", type=str, default="10,28,50",
                        help="Comma-separated N values. Must correspond to real point-in-time "
                             "universes available under data/. N>50 requires building "
                             "sp100/sp200_pit2013 first; the prior MockGB shortcut was removed.")
    parser.add_argument("--n_train_steps", type=int, default=5000,
                        help="Number of SAC updates to time. Bumped from 2000 to amortize "
                             "Python/optimizer fixed-cost overhead at small batch sizes.")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.smoke_test:
        n_warmup_inf, n_timed_inf = 5, 20
        n_train_steps = 10
        sweep_points = [10, 28, 50]
    else:
        n_warmup_inf, n_timed_inf = 100, 1000
        n_train_steps = args.n_train_steps
        sweep_points = [int(x) for x in args.sweep_points.split(",")]

    device = resolve_device(force_cpu=args.cpu)
    
    unique_topic = f"r2c3_complexity_sweep_logs_{uuid.uuid4().hex[:8]}"
    logger = RemoteLogger(topic=unique_topic)
    
    logger.log(f"Starting Complexity Sweep on device: {device} (CUDA: {torch.cuda.is_available()})")
    logger.log(f"Live logs streaming to: https://ntfy.sh/{unique_topic}")
    
    if device.type != "cuda":
        logger.log("WARNING: running on CPU. VRAM column will be NaN. Manuscript-grade "
              "numbers must be regenerated on a Kaggle P100/T4 session.")

    rows = []
    theory_tokens = "O(N^2 d) GNN + O(N K=32) IQN + O(N log N) softmax"

    for n in sweep_points:
        logger.log(f"\n=== N = {n} ===")
        gb = build_sub_universe(n, args.data_dir)
        agent, state_dim, action_dim, num_nodes, adj = make_agent(gb, device)

        inf_ms = time_inference(agent, state_dim, num_nodes, device, adj,
                                n_warmup=n_warmup_inf, n_timed=n_timed_inf)
        train_s, peak_vram = time_training(agent, state_dim, action_dim, device,
                                           n_steps=n_train_steps)

        # Project per-1k-step training time to a full 75k-step Kaggle run hour count.
        # train_s is for n_train_steps update calls — normalize to per-1k.
        train_s_per_1k = (train_s / n_train_steps) * 1000.0
        projected_h_at_n = (train_s_per_1k * 75.0) / 3600.0

        row = {
            "N": n,
            "num_nodes_total": num_nodes,
            "theory_O": theory_tokens,
            "train_time_per_1k_steps_s": round(train_s_per_1k, 4),
            "inference_latency_ms": round(inf_ms, 4),
            "peak_vram_gb": (None if not np.isfinite(peak_vram) else round(peak_vram, 4)),
            "projected_train_time_h_at_N": round(projected_h_at_n, 4),
            "device": f"{device} (measured)",
            "n_train_steps": n_train_steps,
            "n_inference_calls": n_timed_inf,
        }
        rows.append(row)
        logger.log(f"  inference: {inf_ms:.3f} ms/call   train: {train_s_per_1k:.2f} s/1k steps   "
              f"peak_vram: {peak_vram}")

        # Free GPU memory between points so peak measurement is per-N
        del agent
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Power-law fits on measured rows only
    ns_arr = np.array([r["N"] for r in rows], dtype=float)
    train_arr = np.array([r["train_time_per_1k_steps_s"] for r in rows], dtype=float)
    inf_arr = np.array([r["inference_latency_ms"] for r in rows], dtype=float)
    vram_arr = np.array(
        [(r["peak_vram_gb"] if r["peak_vram_gb"] is not None else float("nan"))
         for r in rows], dtype=float)

    alpha_t, beta_t, r2_t, train_proj = fit_and_extrapolate(ns_arr, train_arr, EXTRAPOLATE_POINTS)
    alpha_i, beta_i, r2_i, inf_proj = fit_and_extrapolate(ns_arr, inf_arr, EXTRAPOLATE_POINTS)
    alpha_v, beta_v, r2_v, vram_proj = fit_and_extrapolate(ns_arr, vram_arr, EXTRAPOLATE_POINTS)

    R2_MIN = 0.5
    for label, r2 in [("train", r2_t), ("infer", r2_i), ("vram", r2_v)]:
        if np.isfinite(r2) and r2 < R2_MIN:
            logger.log(
                f"WARNING: {label} power-law fit has R^2={r2:.2f} < {R2_MIN}. "
                f"Measured points do not follow a clean power law — extrapolation "
                f"to N>{int(ns_arr.max())} is not trustworthy. Likely cause: "
                f"fixed-cost overhead (LSTM/IQN/Python) dominates the GNN O(N^2 d) "
                f"term at this N range. Report measured values only."
            )

    # Append extrapolated rows for the CSV (clearly tagged)
    for n_ext, t_ext, i_ext, v_ext in zip(EXTRAPOLATE_POINTS, train_proj, inf_proj, vram_proj):
        rows.append({
            "N": n_ext,
            "num_nodes_total": None,
            "theory_O": theory_tokens,
            "train_time_per_1k_steps_s": round(t_ext, 4) if np.isfinite(t_ext) else None,
            "inference_latency_ms": round(i_ext, 4) if np.isfinite(i_ext) else None,
            "peak_vram_gb": round(v_ext, 4) if np.isfinite(v_ext) else None,
            "projected_train_time_h_at_N": (
                round((t_ext * 75.0) / 3600.0, 4) if np.isfinite(t_ext) else None
            ),
            "device": "extrapolated",
            "n_train_steps": None,
            "n_inference_calls": None,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.results_dir, "complexity_sweep_sp50_pit2013.csv")
    df.to_csv(csv_path, index=False)
    logger.log(f"\nWrote {csv_path}")
    print(df.to_string(index=False))

    fig_path = os.path.join(args.results_dir, "Figure_Scaling_Curve.png")
    eps_path = plot_scaling_curve(
        rows[:len(sweep_points)],
        alpha_t, beta_t, r2_t,
        alpha_i, beta_i, r2_i,
        alpha_v, beta_v, r2_v,
        fig_path,
    )
    logger.log(f"Wrote {fig_path}")
    logger.log(f"Wrote {eps_path}")

    logger.log(f"\nFit (train) : alpha={alpha_t:.4e}, beta={beta_t:.4f}, R^2={r2_t:.4f}")
    logger.log(f"Fit (infer) : alpha={alpha_i:.4e}, beta={beta_i:.4f}, R^2={r2_i:.4f}")
    logger.log(f"Fit (vram)  : alpha={alpha_v:.4e}, beta={beta_v:.4f}, R^2={r2_v:.4f}")


if __name__ == "__main__":
    main()
