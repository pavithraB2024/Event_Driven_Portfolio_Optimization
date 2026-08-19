"""
NLP Error Sensitivity Analysis for Event-Driven GNN-SAC
=========================================================
This script evaluates the sensitivity of the trained GNN-SAC model to event extraction
and classification errors during inference. It perturbs a fraction of news events in
the news feed by randomly mislinking entities and flipping sentiment polarity, and then
quantifies the impact on final portfolio performance.
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
from tqdm import tqdm

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.data_pipeline.graph_builder import DynamicGraphBuilder
from src.agents.gnn_sac_agent import GNNSACAgent
from src.environments.portfolio_env import PortfolioEnv
from scripts.evaluation.evaluate_gnn_sac import load_robust_data, patch_env_for_graph, run_evaluation_loop, calculate_metrics

PORTFOLIO_NAME = "28stocks"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "checkpoints", "gnn_sac_28stocks_step_40000.pth")
NODE_FEAT_PATH = os.path.join(DATA_DIR, "28stocks_data.npy")
ORIGINAL_NEWS_PATH = os.path.join(DATA_DIR, "processed", "news_with_events_28stocks.csv")
TEMP_NEWS_PATH = os.path.join(DATA_DIR, "processed", "news_with_events_28stocks_temp_perturbed.csv")

def reset_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
         torch.cuda.manual_seed_all(seed)

def perturb_news_data(news_df, error_rate, tickers, seed=42):
    """
    Perturbs a fraction of event extractions and classifications.
    - Entity Extraction Error: entity_a and stock are replaced with a random ticker from the portfolio
    - Sentiment Polarity Flip: sentiment_scalar sign is flipped
    """
    if error_rate == 0.0:
        return news_df.copy()
        
    rng = np.random.default_rng(seed)
    perturbed_df = news_df.copy()
    n_rows = len(perturbed_df)
    n_corrupt = int(n_rows * error_rate)
    
    if n_corrupt == 0:
        return perturbed_df
        
    # Select indices to corrupt
    corrupt_indices = rng.choice(n_rows, size=n_corrupt, replace=False)
    
    for idx in corrupt_indices:
        current_stock = perturbed_df.iloc[idx].get('stock')
        # Choose a random ticker different from the current one
        choices = [t for t in tickers if t != current_stock]
        if not choices:
            choices = tickers
        new_ticker = rng.choice(choices)
        
        # Modify both stock and entity_a fields
        perturbed_df.iat[idx, perturbed_df.columns.get_loc('stock')] = new_ticker
        if 'entity_a' in perturbed_df.columns:
            perturbed_df.iat[idx, perturbed_df.columns.get_loc('entity_a')] = new_ticker
            
        # Flip sentiment polarity
        if 'sentiment_scalar' in perturbed_df.columns:
            val = perturbed_df.iat[idx, perturbed_df.columns.get_loc('sentiment_scalar')]
            perturbed_df.iat[idx, perturbed_df.columns.get_loc('sentiment_scalar')] = -val
            
    return perturbed_df

def compute_graphs_for_perturbation(perturbed_news_path, env_data):
    """Recomputes event-aware graphs with perturbed news."""
    np.random.seed(42)
    gb = DynamicGraphBuilder(DATA_DIR, PORTFOLIO_NAME, perturbed_news_path,
                             decay_rate=0.1, inject_sentiment=True)
                             
    base_node_features = np.load(NODE_FEAT_PATH)
    
    # Normalize features to prevent NaN outputs in GNN encoder
    mu = np.nanmean(base_node_features, axis=(0, 1), keepdims=True)
    sd = np.nanstd(base_node_features, axis=(0, 1), keepdims=True) + 1e-6
    base_node_features = np.clip((base_node_features - mu) / sd, -5, 5)
    base_node_features = np.nan_to_num(base_node_features, nan=0.0, posinf=5.0, neginf=-5.0)
    
    dates = env_data.index
    T = min(len(dates), len(base_node_features))
    dates = dates[:T]
    base_feat_tensor = torch.tensor(base_node_features[:T], dtype=torch.float32)
    
    all_features = []
    for t in tqdm(range(T), desc="  Building perturbed graphs", leave=False):
        try:
            g = gb.get_graph(dates[t], base_feat_tensor[t])
            feats = g.x.numpy()
        except Exception:
            feats = np.zeros((gb.num_nodes, base_node_features.shape[2]))
        feats = np.nan_to_num(feats, nan=0.0, posinf=5.0, neginf=-5.0)
        all_features.append(feats)
        
    return np.array(all_features), gb

def main():
    parser = argparse.ArgumentParser(description="Run NLP error sensitivity sweep on Event-Driven GNN-SAC.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for replication")
    args = parser.parse_args()
    
    reset_seed(args.seed)
    
    print("\n" + "=" * 70)
    print("  NLP ERROR SENSITIVITY STUDY ON EVENT-DRIVEN GNN-SAC  ")
    print("=" * 70)
    
    # 1. Load Data
    env_data = load_robust_data(DATA_DIR, PORTFOLIO_NAME)
    tickers_file = os.path.join(DATA_DIR, '28stocks_tickers.txt')
    with open(tickers_file, 'r') as f:
        tickers = f.read().strip().split(',')
    tickers = [t.strip() for t in tickers if t.strip()]
    
    dates = env_data.index
    price_cols = [c for c in env_data.columns if c.startswith('price_')]
    len(price_cols)
    
    # 2. Load Original News
    original_news_df = pd.read_csv(ORIGINAL_NEWS_PATH)
    
    # 3. Load Macro Data (VIX)
    try:
        vix_df = pd.read_csv(os.path.join(DATA_DIR, "raw", "vix.csv"), index_col=0, parse_dates=True)
        vix_series = vix_df.reindex(env_data.index).ffill().fillna(20.0).iloc[:, 0]
        vix_mean_series = vix_series.rolling(60, min_periods=1).mean()
        nextgen_data = {
            'vix': vix_series.values,
            'vix_mean': vix_mean_series.values
        }
    except Exception as e:
        print(f"Warning: VIX data not found: {e}")
        nextgen_data = {
            'vix': np.full(len(env_data), 20.0),
            'vix_mean': np.full(len(env_data), 20.0)
        }
        
    # 4. Sweep NLP Error Rates
    error_rates = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    results = []
    
    for err in error_rates:
        print(f"\nEvaluating GNN-SAC under {err*100:.1f}% event extraction error rate...")
        
        # Perturb news data
        perturbed_news = perturb_news_data(original_news_df, err, tickers, seed=args.seed)
        perturbed_news.to_csv(TEMP_NEWS_PATH, index=False)
        
        # Build graphs
        all_features, gb = compute_graphs_for_perturbation(TEMP_NEWS_PATH, env_data)
        
        # Set up benchmark slice (Nov 4, 2013 - March 14, 2014) with 60 days warmup
        benchmark_start = "2013-11-04"
        end_date = "2014-03-14"
        benchmark_mask = (dates >= benchmark_start) & (dates <= end_date)
        
        benchmark_start_idx = np.argmax(benchmark_mask)
        warmup_size = 60
        data_start_idx = max(0, benchmark_start_idx - warmup_size)
        end_mask_idx = len(dates) - 1 - np.argmax(benchmark_mask[::-1])
        
        test_data = env_data.iloc[data_start_idx:end_mask_idx + 1]
        test_features = all_features[data_start_idx:end_mask_idx + 1]
        test_data.attrs['graph_features'] = test_features
        
        # Re-slice nextgen VIX data to match test_data
        test_nextgen = {
            'vix': nextgen_data['vix'][data_start_idx:end_mask_idx + 1],
            'vix_mean': nextgen_data['vix_mean'][data_start_idx:end_mask_idx + 1]
        }
        
        # Initialize env
        env = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
        feat_dim = patch_env_for_graph(env, test_features)
        
        # Load agent
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        policy_sd = checkpoint.get('policy_state_dict', checkpoint)
        
        ckpt_num_nodes = gb.num_nodes
        lstm_w = policy_sd.get('lstm.weight_ih_l0')
        if lstm_w is not None:
            ckpt_num_nodes = lstm_w.shape[1] // 64
            
        ckpt_action_dim = env.action_space.shape[0]
        mean_w = policy_sd.get('mean.weight')
        if mean_w is not None:
            ckpt_action_dim = mean_w.shape[0]
            
        ckpt_state_dim = ckpt_num_nodes * feat_dim
        
        agent = GNNSACAgent(
            state_dim=ckpt_state_dim,
            action_dim=ckpt_action_dim,
            node_features_dim=feat_dim,
            num_nodes=ckpt_num_nodes,
            adj_matrix=gb.mixed_adj_template[:ckpt_num_nodes, :ckpt_num_nodes],
            device=DEVICE,
            hidden_dims=[256, 256],
            lstm_hidden_size=64, num_lstm_layers=1, use_graph=True,
            use_nextgen=True,
            learning_rate=3e-4,
            use_tgn=False,
            use_attention_gating=True,
            use_momentum_masking=True,
            use_feature_gating=False,
            use_dual_head_iqn=False,
            use_nlp_alpha=False,
            residual_alpha=0.5
        )
        agent.load(MODEL_PATH)
        
        # Evaluate
        portfolio_values, weights_history = run_evaluation_loop(
            env, agent, f"GNN-SAC (Error={err:.2f})", deterministic=True, nextgen_data=test_nextgen
        )
        
        metrics = calculate_metrics(portfolio_values, weights_history)
        
        results.append({
            "error_rate": err,
            "sharpe": metrics["Sharpe"],
            "roi": metrics["ROI"] * 100.0,
            "mdd": metrics["MDD"] * 100.0
        })
        
        print(f"Results for Error Rate {err*100:.0f}%: Sharpe = {metrics['Sharpe']:.4f}, ROI = {metrics['ROI']*100:.2f}%, MDD = {metrics['MDD']*100:.2f}%")
        
        # Clean up temp file
        if os.path.exists(TEMP_NEWS_PATH):
            os.remove(TEMP_NEWS_PATH)
            
    # 5. Save Results
    df_results = pd.DataFrame(results)
    output_csv = os.path.join(BASE_DIR, "results", "nlp_error_sensitivity_results_28stocks.csv")
    df_results.to_csv(output_csv, index=False)
    print(f"\nSaved sensitivity results to {output_csv}")
    
    # 6. Format Markdown Table
    print("\n" + "=" * 50)
    print("SENSITIVITY ANALYSIS SUMMARY TABLE (MARKDOWN)")
    print("=" * 50)
    print("| Error Rate (%) | Sharpe Ratio | ROI (%) | Max Drawdown (%) |")
    print("|----------------|--------------|---------|------------------|")
    for r in results:
        print(f"| {r['error_rate']*100:14.0f}% | {r['sharpe']:12.4f} | {r['roi']:7.2f}% | {r['mdd']:16.2f}% |")
        
    # 7. Format LaTeX Table
    print("\n" + "=" * 50)
    print("SENSITIVITY ANALYSIS SUMMARY TABLE (LATEX)")
    print("=" * 50)
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Sensitivity Analysis of Event-Driven GNN-SAC to Inference-Time NLP Extraction and Sentiment Classification Errors}")
    print(r"\label{tab:nlp_error_sensitivity}")
    print(r"\renewcommand{\arraystretch}{1.2}")
    print(r"\small")
    print(r"\begin{tabular}{c c c c}")
    print(r"\hline")
    print(r"NLP Error Rate (\%) & Sharpe Ratio & ROI (\%) & MDD (\%) \\")
    print(r"\hline")
    for r in results:
        bold_prefix = r"\textbf{" if r['error_rate'] == 0.0 else ""
        bold_suffix = "}" if r['error_rate'] == 0.0 else ""
        print(f"{r['error_rate']*100:.0f}\\% & {bold_prefix}{r['sharpe']:.4f}{bold_suffix} & {bold_prefix}{r['roi']:.2f}\\%{bold_suffix} & {bold_prefix}{r['mdd']:.2f}\\%{bold_suffix} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    
    # 8. Plot Figure
    plt.figure(figsize=(8, 5))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('NLP Event Extraction Error Rate (%)', fontweight='bold')
    ax1.set_ylabel('Sharpe Ratio', color=color, fontweight='bold')
    ax1.plot(df_results['error_rate'] * 100, df_results['sharpe'], marker='o', color=color, linewidth=2.5, label='Sharpe Ratio')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Maximum Drawdown (%)', color=color, fontweight='bold')
    ax2.plot(df_results['error_rate'] * 100, df_results['mdd'], marker='s', color=color, linewidth=2.5, linestyle='--', label='Max Drawdown')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Performance Degradation vs. NLP Event Extraction & Sentiment Classification Error Rate', fontsize=12, fontweight='bold')
    fig.tight_layout()
    
    plot_path = os.path.join(BASE_DIR, "results", "Figure_NLP_Error_Sensitivity.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved sensitivity figure to {plot_path}")

if __name__ == "__main__":
    main()
