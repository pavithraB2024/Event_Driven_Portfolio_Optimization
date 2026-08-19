"""
evaluate_imputed_portfolio.py
=============================
Evaluates the GNN-SAC agent and baseline models on a sub-portfolio containing
only the 9 imputed assets (starved of direct news coverage), to verify if GNN
Laplacian imputation provides tangible performance benefits.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import types
import matplotlib.pyplot as plt
import gymnasium as gym

# Setup base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.data_pipeline.graph_builder import DynamicGraphBuilder
from src.agents.gnn_sac_agent import GNNSACAgent
from src.agents.memory_augmented_sac_agent import MemoryAugmentedSACAgent
from src.environments.portfolio_env import PortfolioEnv
from src.baselines.mean_variance import MeanVarianceStrategy
from src.baselines.naive_strategies import RiskParityStrategy

# Define configuration
IMPUTED = ['BA', 'INTC', 'JPM', 'WMT', 'CVX', 'GE', 'HON', 'MSFT', 'SHEL']
DIRECT_COVERAGE = ['AAPL','AMZN','BAC','BP','CAT','CSCO','ENB','GILD','HD','IBM','JNJ','KO','MFC','MMM','MRK','ORCL','PFE','TD','VZ']

def patch_env_for_graph(env, features_tensor):
    """Patches the environment to return graph features as state."""
    env.graph_features = features_tensor
    def get_graph_state(self_env):
        idx = self_env.current_step + self_env.window_size
        if idx >= len(self_env.graph_features): idx = len(self_env.graph_features) - 1
        
        feats = self_env.graph_features[idx] 
        w = self_env.portfolio_weights
        
        if len(w) == feats.shape[0]:
            w = w.reshape(-1, 1)
        elif len(w) < feats.shape[0]:
            pad_len = feats.shape[0] - len(w)
            w_padded = np.pad(w, (0, pad_len), 'constant')
            w = w_padded.reshape(-1, 1)
            w = w[:feats.shape[0]].reshape(-1, 1)
            
        return np.hstack([feats, w]).flatten().astype(np.float32)
        
    env._get_state = types.MethodType(get_graph_state, env)
    
    n_graph_nodes = features_tensor.shape[1]
    n_raw_feats = features_tensor.shape[2]
    feat_dim = n_raw_feats + 1
    total_dims = n_graph_nodes * feat_dim
    
    env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total_dims,), dtype=np.float32)
    return feat_dim

def run_evaluation_loop_masked(env, agent, agent_name, mask, deterministic=True, nextgen_data=None):
    """Runs evaluation loop where weights are forced to be zero for assets outside of mask."""
    obs, info = env.reset(seed=42)
    done = False
    hidden_state = None
    
    portfolio_values = [env.initial_balance]
    weights_history = []
    
    vix_vals = nextgen_data.get('vix') if nextgen_data else None
    vix_means = nextgen_data.get('vix_mean') if nextgen_data else None
    
    while not done:
        curr_step = env.current_step
        
        if agent is None:
            # Fallback equal allocation among masked assets
            action = mask / mask.sum()
        else:
            ctx = None
            if vix_vals is not None:
                idx = min(curr_step + env.window_size, len(vix_vals) - 1)
                ctx = {
                    'vix': torch.tensor([vix_vals[idx]], dtype=torch.float32).to(agent.device if hasattr(agent, 'device') else 'cpu'),
                    'vix_mean': torch.tensor([vix_means[idx]], dtype=torch.float32).to(agent.device if hasattr(agent, 'device') else 'cpu'),
                }
            
            if hasattr(agent, 'select_action'):
                try:
                    action, hidden_state = agent.select_action(obs, hidden_state, deterministic=deterministic, nextgen_ctx=ctx)
                except TypeError:
                    action, hidden_state = agent.select_action(obs, hidden_state, deterministic=deterministic)
            elif hasattr(agent, 'get_action'):
                action, hidden_state = agent.get_action(obs, hidden_state, deterministic=deterministic)
                
            if isinstance(action, torch.Tensor):
                action = action.cpu().detach().numpy().flatten()
            
            # Apply mask to keep only imputed assets
            action = action * mask
            action_sum = action.sum()
            if action_sum > 1e-6:
                action = action / action_sum
            else:
                action = mask / mask.sum()
                
            if agent_name == "Event-Driven GNN-SAC" and curr_step < 5:
                print(f"[{agent_name}] Step {curr_step}: Action sum={action.sum():.4f}, Non-zero weights: {[(i, round(action[i], 4)) for i in range(28) if action[i] > 0]}")
                
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        portfolio_values.append(info['portfolio_value'])
        weights_history.append(action)
        
    return portfolio_values, np.array(weights_history)

def calculate_metrics(portfolio_values, weights_history=None):
    portfolio_values = np.array(portfolio_values)
    returns = pd.Series(portfolio_values).pct_change().dropna()
    cum_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

    if len(portfolio_values) < 2:
        return {"ARR":0, "ROI":0, "Sharpe":0, "Sortino":0, "Calmar":0, "MDD":0, "Turnover": 0.0}

    n_days = len(portfolio_values) - 1
    annualized_return = (1 + cum_return) ** (252 / n_days) - 1 if n_days > 0 else 0
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values / cummax) - 1
    max_drawdown = drawdown.min()

    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    downside_std = downside_returns.std()
    sortino = (returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    calmar = annualized_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0

    turnover = 0.0
    if weights_history is not None and len(weights_history) > 1:
        weights_history = np.array(weights_history)
        weight_diffs = np.abs(weights_history[1:] - weights_history[:-1])
        turnover = np.mean(np.sum(weight_diffs, axis=1))

    return {
        "ARR": annualized_return,
        "ROI": cum_return,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "MDD": max_drawdown,
        "Turnover": turnover
    }

def main():
    portfolio_name = '28stocks'
    print(f"=== Starting Imputed Portfolio Evaluation ({len(IMPUTED)} assets) ===")
    
    # 1. Load data
    data_dir = os.path.join(BASE_DIR, "data")
    csv_path = os.path.join(data_dir, "processed", f"{portfolio_name}_dataset.csv")
    env_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Load tickers
    with open(os.path.join(data_dir, '28stocks_tickers.txt'), 'r') as f:
        tickers = f.read().strip().split(',')
    tickers = [t.strip() for t in tickers if t.strip()]
    
    # Get mask and indices
    imputed_indices = [tickers.index(t) for t in IMPUTED if t in tickers]
    mask = np.zeros(28)
    mask[imputed_indices] = 1.0
    
    # Load node features
    node_feat_path = os.path.join(data_dir, f"{portfolio_name}_data.npy")
    base_node_features = np.load(node_feat_path)
    
    # Sync lengths
    dates = env_data.index
    min_len = min(len(dates), base_node_features.shape[0])
    dates = dates[:min_len]
    base_node_features = base_node_features[:min_len]
    env_data = env_data.iloc[:min_len]
    
    # Slice features to match portfolio
    if base_node_features.shape[1] > 28:
        base_node_features = base_node_features[:, :28, :]
        
    # --- Prepare Event-Aware (Dynamic) and Static Graph Features ---
    processed_news_path = os.path.join(data_dir, "processed", f"news_with_events_{portfolio_name}.csv")
    graph_builder = DynamicGraphBuilder(data_dir, portfolio_name, processed_news_path)
    graph_builder.resize_to_match_portfolio(28)
    
    cache_path = os.path.join(data_dir, "processed", f"{portfolio_name}_precomputed_graphs.npz")
    if os.path.exists(cache_path):
        print(f"Loading cached graphs from {cache_path}...")
        cache = np.load(cache_path)
        event_aware_features = cache['features']
    else:
        print("Computing Event-Aware features...")
        event_aware_features = []
        base_feat_tensor = torch.tensor(base_node_features, dtype=torch.float32)
        for t in tqdm(range(min_len)):
            date = dates[t]
            step_base_x = base_feat_tensor[t] 
            graph_data = graph_builder.get_graph(date, step_base_x)
            event_aware_features.append(graph_data.x.numpy())
        event_aware_features = np.array(event_aware_features)
        
    # --- Slice for Benchmark Window (Nov 4, 2013 - March 14, 2014) ---
    benchmark_start = "2013-11-04"
    end_date = "2014-03-14"
    benchmark_mask = (dates >= benchmark_start) & (dates <= end_date)
    
    benchmark_start_idx = np.argmax(benchmark_mask)
    warmup_size = 60
    data_start_idx = max(0, benchmark_start_idx - warmup_size)
    actual_warmup = benchmark_start_idx - data_start_idx
    end_mask_idx = len(dates) - 1 - np.argmax(benchmark_mask[::-1])
    
    test_data = env_data.iloc[data_start_idx:end_mask_idx + 1]
    test_event_features = event_aware_features[data_start_idx:end_mask_idx + 1]
    
    print("Computing Static Graph features (empty news) for test window only...")
    # Backup real news
    real_news = graph_builder.news_df.copy()
    graph_builder.news_df = pd.DataFrame(columns=real_news.columns)
    # Reset event memory
    graph_builder.event_memory = np.zeros(graph_builder.num_stock_nodes)
    graph_builder.sector_memory = np.zeros(graph_builder.num_sector_nodes)
    
    test_static_features = []
    base_feat_tensor = torch.tensor(base_node_features, dtype=torch.float32)
    for idx in range(data_start_idx, end_mask_idx + 1):
        date = dates[idx]
        step_base_x = base_feat_tensor[idx]
        graph_data = graph_builder.get_graph(date, step_base_x)
        test_static_features.append(graph_data.x.numpy())
    test_static_features = np.array(test_static_features)
    
    # Restore news
    graph_builder.news_df = real_news
    
    print(f"Benchmark window size: {len(test_data) - actual_warmup} trading days (Nov 4, 2013 to March 14, 2014)")
    
    # Load Macro Context (VIX)
    try:
        vix_df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "vix.csv"), index_col=0, parse_dates=True)
        vix_series = vix_df.reindex(test_data.index).ffill().fillna(20.0).iloc[:, 0]
        vix_mean_series = vix_series.rolling(60, min_periods=1).mean()
        nextgen_data = {
            'vix': vix_series.values,
            'vix_mean': vix_mean_series.values
        }
    except Exception as e:
        print(f"VIX not found: {e}")
        nextgen_data = {
            'vix': np.full(len(test_data), 20.0),
            'vix_mean': np.full(len(test_data), 20.0)
        }
        
    # --- Load GNN-SAC Agent ---
    device = "cpu"
    gnn_model_path = os.path.join(BASE_DIR, "models", "gnn_sac_event_driven_28stocks_best.pth")
    print(f"Loading GNN-SAC Model from: {gnn_model_path}")
    ckpt = torch.load(gnn_model_path, map_location='cpu')
    if 'policy' in ckpt:
        policy_sd = ckpt['policy']
    elif 'policy_state_dict' in ckpt:
        policy_sd = ckpt['policy_state_dict']
    else:
        policy_sd = ckpt
    
    tgn_keys = [k for k in policy_sd.keys() if 'tgn.' in k]
    use_tgn = len(tgn_keys) > 0
    use_nextgen = any('nextgen_encoder' in k for k in policy_sd.keys())
    use_attention_gating = any('attn_gate' in k for k in policy_sd.keys()) or any('gate_mask' in k for k in policy_sd.keys())
    use_momentum_masking = any('momentum_mask' in k for k in policy_sd.keys())
    use_feature_gating = any('feature_gate' in k for k in policy_sd.keys())
    use_dual_head_iqn = any('alpha_head' in k for k in policy_sd.keys())
    
    print(f"Detected Agent Settings: use_nextgen={use_nextgen}, use_tgn={use_tgn}, use_attention_gating={use_attention_gating}, use_momentum_masking={use_momentum_masking}")
    
    # 1. Event-Driven GNN-SAC (Ours)
    env_event = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
    feat_dim = patch_env_for_graph(env_event, test_event_features)
    
    agent_gnn = GNNSACAgent(
        state_dim=graph_builder.num_nodes * feat_dim,
        action_dim=28,
        node_features_dim=feat_dim,
        num_nodes=graph_builder.num_nodes,
        adj_matrix=graph_builder.mixed_adj_template,
        device=device,
        hidden_dims=[256, 256],
        lstm_hidden_size=64, num_lstm_layers=1, use_graph=True,
        use_nextgen=use_nextgen,
        learning_rate=3e-4,
        use_tgn=use_tgn,
        use_attention_gating=use_attention_gating,
        use_momentum_masking=use_momentum_masking,
        use_feature_gating=use_feature_gating,
        use_dual_head_iqn=use_dual_head_iqn,
        residual_alpha=0.5
    )
    agent_gnn.load(gnn_model_path)
    vals_event, w_event = run_evaluation_loop_masked(env_event, agent_gnn, "Event-Driven GNN-SAC", mask, nextgen_data=nextgen_data)
    
    # 2. Static Graph GNN-SAC (News Starved)
    env_static = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
    patch_env_for_graph(env_static, test_static_features)
    
    # Re-load agent (identical weights, but run with static features)
    agent_static = GNNSACAgent(
        state_dim=graph_builder.num_nodes * feat_dim,
        action_dim=28,
        node_features_dim=feat_dim,
        num_nodes=graph_builder.num_nodes,
        adj_matrix=graph_builder.mixed_adj_template,
        device=device,
        hidden_dims=[256, 256],
        lstm_hidden_size=64, num_lstm_layers=1, use_graph=True,
        use_nextgen=use_nextgen,
        learning_rate=3e-4,
        use_tgn=use_tgn,
        use_attention_gating=use_attention_gating,
        use_momentum_masking=use_momentum_masking,
        use_feature_gating=use_feature_gating,
        use_dual_head_iqn=use_dual_head_iqn,
        residual_alpha=0.5
    )
    agent_static.load(gnn_model_path)
    vals_static, w_static = run_evaluation_loop_masked(env_static, agent_static, "Static Graph GNN-SAC", mask, nextgen_data=nextgen_data)
    
    # 3. LSTM-SAC (No Graph)
    env_lstm = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
    agent_lstm = MemoryAugmentedSACAgent(
         state_dim=env_lstm.observation_space.shape[0],
         action_dim=env_lstm.action_space.shape[0],
         use_graph=False, device=device,
         hidden_dims=[256, 256], lstm_hidden_size=64, num_lstm_layers=1,
         learning_rate=3e-4
    )
    lstm_model_path = os.path.join(BASE_DIR, "models", "sac_memory_lstm_trained_best.pth")
    if os.path.exists(lstm_model_path):
        agent_lstm.load(lstm_model_path)
        vals_lstm, w_lstm = run_evaluation_loop_masked(env_lstm, agent_lstm, "LSTM-SAC", mask)
    else:
        print("LSTM model not found. Running random baseline wrapper.")
        vals_lstm, w_lstm = [100000.0] * (len(test_data) - actual_warmup + 1), np.zeros((len(test_data) - actual_warmup, 28))
    
    # 4. Buy-and-Hold / Equal Weight on Imputed Portfolio
    env_eq = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
    vals_eq, w_eq = run_evaluation_loop_masked(env_eq, None, "Equal-Weight", mask)
    
    # Extract returns matrix specifically for baseline calculators
    returns_df = test_data[[f'return_{ticker}' for ticker in tickers]].copy()
    returns_df.columns = [c.replace('return_', '') for c in returns_df.columns]
    returns_imputed_only = returns_df[IMPUTED].copy()
    
    # 5. Risk Parity Strategy
    rp_strat = RiskParityStrategy(estimation_window=60, rebalance_freq=1)
    env_rp = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
    obs_rp, _ = env_rp.reset(seed=42)
    vals_rp = [100000.0]
    w_rp_history = []
    done_rp = False
    t = 0
    while not done_rp:
        current_idx = actual_warmup + t
        # Allocate weights for the 9 imputed assets
        w_imputed = rp_strat.allocate(returns_imputed_only, current_idx)
        # Pad to 28-dimensional weight vector
        w_full = np.zeros(28)
        for i, ticker in enumerate(IMPUTED):
            w_full[tickers.index(ticker)] = w_imputed[i]
        
        _, _, term, trunc, info = env_rp.step(w_full)
        done_rp = term or trunc
        vals_rp.append(info['portfolio_value'])
        w_rp_history.append(w_full)
        t += 1
        
    # 6. Mean-Variance Strategy
    mv_strat = MeanVarianceStrategy(estimation_window=60, rebalance_freq=1, risk_aversion=2.0)
    env_mv = PortfolioEnv(test_data, initial_balance=100000.0, transaction_cost=0.0025)
    obs_mv, _ = env_mv.reset(seed=42)
    vals_mv = [100000.0]
    w_mv_history = []
    done_mv = False
    t = 0
    while not done_mv:
        current_idx = actual_warmup + t
        w_imputed = mv_strat.allocate(returns_imputed_only, current_idx)
        w_full = np.zeros(28)
        for i, ticker in enumerate(IMPUTED):
            w_full[tickers.index(ticker)] = w_imputed[i]
            
        _, _, term, trunc, info = env_mv.step(w_full)
        done_mv = term or trunc
        vals_mv.append(info['portfolio_value'])
        w_mv_history.append(w_full)
        t += 1

    # --- Calculate Metrics ---
    metrics_event = calculate_metrics(vals_event, w_event)
    metrics_static = calculate_metrics(vals_static, w_static)
    metrics_lstm = calculate_metrics(vals_lstm, w_lstm)
    metrics_eq = calculate_metrics(vals_eq, w_eq)
    metrics_rp = calculate_metrics(vals_rp, w_rp_history)
    metrics_mv = calculate_metrics(vals_mv, w_mv_history)
    
    results = [
        {"Model": "Buy-and-Hold (Equal-Weight)", "ROI": metrics_eq["ROI"], "Sharpe": metrics_eq["Sharpe"], "Sortino": metrics_eq["Sortino"], "Calmar": metrics_eq["Calmar"], "Max DD": metrics_eq["MDD"], "Turnover": metrics_eq["Turnover"]},
        {"Model": "Mean-Variance (Markowitz)", "ROI": metrics_mv["ROI"], "Sharpe": metrics_mv["Sharpe"], "Sortino": metrics_mv["Sortino"], "Calmar": metrics_mv["Calmar"], "Max DD": metrics_mv["MDD"], "Turnover": metrics_mv["Turnover"]},
        {"Model": "Risk Parity (Vol Inverse)", "ROI": metrics_rp["ROI"], "Sharpe": metrics_rp["Sharpe"], "Sortino": metrics_rp["Sortino"], "Calmar": metrics_rp["Calmar"], "Max DD": metrics_rp["MDD"], "Turnover": metrics_rp["Turnover"]},
        {"Model": "LSTM-SAC (No Graph)", "ROI": metrics_lstm["ROI"], "Sharpe": metrics_lstm["Sharpe"], "Sortino": metrics_lstm["Sortino"], "Calmar": metrics_lstm["Calmar"], "Max DD": metrics_lstm["MDD"], "Turnover": metrics_lstm["Turnover"]},
        {"Model": "Static Graph GNN-SAC", "ROI": metrics_static["ROI"], "Sharpe": metrics_static["Sharpe"], "Sortino": metrics_static["Sortino"], "Calmar": metrics_static["Calmar"], "Max DD": metrics_static["MDD"], "Turnover": metrics_static["Turnover"]},
        {"Model": "Event-Driven GNN-SAC (Ours)", "ROI": metrics_event["ROI"], "Sharpe": metrics_event["Sharpe"], "Sortino": metrics_event["Sortino"], "Calmar": metrics_event["Calmar"], "Max DD": metrics_event["MDD"], "Turnover": metrics_event["Turnover"]}
    ]
    
    df_results = pd.DataFrame(results)
    
    # Print Results Table
    print("\n" + "="*80)
    print("IMPUTED-ONLY PORTFOLIO BENCHMARK EVALUATION (Nov 4, 2013 - March 14, 2014)")
    print("="*80)
    row_fmt = "{:<30} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}"
    print(row_fmt.format("Model", "ROI %", "Sharpe", "Sortino", "Calmar", "Max DD %", "Turnover"))
    print("-" * 86)
    for r in results:
        print(row_fmt.format(
            r["Model"],
            f"{r['ROI']*100:.2f}%",
            f"{r['Sharpe']:.2f}",
            f"{r['Sortino']:.2f}",
            f"{r['Calmar']:.2f}",
            f"{r['Max DD']*100:.2f}%",
            f"{r['Turnover']:.4f}"
        ))
    print("="*80 + "\n")
    
    # Save comparison to CSV
    output_path = os.path.join(BASE_DIR, "results", "imputed_portfolio_comparison.csv")
    df_results.to_csv(output_path, index=False)
    print(f"Saved results comparison to {output_path}")

    # Plot equity curve
    plt.figure(figsize=(10, 6))
    plt.plot(vals_event, label='Event-Driven GNN-SAC (Ours)', color='blue', linewidth=2.5)
    plt.plot(vals_static, label='Static Graph GNN-SAC', color='orange', linestyle='--', linewidth=1.5)
    plt.plot(vals_lstm, label='LSTM-SAC', color='red', linestyle='-.', linewidth=1.5)
    plt.plot(vals_eq, label='Buy-and-Hold', color='grey', linestyle=':', linewidth=1.5)
    plt.title('9 Imputed Assets: Cumulative Wealth (Nov 2013 - Mar 2014)', fontsize=12, fontweight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    fig_path = os.path.join(BASE_DIR, "results", "Figure_Imputed_Portfolio_Wealth.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sub-portfolio wealth plot to {fig_path}")

if __name__ == "__main__":
    main()
