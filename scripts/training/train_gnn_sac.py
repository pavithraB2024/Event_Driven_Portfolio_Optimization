import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import types
import matplotlib.pyplot as plt
import gymnasium as gym
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import argparse
import uuid
from src.utils.math_utils import get_annealed_cvar
from src.utils.remote_logger import RemoteLogger


class RunningMeanStd:
    """Online running mean and standard deviation for reward normalization."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        batch_mean = float(x)
        self.count += 1
        delta = batch_mean - self.mean
        self.mean += delta / self.count
        self.var = self.var * (1 - 1/self.count) + (delta ** 2) / self.count

    def normalize(self, x):
        return (x - self.mean) / (max(np.sqrt(self.var), 1e-8))

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode, encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

# Redirect Output
# Redirect Output setup moved to main execution block

# Add main repo path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.data_pipeline.graph_builder import DynamicGraphBuilder
from src.agents.gnn_sac_agent import GNNSACAgent
from src.environments.portfolio_env import PortfolioEnv

def train_gnn_sac(
    portfolio_name='28stocks',
    total_timesteps=150000,
    learning_rate=None,
    batch_size=None,
    alpha_init=None,
    use_tgn=None,
    save_freq=10000,
    use_nextgen=False,
    reward_type_override=None,
    early_stopping_patience=20,
    auto_tune_alpha=None,
    # Phase 0: LR warmup
    warmup_steps=5000,
    lr_min=1e-6,
    # Phase 2+5: Environment & Agent enhancements
    noise_augmentation=False,
    noise_std=0.001,
    random_start_offset=False,
    max_start_offset_pct=0.1,
    diversification_bonus=0.0,
    target_update_interval=2,
    horizon_blend_lambda=1.0,
    use_edge_pruning=False,
    use_nlp_alpha=False,
    nlp_alpha_multiplier=0.5,
    use_dual_head_iqn=False,
    use_action_masking=False,
    # Phase 2: Advanced Stabilizers (D1-D3)
    use_cvar_loss=False,
    cvar_multiplier=1.0,
    cvar_tau=0.1,
    use_attention_gating=False,
    attn_gate_threshold=2.0,
    use_momentum_masking=False,
    momentum_threshold=-0.2,
    residual_alpha=0.5,
    use_feature_gating=False,
    cvar_anneal_start=50000,
    cvar_anneal_end=100000,
    # isolated sub-portfolio experiment
    restrict_to_assets=None,        # list[str] of ticker symbols
    model_suffix='',                # appended to portfolio_name in save paths
    disable_event_channel=False,    # set λ_mix=0 -> static-template adjacency only
):
    # Dataset A (28-stocks) settings — tuned for outperformance v3.1 (Sharpe Max)
    learning_rate  = learning_rate  or 5e-5
    batch_size     = batch_size     or 256
    alpha_init     = alpha_init     or 0.02    # Ultra-low entropy → minimum exploration → tighter MDD
    use_tgn        = False if use_tgn is None else use_tgn
    auto_tune_alpha = False if auto_tune_alpha is None else auto_tune_alpha
    gamma          = 0.995                     # longer horizon → higher Sharpe
    tau            = 0.005
    random_steps   = 1000
    # ─────────────────────────────────────────────────────────────
    # derive an effective name (suffix is appended to save paths so
    # restricted-universe runs don't overwrite the headline 28-asset checkpoints).
    effective_name = f"{portfolio_name}{model_suffix}"
    
    unique_topic = f"r2c3_gnn_sac_logs_{effective_name}_{uuid.uuid4().hex[:8]}"
    logger = RemoteLogger(topic=unique_topic)
    
    logger.log(f"=== Training Event-Driven GNN-SAC for {portfolio_name} (save-name={effective_name}) ===")
    logger.log(f"Live logs streaming to: https://ntfy.sh/{unique_topic}")
    if restrict_to_assets:
        print(f"restricting allocation universe to {restrict_to_assets}")
    if disable_event_channel:
        print("event-channel disabled (Static-Graph baseline, λ_mix=0)")
    data_dir = os.path.join(BASE_DIR, "data")
    processed_news_path = os.path.join(data_dir, "processed", f"news_with_events_{portfolio_name}.csv")
    
    # 1. Load Base Data (Price/Features)
    # 1. Load Base Data (Price/Features)
    # Try specific dataset file first, fallback to enhanced/default
    env_data_path = os.path.join(data_dir, "processed", f"{portfolio_name}_dataset.csv")
    if not os.path.exists(env_data_path) and portfolio_name == '28stocks':
        env_data_path = os.path.join(data_dir, "processed", "complete_dataset_enhanced.csv")
        
    if not os.path.exists(env_data_path):
        print(f"Error: {env_data_path} not found.")
        return

    env_data = pd.read_csv(env_data_path, index_col=0, parse_dates=True)
    
    # Load Node Features (Base)
    node_feat_path = os.path.join(data_dir, f"{portfolio_name}_data.npy")
    if not os.path.exists(node_feat_path):
        print(f"Error: {node_feat_path} not found.")
        return
    base_node_features = np.load(node_feat_path) # (Time, Nodes, Feats)
    
    # --- Redundancy Fix (Dynamic Slicing) ---
    # Determine actual portfolio size from environment data
    price_cols = [c for c in env_data.columns if c.startswith('price_')]
    n_assets = len(price_cols)
    print(f"Verified Portfolio Assets: {n_assets}")
    
    if base_node_features.shape[1] < n_assets:
        print(f"Error: Node features ({base_node_features.shape[1]}) < Portfolio Assets ({n_assets})")
        return
    
    # Allow Heterogeneous Graph (Nodes >= Assets)
    if base_node_features.shape[1] > n_assets:
        print(f"Heterogeneous Graph Detected: {base_node_features.shape[1]} Nodes (Includes {n_assets} Tradable + Informational Nodes)")
    # ----------------------------------------
    
    print(f"Loaded Base Node Features: {base_node_features.shape}")
    
    # Robust Normalization (Z-Score + Clipping)
    feat_mean = np.nanmean(base_node_features, axis=(0, 1), keepdims=True)
    feat_std = np.nanstd(base_node_features, axis=(0, 1), keepdims=True) + 1e-6
    base_node_features = (base_node_features - feat_mean) / feat_std
    base_node_features = np.clip(base_node_features, -5.0, 5.0)
    
    print(f"Features Normalized & Clipped [-5, 5]. New Stats - Mean: {np.nanmean(base_node_features):.4f}, Std: {np.nanstd(base_node_features):.4f}")
    
    # Validation
    dates = env_data.index
    min_len = min(len(dates), base_node_features.shape[0])
    dates = dates[:min_len]
    base_node_features = base_node_features[:min_len]
    env_data = env_data.iloc[:min_len]
    
    # Mock Data Check (Fallback)
    if not os.path.exists(processed_news_path):
        print(f"[Safe Mode] Processed news not found at {processed_news_path}.")
        print("Generating MOCK event data for verification...")
        mock_news = []
        tickers_path = os.path.join(data_dir, f"{portfolio_name}_tickers.txt")
        # Try to read tickers, if fail, define standard list
        try:
            with open(tickers_path, 'r') as f:
                tickers = f.read().strip().split(',')
        except:
             tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"] # Fallback
             
        for d in dates[:100]: 
             for t in tickers:
                 if np.random.rand() < 0.1: 
                     mock_news.append({
                         'date': d,
                         'stock': t,
                         'sentiment_scalar': np.random.uniform(-1, 1),
                         'event_type': np.random.choice(["Mergers and Acquisitions", "Earnings Report", "Stock Movement"])
                     })
        
        mock_df = pd.DataFrame(mock_news)
        mock_df.to_csv(processed_news_path, index=False)
        print(f"Mock news saved to {processed_news_path}")
    
    # 2. Dynamic Graph Builder (Pre-computation)
    print("Initializing Dynamic Graph Builder...")
    graph_builder = DynamicGraphBuilder(data_dir, portfolio_name, processed_news_path)
    if graph_builder.num_stock_nodes > n_assets:
        graph_builder.resize_to_match_portfolio(n_assets)
    
    
    print("Pre-computing Event-Aware Node Features & Dynamic Adjacency...")
    cache_path = os.path.join(data_dir, f"processed/{portfolio_name}_precomputed_graphs.npz")
    
    if os.path.exists(cache_path):
        print(f"Loading cached graphs from {cache_path}...")
        cache = np.load(cache_path)
        event_aware_features = cache['features']
        dynamic_adjs_np = cache['adjs']
    else:
        # We iterate and fuse features
        event_aware_features = []
        dynamic_adjs = []
        
        # Convert base features to Tensor for builder
        base_feat_tensor = torch.tensor(base_node_features, dtype=torch.float32)
        
        for t in tqdm(range(min_len)):
            date = dates[t]
            step_base_x = base_feat_tensor[t] 
            
            # Get Graph (Fused)
            graph_data = graph_builder.get_graph(date, step_base_x)
            
            # Store fused features [N, F+2]
            event_aware_features.append(graph_data.x.numpy())
            
            # Store Dynamic Adjacency [N, N]
            dynamic_adjs.append(graph_data.adj.numpy())
            
        event_aware_features = np.array(event_aware_features)
        dynamic_adjs_np = np.array(dynamic_adjs)
        
        # Save cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, features=event_aware_features, adjs=dynamic_adjs_np)
        print(f"Saved computed graphs to {cache_path}")

    
    print(f"Fused Features Shape: {event_aware_features.shape}")
    print(f"Dynamic Adjacency Shape: {dynamic_adjs_np.shape}")

    # Static-Graph baseline — replace blended dynamic adjacency with
    # the static template tiled across time. Event-channel contribution -> 0.
    if disable_event_channel:
        static_adj = graph_builder.mixed_adj_template.astype(np.float32)
        dynamic_adjs_np = np.broadcast_to(
            static_adj, (dynamic_adjs_np.shape[0],) + static_adj.shape
        ).copy()
        print(f"Static-Graph adj broadcast: {dynamic_adjs_np.shape} (event channel zeroed)")
    
    # 3. Setup Environment with Patch
    # CRITICAL: Use Sun et al. (2024) exact train/val/test split for fair comparison
    # Training:   2002-01-02 to 2012-12-05
    # Validation: 2012-12-06 to 2013-11-03
    # Test:       2013-11-04 to 2014-03-14
    # The test set includes 60 extra warm-up days before benchmark start
    # so that PortfolioEnv (window_size=60) doesn't consume benchmark days.
    
    train_end = "2012-12-05"
    val_start = "2012-12-06"
    val_end = "2013-11-03"
    benchmark_start = "2013-11-04"
    benchmark_end = "2014-03-14"

    train_mask = (dates <= train_end)
    val_mask = (dates >= val_start) & (dates <= val_end)

    if train_mask.sum() == 0:
        n = len(env_data)
        train_n = int(n * 0.7)
        val_n = int(n * 0.15)
        train_data = env_data.iloc[:train_n]
        val_warmup_start = max(0, train_n - 60)
        val_data = env_data.iloc[val_warmup_start:train_n + val_n]
    else:
        train_data = env_data.loc[train_mask]
        val_start_idx = np.argmax(val_mask.values)
        val_warmup_start = max(0, val_start_idx - 60)
        val_end_idx = len(dates) - 1 - np.argmax(val_mask.values[::-1])
        val_data = env_data.iloc[val_warmup_start:val_end_idx + 1]
    
    print(f"Train: {len(train_data)} rows ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"Val:   {len(val_data)} rows ({val_data.index[0]} to {val_data.index[-1]}) [incl. 60-day warmup]")

    # Attach features and ADJs using index alignment
    if train_mask.sum() == 0:
        train_n = len(train_data)
        train_data.attrs['graph_features'] = event_aware_features[:train_n]
        train_data.attrs['dynamic_adjs'] = dynamic_adjs_np[:train_n]
        val_end_abs = val_warmup_start + len(val_data)
        val_data.attrs['graph_features'] = event_aware_features[val_warmup_start:val_end_abs]
        val_data.attrs['dynamic_adjs'] = dynamic_adjs_np[val_warmup_start:val_end_abs]
    else:
        train_idx_end = np.sum(train_mask)
        train_data.attrs['graph_features'] = event_aware_features[:train_idx_end]
        train_data.attrs['dynamic_adjs'] = dynamic_adjs_np[:train_idx_end]
        val_data.attrs['graph_features'] = event_aware_features[val_warmup_start:val_end_idx + 1]
        val_data.attrs['dynamic_adjs'] = dynamic_adjs_np[val_warmup_start:val_end_idx + 1]
    
    # Define Patch function
    def patch_env_for_graph(env, features_tensor):
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

    # Resolve reward type
    # For Dataset A (28stocks), Differential Sharpe Ratio (DSR) is superior for Sharpe optimization
    if portfolio_name in ['28stocks', 'dataset_a'] and reward_type_override is None:
        active_reward_type = 'log_utility'
    else:
        active_reward_type = reward_type_override or 'log_utility'
    print(f"Reward type: {active_reward_type}")
    if active_reward_type == 'aggressive':
        print("  → Sun et al. compatible mode: raw fractional return, NO penalties")
    
    # Create Env with Phase 2/3/5.3 enhancements
    env = PortfolioEnv(
        data=train_data,
        initial_balance=100000.0,
        transaction_cost=0.0025,
        reward_type=active_reward_type,
        max_weight_per_asset=0.15,
        noise_augmentation=noise_augmentation,
        noise_std=noise_std,
        random_start_offset=random_start_offset,
        max_start_offset_pct=max_start_offset_pct,
        diversification_bonus=diversification_bonus,
        horizon_blend_lambda=horizon_blend_lambda,
    )
    
    feat_dim = patch_env_for_graph(env, train_data.attrs['graph_features'])
    
    # 4. Agent
    # Flexible Device Strategy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing GNNSACAgent on {device}...")
    
    # Phase 4: Construct sector map tensor mapping each asset to a continuous sector ID [0..num_sectors-1]
    sector_names_unique = sorted(list(set(graph_builder.sector_map.get(t, "Other") for t in graph_builder.tickers)))
    sector_name_to_id = {name: i for i, name in enumerate(sector_names_unique)}
    sector_map_array = [sector_name_to_id[graph_builder.sector_map.get(t, "Other")] for t in graph_builder.tickers]
    sector_map_tensor = torch.tensor(sector_map_array, dtype=torch.long, device=device)

    # build the logit-level action mask from --restrict_to_assets.
    sector_mask_tensor = None
    stock_mask_tensor = None
    if restrict_to_assets:
        from src.agents.gnn_sac_agent import build_action_masks_from_indices
        tickers_lower = [t.upper() for t in graph_builder.tickers]
        allowed = [t.upper() for t in restrict_to_assets]
        missing = [t for t in allowed if t not in tickers_lower]
        if missing:
            print(f"WARNING:  restrict_to_assets contains tickers not in portfolio: {missing}")
        allowed_idx = [i for i, t in enumerate(tickers_lower) if t in allowed]
        if len(allowed_idx) == 0:
            raise ValueError(
                f"restrict_to_assets={restrict_to_assets} matches no tickers in portfolio {tickers_lower}"
            )
        sector_mask_tensor, stock_mask_tensor_full = build_action_masks_from_indices(
            allowed_idx, sector_map_tensor.cpu(), len(sector_names_unique),
        )
        # Pad/truncate stock_mask to match action_dim (= n_assets)
        action_dim_local = env.action_space.shape[0]
        if stock_mask_tensor_full.shape[0] != action_dim_local:
            buf = torch.zeros(action_dim_local, dtype=torch.bool)
            n_copy = min(stock_mask_tensor_full.shape[0], action_dim_local)
            buf[:n_copy] = stock_mask_tensor_full[:n_copy]
            stock_mask_tensor_full = buf
        stock_mask_tensor = stock_mask_tensor_full.to(device)
        sector_mask_tensor = sector_mask_tensor.to(device)
        print(f"allowed indices = {allowed_idx} "
              f"(sectors enabled = {int(sector_mask_tensor.sum().item())}/"
              f"{len(sector_names_unique)})")

    agent = GNNSACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        node_features_dim=feat_dim,
        num_nodes=graph_builder.num_nodes,
        adj_matrix=graph_builder.mixed_adj_template,
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_capacity=200000,
        lstm_hidden_size=64,
        num_lstm_layers=1,
        hidden_dims=[256, 256],
        use_graph=True,
        use_tgn=use_tgn,
        use_nextgen=use_nextgen,
        alpha=alpha_init,
        auto_tune_alpha=auto_tune_alpha,
        gamma=gamma,
        tau=tau,
        target_update_interval=target_update_interval,
        num_stock_nodes=graph_builder.num_stock_nodes,
        num_sector_nodes=len(graph_builder.sector_map_local) if hasattr(graph_builder, 'sector_map_local') else None,
        num_event_nodes=graph_builder.num_event_nodes if hasattr(graph_builder, 'num_event_nodes') else None,
        sector_map=sector_map_tensor,
        # Advanced Stabilizers
        use_attention_gating=use_attention_gating,
        attn_gate_threshold=attn_gate_threshold,
        use_momentum_masking=use_momentum_masking,
        momentum_threshold=momentum_threshold,
        residual_alpha=residual_alpha,
        cvar_multiplier=cvar_multiplier,
        cvar_tau=cvar_tau,
        use_cvar_loss=use_cvar_loss,
        # Phase 1 architecture flags (must be in kwargs for __init__)
        use_edge_pruning=use_edge_pruning,
        use_nlp_alpha=use_nlp_alpha,
        use_dual_head_iqn=use_dual_head_iqn,
        use_action_masking=use_action_masking,
        use_feature_gating=use_feature_gating,
        # logit-level allocation-universe restriction (None = no mask)
        sector_mask=sector_mask_tensor,
        stock_mask=stock_mask_tensor,
    )
    
    # --- Phase 6: Resume from Checkpoint ---
    final_model_path = os.path.join(BASE_DIR, "models", f"gnn_sac_event_driven_{effective_name}.pth")
    if os.path.exists(final_model_path):
        print(f"Found existing model at {final_model_path}. Resuming training...")
        try:
            agent.load(final_model_path)
            print("Successfully loaded model state.")
        except Exception as e:
            print(f"Warning: Failed to load existing model: {e}")
    
    
    # Configure Action Masking indices based on actual tickers if used
    if use_action_masking:
        eq_idx = [i for i, t in enumerate(graph_builder.tickers) if graph_builder.sector_map.get(t, "Other") not in ['FixedIncome', 'Commodity']]
        sh_idx = [i for i, t in enumerate(graph_builder.tickers) if graph_builder.sector_map.get(t, "Other") in ['FixedIncome', 'Commodity']]
        agent.policy.equity_indices = eq_idx
        agent.policy.safe_haven_indices = sh_idx
        agent.policy.use_action_masking = True

    if use_nextgen:
        print('[NextGen] Architecture enabled: C1-C7 + IQN distributional critic + regime entropy')
    
    # 5. Training Loop with LR Scheduling + Reward Normalization + Validation
    print("Starting Training with Dynamic Graphs...")
    state, _ = env.reset(seed=42)
    hidden_state = None
    rewards = []
    reward_normalizer = RunningMeanStd()
    
    # Helper to get current adj
    train_adjs = train_data.attrs['dynamic_adjs']
    
    # --- Phase 2 Fix: Load Macro Context (VIX, Treasury) ---
    try:
        vix_df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "vix.csv"), index_col=0, parse_dates=True)
        vix_series = vix_df.reindex(train_data.index).ffill().fillna(20.0).iloc[:, 0]
        vix_mean_series = vix_series.rolling(60, min_periods=1).mean()
        
        # Prepare context for training loop
        vix_vals = vix_series.values
        vix_means = vix_mean_series.values
        
        # Prepare context for validation loop
        vix_series_val = vix_df.reindex(val_data.index).ffill().fillna(20.0).iloc[:, 0]
        vix_mean_series_val = vix_series_val.rolling(60, min_periods=1).mean()
        vix_vals_val = vix_series_val.values
        vix_means_val = vix_mean_series_val.values
    except Exception as e:
        print(f"Warning: Macro data not found for context: {e}")
        vix_vals = np.full(len(train_data), 20.0)
        vix_means = np.full(len(train_data), 20.0)

    # Ensure checkpoint directory exists
    checkpoint_dir = os.path.join(BASE_DIR, "models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Phase 0: LR Cosine Annealing with Linear Warmup ---
    lr_decay_steps = total_timesteps - warmup_steps
    schedulers = []
    for opt in [agent.policy_optimizer, agent.q1_optimizer, agent.q2_optimizer]:
        sched = CosineAnnealingLR(opt, T_max=max(lr_decay_steps, 1), eta_min=lr_min)
        schedulers.append(sched)
    # Include IQN optimizer if nextgen
    if use_nextgen and hasattr(agent, 'iqn_optimizer'):
        iqn_sched = CosineAnnealingLR(agent.iqn_optimizer, T_max=max(lr_decay_steps, 1), eta_min=lr_min)
        schedulers.append(iqn_sched)
    print(f"LR Schedule: Linear warmup {warmup_steps} steps → Cosine decay to {lr_min:.1e} over {lr_decay_steps} steps")

    # --- Validation-Based Best Model Tracking + Early Stopping ---
    best_val_sharpe = -np.inf
    best_model_path = os.path.join(BASE_DIR, "models", f"gnn_sac_event_driven_{effective_name}_best.pth")
    early_stop_patience = early_stopping_patience
    early_stop_counter = 0
    nan_count = 0
    NAN_HALT_THRESHOLD = 50  # Halt after this many consecutive NaN updates

    # Validation env setup
    env_val = PortfolioEnv(
        data=val_data,
        initial_balance=100000.0,
        transaction_cost=0.0025,
        reward_type=active_reward_type,
        max_weight_per_asset=0.15,
    )
    val_feats = val_data.attrs['graph_features']
    if len(val_feats) > 0:
        patch_env_for_graph(env_val, val_feats)
    val_adjs = val_data.attrs['dynamic_adjs']

    for step in tqdm(range(total_timesteps), desc="RL Training"):
        # Phase 5.4: CVaR Curriculum Learning
        if use_cvar_loss:
            agent.cvar_multiplier = get_annealed_cvar(
                current_step=step,
                start_step=cvar_anneal_start,
                end_step=cvar_anneal_end,
                max_cvar=cvar_multiplier
            )
        
        # Get Current Dyn Adj
        idx = env.current_step + env.window_size
        if idx >= len(train_adjs): idx = len(train_adjs) - 1
        current_adj_np = train_adjs[idx]
        current_adj = torch.tensor(current_adj_np, dtype=torch.float32).to(device).unsqueeze(0) # [1, N, N]
        
        # Phase 2 Fix: Construct NextGen Context
        curr_idx = env.current_step + env.window_size
        if curr_idx >= len(vix_vals): curr_idx = len(vix_vals) - 1
        
        ctx = {
            'vix': torch.tensor([vix_vals[curr_idx]], dtype=torch.float32).to(device),
            'vix_mean': torch.tensor([vix_means[curr_idx]], dtype=torch.float32).to(device),
            # Add other components if needed (regime_feats, vol_vec, etc.)
        } if use_nextgen else None

        if step < random_steps:
            action = env.action_space.sample()
            hidden_state = None
        else:
            # Pass dynamic adj + nextgen_ctx to select_action
            action, hidden_state = agent.select_action(state, hidden_state, deterministic=False, adj=current_adj, nextgen_ctx=ctx)
            
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        # Normalize reward for stable critic learning
        # SKIP normalization for 'aggressive' mode — raw return signal is cleaner
        # Phase 2 Fix: DSR is already a normalized ratio signal, skip RunningMeanStd
        if active_reward_type in ('aggressive', 'dsr'):
            norm_reward = reward
        else:
            reward_normalizer.update(reward)
            norm_reward = reward_normalizer.normalize(reward)
        
        agent.replay_buffer.push(state, action, norm_reward, next_state, float(done))
        state = next_state
        rewards.append(reward)  # Track raw reward for plotting
        
        if step >= random_steps:
            # Pass context to update if needed
            update_info = agent.update(nextgen_ctx=ctx)
            
            # --- Phase 2 Fix: Clear CUDA Cache to prevent OOM ---
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            # Memory Optimization: Periodic cache clear for small GPUs
            if step % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ── NaN Loss Detection ────────────────────────────────────
            if update_info and any(
                isinstance(v, float) and (np.isnan(v) or np.isinf(v))
                for v in update_info.values()
            ):
                nan_count += 1
                if nan_count >= NAN_HALT_THRESHOLD:
                    logger.log(f"\nCRITICAL: NaN/Inf loss detected {nan_count} consecutive times at step {step+1}.")
                    print(f"   Losses: {update_info}")
                    print("   HALTING training to prevent saving broken checkpoints.")
                    print("   → Check learning rate, gradient clipping, and encoder NaN guards.")
                    ckpt_path = os.path.join(checkpoint_dir, f"gnn_sac_{effective_name}_step_{step+1}_NAN_HALT.pth")
                    agent.save(ckpt_path)
                    break
            else:
                nan_count = 0  # Reset on healthy update
            # ── End NaN Detection ─────────────────────────────────────
            # Phase 0: Step LR schedulers with warmup
            if step < warmup_steps:
                # Linear warmup: scale LR from 0 to learning_rate
                warmup_factor = (step + 1) / warmup_steps
                opts = [agent.policy_optimizer]
                if use_nextgen:
                    opts.append(agent.iqn_optimizer)
                else:
                    opts.extend([agent.q1_optimizer, agent.q2_optimizer])
                
                for opt in opts:
                    for pg in opt.param_groups:
                        pg['lr'] = learning_rate * warmup_factor
            else:
                for sched in schedulers:
                    sched.step()
            
        if done:
            state, _ = env.reset()
            hidden_state = None

        # Checkpointing + Validation-Based Best Selection
        if (step + 1) % save_freq == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"gnn_sac_{effective_name}_step_{step+1}.pth")
            agent.save(ckpt_path)

            # --- Validation Evaluation ---
            try:
                val_obs, _ = env_val.reset(seed=42)
                val_hidden = None
                val_done = False
                val_values = [100000.0]
                while not val_done:
                    v_idx = min(env_val.current_step + env_val.window_size, len(val_adjs) - 1)
                    v_adj = torch.tensor(val_adjs[v_idx], dtype=torch.float32).to(device).unsqueeze(0)
                    
                    # Phase 2 Fix: Pass context to validation
                    val_ctx = {
                        'vix': torch.tensor([vix_vals_val[v_idx]], dtype=torch.float32).to(device),
                        'vix_mean': torch.tensor([vix_means_val[v_idx]], dtype=torch.float32).to(device),
                    } if use_nextgen else None
                    
                    val_action, val_hidden = agent.select_action(val_obs, val_hidden, deterministic=True, adj=v_adj, nextgen_ctx=val_ctx)
                    val_obs, _, v_term, v_trunc, v_info = env_val.step(val_action)
                    val_done = v_term or v_trunc
                    val_values.append(v_info['portfolio_value'])

                val_returns = pd.Series(val_values).pct_change().dropna()
                val_sharpe = float((val_returns.mean() / val_returns.std()) * np.sqrt(252)) if val_returns.std() > 1e-9 else 0.0
                val_roi = (val_values[-1] - val_values[0]) / val_values[0] * 100
                current_lr = schedulers[0].get_last_lr()[0]
                logger.log(f"  Step {step+1}: Val Sharpe={val_sharpe:.4f}, Val ROI={val_roi:.2f}%, LR={current_lr:.2e}")

                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    early_stop_counter = 0
                    agent.save(best_model_path)
                    print(f"  * New Best Validation Sharpe: {val_sharpe:.4f} → Saved to {best_model_path}")
                else:
                    early_stop_counter += 1
                    print(f"  No improvement ({early_stop_counter}/{early_stop_patience})")
                    if early_stop_counter >= early_stop_patience:
                        logger.log(f"  EARLY STOPPING at step {step+1} — Val Sharpe has not improved for {early_stop_patience} checkpoints")
                        print(f"     Best Val Sharpe: {best_val_sharpe:.4f} (saved at {best_model_path})")
                        break
            except Exception as e:
                print(f"  Validation error at step {step+1}: {e}")
            
    # Save Final Model
    save_path = os.path.join(BASE_DIR, "models", f"gnn_sac_event_driven_{effective_name}.pth")
    agent.save(save_path)
    print(f"Model saved to {save_path}")

    # Plot
    plt.plot(pd.Series(rewards).rolling(100).mean())
    plt.title("Event-Driven GNN-SAC Training Rewards")
    png_path = os.path.join(BASE_DIR, "results", f"gnn_sac_training_{effective_name}.png")
    eps_path = os.path.join(BASE_DIR, "results", f"gnn_sac_training_{effective_name}.eps")
    plt.savefig(png_path)
    plt.savefig(eps_path, format="eps", bbox_inches="tight")
    print(f"Saved training plot to {png_path} and {eps_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN-SAC Agent")
    parser.add_argument("--portfolio_name",    type=str,   default='28stocks', help="Name of the portfolio/dataset")
    parser.add_argument("--total_timesteps",   type=int,   default=200000,     help="Total training timesteps")
    parser.add_argument("--batch_size",        type=int,   default=None,       help="Batch size (auto per dataset)")
    parser.add_argument("--learning_rate",     type=float, default=None,       help="Learning rate (auto per dataset)")
    parser.add_argument("--alpha_init",        type=float, default=None,       help="SAC entropy coefficient")
    parser.add_argument("--use_tgn",           action="store_true",            help="Enable legacy TGN encoder")
    parser.add_argument("--save_freq",         type=int,   default=10000,      help="Checkpoint save frequency")
    # Next-Gen architecture flags
    parser.add_argument("--use_nextgen",       action="store_true",
                        help="Enable next-gen EventGraph (C1-C7: regime-lambda, vol-attention, "
                             "shock-gating, severity-edges, DD-gate, decay-TGN, hierarchical-HGT) "
                             "+ IQN distributional critic + regime-entropy scheduler")
    parser.add_argument("--use_edge_pruning",  action="store_true", help="Enable dynamic edge pruning based on momentum")
    parser.add_argument("--use_nlp_alpha",     action="store_true", help="Enable NLP-conditioned dynamic entropy scaling")
    parser.add_argument("--use_dual_head_iqn", action="store_true", help="Enable dual-head IQN critic for tail-risk")
    parser.add_argument("--use_action_masking",action="store_true", help="Enable macro-conditioned action masking")
    parser.add_argument("--reward_type",       type=str,   default=None,
                        choices=['log_utility', 'sharpe', 'return', 'risk_adjusted', 'aggressive', 'dsr'],
                        help="Override reward type. 'dsr' = Differential Sharpe Ratio (recommended).")
    
    # Advanced Stabilizers (D1-D3)
    parser.add_argument("--use_cvar_loss",      action="store_true", default=True, help="Enable CVaR-regulated critic loss (D1)")
    parser.add_argument("--cvar_multiplier",     type=float, default=1.0,  help="Weight for CVaR penalty")
    parser.add_argument("--cvar_tau",            type=float, default=0.1,  help="CVaR quantile threshold (e.g. 0.1 for 10% tail)")
    parser.add_argument("--cvar_anneal_start",   type=int,   default=50000,    help="Step to start ramping up CVaR penalty")
    parser.add_argument("--cvar_anneal_end",     type=int,   default=100000,   help="Step to reach max CVaR penalty")
    parser.add_argument("--use_attention_gating",action="store_true", help="Enable adaptive attention gating (D2)")
    parser.add_argument("--attn_gate_threshold", type=float, default=2.0,  help="Z-score threshold for attention gating")
    parser.add_argument("--use_momentum_masking",action="store_true", help="Enable local momentum action masking (D3)")
    parser.add_argument("--momentum_threshold",  type=float, default=-0.2, help="Threshold for momentum suppression")
    parser.add_argument("--residual_alpha",      type=float, default=0.5,  help="D4 residual alpha")
    parser.add_argument("--use_feature_gating", action="store_true", help="Enable D6 learnable feature gating (soft-SHAP)")

    # Isolated sub-portfolio experiment flags
    parser.add_argument("--restrict_to_assets", type=str, default="",
                        help="comma-separated ticker symbols defining the allocation universe "
                             "(logit-level mask). Empty = no restriction. "
                             "Example: BA,INTC,JPM,WMT,CVX,GE,HON,MSFT,SHEL")
    parser.add_argument("--model_suffix", type=str, default="",
                        help="suffix appended to portfolio_name in checkpoint / log paths so "
                             "restricted-universe runs don't overwrite headline checkpoints. "
                             "Example: _imputed9")
    parser.add_argument("--disable_event_channel", action="store_true",
                        help="Static-Graph baseline: replace dynamic event-blended adjacency "
                             "with the static template (λ_mix=0).")

    parser.add_argument("--config",            type=str,   default="",         help="Path to YAML configuration file")

    args = parser.parse_args()

    # Load from config file if provided
    kwargs = {
        'portfolio_name': args.portfolio_name,
        'total_timesteps': args.total_timesteps,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'alpha_init': args.alpha_init,
        'use_tgn': args.use_tgn,
        'save_freq': args.save_freq,
        'use_nextgen': args.use_nextgen,
        'reward_type_override': args.reward_type,
        'use_edge_pruning': args.use_edge_pruning,
        'use_nlp_alpha': args.use_nlp_alpha,
        'use_dual_head_iqn': args.use_dual_head_iqn,
        'use_action_masking': args.use_action_masking,
        'use_cvar_loss': args.use_cvar_loss,
        'cvar_multiplier': args.cvar_multiplier,
        'cvar_tau': args.cvar_tau,
        'use_attention_gating': args.use_attention_gating,
        'attn_gate_threshold': args.attn_gate_threshold,
        'use_momentum_masking': args.use_momentum_masking,
        'momentum_threshold': args.momentum_threshold,
        'residual_alpha': args.residual_alpha,
        'use_feature_gating': args.use_feature_gating,
        'cvar_anneal_start': args.cvar_anneal_start,
        'cvar_anneal_end': args.cvar_anneal_end,
        # isolated sub-portfolio experiment
        'restrict_to_assets': [t.strip() for t in args.restrict_to_assets.split(',') if t.strip()] or None,
        'model_suffix': args.model_suffix,
        'disable_event_channel': args.disable_event_channel,
    }

    if args.config:
        print(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            if 'experiment' in cfg:
                if 'use_nextgen' in cfg['experiment']:
                    kwargs['use_nextgen'] = cfg['experiment']['use_nextgen']
            if 'data' in cfg:
                if 'portfolio_name' in cfg['data']:
                    kwargs['portfolio_name'] = cfg['data']['portfolio_name']
            if 'training' in cfg:
                tcfg = cfg['training']
                kwargs['total_timesteps'] = tcfg.get('max_steps', kwargs['total_timesteps'])
                kwargs['batch_size'] = tcfg.get('batch_size', kwargs['batch_size'])
                kwargs['learning_rate'] = tcfg.get('learning_rate', kwargs['learning_rate'])
                kwargs['alpha_init'] = tcfg.get('alpha', kwargs['alpha_init'])
                if 'auto_tune_alpha' in tcfg:
                    kwargs['auto_tune_alpha'] = tcfg.get('auto_tune_alpha')
                kwargs['early_stopping_patience'] = tcfg.get('early_stopping_patience', 20)
                # Phase 0: LR warmup
                kwargs['warmup_steps'] = tcfg.get('warmup_steps', 5000)
                kwargs['lr_min'] = tcfg.get('lr_min', 1e-6)
                # Phase 5.2: Target update interval
                kwargs['target_update_interval'] = tcfg.get('target_update_interval', 2)
            # Phase 2/3/5.3: Environment enhancements
            if 'environment' in cfg:
                ecfg = cfg['environment']
                kwargs['reward_type_override'] = kwargs.get('reward_type_override') or ecfg.get('reward_type')
                kwargs['noise_augmentation'] = ecfg.get('noise_augmentation', False)
                kwargs['noise_std'] = ecfg.get('noise_std', 0.001)
                kwargs['random_start_offset'] = ecfg.get('random_start_offset', False)
                kwargs['max_start_offset_pct'] = ecfg.get('max_start_offset_pct', 0.1)
                kwargs['diversification_bonus'] = ecfg.get('diversification_bonus', 0.0)
                kwargs['horizon_blend_lambda'] = ecfg.get('horizon_blend_lambda', 1.0)

    # Redirect Output
    if not os.path.exists("results"):
        os.makedirs("results")
    _log_name = f"{kwargs['portfolio_name']}{kwargs.get('model_suffix') or ''}"
    sys.stdout = Tee(os.path.join("results", f"log_train_gnn_sac_{_log_name}.txt"), "w")

    train_gnn_sac(**kwargs)

