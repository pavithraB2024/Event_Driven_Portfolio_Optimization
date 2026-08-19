"""Train the GraphSAGE-PPO baseline inside this repo's env/split/cost model.

Implements the direct DRL-baseline replication asked for in Reviewer 2,
Comment 8: rather than copying literature numbers, the GraphSAGE-PPO agent
is trained on the **same** 28-stock universe, **same** Soleymani & Paquet
(2021) train/val/benchmark split, and **same** ``PortfolioEnv``
transaction-cost model used by the proposed GNN-SAC agent.

The script intentionally reuses ``train_gnn_sac``'s graph construction
(precomputed ``*_precomputed_graphs.npz``) so that the only thing varying
between the two baselines is the agent — encoder family + objective —
and not the data pipeline.

Run from the repo root with ``PYTHONPATH=.`` (matches the rest of the
training/evaluation scripts):

    PYTHONPATH=. python scripts/training/train_graphsage_ppo.py \\
        --portfolio_name 28stocks --total_timesteps 200000

The resulting checkpoint is dropped into
``models/graphsage_ppo_event_driven_<portfolio>.pth`` so the evaluation
pipeline can pick it up next to ``gnn_sac_event_driven_<portfolio>.pth``.
"""

from __future__ import annotations

import argparse
import os
import sys
import types

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.agents.graphsage_ppo_agent import GraphSAGEPPOAgent, PPOConfig
from src.data_pipeline.graph_builder import DynamicGraphBuilder
from src.environments.portfolio_env import PortfolioEnv


def _patch_env_for_graph(env: PortfolioEnv, features_tensor: np.ndarray) -> int:
    """Replicate the state layout used by train_gnn_sac so the encoder sees
    identical inputs to the GNN-SAC pipeline. Returns per-node feat dim."""
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
            w_padded = np.pad(w, (0, feats.shape[0] - len(w)), "constant")
            w = w_padded.reshape(-1, 1)
        else:
            w = w[: feats.shape[0]].reshape(-1, 1)
        return np.hstack([feats, w]).flatten().astype(np.float32)

    env._get_state = types.MethodType(get_graph_state, env)
    n_graph_nodes = features_tensor.shape[1]
    feat_dim = features_tensor.shape[2] + 1  # +1 for the appended weight column
    total_dims = n_graph_nodes * feat_dim
    env.observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(total_dims,), dtype=np.float32
    )
    return feat_dim


def train_graphsage_ppo(
    portfolio_name: str = "28stocks",
    total_timesteps: int = 200_000,
    rollout_length: int = 2048,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.005,
    value_coef: float = 0.5,
    update_epochs: int = 10,
    minibatch_size: int = 64,
    save_freq: int = 20_000,
    seed: int = 42,
):
    print("=== Training GraphSAGE-PPO baseline (replication) ===")
    data_dir = os.path.join(BASE_DIR, "data")
    processed_news_path = os.path.join(
        data_dir, "processed", f"news_with_events_{portfolio_name}.csv"
    )

    env_data_path = os.path.join(data_dir, "processed", f"{portfolio_name}_dataset.csv")
    if not os.path.exists(env_data_path) and portfolio_name == "28stocks":
        env_data_path = os.path.join(
            data_dir, "processed", "complete_dataset_enhanced.csv"
        )
    if not os.path.exists(env_data_path):
        raise FileNotFoundError(f"Dataset not found at {env_data_path}")

    env_data = pd.read_csv(env_data_path, index_col=0, parse_dates=True)

    node_feat_path = os.path.join(data_dir, f"{portfolio_name}_data.npy")
    if not os.path.exists(node_feat_path):
        raise FileNotFoundError(f"Node features not found at {node_feat_path}")
    base_node_features = np.load(node_feat_path)

    price_cols = [c for c in env_data.columns if c.startswith("price_")]
    n_assets = len(price_cols)
    print(f"Verified Portfolio Assets: {n_assets}")
    if base_node_features.shape[1] < n_assets:
        raise ValueError(
            f"Node features ({base_node_features.shape[1]}) < portfolio assets ({n_assets})"
        )

    # Match z-score normalization used by GNN-SAC so encoder inputs are identical.
    feat_mean = np.nanmean(base_node_features, axis=(0, 1), keepdims=True)
    feat_std = np.nanstd(base_node_features, axis=(0, 1), keepdims=True) + 1e-6
    base_node_features = (base_node_features - feat_mean) / feat_std
    base_node_features = np.clip(base_node_features, -5.0, 5.0)

    dates = env_data.index
    min_len = min(len(dates), base_node_features.shape[0])
    dates = dates[:min_len]
    base_node_features = base_node_features[:min_len]
    env_data = env_data.iloc[:min_len]

    # Reuse the same graph cache the proposed model trains on.
    print("Initializing Dynamic Graph Builder...")
    graph_builder = DynamicGraphBuilder(data_dir, portfolio_name, processed_news_path)
    if graph_builder.num_stock_nodes > n_assets:
        graph_builder.resize_to_match_portfolio(n_assets)

    cache_path = os.path.join(
        data_dir, "processed", f"{portfolio_name}_precomputed_graphs.npz"
    )
    if os.path.exists(cache_path):
        print(f"Loading cached graphs from {cache_path}...")
        cache = np.load(cache_path)
        event_aware_features = cache["features"]
    else:
        print("Building graph cache (first run only)...")
        event_aware_features = []
        base_feat_tensor = torch.tensor(base_node_features, dtype=torch.float32)
        for t in tqdm(range(min_len)):
            graph_data = graph_builder.get_graph(dates[t], base_feat_tensor[t])
            event_aware_features.append(graph_data.x.numpy())
        event_aware_features = np.array(event_aware_features)

    # Same split as the manuscript / train_gnn_sac.py.
    train_end = "2012-12-05"
    val_start = "2012-12-06"
    val_end = "2013-11-03"

    train_mask = dates <= train_end
    val_mask = (dates >= val_start) & (dates <= val_end)
    train_data = env_data.loc[train_mask]
    val_start_idx = int(np.argmax(val_mask))
    val_warmup_start = max(0, val_start_idx - 60)
    val_end_idx = len(dates) - 1 - int(np.argmax(val_mask[::-1]))
    val_data = env_data.iloc[val_warmup_start : val_end_idx + 1]

    train_idx_end = int(np.sum(train_mask))
    train_data.attrs["graph_features"] = event_aware_features[:train_idx_end]
    val_data.attrs["graph_features"] = event_aware_features[val_warmup_start : val_end_idx + 1]

    print(f"Train: {len(train_data)} rows ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"Val:   {len(val_data)} rows ({val_data.index[0]} to {val_data.index[-1]})")

    env = PortfolioEnv(
        data=train_data,
        initial_balance=100_000.0,
        transaction_cost=0.0025,
        reward_type="log_utility",
        max_weight_per_asset=0.15,
    )
    feat_dim = _patch_env_for_graph(env, train_data.attrs["graph_features"])
    num_nodes = train_data.attrs["graph_features"].shape[1]

    # Static adjacency from the mixed sector/correlation template — matches the
    # GraphSAGE-PPO paper's "Asset Graph" definition (no event channel) so the
    # comparison against our event-driven GNN-SAC is the actual ablation
    # variable.
    static_adj = graph_builder.mixed_adj_template[:num_nodes, :num_nodes]
    if isinstance(static_adj, torch.Tensor):
        static_adj = static_adj.detach().cpu().numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    agent = GraphSAGEPPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        node_features_dim=feat_dim,
        num_nodes=num_nodes,
        adj_matrix=static_adj,
        config=PPOConfig(
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
        ),
        device=str(device),
    )

    np.random.seed(seed)
    torch.manual_seed(seed)

    state, _ = env.reset(seed=seed)
    rollout_rewards: list[float] = []
    episode_reward = 0.0
    episode_returns: list[float] = []
    best_val_sharpe = -np.inf
    best_path = os.path.join(
        BASE_DIR, "models", f"graphsage_ppo_event_driven_{portfolio_name}_best.pth"
    )
    final_path = os.path.join(
        BASE_DIR, "models", f"graphsage_ppo_event_driven_{portfolio_name}.pth"
    )

    # Setup validation environment up front.
    env_val = PortfolioEnv(
        data=val_data,
        initial_balance=100_000.0,
        transaction_cost=0.0025,
        reward_type="log_utility",
        max_weight_per_asset=0.15,
    )
    if len(val_data.attrs["graph_features"]) > 0:
        _patch_env_for_graph(env_val, val_data.attrs["graph_features"])

    for step in tqdm(range(total_timesteps), desc="PPO training"):
        action, _ = agent.select_action(state, deterministic=False)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        agent.store_transition(state, action, reward, float(done), adj=static_adj)
        state = next_state
        episode_reward += reward

        if done:
            episode_returns.append(episode_reward)
            episode_reward = 0.0
            state, _ = env.reset()

        if (step + 1) % rollout_length == 0:
            metrics = agent.update(last_state=state)
            rollout_rewards.append(float(np.mean(episode_returns[-5:]) if episode_returns else 0.0))
            if metrics:
                print(
                    f"  step {step + 1}: policy_loss={metrics.get('policy_loss', 0.0):.4f} "
                    f"value_loss={metrics.get('value_loss', 0.0):.4f} "
                    f"entropy={metrics.get('entropy', 0.0):.4f} "
                    f"kl={metrics.get('approx_kl', 0.0):.4f}"
                )

        if (step + 1) % save_freq == 0:
            # Validation eval — keep the best-Sharpe checkpoint.
            try:
                val_state, _ = env_val.reset(seed=seed)
                vals = [100_000.0]
                done_v = False
                while not done_v:
                    v_action, _ = agent.select_action(val_state, deterministic=True)
                    val_state, _, term_v, trunc_v, info_v = env_val.step(v_action)
                    done_v = term_v or trunc_v
                    vals.append(info_v["portfolio_value"])
                val_returns = pd.Series(vals).pct_change().dropna()
                val_sharpe = (
                    float((val_returns.mean() / val_returns.std()) * np.sqrt(252))
                    if val_returns.std() > 1e-9
                    else 0.0
                )
                val_roi = (vals[-1] - vals[0]) / vals[0] * 100
                print(
                    f"  [val] step={step + 1} sharpe={val_sharpe:.4f} roi={val_roi:.2f}%"
                )
                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    agent.save(best_path)
                    print(f"  -> new best val sharpe; saved to {best_path}")
            except Exception as exc:  # pragma: no cover - logging only
                print(f"  validation error at step {step + 1}: {exc}")

    agent.save(final_path)
    print(f"Final model saved to {final_path}")
    if best_val_sharpe > -np.inf:
        print(f"Best val Sharpe: {best_val_sharpe:.4f} (saved to {best_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GraphSAGE-PPO baseline ")
    parser.add_argument("--portfolio_name", type=str, default="28stocks")
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--rollout_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.005)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--save_freq", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_graphsage_ppo(**vars(args))
