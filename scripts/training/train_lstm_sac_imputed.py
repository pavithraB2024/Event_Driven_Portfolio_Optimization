"""
LSTM-SAC training script.

Trains a ``MemoryAugmentedSACAgent`` (no graph encoder). Optionally restricts
the actor to a user-specified subset of assets via a logit-level stock_mask
(enforced inside ``LSTMGaussianPolicy.sample`` / ``get_action``).

Mirrors ``train_gnn_sac.py``'s data + split contract (Sun et al. 2024).

Usage (full 28-stock baseline for Table 5):
    PYTHONPATH=. python scripts/training/train_lstm_sac_imputed.py \\
        --portfolio_name 28stocks \\
        --total_timesteps 200000

Usage (9-stock sub-portfolio ablation):
    PYTHONPATH=. python scripts/training/train_lstm_sac_imputed.py \\
        --portfolio_name 28stocks \\
        --restrict_to_assets BA,INTC,JPM,WMT,CVX,GE,HON,MSFT,SHEL \\
        --model_suffix _imputed9_lstm \\
        --total_timesteps 200000
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.agents.memory_augmented_sac_agent import MemoryAugmentedSACAgent
from src.environments.portfolio_env import PortfolioEnv


class RunningMeanStd:
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        batch_mean = float(x)
        self.count += 1
        delta = batch_mean - self.mean
        self.mean += delta / self.count
        self.var = self.var * (1 - 1 / self.count) + (delta ** 2) / self.count

    def normalize(self, x):
        return (x - self.mean) / max(np.sqrt(self.var), 1e-8)


def _load_env_data(data_dir, portfolio_name):
    candidate = os.path.join(data_dir, "processed", f"{portfolio_name}_dataset.csv")
    if not os.path.exists(candidate) and portfolio_name == "28stocks":
        candidate = os.path.join(data_dir, "processed", "complete_dataset_enhanced.csv")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"env data not found: {candidate}")
    return pd.read_csv(candidate, index_col=0, parse_dates=True)


def _read_tickers(data_dir, portfolio_name, env_data):
    tpath = os.path.join(data_dir, f"{portfolio_name}_tickers.txt")
    if os.path.exists(tpath):
        with open(tpath) as f:
            return [t.strip() for t in f.read().strip().split(",") if t.strip()]
    return [c.replace("price_", "") for c in env_data.columns if c.startswith("price_")]


def _build_stock_mask(tickers, restrict_to_assets, device):
    tickers_upper = [t.upper() for t in tickers]
    allowed = {t.upper() for t in restrict_to_assets}
    mask = torch.tensor(
        [t in allowed for t in tickers_upper], dtype=torch.bool, device=device
    )
    missing = sorted(allowed - set(tickers_upper))
    if missing:
        print(f"WARNING:  restrict_to_assets contains tickers not in portfolio: {missing}")
    if not mask.any():
        raise ValueError(
            f"restrict_to_assets={restrict_to_assets} matches no tickers in {tickers_upper}"
        )
    return mask


def train_lstm_sac_imputed(
    portfolio_name="28stocks",
    restrict_to_assets=None,
    model_suffix="",
    total_timesteps=200000,
    learning_rate=1e-4,
    batch_size=8,
    save_freq=10000,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = os.path.join(BASE_DIR, "data")
    env_data = _load_env_data(data_dir, portfolio_name)
    tickers = _read_tickers(data_dir, portfolio_name, env_data)
    print(f"LSTM-SAC: portfolio={portfolio_name}, |tickers|={len(tickers)}")

    # Sun et al. (2024) split — keep identical to train_gnn_sac.py
    dates = env_data.index
    train_end = "2012-12-05"
    val_start = "2012-12-06"
    val_end = "2013-11-03"

    train_data = env_data.loc[dates <= train_end]
    val_mask = (dates >= val_start) & (dates <= val_end)
    val_start_idx = int(np.argmax(val_mask))
    val_warmup_start = max(0, val_start_idx - 60)
    val_end_idx = len(dates) - 1 - int(np.argmax(val_mask[::-1]))
    val_data = env_data.iloc[val_warmup_start: val_end_idx + 1]
    print(f"Train rows={len(train_data)}  Val rows={len(val_data)}")

    env = PortfolioEnv(
        data=train_data,
        initial_balance=100000.0,
        transaction_cost=0.0025,
        reward_type="log_utility",
        max_weight_per_asset=0.15,
    )
    env_val = PortfolioEnv(
        data=val_data,
        initial_balance=100000.0,
        transaction_cost=0.0025,
        reward_type="log_utility",
        max_weight_per_asset=0.15,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    stock_mask = None
    if restrict_to_assets:
        stock_mask = _build_stock_mask(tickers[:action_dim], restrict_to_assets, device)
        n_allowed = int(stock_mask.sum().item())
        print(f"stock_mask: {n_allowed}/{action_dim} assets allowed")

    agent = MemoryAugmentedSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lstm_hidden_size=64,
        num_lstm_layers=1,
        hidden_dims=[256, 256],
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_capacity=200000,
        auto_tune_alpha=False,
        alpha=0.02,
        gamma=0.995,
        tau=0.005,
        use_graph=False,
        stock_mask=stock_mask,
    )

    effective_name = f"{portfolio_name}{model_suffix}"
    ckpt_dir = os.path.join(BASE_DIR, "models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(BASE_DIR, "models", f"lstm_sac_{effective_name}.pth")
    best_path = os.path.join(BASE_DIR, "models", f"lstm_sac_{effective_name}_best.pth")
    print(f"Save base: {final_path}")

    rms = RunningMeanStd()
    state, _ = env.reset(seed=seed)
    hidden_state = None
    random_steps = 1000
    best_val_sharpe = -np.inf

    for step in tqdm(range(total_timesteps), desc="LSTM-SAC train"):
        if step < random_steps:
            action = env.action_space.sample()
            hidden_state = None
        else:
            action, hidden_state = agent.select_action(state, hidden_state, deterministic=False)

        next_state, reward, term, trunc, _ = env.step(action)
        rms.update(reward)
        norm_reward = float(np.clip(rms.normalize(reward), -10, 10))
        agent.replay_buffer.push(state, action, norm_reward, next_state, float(term or trunc))
        state = next_state

        if step >= random_steps and len(agent.replay_buffer) >= agent.batch_size:
            agent.update()

        if term or trunc:
            state, _ = env.reset()
            hidden_state = None

        if (step + 1) % save_freq == 0:
            ck = os.path.join(ckpt_dir, f"lstm_sac_{effective_name}_step_{step+1}.pth")
            agent.save(ck)

            try:
                vo, _ = env_val.reset(seed=seed)
                vh, vd = None, False
                vvals = [100000.0]
                while not vd:
                    va, vh = agent.select_action(vo, vh, deterministic=True)
                    vo, _, vt, vtr, vi = env_val.step(va)
                    vd = vt or vtr
                    vvals.append(vi.get("portfolio_value", vvals[-1]))
                vrets = pd.Series(vvals).pct_change().dropna()
                vsh = float((vrets.mean() / vrets.std()) * np.sqrt(252)) if vrets.std() > 1e-9 else 0.0
                vroi = (vvals[-1] - vvals[0]) / vvals[0] * 100
                print(f"  step {step+1}: ValSharpe={vsh:.4f} ValROI={vroi:.2f}%")
                if vsh > best_val_sharpe:
                    best_val_sharpe = vsh
                    agent.save(best_path)
                    print(f"  * best -> {best_path}")
            except Exception as e:
                print(f"  validation error at step {step+1}: {e}")

    agent.save(final_path)
    print(f"Final checkpoint -> {final_path}")
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM-SAC training (no graph) with stock_mask")
    parser.add_argument("--portfolio_name", type=str, default="28stocks")
    parser.add_argument("--restrict_to_assets", type=str, default="",
                        help="Comma-separated ticker symbols defining the allocation universe")
    parser.add_argument("--model_suffix", type=str, default="")
    parser.add_argument("--total_timesteps", type=int, default=200000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_freq", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    restrict = [t.strip() for t in args.restrict_to_assets.split(",") if t.strip()] or None
    train_lstm_sac_imputed(
        portfolio_name=args.portfolio_name,
        restrict_to_assets=restrict,
        model_suffix=args.model_suffix,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        save_freq=args.save_freq,
        seed=args.seed,
    )
