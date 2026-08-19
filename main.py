"""
Event-Driven Graph Reinforcement Learning for Risk-Aware Portfolio Optimization
================================================================================
CLI entry point for model loading, inference, training, and data preparation.

Usage:
    python main.py                  # Interactive menu
    python main.py --mode load      # Load default (Event-Driven GNN-SAC) model
    python main.py --mode train     # Training pipeline menu
    python main.py --mode prepare   # Data preparation pipeline menu
"""

import argparse
import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ────────────────────────────────────────────────────────────────────
# Model Registry
# ────────────────────────────────────────────────────────────────────

MODELS = {
    "1": {
        "name": "Event-Driven GNN-SAC (Proposed)",
        "key": "gnn_sac_event",
        "description": (
            "Main proposed model: GNN-SAC with FinBERT-MRC event-driven "
            "heterogeneous graph, dual-head IQN critic, CVaR curriculum, "
            "adaptive attention gating, and momentum masking."
        ),
        "checkpoint_pattern": "models/gnn_sac_event_driven_{portfolio}_best.pth",
        "agent_module": "src.agents.gnn_sac_agent",
        "agent_class": "GNNSACAgent",
        "requires_graph": True,
        "train_script": "scripts/training/train_gnn_sac.py",
        "train_args": [
            "--use_nextgen",
            "--use_attention_gating",
            "--use_momentum_masking",
            "--use_cvar_loss",
            "--cvar_anneal_start", "50000",
            "--cvar_anneal_end", "100000",
        ],
    },
    "2": {
        "name": "Static-Graph GNN-SAC (Ablation)",
        "key": "gnn_sac_static",
        "description": (
            "GNN-SAC with static correlation graph only (lambda_mix=0). "
            "No event channel — isolates the contribution of the "
            "event-driven topology."
        ),
        "checkpoint_pattern": "models/gnn_sac_event_driven_{portfolio}_imputed9_static_best.pth",
        "agent_module": "src.agents.gnn_sac_agent",
        "agent_class": "GNNSACAgent",
        "requires_graph": True,
        "train_script": "scripts/training/train_gnn_sac.py",
        "train_args": [
            "--use_nextgen",
            "--use_attention_gating",
            "--use_momentum_masking",
            "--use_cvar_loss",
            "--disable_event_channel",
        ],
    },
    "3": {
        "name": "GraphSAGE-PPO (Baseline Replication)",
        "key": "graphsage_ppo",
        "description": (
            "Replicated GraphSAGE-PPO baseline (Sun et al., 2024) trained "
            "under the same data split, graph, and transaction-cost model "
            "as the proposed GNN-SAC agent."
        ),
        "checkpoint_pattern": "models/graphsage_ppo_event_driven_{portfolio}_best.pth",
        "agent_module": "src.agents.graphsage_ppo_agent",
        "agent_class": "GraphSAGEPPOAgent",
        "requires_graph": True,
        "train_script": "scripts/training/train_graphsage_ppo.py",
        "train_args": [],
    },
    "4": {
        "name": "LSTM-SAC (Memory-Augmented Baseline)",
        "key": "lstm_sac",
        "description": (
            "Memory-Augmented SAC with LSTM encoder — no graph structure. "
            "Tests whether temporal memory alone suffices without "
            "relational topology."
        ),
        "checkpoint_pattern": "models/lstm_sac_{portfolio}_best.pth",
        "agent_module": "src.agents.memory_augmented_sac_agent",
        "agent_class": "MemoryAugmentedSACAgent",
        "requires_graph": False,
        "train_script": "scripts/training/train_lstm_sac_imputed.py",
        "train_args": [],
    },
    "5": {
        "name": "FinBERT-MRC (NLP Event Extraction Pipeline)",
        "key": "finbert_mrc",
        "description": (
            "Three-stage NLP pipeline: FinBERT sentiment -> BART-MNLI "
            "event classification -> FinancialBERT-SQuAD2 entity extraction. "
            "Produces structured event tuples for graph rewiring."
        ),
        "checkpoint_pattern": "models/finbert_mrc_custom/",
        "agent_module": None,
        "agent_class": None,
        "requires_graph": False,
        "train_script": "scripts/training/train_finbert_mrc.py",
        "train_args": ["--epochs", "3"],
    },
}

DATA_PREP_STEPS = {
    "1": {
        "name": "Prepare 28-Stock Dataset",
        "description": "Download historical prices and generate base features for the 28-stock universe.",
        "script": "scripts/data_prep/prepare_28stocks_dataset.py",
        "args": [],
        "output": "data/processed/28stocks_dataset.csv",
    },
    "2": {
        "name": "Prepare S&P 50 Point-in-Time Dataset",
        "description": "Build the 50-asset point-in-time S&P 500 slice for scalability experiments.",
        "script": "scripts/data_prep/prepare_sp50_pit2013.py",
        "args": [],
        "output": "data/processed/sp50_pit2013_dataset.csv",
    },
    "3": {
        "name": "Generate FinBERT-MRC Training Data",
        "description": "Generate SQuAD-formatted synthetic QA data for MRC fine-tuning.",
        "script": "scripts/data_prep/generate_mrc_dataset.py",
        "args": [],
        "output": "data/finbert_mrc_train_synthetic.json",
    },
    "4": {
        "name": "Ingest News Headlines",
        "description": "Filter and align raw financial news headlines to the portfolio tickers.",
        "script": "src/data_pipeline/news_ingester.py",
        "args": ["--portfolio", "28stocks"],
        "output": "data/processed/filtered_news_28stocks.csv",
    },
    "5": {
        "name": "Process News with FinBERT-MRC",
        "description": "Run the three-stage NLP pipeline to extract event tuples from headlines.",
        "script": "src/data_pipeline/news_processor.py",
        "args": ["--portfolio", "28stocks", "--force"],
        "output": "data/processed/news_with_events_28stocks.csv",
    },
    "6": {
        "name": "Build Heterogeneous Graphs",
        "description": "Construct static + event-driven adjacency matrices and node features.",
        "script": "scripts/data_prep/build_graphs.py",
        "args": [],
        "output": "data/28stocks_data.npy",
    },
    "7": {
        "name": "Full Data Pipeline (Steps 1-6)",
        "description": "Run the complete data preparation sequence end-to-end.",
        "script": None,
        "args": [],
        "output": None,
    },
}

TRAIN_CONFIGS = {
    "1": {
        "name": "Full Training (150k steps, manuscript config)",
        "timesteps": 150000,
        "extra_args": ["--batch_size", "256", "--learning_rate", "5e-5", "--alpha_init", "0.02"],
    },
    "2": {
        "name": "Quick Training (20k steps, ablation/debug)",
        "timesteps": 20000,
        "extra_args": ["--batch_size", "256", "--learning_rate", "5e-5"],
    },
    "3": {
        "name": "Smoke Test (1k steps, verify pipeline)",
        "timesteps": 1000,
        "extra_args": [],
    },
}

PORTFOLIO_CHOICES = {
    "1": {"name": "28stocks", "label": "28-Stock Universe (Soleymani & Paquet, 2021)"},
    "2": {"name": "sp50_pit2013", "label": "S&P 50 Point-in-Time (Scalability Experiment)"},
}


# ────────────────────────────────────────────────────────────────────
# Display helpers
# ────────────────────────────────────────────────────────────────────

def print_banner():
    print()
    print("=" * 72)
    print("  Event-Driven Graph RL for Risk-Aware Portfolio Optimization")
    print("  GNN-SAC with FinBERT-MRC | Dual-Head IQN | CVaR Curriculum")
    print("=" * 72)
    print()


def print_menu(title, options, show_back=True):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    for key in sorted(options.keys(), key=lambda k: (not k.isdigit(), k)):
        item = options[key]
        name = item if isinstance(item, str) else item.get("name", item.get("label", key))
        print(f"  [{key}]  {name}")
    if show_back:
        print(f"  [0]  Back / Exit")
    print()


def prompt_choice(valid_keys, prompt_text="Select an option: "):
    while True:
        choice = input(prompt_text).strip()
        if choice in valid_keys or choice == "0":
            return choice
        print(f"  Invalid choice '{choice}'. Please try again.")


# ────────────────────────────────────────────────────────────────────
# Load model
# ────────────────────────────────────────────────────────────────────

def resolve_checkpoint(pattern, portfolio):
    path = os.path.join(BASE_DIR, pattern.format(portfolio=portfolio))
    if os.path.exists(path):
        return path
    return None


def _inspect_finbert_dir(model_dir):
    """List contents and key config of a HuggingFace model directory."""
    import json as _json
    files = os.listdir(model_dir) if os.path.isdir(model_dir) else []
    print(f"  Directory contents: {files}")
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = _json.load(f)
            print(f"  Architecture: {cfg.get('architectures', ['unknown'])}")
            print(f"  Hidden size: {cfg.get('hidden_size', '?')}")
            print(f"  Vocab size: {cfg.get('vocab_size', '?')}")
            print(f"  Model type: {cfg.get('model_type', '?')}")
        except Exception:
            pass
    bin_files = [f for f in files if f.endswith((".bin", ".safetensors"))]
    if bin_files:
        for bf in bin_files:
            size_mb = os.path.getsize(os.path.join(model_dir, bf)) / (1024 * 1024)
            print(f"  Weights: {bf} ({size_mb:.1f} MB)")
    print(f"  Loaded successfully.")


def load_model_interactive():
    print_menu("Select Model to Load", {k: v["name"] for k, v in MODELS.items()})
    print("  [C]  Load a custom model checkpoint")
    choice = prompt_choice(list(MODELS.keys()) + ["C", "c"])
    if choice == "0":
        return

    if choice.upper() == "C":
        load_custom_model()
        return

    model_info = MODELS[choice]
    print(f"\n  Model: {model_info['name']}")
    print(f"  {model_info['description']}\n")

    print_menu("Select Portfolio Universe", {k: v["label"] for k, v in PORTFOLIO_CHOICES.items()}, show_back=True)
    pchoice = prompt_choice(list(PORTFOLIO_CHOICES.keys()))
    if pchoice == "0":
        return
    portfolio = PORTFOLIO_CHOICES[pchoice]["name"]

    checkpoint = resolve_checkpoint(model_info["checkpoint_pattern"], portfolio)
    if checkpoint is None:
        print(f"\n  Checkpoint not found: {model_info['checkpoint_pattern'].format(portfolio=portfolio)}")
        print("  Train this model first, or provide a custom checkpoint path.")
        return

    print(f"\n  Loading checkpoint: {checkpoint}")

    if model_info["agent_module"] is None:
        print(f"  This is the NLP pipeline model (not a trading agent).")
        _inspect_finbert_dir(checkpoint)
        print(f"  Use it via: src/data_pipeline/news_processor.py")
        return

    try:
        import torch
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            keys_preview = list(state.keys())[:10]
            print(f"  Checkpoint loaded successfully.")
            print(f"  Keys: {keys_preview}{'...' if len(state.keys()) > 10 else ''}")
            for key_name, label in [
                ("actor_state_dict", "Actor"),
                ("critic1_state_dict", "Critic-1"),
                ("critic2_state_dict", "Critic-2"),
                ("policy_state_dict", "Policy"),
                ("policy", "Policy"),
            ]:
                if key_name in state:
                    params = sum(
                        p.numel() for p in
                        [v for v in state[key_name].values() if hasattr(v, "numel")]
                    )
                    print(f"  {label} parameters: {params:,}")
            if "step" in state:
                print(f"  Training step: {state['step']:,}")
            if "best_sharpe" in state:
                print(f"  Best Sharpe (at save): {state['best_sharpe']:.4f}")
        else:
            print(f"  Checkpoint loaded (type: {type(state).__name__}).")
    except Exception as e:
        print(f"  Error loading checkpoint: {e}")

    # Offer to reproduce results
    print(f"\n{'─' * 60}")
    print(f"  What would you like to do with this model?")
    print(f"{'─' * 60}")
    print(f"  [1]  Reproduce benchmark results (90-day evaluation window)")
    print(f"  [2]  Run crisis stress test (COVID-19 / China / Volpocalypse)")
    print(f"  [3]  Run XAI node importance analysis")
    print(f"  [4]  Just inspect (done)")
    print()
    action = prompt_choice(["1", "2", "3", "4"])

    if action == "0" or action == "4":
        return

    eval_args_base = ["--portfolio_name", portfolio]
    if model_info["key"] in ("gnn_sac_event", "gnn_sac_static"):
        eval_args_base += [
            "--use_nextgen",
            "--use_attention_gating",
            "--use_momentum_masking",
        ]

    if action == "1":
        print(f"\n  Reproducing benchmark results (Table 5 in manuscript)...")
        print(f"  Evaluation window: Nov 4, 2013 - Mar 14, 2014")
        print(f"  Checkpoint: {checkpoint}\n")
        eval_args = eval_args_base + ["--benchmark_28stocks"]
        run_subprocess("scripts/evaluation/evaluate_gnn_sac.py", eval_args)
        print(f"\n  Results saved to: results/gnn_sac_eval_details_{portfolio}.csv")
        print(f"  Wealth plot:      results/Figure_6.2_Wealth_{portfolio}.png")

    elif action == "2":
        print(f"\n  Running crisis stress test...")
        print(f"  Checkpoint: {checkpoint}\n")
        run_subprocess("scripts/analysis/run_crisis_evaluation_gnn.py", eval_args_base)
        print(f"\n  Results saved to: results/crisis_metrics_{portfolio}.csv")
        print(f"  Wealth curves:    results/Figure_7.2_Crisis_Wealth.png")

    elif action == "3":
        date = input("  XAI analysis date [2020-03-25]: ").strip() or "2020-03-25"
        print(f"\n  Running XAI perturbation analysis for {date}...")
        print(f"  Checkpoint: {checkpoint}\n")
        run_subprocess("scripts/visualization/explain_decision.py",
                       eval_args_base + ["--date", date])


def load_custom_model():
    print("\n  Enter the path to your custom model checkpoint (.pth file):")
    path = input("  Path: ").strip()

    if not path:
        print("  No path provided.")
        return

    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)

    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return

    try:
        import torch
        state = torch.load(path, map_location="cpu", weights_only=False)
        print(f"\n  Custom checkpoint loaded: {path}")
        if isinstance(state, dict):
            print(f"  Keys: {list(state.keys())[:15]}")
            for key in ["actor_state_dict", "critic1_state_dict", "policy_state_dict"]:
                if key in state:
                    params = sum(
                        p.numel() for p in
                        [v for v in state[key].values() if hasattr(v, "numel")]
                    )
                    print(f"  {key}: {params:,} parameters")
        else:
            print(f"  Type: {type(state).__name__}")
    except Exception as e:
        print(f"  Error loading checkpoint: {e}")


# ────────────────────────────────────────────────────────────────────
# Training pipeline
# ────────────────────────────────────────────────────────────────────

def run_subprocess(script, args, cwd=BASE_DIR):
    cmd = [sys.executable, os.path.join(BASE_DIR, script)] + args
    print(f"\n  Running: {' '.join(cmd)}\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = BASE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"\n  Process exited with code {result.returncode}")
    return result.returncode


def train_model_interactive():
    print_menu("Select Model to Train", {k: v["name"] for k, v in MODELS.items()})
    print("  [C]  Train with a custom CSV dataset")
    choice = prompt_choice(list(MODELS.keys()) + ["C", "c"])
    if choice == "0":
        return

    if choice.upper() == "C":
        train_custom_csv()
        return

    model_info = MODELS[choice]
    print(f"\n  Model: {model_info['name']}")
    print(f"  {model_info['description']}\n")

    if model_info["key"] == "finbert_mrc":
        print("  This will fine-tune the FinBERT-MRC event extraction model.")
        confirm = input("  Proceed? [y/N]: ").strip().lower()
        if confirm != "y":
            return
        run_subprocess(model_info["train_script"], model_info["train_args"])
        return

    print_menu("Select Portfolio Universe", {k: v["label"] for k, v in PORTFOLIO_CHOICES.items()}, show_back=True)
    pchoice = prompt_choice(list(PORTFOLIO_CHOICES.keys()))
    if pchoice == "0":
        return
    portfolio = PORTFOLIO_CHOICES[pchoice]["name"]

    print_menu("Select Training Duration", {k: v["name"] for k, v in TRAIN_CONFIGS.items()}, show_back=True)
    tchoice = prompt_choice(list(TRAIN_CONFIGS.keys()))
    if tchoice == "0":
        return
    train_cfg = TRAIN_CONFIGS[tchoice]

    args = [
        "--portfolio_name", portfolio,
        "--total_timesteps", str(train_cfg["timesteps"]),
        "--save_freq", "10000",
    ] + model_info["train_args"] + train_cfg["extra_args"]

    print(f"\n  Configuration:")
    print(f"    Model:      {model_info['name']}")
    print(f"    Portfolio:  {PORTFOLIO_CHOICES[pchoice]['label']}")
    print(f"    Duration:   {train_cfg['name']}")
    print(f"    Script:     {model_info['train_script']}")

    confirm = input("\n  Start training? [y/N]: ").strip().lower()
    if confirm != "y":
        return

    run_subprocess(model_info["train_script"], args)


def train_custom_csv():
    print("\n" + "=" * 60)
    print("  Custom CSV Training Pipeline")
    print("=" * 60)
    print()
    print("  This pipeline trains the Event-Driven GNN-SAC model on your")
    print("  own dataset. Your CSV must contain columns in the format:")
    print()
    print("    - Date index (parseable datetime)")
    print("    - price_TICKER columns (e.g., price_AAPL, price_MSFT)")
    print("    - VIX column (implied volatility index)")
    print()
    print("  Optional columns:")
    print("    - return_TICKER  (auto-computed if missing)")
    print("    - volume_TICKER  (enhances graph construction)")
    print()

    csv_path = input("  Path to price CSV: ").strip()
    if not csv_path:
        print("  No path provided.")
        return

    csv_path = os.path.expanduser(csv_path)
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(BASE_DIR, csv_path)

    if not os.path.exists(csv_path):
        print(f"  File not found: {csv_path}")
        return

    try:
        import pandas as pd
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        price_cols = [c for c in df.columns if c.startswith("price_")]
        tickers = [c.replace("price_", "") for c in price_cols]
        print(f"\n  Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Tickers found ({len(tickers)}): {', '.join(tickers[:15])}")
        if len(tickers) > 15:
            print(f"    ... and {len(tickers) - 15} more")

        if len(tickers) < 2:
            print("  Error: Need at least 2 price_TICKER columns.")
            return

        has_vix = "VIX" in df.columns
        if not has_vix:
            print("  Warning: No VIX column found. A synthetic VIX proxy will be computed.")
    except Exception as e:
        print(f"  Error reading CSV: {e}")
        return

    news_csv = None
    use_news = input("\n  Do you have a news CSV for event extraction? [y/N]: ").strip().lower()
    if use_news == "y":
        news_csv = input("  Path to news CSV: ").strip()
        if news_csv:
            news_csv = os.path.expanduser(news_csv)
            if not os.path.isabs(news_csv):
                news_csv = os.path.join(BASE_DIR, news_csv)
            if not os.path.exists(news_csv):
                print(f"  News file not found: {news_csv}. Proceeding without news.")
                news_csv = None
            else:
                print(f"  News CSV: {news_csv}")

    portfolio_name = input("\n  Portfolio name (used for output files) [custom]: ").strip() or "custom"

    print_menu("Select Training Duration", {k: v["name"] for k, v in TRAIN_CONFIGS.items()}, show_back=True)
    tchoice = prompt_choice(list(TRAIN_CONFIGS.keys()))
    if tchoice == "0":
        return
    train_cfg = TRAIN_CONFIGS[tchoice]

    # ── Step 1: Copy CSV to expected location ──
    dest = os.path.join(BASE_DIR, "data", "processed", f"{portfolio_name}_dataset.csv")
    print(f"\n  Step 1: Staging dataset -> {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    import shutil
    if os.path.abspath(csv_path) != os.path.abspath(dest):
        shutil.copy2(csv_path, dest)
        print(f"    Copied.")
    else:
        print(f"    Already in place.")

    # ── Step 2: Generate tickers file ──
    tickers_file = os.path.join(BASE_DIR, "data", f"{portfolio_name}_tickers.txt")
    with open(tickers_file, "w") as f:
        f.write(",".join(tickers))
    print(f"  Step 2: Wrote tickers file -> {tickers_file}")

    # ── Step 3: Ensure return columns exist ──
    import pandas as pd
    df = pd.read_csv(dest, index_col=0, parse_dates=True)
    added_returns = False
    for t in tickers:
        rcol = f"return_{t}"
        pcol = f"price_{t}"
        if rcol not in df.columns and pcol in df.columns:
            df[rcol] = df[pcol].pct_change().fillna(0)
            added_returns = True
    if not has_vix:
        returns = df[[f"return_{t}" for t in tickers if f"return_{t}" in df.columns]]
        df["VIX"] = returns.std(axis=1).rolling(20, min_periods=1).mean() * 100 * (252 ** 0.5)
        df["VIX"] = df["VIX"].fillna(15.0)
        print("  Step 3: Synthesized VIX proxy from return volatility.")

    if added_returns:
        df.to_csv(dest)
        print("  Step 3: Added missing return columns.")
    else:
        print("  Step 3: Return columns already present.")

    # ── Step 4: Build graph features ──
    print("  Step 4: Building node features and graph structure...")
    run_subprocess("scripts/data_prep/build_graphs.py", [])

    node_feat_path = os.path.join(BASE_DIR, "data", f"{portfolio_name}_data.npy")
    if not os.path.exists(node_feat_path):
        print(f"    Graph builder did not produce {node_feat_path}.")
        print(f"    Generating minimal node features from price data...")
        import numpy as np
        n_days = len(df)
        n_assets = len(tickers)
        n_sectors = 7
        n_macro = 1
        n_events = 5
        n_nodes = n_assets + n_sectors + n_macro + n_events
        n_feats = 8
        features = np.zeros((n_days, n_nodes, n_feats), dtype=np.float32)
        for i, t in enumerate(tickers):
            pcol = f"price_{t}"
            rcol = f"return_{t}"
            if pcol in df.columns:
                prices = df[pcol].values
                features[:, i, 0] = prices / (prices[0] + 1e-8)
            if rcol in df.columns:
                features[:, i, 1] = df[rcol].values
        if "VIX" in df.columns:
            vix_vals = df["VIX"].values
            features[:, n_assets + n_sectors, 0] = vix_vals / (vix_vals.mean() + 1e-8)
        np.save(node_feat_path, features)
        print(f"    Saved node features: {features.shape}")

    # ── Step 5: Handle news data ──
    if news_csv:
        news_dest = os.path.join(BASE_DIR, "data", "processed", f"news_with_events_{portfolio_name}.csv")
        if not os.path.exists(news_dest):
            print("  Step 5: Processing news through FinBERT-MRC pipeline...")
            filtered_dest = os.path.join(BASE_DIR, "data", "processed", f"filtered_news_{portfolio_name}.csv")
            shutil.copy2(news_csv, filtered_dest)
            run_subprocess("src/data_pipeline/news_processor.py", [
                "--portfolio", portfolio_name, "--force"
            ])
        else:
            print("  Step 5: News events file already exists.")
    else:
        print("  Step 5: No news data — training with static graph only (lambda_mix=0).")

    # ── Step 6: Train ──
    print(f"\n  Step 6: Training Event-Driven GNN-SAC on '{portfolio_name}'...")
    train_args = [
        "--portfolio_name", portfolio_name,
        "--total_timesteps", str(train_cfg["timesteps"]),
        "--save_freq", "10000",
        "--use_nextgen",
        "--use_attention_gating",
        "--use_momentum_masking",
        "--use_cvar_loss",
        "--cvar_anneal_start", "50000",
        "--cvar_anneal_end", "100000",
        "--batch_size", "256",
        "--learning_rate", "5e-5",
        "--alpha_init", "0.02",
    ]
    if not news_csv:
        train_args.append("--disable_event_channel")

    confirm = input("\n  Start training? [y/N]: ").strip().lower()
    if confirm != "y":
        return

    run_subprocess("scripts/training/train_gnn_sac.py", train_args)


# ────────────────────────────────────────────────────────────────────
# Data preparation pipeline
# ────────────────────────────────────────────────────────────────────

def prepare_data_interactive():
    print_menu("Data Preparation Pipeline", {k: v["name"] for k, v in DATA_PREP_STEPS.items()})
    print("  [C]  Prepare from a custom CSV")
    choice = prompt_choice(list(DATA_PREP_STEPS.keys()) + ["C", "c"])
    if choice == "0":
        return

    if choice.upper() == "C":
        prepare_custom_csv()
        return

    step = DATA_PREP_STEPS[choice]
    print(f"\n  {step['name']}")
    print(f"  {step['description']}")
    if step["output"]:
        exists = os.path.exists(os.path.join(BASE_DIR, step["output"]))
        status = "exists" if exists else "not yet generated"
        print(f"  Output: {step['output']} ({status})")

    if choice == "7":
        confirm = input("\n  Run full pipeline (steps 1-6)? [y/N]: ").strip().lower()
        if confirm != "y":
            return
        for k in ["1", "3", "4", "5", "6"]:
            s = DATA_PREP_STEPS[k]
            print(f"\n{'─' * 40}")
            print(f"  Running: {s['name']}")
            print(f"{'─' * 40}")
            rc = run_subprocess(s["script"], s["args"])
            if rc != 0:
                print(f"\n  Step {k} failed. Stopping pipeline.")
                return
        print("\n  Full data pipeline completed.")
        return

    confirm = input("\n  Run this step? [y/N]: ").strip().lower()
    if confirm != "y":
        return

    run_subprocess(step["script"], step["args"])


def prepare_custom_csv():
    print("\n" + "=" * 60)
    print("  Custom CSV Data Preparation")
    print("=" * 60)
    print()
    print("  Provide a CSV file with daily OHLCV or at minimum Close prices.")
    print("  The script will generate the price_TICKER / return_TICKER format")
    print("  required by the training pipeline.")
    print()
    print("  Accepted formats:")
    print("    A) Multi-ticker CSV: Date index, columns like AAPL_Close, MSFT_Close ...")
    print("    B) Already formatted: Date index, price_AAPL, price_MSFT ...")
    print("    C) yfinance-style MultiIndex CSV")
    print()

    csv_path = input("  Path to CSV: ").strip()
    if not csv_path:
        return

    csv_path = os.path.expanduser(csv_path)
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(BASE_DIR, csv_path)

    if not os.path.exists(csv_path):
        print(f"  File not found: {csv_path}")
        return

    try:
        import pandas as pd
        import numpy as np

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"\n  Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Columns (first 20): {list(df.columns[:20])}")

        price_cols = [c for c in df.columns if c.startswith("price_")]
        if price_cols:
            print(f"\n  Already in pipeline format ({len(price_cols)} tickers detected).")
            portfolio_name = input("  Portfolio name [custom]: ").strip() or "custom"
            dest = os.path.join(BASE_DIR, "data", "processed", f"{portfolio_name}_dataset.csv")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            import shutil
            shutil.copy2(csv_path, dest)
            tickers = [c.replace("price_", "") for c in price_cols]
            tickers_file = os.path.join(BASE_DIR, "data", f"{portfolio_name}_tickers.txt")
            with open(tickers_file, "w") as f:
                f.write(",".join(tickers))
            print(f"  Saved to: {dest}")
            print(f"  Tickers: {tickers_file}")
            return

        close_cols = [c for c in df.columns if "close" in c.lower() or "adj" in c.lower()]
        if not close_cols:
            close_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.float32, np.int64]]
            if len(close_cols) > 50:
                close_cols = close_cols[:50]

        if not close_cols:
            print("  Could not identify price columns. Please reformat your CSV.")
            return

        print(f"\n  Detected {len(close_cols)} potential price columns:")
        for c in close_cols[:10]:
            print(f"    {c}")
        if len(close_cols) > 10:
            print(f"    ... and {len(close_cols) - 10} more")

        confirm = input("\n  Convert these columns to pipeline format? [y/N]: ").strip().lower()
        if confirm != "y":
            return

        portfolio_name = input("  Portfolio name [custom]: ").strip() or "custom"

        out = pd.DataFrame(index=df.index)
        tickers = []
        for col in close_cols:
            parts = col.replace("_Close", "").replace("_close", "").replace("_Adj Close", "")
            parts = parts.replace("_adj_close", "").replace(" ", "_").upper()
            ticker = parts.strip("_")
            if not ticker:
                ticker = col.upper().replace(" ", "_")
            tickers.append(ticker)
            out[f"price_{ticker}"] = df[col]
            out[f"return_{ticker}"] = df[col].pct_change().fillna(0)

        returns = out[[c for c in out.columns if c.startswith("return_")]]
        out["VIX"] = returns.std(axis=1).rolling(20, min_periods=1).mean() * 100 * (252 ** 0.5)
        out["VIX"] = out["VIX"].fillna(15.0)

        out.dropna(subset=[f"price_{tickers[0]}"], inplace=True)

        dest = os.path.join(BASE_DIR, "data", "processed", f"{portfolio_name}_dataset.csv")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        out.to_csv(dest)

        tickers_file = os.path.join(BASE_DIR, "data", f"{portfolio_name}_tickers.txt")
        with open(tickers_file, "w") as f:
            f.write(",".join(tickers))

        print(f"\n  Conversion complete:")
        print(f"    Dataset:  {dest} ({out.shape[0]} rows, {len(tickers)} assets)")
        print(f"    Tickers:  {tickers_file}")
        print(f"\n  Next: select 'Train Model' from the main menu to train on this dataset.")

    except Exception as e:
        print(f"  Error processing CSV: {e}")


# ────────────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────────────

def evaluate_interactive():
    print_menu("Evaluation & Analysis", {
        "1": "Benchmark Evaluation (90-day window)",
        "2": "Crisis Stress Test (COVID-19 / China / Volpocalypse)",
        "3": "Run Traditional Baselines (Mean-Var, Risk Parity, etc.)",
        "4": "XAI Node Importance Analysis",
        "5": "Hyperparameter Sensitivity Sweep",
    })
    choice = prompt_choice(["1", "2", "3", "4", "5"])
    if choice == "0":
        return

    print_menu("Select Portfolio Universe", {k: v["label"] for k, v in PORTFOLIO_CHOICES.items()}, show_back=True)
    pchoice = prompt_choice(list(PORTFOLIO_CHOICES.keys()))
    if pchoice == "0":
        return
    portfolio = PORTFOLIO_CHOICES[pchoice]["name"]

    if choice == "1":
        run_subprocess("scripts/evaluation/evaluate_gnn_sac.py", [
            "--portfolio_name", portfolio,
            "--benchmark_28stocks",
            "--use_nextgen",
            "--use_attention_gating",
            "--use_momentum_masking",
        ])
    elif choice == "2":
        run_subprocess("scripts/analysis/run_crisis_evaluation_gnn.py", [
            "--portfolio_name", portfolio,
        ])
    elif choice == "3":
        run_subprocess("scripts/evaluation/run_baseline_backtests.py", [])
    elif choice == "4":
        run_subprocess("scripts/visualization/explain_decision.py", [
            "--portfolio_name", portfolio,
            "--date", "2020-03-25",
        ])
    elif choice == "5":
        run_subprocess("scripts/analysis/run_sensitivity.py", [
            "--sweep", "both",
            "--steps", "10000",
        ])


# ────────────────────────────────────────────────────────────────────
# Main menu
# ────────────────────────────────────────────────────────────────────

MAIN_MENU = {
    "1": "Load Model (inspect a trained checkpoint)",
    "2": "Train Model (select model + dataset + duration)",
    "3": "Data Preparation (download, preprocess, build graphs)",
    "4": "Evaluate & Analyze (benchmark, stress test, XAI)",
}


def main_interactive():
    print_banner()
    while True:
        print_menu("Main Menu", MAIN_MENU, show_back=False)
        print("  [0]  Exit")
        choice = prompt_choice(list(MAIN_MENU.keys()))
        if choice == "0":
            print("\n  Goodbye.\n")
            break
        elif choice == "1":
            load_model_interactive()
        elif choice == "2":
            train_model_interactive()
        elif choice == "3":
            prepare_data_interactive()
        elif choice == "4":
            evaluate_interactive()


# ────────────────────────────────────────────────────────────────────
# CLI argument parsing for non-interactive use
# ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Event-Driven GNN-SAC Portfolio Optimization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                       Interactive menu
  python main.py --mode load                           Load default model (interactive)
  python main.py --mode load --model 1 --reproduce benchmark
                                                       Reproduce Table 5 benchmark results
  python main.py --mode load --model 1 --reproduce crisis
                                                       Run crisis stress test (Table 8)
  python main.py --mode load --model 3                 Load GraphSAGE-PPO checkpoint
  python main.py --mode train                          Training pipeline
  python main.py --mode prepare                        Data preparation
  python main.py --mode evaluate                       Evaluation & analysis
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["load", "train", "prepare", "evaluate", "interactive"],
        default="interactive",
        help="Operation mode (default: interactive menu)",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Model number: 1=GNN-SAC, 2=Static-Graph, 3=GraphSAGE-PPO, 4=LSTM-SAC, 5=FinBERT-MRC",
    )
    parser.add_argument(
        "--portfolio",
        choices=[v["name"] for v in PORTFOLIO_CHOICES.values()],
        help="Portfolio universe name",
    )
    parser.add_argument(
        "--reproduce",
        choices=["benchmark", "crisis", "xai"],
        help="Reproduce manuscript results: benchmark (Table 5), crisis (Table 8), or xai (Table 9)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "interactive":
        main_interactive()
    elif args.mode == "load":
        if args.model:
            model_info = MODELS[args.model]
            portfolio = args.portfolio or "28stocks"
            checkpoint = resolve_checkpoint(model_info["checkpoint_pattern"], portfolio)
            print(f"\n  Model: {model_info['name']}")
            print(f"  {model_info['description']}")
            print(f"  Portfolio: {portfolio}")
            if checkpoint:
                print(f"  Checkpoint: {checkpoint}")

                if model_info["agent_module"] is None:
                    # FinBERT-MRC is a HuggingFace model directory, not a .pth
                    print(f"  This is the NLP pipeline model (not a trading agent).")
                    _inspect_finbert_dir(checkpoint)
                else:
                    try:
                        import torch
                        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                        if isinstance(state, dict):
                            print(f"  Keys: {list(state.keys())[:10]}")
                            for key_name, label in [
                                ("actor_state_dict", "Actor"),
                                ("critic1_state_dict", "Critic-1"),
                                ("policy_state_dict", "Policy"),
                                ("policy", "Policy"),
                            ]:
                                if key_name in state:
                                    params = sum(
                                        p.numel() for p in
                                        [v for v in state[key_name].values() if hasattr(v, "numel")]
                                    )
                                    print(f"  {label} parameters: {params:,}")
                            if "step" in state:
                                print(f"  Training step: {state['step']:,}")
                            if "best_sharpe" in state:
                                print(f"  Best Sharpe (at save): {state['best_sharpe']:.4f}")
                        print("  Loaded successfully.")
                    except Exception as e:
                        print(f"  Error: {e}")

                if args.reproduce:
                    eval_args = ["--portfolio_name", portfolio]
                    if model_info["key"] in ("gnn_sac_event", "gnn_sac_static"):
                        eval_args += [
                            "--use_nextgen",
                            "--use_attention_gating",
                            "--use_momentum_masking",
                        ]

                    if args.reproduce == "benchmark":
                        print(f"\n  Reproducing benchmark results...")
                        print(f"  Window: Nov 4, 2013 - Mar 14, 2014\n")
                        run_subprocess("scripts/evaluation/evaluate_gnn_sac.py",
                                       eval_args + ["--benchmark_28stocks"])
                    elif args.reproduce == "crisis":
                        print(f"\n  Running crisis stress test...\n")
                        run_subprocess("scripts/analysis/run_crisis_evaluation_gnn.py", eval_args)
                    elif args.reproduce == "xai":
                        print(f"\n  Running XAI node importance (March 25, 2020)...\n")
                        run_subprocess("scripts/visualization/explain_decision.py",
                                       eval_args + ["--date", "2020-03-25"])
            else:
                print(f"  Checkpoint not found: {model_info['checkpoint_pattern'].format(portfolio=portfolio)}")
        else:
            load_model_interactive()
    elif args.mode == "train":
        train_model_interactive()
    elif args.mode == "prepare":
        prepare_data_interactive()
    elif args.mode == "evaluate":
        evaluate_interactive()


if __name__ == "__main__":
    main()
