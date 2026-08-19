# Event-Driven Graph Reinforcement Learning for Risk-Aware Portfolio Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-140%20passing-brightgreen.svg)](#testing)

This repository is the source code for the paper *"Event-Driven Graph Reinforcement Learning for Risk-Aware Portfolio Optimization"*. It contains the implementation of the proposed GNN-SAC agent, the FinBERT-MRC event extraction pipeline, the heterogeneous dynamic graph environment, and the regression-tested driver `main.py` that reproduces every table and figure reported in the paper.

The study uses a **28-stock U.S. large-cap universe** across 7 sectors over the period January 2002 to April 2022 (~5,000 trading days). The held-out benchmark window (November 4, 2013 -- March 14, 2014, 90 trading days) follows the evaluation protocol of Soleymani & Paquet (2021). A scalability experiment extends the framework to an **S&P 50 point-in-time** universe.

---

## Quick Start & Installation

Tested on Python 3.10 and 3.12.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# For development tools (linting, experiment trackers, dashboard):
# pip install -r requirements-dev.txt
```

PyTorch Geometric wheels (`torch_geometric`, `torch_scatter`, `torch_sparse`) are CUDA-version-specific. If installation fails, match your CUDA version using the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

*(Note: On systems with PEP 668 restrictions, use `uv pip install -r requirements.txt`)*

```bash
# 3. Set up external API keys (required for data preparation only)
export FRED_API_KEY="your_fred_api_key"   # Free key from https://fred.stlouisfed.org/docs/api/api_key.html

# 4. Run the test suite to verify your environment
python -m pytest tests/ -q
```

---

## Reproducing the Reported Results: End-to-End Pipeline

### Step 1: Data Preparation

The primary manuscript results use a 28-stock portfolio universe. Run the dataset preparation script, which downloads raw OHLCV data from Yahoo Finance, computes technical features, and exports the final feature matrix.

```bash
python scripts/data_prep/prepare_28stocks_dataset.py
```
*Output: `data/processed/28stocks_dataset.csv`*

### Step 2: FinBERT-MRC Event Extraction (Optional -- required for event-driven mode)

The three-stage NLP pipeline extracts structured event tuples from financial headlines: FinBERT sentiment, BART-MNLI event classification, and FinancialBERT-SQuAD2 entity extraction.

```bash
# 2a. Generate synthetic MRC training data
python scripts/data_prep/generate_mrc_dataset.py

# 2b. Fine-tune FinBERT-MRC
python scripts/training/train_finbert_mrc.py --epochs 3

# 2c. Ingest and process financial news
python src/data_pipeline/news_ingester.py --portfolio 28stocks
python src/data_pipeline/news_processor.py --portfolio 28stocks --force
```
*Output: `models/finbert_mrc_custom/`, `data/processed/news_with_events_28stocks.csv`*

### Step 3: Model Training

With the dataset generated, you can train the four reinforcement learning agents evaluated in the manuscript. Each command takes roughly 2 to 6 hours on a single mid-range GPU (longer on CPU) and writes the expected `.pth` checkpoint to the `models/` directory.

```bash
# 1. Event-Driven GNN-SAC (proposed) -- manuscript headline agent
python scripts/training/train_gnn_sac.py \
    --portfolio_name 28stocks \
    --total_timesteps 150000 \
    --save_freq 10000 \
    --use_nextgen \
    --use_attention_gating \
    --use_momentum_masking \
    --use_cvar_loss \
    --cvar_anneal_start 50000 \
    --cvar_anneal_end 100000 \
    --cvar_multiplier 1.0 \
    --alpha_init 0.02 \
    --batch_size 256 \
    --learning_rate 5e-5

# 2. Static-Graph GNN-SAC (ablation, lambda_mix=0)
python scripts/training/train_gnn_sac.py \
    --portfolio_name 28stocks \
    --total_timesteps 150000 \
    --use_nextgen \
    --use_attention_gating \
    --use_momentum_masking \
    --use_cvar_loss \
    --disable_event_channel

# 3. GraphSAGE-PPO (Sun et al. 2024 replication)
python scripts/training/train_graphsage_ppo.py \
    --portfolio_name 28stocks \
    --total_timesteps 150000

# 4. LSTM-SAC (memory-augmented baseline)
python scripts/training/train_lstm_sac_imputed.py \
    --portfolio_name 28stocks \
    --total_timesteps 150000
```

The canonical checkpoint is **step 40,000** (during the return-seeking phase, before the CVaR curriculum activates at step 50k). Once the curriculum ramps up, the CVaR penalty pushes the agent into an overly defensive local optimum.

### Step 4: Result Reproduction (`main.py`)

Run `main.py` to evaluate the trained agents against classical baselines. The script provides both an interactive menu and non-interactive CLI commands.

```bash
# Interactive mode
python main.py

# Non-interactive: reproduce benchmark results (Table 5)
python main.py --mode load --model 1 --reproduce benchmark

# Non-interactive: reproduce crisis stress test (Table 8)
python main.py --mode load --model 1 --reproduce crisis

# Non-interactive: reproduce XAI analysis (Table 9)
python main.py --mode load --model 1 --reproduce xai
```

| Table | Contents | Source |
|---|---|---|
| **Table 5** | Headline comparison: GNN-SAC, GraphSAGE-PPO, LSTM-SAC, Buy-and-Hold | Live evaluation of trained checkpoints |
| **Table 7** | Architecture ablation ladder (EXP-0 through EXP-5) | Ablation study scripts |
| **Table 8** | Crisis-period performance (COVID-19, China crash, Volpocalypse) | Crisis stress test script |
| **Table 9** | XAI node importance and attention weights | SHAP attribution analysis |

*Checkpoints are not shipped with this repository. Run the training scripts (Step 3) first, then use `main.py` to evaluate. If a checkpoint is missing, `main.py` reports the expected file path.*

#### Regenerating Individual Evaluation Results

```bash
# Benchmark evaluation (Section 5.2 / Figure 6.2)
PYTHONPATH=. python scripts/evaluation/evaluate_gnn_sac.py \
    --portfolio_name 28stocks \
    --benchmark_28stocks \
    --use_nextgen \
    --use_attention_gating \
    --use_momentum_masking

# Classical baselines (Mean-Variance, Risk Parity, Merton, Buy-and-Hold)
PYTHONPATH=. python scripts/evaluation/run_baseline_backtests.py --portfolio_name 28stocks

# Ablation study (6-tier architecture ladder)
python scripts/analysis/run_all_ablations.py --portfolio_name 28stocks --steps 200000

# Crisis stress test (COVID-19 crash window)
PYTHONPATH=. python scripts/analysis/run_crisis_evaluation_gnn.py --portfolio_name 28stocks

# Hyperparameter sensitivity sweeps (delta and xi)
PYTHONPATH=. python scripts/analysis/run_sensitivity.py --sweep both --steps 10000
```

---

## Data

The repository provides scripts for two levels of data reproducibility:

### 1. 28-Stock Universe (Tables 5--9)
Generated via `python scripts/data_prep/prepare_28stocks_dataset.py` (as shown in Step 1).

### 2. S&P 50 Point-in-Time Scalability Experiment (Section 6.6)
To regenerate the 50-asset point-in-time dataset used in Section 6.6, run:

```bash
python scripts/data_prep/prepare_sp50_pit2013.py
```
*Output: `data/processed/sp50_pit2013_dataset.csv`*

### 28-Stock Portfolio Details

| Property | Value |
|----------|-------|
| Assets | 28 U.S. large-cap stocks across 7 sectors |
| History | Jan 2002 -- Apr 2022 (~5,000 trading days) |
| Node features | 20 dimensions (economic + technical + price lags) |
| Train period | Up to Dec 5, 2012 |
| Validation | Jan 1, 2013 -- Oct 31, 2013 |
| Benchmark | Nov 4, 2013 -- Mar 14, 2014 (90 days) |

**Constituent stocks:** AAPL, AMZN, BA, BAC, BP, CAT, CSCO, CVX, ENB, GE, GILD, HD, HON, IBM, INTC, JNJ, JPM, KO, MFC, MMM, MRK, MSFT, ORCL, PFE, SHEL, TD, VZ, WMT.

**Data sources:**
- Price data: Yahoo Finance (downloaded by data prep scripts)
- Financial news: [Kaggle Daily Financial News Dataset](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests)
- Macroeconomic indicators: FRED (Federal Reserve Economic Data)

---

## Benchmark Results

Evaluated on the 90-day window (Nov 4, 2013 -- Mar 14, 2014) at step-40k checkpoint:

| Model | ROI | Sharpe | Sortino | Calmar | Max DD | Turnover |
|-------|-----|--------|---------|--------|--------|----------|
| Buy-and-Hold (Market) | 4.89% | 1.34 | 2.20 | 2.47 | -5.93% | 0.0000 |
| LSTM-SAC | 5.79% | 1.57 | 2.63 | 3.06 | -5.71% | 0.0123 |
| GraphSAGE-PPO (Replicated) | 4.79% | 1.31 | 2.15 | 2.41 | -5.96% | 0.0199 |
| **Event-Driven GNN-SAC** | **8.05%** | **2.11** | **3.59** | **4.40** | **-5.60%** | 0.0230 |

---

---

## Configuration

`scripts/config.yaml` is the source of truth for training hyperparameters. Key settings:

```yaml
data:
  train_end: "2012-12-05"
  val_start: "2013-01-01"
  val_end: "2013-10-31"
  benchmark_start: "2013-11-04"
  benchmark_end: "2014-03-14"

training:
  max_steps: 150000
  batch_size: 256
  learning_rate: 0.00005
  gamma: 0.995
  tau: 0.005
  alpha: 0.02
```

---

## Repository Structure

```
main.py                 # CLI entry point (interactive + non-interactive)

src/
  agents/               # GNN-SAC, LSTM-SAC, GraphSAGE-PPO agents
  backtesting/          # Backtest engine and strategy adapters
  baselines/            # Mean-Variance, Merton, Buy-and-Hold, Risk Parity
  data_pipeline/        # News ingestion, FinBERT-MRC processing, graph builder
  environments/         # Portfolio MDP environment and feature extraction
  regime_detection/     # GMM/HMM regime classifiers
  optimization/         # Portfolio optimization utilities
  utils/                # Shared utilities
  visualization/        # Plotting helpers

scripts/
  training/             # Training: GNN-SAC, FinBERT-MRC, GraphSAGE-PPO, LSTM-SAC
  evaluation/           # Benchmark evaluation, baseline backtests
  analysis/             # Ablations, crisis stress tests, SHAP attribution
  data_prep/            # Dataset preparation (28-stock, S&P 50)
  visualization/        # Figure generation, XAI decision explanations

models/                 # Model definitions and FinBERT-MRC config (checkpoints generated by training)
results/                # Figures, tables, evaluation CSVs
papers/elsarticle/      # Manuscript LaTeX sources
data/                   # Processed datasets and regime labels
tests/                  # Unit and integration tests (12 test files)
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full text.

---

## Acknowledgements

Market data are provided by Yahoo Finance (via the `yfinance` package) and the Federal Reserve Economic Data service (FRED). Financial news data are sourced from the Kaggle Daily Financial News Dataset. The implementation builds on PyTorch, PyTorch Geometric, Gymnasium, and HuggingFace Transformers.


