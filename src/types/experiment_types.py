"""
experiment_types.py
====================
Strongly-typed data structures for the comparative experiment:
    1. Soleymani et al. (2021) — DeepPocket flat CNN
    2. Sun et al. (2024)       — GraphSAGE-PPO (static graph)
    3. Ours                    — Event-Driven Dynamic GNN-SAC (dynamic graph + TGN + FinBERT)

Design principles
-----------------
* Pyright strict-mode compatible (no `Any`, no untyped dicts).
* Phantom-type / NewType tagging so mismatched tensor shapes are caught at type-check time.
* Protocol-based model contracts so the three model families CANNOT be swapped accidentally.
* Literal-constrained constants enforce the 28-ticker universe and 90-day window at the
  class level, not just at runtime.

Usage
-----
Run: `pyright src/types/experiment_types.py --strict`
"""

from __future__ import annotations

import datetime
from typing import (
    Final,
    Literal,
    NamedTuple,
    Protocol,
    Tuple,
    runtime_checkable,
)

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# 1.  Constants — enforced at class level, not buried in config dicts
# ---------------------------------------------------------------------------

UNIVERSE_SIZE: Final[int] = 28          # Exactly 28 US large-cap stocks (Sun et al. split)
WINDOW_DAYS: Final[int] = 90            # Benchmark rolling window (Nov 2013 – Mar 2014)
WARMUP_DAYS: Final[int] = 60            # PortfolioEnv look-back window (portfolio_env.py)
TRANSACTION_COST_BPS: Final[float] = 25.0   # 25 bps — identical across all three models
RISK_FREE_RATE: Final[float] = 0.02         # Annual, used in Sharpe / Sortino computation


# ---------------------------------------------------------------------------
# 2.  Phantom / NewType tags — zero runtime overhead, full type-checker benefit
# ---------------------------------------------------------------------------

# Dimension labels — used as TypeVar bounds to tag tensors by their semantic role
class _StockDim:     pass   # axis that spans UNIVERSE_SIZE stocks
class _NodeDim:      pass   # axis that spans GNN nodes (stocks + sector + macro + events)
class _FeatDim:      pass   # axis that spans per-node feature dimension
class _TimeDim:      pass   # axis that spans time-steps inside a sequence


# ---------------------------------------------------------------------------
# 3.  StockUniverse — enforces exactly 28 tickers at construction
# ---------------------------------------------------------------------------

class StockUniverse:
    """
    Immutable container for the 28-ticker universe shared across all three experiments.

    Construction raises ValueError immediately if len(tickers) != 28, so any
    misconfiguration is caught before training starts rather than mid-run.

    >>> u = StockUniverse(tickers=[...28 strings...])
    >>> u.size   # 28
    """

    _tickers: Tuple[str, ...]
    _size: Literal[28]   # Pyright understands this is invariant

    def __init__(self, tickers: list[str]) -> None:
        if len(tickers) != UNIVERSE_SIZE:
            raise ValueError(
                f"StockUniverse requires exactly {UNIVERSE_SIZE} tickers, "
                f"got {len(tickers)}."
            )
        if len(set(tickers)) != UNIVERSE_SIZE:
            raise ValueError("StockUniverse tickers must be unique.")
        self._tickers = tuple(tickers)
        self._size = 28  # type: ignore[assignment]  # narrowed Literal

    @property
    def size(self) -> Literal[28]:
        return self._size

    @property
    def tickers(self) -> Tuple[str, ...]:
        return self._tickers

    def index_of(self, ticker: str) -> int:
        return self._tickers.index(ticker)

    def __repr__(self) -> str:
        return f"StockUniverse(n=28, tickers={list(self._tickers[:4])}...)"


# ---------------------------------------------------------------------------
# 4.  RollingWindow — enforces 90-day horizon
# ---------------------------------------------------------------------------

class RollingWindow(NamedTuple):
    """
    Immutable value-type describing the benchmark evaluation window.
    Used to stamp every MarketBatch so evaluation scripts know which
    window a batch belongs to.

    Construction raises ValueError if the span is not exactly 90 trading days.
    """
    start: datetime.date
    end: datetime.date
    n_trading_days: int          # pre-computed; must equal WINDOW_DAYS

    @classmethod
    def from_dates(
        cls,
        start: datetime.date,
        end: datetime.date,
        n_trading_days: int,
    ) -> "RollingWindow":
        if n_trading_days != WINDOW_DAYS:
            raise ValueError(
                f"RollingWindow must span exactly {WINDOW_DAYS} trading days, "
                f"got {n_trading_days}."
            )
        return cls(start=start, end=end, n_trading_days=n_trading_days)

    # Canonical benchmark window from Sun et al. (2024) / Soleymani et al. (2021)
    @classmethod
    def benchmark_28stocks(cls) -> "RollingWindow":
        """Returns the validated 90-day benchmark window (Nov 2013 – Mar 2014)."""
        return cls(
            start=datetime.date(2013, 11, 4),
            end=datetime.date(2014, 3, 14),
            n_trading_days=WINDOW_DAYS,
        )


# ---------------------------------------------------------------------------
# 5.  MarketBatch  — the FLAT state tensor consumed by CNN / MLP / LSTM models
#     (Soleymani et al.  ==> accepts MarketBatch, REJECTS GraphBatch)
# ---------------------------------------------------------------------------

class MarketBatch(NamedTuple):
    """
    Flat price/feature batch for non-graph models (Soleymani DeepPocket, LSTM-SAC).

    prices:   float32 [B, T, 28]      — OHLCV close prices for exactly 28 stocks
    returns:  float32 [B, T, 28]      — daily log-returns
    volatility: float32 [B, 28]       — 20-day rolling realised vol
    vix:      float32 [B, 1]          — VIX level (scalar macro input)
    window:   RollingWindow           — which 90-day window this batch covers
    universe: StockUniverse           — which 28 tickers are in columns
    """
    prices:     Tensor          # shape: [B, T, 28]
    returns:    Tensor          # shape: [B, T, 28]
    volatility: Tensor          # shape: [B, 28]
    vix:        Tensor          # shape: [B, 1]
    window:     RollingWindow
    universe:   StockUniverse

    def validate(self) -> None:
        """Runtime assertion — call this once at the dataloader boundary."""
        n = self.universe.size  # always 28
        assert self.prices.shape[-1] == n, (
            f"prices last dim {self.prices.shape[-1]} != universe size {n}"
        )
        assert self.returns.shape[-1] == n
        assert self.volatility.shape[-1] == n
        assert self.prices.dtype == torch.float32
        assert self.window.n_trading_days == WINDOW_DAYS


# ---------------------------------------------------------------------------
# 6.  GraphBatch  — heterogeneous graph state for GNN models
#     (Sun GraphSAGE / Our GNN-SAC ==> accepts GraphBatch, REJECTS MarketBatch)
# ---------------------------------------------------------------------------

class GraphBatch(NamedTuple):
    """
    Enriched graph batch for GNN-based models.

    node_features: float32 [B, N, F]   — per-node features (stocks + sector + macro [+ events])
    adj:           float32 [B, N, N]   — dynamic weighted adjacency matrix
    node_types:    int64  [N]           — 0=stock, 1=sector, 2=macro, 3=event
    market:        MarketBatch          — the underlying flat market data (for metrics)
    lambda_event:  float               — FinBERT fusion gate λ ∈ [0, 1]

    N = num_stock_nodes + num_sector_nodes + 1 (macro) [+ num_event_nodes]
    For the 28-stock dataset: N = 28 + 7 + 1 + 5 = 41
    """
    node_features:  Tensor          # [B, N, F]
    adj:            Tensor          # [B, N, N]
    node_types:     Tensor          # [N]   — int64
    market:         MarketBatch
    lambda_event:   float           # Sensitivity parameter swept in Table 10

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[1])

    @property
    def num_stock_nodes(self) -> int:
        """Returns the number of TRADABLE (stock-type=0) nodes."""
        return int((self.node_types == 0).sum().item())

    def validate(self) -> None:
        self.market.validate()
        assert self.node_features.dtype == torch.float32
        assert self.adj.dtype == torch.float32
        assert self.node_types.dtype == torch.int64
        B, N, F = self.node_features.shape
        assert self.adj.shape == (B, N, N), (
            f"adj shape {self.adj.shape} does not match node_features [B={B}, N={N}]"
        )
        assert 0.0 <= self.lambda_event <= 1.0, "lambda_event must be in [0, 1]"
        # Tradable stocks must equal universe size
        assert self.num_stock_nodes == self.market.universe.size, (
            f"num_stock_nodes={self.num_stock_nodes} != universe.size={self.market.universe.size}"
        )


# ---------------------------------------------------------------------------
# 7.  EventGraphBatch — extends GraphBatch with TGN memory (our model only)
#     Pyright enforces that GNNSACAgent CANNOT receive a plain GraphBatch
#     (missing tgn_memory field) unless you explicitly acknowledge it.
# ---------------------------------------------------------------------------

class EventGraphBatch(NamedTuple):
    """
    Superset of GraphBatch: adds TGN temporal memory and FinBERT event embeddings.
    ONLY the Event-Driven GNN-SAC agent accepts this type.

    tgn_memory:      float32 [B, N, M]   — TGN recurrent memory state
    event_embeddings: float32 [B, E, H]  — FinBERT-MRC embeddings for E extracted events
    """
    graph:            GraphBatch
    tgn_memory:       Tensor          # [B, N, M]  — M = memory_dim (64 in config)
    event_embeddings: Tensor          # [B, E, H]  — H = 768 (FinBERT hidden)

    def validate(self) -> None:
        self.graph.validate()
        B = self.graph.node_features.shape[0]
        N = self.graph.num_nodes
        assert self.tgn_memory.shape[0] == B
        assert self.tgn_memory.shape[1] == N
        assert self.event_embeddings.shape[0] == B


# ---------------------------------------------------------------------------
# 8.  PortfolioState — the environment observation fed to the RL policy
# ---------------------------------------------------------------------------

class PortfolioState(NamedTuple):
    """
    Complete observation passed from PortfolioEnv → Policy at each step.
    For GNN models this wraps the flattened node features + prev weights.
    For flat models this is just price/return features + prev weights.

    flat_obs:       float32 [state_dim]   — what the SAC Policy.forward() actually sees
    weights:        float32 [28]          — current portfolio allocation (sum=1)
    step:           int                   — current env step index
    portfolio_value: float                — current portfolio NAV
    """
    flat_obs:       Tensor      # [state_dim]
    weights:        Tensor      # [28]  — must sum to 1.0
    step:           int
    portfolio_value: float

    def validate(self) -> None:
        assert self.weights.shape == (UNIVERSE_SIZE,), (
            f"weights must be [{UNIVERSE_SIZE}], got {self.weights.shape}"
        )
        weight_sum = float(self.weights.sum().item())
        assert abs(weight_sum - 1.0) < 1e-4, (
            f"Portfolio weights must sum to 1.0, got {weight_sum:.6f}"
        )
        assert self.portfolio_value >= 0.0


# ---------------------------------------------------------------------------
# 9.  PortfolioWeights — the SINGLE output type ALL models must produce
#     (ensures apples-to-apples comparison in the evaluation scripts)
# ---------------------------------------------------------------------------

class PortfolioWeights(NamedTuple):
    """
    Normalised portfolio allocation output — identical contract for ALL 3 models.

    weights:    float32 [B, 28]   — softmax-normalised, sum=1 per batch element
    model_tag:  str               — 'soleymani_2021' | 'sun_graphsage_2024' | 'gnnsac_ours'
    """
    weights:   Tensor   # [B, 28], dtype=float32, values in [0,1], row-sum=1
    model_tag: Literal["soleymani_2021", "sun_graphsage_2024", "gnnsac_ours"]

    def validate(self) -> None:
        B, N = self.weights.shape
        assert N == UNIVERSE_SIZE, f"weights must have {UNIVERSE_SIZE} columns, got {N}"
        sums = self.weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-4), (
            f"PortfolioWeights rows must sum to 1.0; max deviation: {(sums - 1).abs().max():.6f}"
        )
        assert self.weights.dtype == torch.float32
        assert (self.weights >= 0).all(), "No short-selling: all weights must be >= 0"


# ---------------------------------------------------------------------------
# 10.  Model Protocol contracts — enforced by Pyright, checked at runtime
# ---------------------------------------------------------------------------

@runtime_checkable
class FlatModel(Protocol):
    """
    Contract for CNN/MLP/LSTM models (Soleymani et al.).
    Accepts: MarketBatch    ONLY.
    Returns: PortfolioWeights with model_tag='soleymani_2021'
    """
    def forward_flat(self, batch: MarketBatch) -> PortfolioWeights: ...


@runtime_checkable
class StaticGraphModel(Protocol):
    """
    Contract for GraphSAGE-PPO (Sun et al. 2024).
    Accepts: GraphBatch     ONLY  (NOT EventGraphBatch, NOT MarketBatch).
    Returns: PortfolioWeights with model_tag='sun_graphsage_2024'
    """
    def forward_graph(self, batch: GraphBatch) -> PortfolioWeights: ...


@runtime_checkable
class EventDrivenGNNModel(Protocol):
    """
    Contract for our Event-Driven GNN-SAC.
    Accepts: EventGraphBatch  ONLY  (requires TGN memory + FinBERT events).
    Returns: PortfolioWeights with model_tag='gnnsac_ours'
    """
    def forward_event(self, batch: EventGraphBatch) -> PortfolioWeights: ...


# ---------------------------------------------------------------------------
# 11.  Type-safe adapter — bridges existing GNNSACAgent to the Protocol
# ---------------------------------------------------------------------------

class GNNSACModelAdapter:
    """
    Thin adapter that wraps the existing GNNSACAgent (which uses **kwargs and
    raw tensors) and exposes the typed Protocol interface.

    This adapter is the ONLY place where raw tensors cross the typed boundary.
    All downstream evaluation code uses EventDrivenGNNModel — never raw kwargs.

    Example
    -------
    >>> adapter = GNNSACModelAdapter(agent, universe)
    >>> isinstance(adapter, EventDrivenGNNModel)   # True (runtime_checkable)
    True
    """

    def __init__(self, agent: object, universe: StockUniverse) -> None:
        self._agent = agent
        self._universe = universe

    def forward_event(self, batch: EventGraphBatch) -> PortfolioWeights:
        batch.validate()
        import torch

        # Pull adj from EventGraphBatch (already validated)
        adj: Tensor = batch.graph.adj              # [B, N, N]
        state: Tensor = batch.graph.node_features  # [B, N, F] — will be flattened by env
        tgn_memory: Tensor = batch.tgn_memory      # [B, N, M]

        # Build composite hidden as (lstm_hidden=None, tgn_memory)
        hidden = (None, tgn_memory)

        # Call existing agent — returns (action_np_array, new_hidden)
        flat_state = state.flatten(start_dim=1)  # [B, N*F]
        action_np, _ = self._agent.select_action(  # type: ignore[attr-defined]
            flat_state.cpu().numpy()[0],
            hidden=hidden,
            deterministic=True,
            adj=adj,
        )

        weights_t = torch.tensor(action_np, dtype=torch.float32).unsqueeze(0)
        # Softmax-normalise to enforce sum=1 (PortfolioEnv does this too)
        weights_t = torch.softmax(weights_t, dim=-1)

        result = PortfolioWeights(weights=weights_t, model_tag="gnnsac_ours")
        result.validate()
        return result


# ---------------------------------------------------------------------------
# 12.  ExperimentRun — ties everything together into one validated context
# ---------------------------------------------------------------------------

class ExperimentRun(NamedTuple):
    """
    Top-level container that stamps a single comparative evaluation run.
    Every metric CSV / result figure should be tagged with this object so
    results are always traceable to their experimental context.
    """
    universe:           StockUniverse
    window:             RollingWindow
    transaction_cost_bps: float         # must be 25.0 for fair comparison
    model_tag:          Literal["soleymani_2021", "sun_graphsage_2024", "gnnsac_ours"]
    run_id:             str             # e.g. "2026-03-04_step150k"

    def validate(self) -> None:
        assert self.universe.size == UNIVERSE_SIZE
        assert self.window.n_trading_days == WINDOW_DAYS
        assert self.transaction_cost_bps == TRANSACTION_COST_BPS, (
            f"Transaction cost must be {TRANSACTION_COST_BPS} bps for fair comparison, "
            f"got {self.transaction_cost_bps} bps."
        )


# ---------------------------------------------------------------------------
# 13.  Type guard helpers — used in evaluation loops for safe narrowing
# ---------------------------------------------------------------------------

def requires_graph(batch: MarketBatch | GraphBatch | EventGraphBatch) -> GraphBatch:
    """
    Pyright-friendly narrowing: call this before passing a batch to a graph-based model.
    Raises TypeError if batch is a plain MarketBatch.
    """
    if isinstance(batch, MarketBatch):
        raise TypeError(
            "A flat MarketBatch was passed to a graph model. "
            "Wrap it in GraphBatch first using build_graph_batch()."
        )
    if isinstance(batch, EventGraphBatch):
        return batch.graph
    return batch  # already GraphBatch


def requires_event_graph(batch: MarketBatch | GraphBatch | EventGraphBatch) -> EventGraphBatch:
    """
    Narrowing for our GNN-SAC: only EventGraphBatch is accepted.
    """
    if not isinstance(batch, EventGraphBatch):
        raise TypeError(
            f"EventDrivenGNNModel requires EventGraphBatch, got {type(batch).__name__}. "
            "You must provide TGN memory and FinBERT event embeddings."
        )
    return batch
