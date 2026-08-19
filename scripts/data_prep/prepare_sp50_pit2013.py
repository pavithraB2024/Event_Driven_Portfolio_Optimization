"""
Build the 50-asset point-in-time S&P 500 universe for the scalability
workstream (manuscript MLWA-D-26-00714).

Universe = top-50 by 2013-11-04 liquidity (close * volume) drawn from the
fja05680/sp500 historical components CSV row for 2013-11-01. The point-in-time
snapshot deliberately includes names that have since been delisted, acquired,
or renamed (the response-letter cites these to disarm the survivorship-bias
criticism in Reviewer 2 Comment 3).

Outputs:
  data/sp50_pit2013_tickers.txt          - 50 comma-separated tickers
  data/sp50_pit2013_delisted_log.txt     - subset of the top-50 no longer in the
                                           current S&P 500 (per the latest CSV row)
  data/processed/sp50_pit2013_dataset.csv - price_TICKER / return_TICKER columns,
                                           same schema as 28stocks_dataset.csv

Usage:
  PYTHONPATH=. python scripts/data_prep/prepare_sp50_pit2013.py

Caveat on ticker reuse:
  A small number of historical tickers (e.g. ALTR) have been reassigned to a
  different company since 2013. yfinance returns the current owner's history,
  which silently corrupts pre-reassignment price data. Confirmed-clean delisted
  tickers from the canonical-plan seed list (BTUUQ, BRCM, CAM, BEAM, AABA,
  ANDV) are not subject to this issue. Out of scope to detect automatically;
  flagged in the delisted log for manual review.
"""
from __future__ import annotations

import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

# Make sibling prepare_data.py importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

from prepare_data import fetch_stock_data  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(REPO_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

COMPONENTS_CSV_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
PIT_DATE = "2013-11-01"
LIQUIDITY_DATE_START = "2013-11-01"
LIQUIDITY_DATE_END = "2013-11-08"  # short window to find a tradable bar around Nov 4
LIQUIDITY_ANCHOR = "2013-11-04"
TOP_N = 50
PRICE_HISTORY_START = "2002-01-01"
PRICE_HISTORY_END = "2022-04-01"
# Tickers with 2013-11-04 close above this dollar amount are flagged as
# suspected ticker-reuse corruption (no S&P 500 constituent traded above
# ~$1500/share in late 2013) and excluded from liquidity ranking.
PRICE_REUSE_GUARD_USD = 1500.0


def download_components_csv() -> pd.DataFrame:
    print(f"Downloading historical S&P 500 components CSV...\n  {COMPONENTS_CSV_URL}")
    with urllib.request.urlopen(COMPONENTS_CSV_URL, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def pit_constituents(df: pd.DataFrame, pit_date: str) -> list[str]:
    if "date" not in df.columns or "tickers" not in df.columns:
        raise RuntimeError(f"Unexpected CSV schema: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    target = pd.to_datetime(pit_date)
    # Pick the row at or immediately before the pit date (snapshots are state-of-index entries).
    eligible = df[df["date"] <= target].sort_values("date")
    if eligible.empty:
        raise RuntimeError(f"No CSV row at or before {pit_date}")
    row = eligible.iloc[-1]
    tickers = [t.strip() for t in str(row["tickers"]).split(",") if t.strip()]
    print(f"  Pit-in-time row date={row['date'].date()}, |constituents|={len(tickers)}")
    return tickers


def latest_constituents(df: pd.DataFrame) -> set[str]:
    df["date"] = pd.to_datetime(df["date"])
    latest = df.sort_values("date").iloc[-1]
    return {t.strip() for t in str(latest["tickers"]).split(",") if t.strip()}


def _yf_aliases(ticker: str) -> list[str]:
    """Return yfinance-compatible aliases for tickers using dots in the CSV
    (e.g. BRK.B -> BRK-B, BF.B -> BF-B). The CSV uses dots; yfinance uses
    hyphens for class-share tickers."""
    candidates = [ticker]
    if "." in ticker:
        candidates.append(ticker.replace(".", "-"))
    return candidates


def rank_by_liquidity(tickers: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Returns (ranked_df, fetch_failures, price_guard_drops). The two lists
    feed the enriched survivorship log."""
    # Build a resolution map: csv-ticker -> yfinance-ticker we actually queried.
    fetch_targets = []
    resolution: dict[str, str] = {}
    for t in tickers:
        for alias in _yf_aliases(t):
            fetch_targets.append(alias)
            resolution[alias] = t  # last alias wins; we pick whichever yields data below

    print(f"\nFetching {LIQUIDITY_DATE_START} -> {LIQUIDITY_DATE_END} OHLCV for {len(fetch_targets)} ticker aliases...")
    raw = fetch_stock_data(fetch_targets, LIQUIDITY_DATE_START, LIQUIDITY_DATE_END)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data for liquidity ranking")

    rows = []
    fetched_csv_tickers: set[str] = set()
    price_guard_drops: list[str] = []

    for alias in fetch_targets:
        csv_ticker = resolution[alias]
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if alias not in raw.columns.get_level_values(0):
                    continue
                tdf = raw[alias]
            else:
                tdf = raw
            close_col = "Close" if "Close" in tdf.columns else ("Adj Close" if "Adj Close" in tdf.columns else None)
            if close_col is None or "Volume" not in tdf.columns:
                continue
            anchor_idx = pd.to_datetime(LIQUIDITY_ANCHOR)
            window = tdf[[close_col, "Volume"]].dropna()
            if window.empty:
                continue
            deltas = np.abs((window.index - anchor_idx).to_numpy().astype("timedelta64[ns]").astype("int64"))
            nearest = int(np.argmin(deltas))
            close = float(window[close_col].iloc[nearest])
            volume = float(window["Volume"].iloc[nearest])
            liquidity = close * volume
            if not np.isfinite(liquidity) or liquidity <= 0:
                continue
            if close > PRICE_REUSE_GUARD_USD:
                # Almost certainly a reused-symbol giving a current security's
                # price. No S&P 500 constituent traded above ~$1500/share in 2013.
                price_guard_drops.append(f"{csv_ticker} (yf={alias}, close=${close:,.2f}, volume={int(volume):,})")
                continue
            fetched_csv_tickers.add(csv_ticker)
            rows.append({
                "ticker": csv_ticker,
                "yfinance_symbol": alias,
                "close": close,
                "volume": volume,
                "liquidity": liquidity,
            })
        except Exception as exc:
            print(f"  skip {alias}: {exc}")
            continue

    # If both BRK.B and BRK-B resolved, keep the highest-liquidity row per csv_ticker.
    df = (
        pd.DataFrame(rows)
        .sort_values("liquidity", ascending=False)
        .drop_duplicates("ticker", keep="first")
        .reset_index(drop=True)
    )
    fetch_failures = [t for t in tickers if t not in fetched_csv_tickers and t not in {d.split()[0] for d in price_guard_drops}]
    print(f"  Ranked {len(df)} tickers with valid liquidity")
    print(f"  yfinance fetch failures (excluded from ranking): {len(fetch_failures)}")
    print(f"  Price-reuse-guard drops (likely ticker reassignment): {len(price_guard_drops)}")
    return df, fetch_failures, price_guard_drops


def build_price_dataset(symbol_map: dict[str, str]) -> pd.DataFrame:
    """symbol_map maps csv_ticker -> yfinance_symbol (e.g. BRK.B -> BRK-B).
    Columns in the output use csv_ticker to stay consistent with the
    tickers file."""
    yf_targets = list(symbol_map.values())
    print(f"\nFetching price history {PRICE_HISTORY_START} -> {PRICE_HISTORY_END} for {len(yf_targets)} tickers...")
    raw = fetch_stock_data(yf_targets, PRICE_HISTORY_START, PRICE_HISTORY_END)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data for price history")

    df = pd.DataFrame(index=raw.index)
    missing: list[str] = []

    def get_close(tdf: pd.DataFrame) -> pd.Series | None:
        for col in ("Adj Close", "Close"):
            if col in tdf.columns:
                return tdf[col]
        return None

    for csv_ticker, yf_symbol in symbol_map.items():
        if isinstance(raw.columns, pd.MultiIndex):
            if yf_symbol not in raw.columns.get_level_values(0):
                missing.append(csv_ticker)
                continue
            tdf = raw[yf_symbol]
        else:
            tdf = raw
        close = get_close(tdf)
        if close is None or close.dropna().empty:
            missing.append(csv_ticker)
            continue
        close = close.ffill().fillna(0.0)
        ret = close.pct_change().fillna(0.0)
        df[f"price_{csv_ticker}"] = close
        df[f"return_{csv_ticker}"] = ret

    if missing:
        print(f"  Warning: {len(missing)} tickers had no fetchable history: {missing}")
    return df


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    components = download_components_csv()
    candidates = pit_constituents(components, PIT_DATE)
    latest = latest_constituents(components)

    ranked, fetch_failures, price_guard_drops = rank_by_liquidity(candidates)
    if len(ranked) < TOP_N:
        raise RuntimeError(f"Only {len(ranked)} ranked tickers; cannot form top-{TOP_N}")
    top = ranked.head(TOP_N).copy()
    top_tickers = top["ticker"].tolist()
    print(f"\nTop-{TOP_N} by 2013-11-04 liquidity:")
    print(top.to_string(index=False))

    # In-top-50 names that are NO LONGER S&P 500 constituents per the latest CSV row.
    delisted_in_top = [t for t in top_tickers if t not in latest]
    # Survivorship-eliminated names: 2013 constituents that BOTH (a) failed to
    # fetch from yfinance AND (b) are no longer in the index. These are the
    # strongest evidence that the data layer below yfinance has survivorship
    # bias even when the universe construction does not.
    survivorship_eliminated = sorted(
        t for t in fetch_failures if t not in latest
    )

    delisted_log_path = os.path.join(DATA_DIR, "sp50_pit2013_delisted_log.txt")
    with open(delisted_log_path, "w") as fh:
        fh.write(
            f"# Survivorship-bias audit log for the sp50_pit2013 universe\n"
            f"# Source: {COMPONENTS_CSV_URL}  pit_date={PIT_DATE}\n"
            f"#\n"
            f"# [A] TOP-50 NAMES SINCE REMOVED FROM S&P 500\n"
            f"#     (in our universe, not in the latest CSV row)\n"
            f"#     Count: {len(delisted_in_top)} / {TOP_N}\n"
        )
        for t in delisted_in_top:
            fh.write(f"{t}\n")
        fh.write(
            f"#\n"
            f"# [B] PRICE-REUSE-GUARD DROPS\n"
            f"#     (2013 constituents whose yfinance close > ${PRICE_REUSE_GUARD_USD:.0f},\n"
            f"#      suggesting the ticker symbol has been reassigned to a different\n"
            f"#      security since delisting; excluded from ranking)\n"
            f"#     Count: {len(price_guard_drops)}\n"
        )
        for line in price_guard_drops:
            fh.write(f"{line}\n")
        fh.write(
            f"#\n"
            f"# [C] SURVIVORSHIP-ELIMINATED CANDIDATES\n"
            f"#     (2013 constituents that BOTH failed to fetch from yfinance AND\n"
            f"#      are no longer in the latest S&P 500 row -- strong indicator of\n"
            f"#      true delisting/acquisition where price data is no longer hosted.\n"
            f"#      These are the marquee survivorship examples for the response letter.)\n"
            f"#     Count: {len(survivorship_eliminated)}\n"
        )
        for t in survivorship_eliminated:
            fh.write(f"{t}\n")
    print(f"\n[A] Top-{TOP_N} removed-from-index ({len(delisted_in_top)}): {delisted_in_top}")
    print(f"[B] Price-reuse-guard drops ({len(price_guard_drops)}): {price_guard_drops}")
    print(f"[C] Survivorship-eliminated candidates ({len(survivorship_eliminated)}): {survivorship_eliminated[:20]}{'...' if len(survivorship_eliminated) > 20 else ''}")
    print(f"  Full log: {delisted_log_path}")

    tickers_path = os.path.join(DATA_DIR, "sp50_pit2013_tickers.txt")
    with open(tickers_path, "w") as fh:
        fh.write(",".join(top_tickers))
    print(f"\nTickers written to {tickers_path}")

    symbol_map = dict(zip(top["ticker"], top["yfinance_symbol"]))
    price_df = build_price_dataset(symbol_map)
    dataset_path = os.path.join(PROCESSED_DIR, "sp50_pit2013_dataset.csv")
    price_df.to_csv(dataset_path)
    print(f"\nDataset written to {dataset_path}  shape={price_df.shape}")
    n_price_cols = sum(1 for c in price_df.columns if c.startswith("price_"))
    n_return_cols = sum(1 for c in price_df.columns if c.startswith("return_"))
    print(f"  price_* columns: {n_price_cols}  return_* columns: {n_return_cols}")


if __name__ == "__main__":
    main()
