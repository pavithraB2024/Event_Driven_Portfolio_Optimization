"""Financial news ingestion from CSV and API sources."""
import pandas as pd
import os
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ticker Alias Maps
# ---------------------------------------------------------------------------
# When a portfolio ticker has zero or very poor coverage in the raw news data,
# we pull news from a "proxy" ticker in the same sector and relabel it.
#   key   = original ticker (the one YOUR portfolio needs)
#   value = proxy ticker   (the one that actually has articles in news.csv)
# ---------------------------------------------------------------------------
TICKER_ALIASES = {
    # 28stocks: No aliases needed. Coverage gaps (MSFT, CVX, GE, HON, SHEL) are
    # handled architecturally in graph_builder.py via Sector Sentiment Aggregation,
    # Temporal Decay Memory, and Correlation-Based Imputation (Solutions 1-3).
}


class NewsIngester:
    def __init__(self, data_path, tickers_path, output_path, portfolio=None):
        self.data_path = data_path
        self.tickers_path = tickers_path
        self.output_path = output_path
        self.portfolio = portfolio
        self.target_tickers = self._load_tickers()

        # Build alias maps for this portfolio
        aliases = TICKER_ALIASES.get(portfolio, {})
        # forward:  original -> proxy   (e.g. MSFT -> ORCL)
        self.alias_forward = aliases
        # reverse:  proxy -> original   (e.g. ORCL -> MSFT)
        self.alias_reverse = {v: k for k, v in aliases.items()}

        if aliases:
            print(f"Ticker aliases active for '{portfolio}':")
            for orig, proxy in aliases.items():
                print(f"  {orig} -> {proxy}")

    def _load_tickers(self):
        if not os.path.exists(self.tickers_path):
            raise FileNotFoundError(f"Tickers file not found: {self.tickers_path}")

        with open(self.tickers_path, 'r') as f:
            content = f.read().strip()
            if not content:
                return set()
            return set(t.strip() for t in content.split(','))

    def load_and_filter(self):
        """Loads the massive news CSV and filters for relevant tickers."""
        print(f"\nLoading news from {self.data_path}...")
        print(f"Filtering for {len(self.target_tickers)} tickers")

        # Build the expanded search set: original tickers + proxy tickers
        search_tickers = set(self.target_tickers)
        for orig, proxy in self.alias_forward.items():
            if orig in search_tickers:
                search_tickers.add(proxy)  # also search for the proxy

        print(f"Search set ({len(search_tickers)} symbols): {sorted(search_tickers)}")

        chunk_size = 100000
        filtered_chunks = []

        try:
            for chunk in tqdm(pd.read_csv(self.data_path, chunksize=chunk_size)):
                if 'stock' not in chunk.columns:
                    continue

                # Filter for both original AND proxy tickers
                mask = chunk['stock'].isin(search_tickers)
                filtered_chunk = chunk[mask].copy()

                if not filtered_chunk.empty:
                    # Relabel proxy tickers back to the original portfolio ticker
                    if self.alias_reverse:
                        filtered_chunk['stock'] = filtered_chunk['stock'].replace(
                            self.alias_reverse
                        )
                    filtered_chunks.append(filtered_chunk)

        except Exception as e:
            print(f"Error reading CSV: {e}")
            return

        if filtered_chunks:
            full_df = pd.concat(filtered_chunks, ignore_index=True)

            # Print per-ticker coverage report
            print(f"\n{'='*50}")
            print(f"Coverage Report ({len(full_df)} total articles)")
            print(f"{'='*50}")
            coverage = full_df['stock'].value_counts()
            for ticker in sorted(self.target_tickers):
                count = coverage.get(ticker, 0)
                source = ""
                if ticker in self.alias_forward:
                    source = f"  (via proxy: {self.alias_forward[ticker]})"
                status = "[OK]" if count > 100 else ("[!!]" if count > 0 else "[--]")
                print(f"  {status} {ticker:8s}: {count:6d} articles{source}")

            zero_tickers = [t for t in self.target_tickers if coverage.get(t, 0) == 0]
            if zero_tickers:
                print(f"\n  Note: {len(zero_tickers)} ticker(s) still have 0 articles "
                      f"(no proxy available): {sorted(zero_tickers)}")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            full_df.to_csv(self.output_path, index=False)
            print(f"\nSaved filtered news to {self.output_path}")
        else:
            print("No relevant news found for the specified tickers.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest and filter news for a specific portfolio"
    )
    parser.add_argument(
        "--portfolio", type=str, default="28stocks",
        help="Portfolio name (e.g. 28stocks)"
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to raw news CSV"
    )

    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Default Paths
    if args.data_path:
        DATA_RAW = args.data_path
    else:
        DATA_RAW = os.path.join(BASE_DIR, "data", "raw", "news.csv")

    TICKERS_FILE = os.path.join(BASE_DIR, "data", f"{args.portfolio}_tickers.txt")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed",
                               f"filtered_news_{args.portfolio}.csv")

    print(f"=== News Ingestion for {args.portfolio} ===")
    print(f"Tickers Source: {TICKERS_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    if not os.path.exists(TICKERS_FILE):
        print(f"Error: Tickers file not found at {TICKERS_FILE}")
        exit(1)

    ingester = NewsIngester(DATA_RAW, TICKERS_FILE, OUTPUT_FILE,
                            portfolio=args.portfolio)
    ingester.load_and_filter()
