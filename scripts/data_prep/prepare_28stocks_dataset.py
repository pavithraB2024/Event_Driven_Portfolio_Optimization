
import sys
import os
import pandas as pd

# Add root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare_data import fetch_stock_data

def prepare_28stocks():
    print("Preparing 28-stock dataset...")
    
    # 1. Load tickers
    tickers_file = os.path.join('data', '28stocks_tickers.txt')
    if not os.path.exists(tickers_file):
        # Fallback if file missing (user provided list)
        tickers = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "BRK-B", "JNJ", "UNH", "XOM", "JPM",
            "NVDA", "PG", "V", "TSLA", "HD", "CVX", "MA", "ABBV", "LLY", "MRK",
            "META", "PEP", "KO", "BAC", "AVGO", "TMO", "COST", "DIS"
        ] # Example list, but usually relies on file
        print("Warning: Tickers file not found. Using default list.")
    else:
        with open(tickers_file, 'r') as f:
            tickers = f.read().strip().split(',')
            
    tickers = [t.strip() for t in tickers if t.strip()]
    print(f"Found {len(tickers)} tickers: {tickers}")
    
    # 2. Fetch Data
    print("Fetching data (2002-01-01 to 2022-04-01)...")
    raw_data = fetch_stock_data(tickers, '2002-01-01', '2022-04-01')
    
    # 3. Process into Environment Format (price_TICKER, return_TICKER)
    df = pd.DataFrame(index=raw_data.index)
    
    # Helper to clean/extract close prices
    def get_prices(ticker_df):
        if 'Adj Close' in ticker_df.columns: return ticker_df['Adj Close']
        if 'Close' in ticker_df.columns: return ticker_df['Close']
        return ticker_df.iloc[:, 0]

    for ticker in tickers:
        try:
            if isinstance(raw_data.columns, pd.MultiIndex):
                if ticker in raw_data.columns:
                    ticker_data = raw_data[ticker]
                    close = get_prices(ticker_data)
                else:
                    # check if ticker is in level 1?
                    # yfinance often returns (PriceType, Ticker) or (Ticker, PriceType) depending on group_by
                    # fetch_stock_data usually uses group_by='ticker'
                    print(f"Warning: {ticker} not found in fetched data.")
                    continue
            else:
                # Flat format? Usually fetch_stock_data returns MultiIndex for >1 ticker
                # Inspect columns just in case
                if f'{ticker}_Adj Close' in raw_data.columns:
                    close = raw_data[f'{ticker}_Adj Close']
                elif ticker in raw_data.columns: # If just prices
                    close = raw_data[ticker]
                else:
                    continue
            
            # Forward fill then fillna(0)
            close = close.ffill().fillna(0.0)
            
            # Calculate returns
            ret = close.pct_change().fillna(0.0)
            
            df[f'price_{ticker}'] = close
            df[f'return_{ticker}'] = ret
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # 4. Save
    output_path = os.path.join('data', 'processed', '28stocks_dataset.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"\nSaved 28-stock dataset to: {output_path}")
    
    # Also save to enhanced path to fix legacy scripts
    enhanced_path = os.path.join('data', 'processed', 'complete_dataset_enhanced.csv')
    df.to_csv(enhanced_path)
    print(f"Saved 28-stock dataset to: {enhanced_path}")
    print(f"Shape: {df.shape}")
    print("Columns:", list(df.columns[:5]), "...")

if __name__ == "__main__":
    prepare_28stocks()
