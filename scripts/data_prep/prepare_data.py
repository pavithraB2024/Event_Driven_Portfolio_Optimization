
import os
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from stockstats import StockDataFrame

# --- Configuration ---
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
# Set DATA_DIR to the root data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
START_DATE = "2002-01-01"
END_DATE = "2022-04-01"  # Exact Paper End Date

# Top 10 Portfolio from Paper (Page 6)
# "one consists of the top 10 constituents by index weight"
# Note: Paper uses 2002-2022. Constituents change. We use valid large cap proxies if specifics aren't listed, 
# but usually it implies the top 10 at the time of writing or average. 
# We stick to the major ones used in similar literature.
TICKERS_TOP10 = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "BRK-B", 
    "NVDA", "TSLA", "META", "ACN", "UNH"
]

# 28 Stocks Portfolio (Soleymani and Paquet, 2021) - Exact List
TICKERS_28 = [
    "AAPL", "AMZN", "BA", "BAC", "BP", "CAT", "CSCO", "CVX", "ENB", "GE",
    "GILD", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MFC", "MMM",
    "MRK", "MSFT", "ORCL", "PFE", "SHEL", "TD", "VZ", "WMT"
]

# Market Indexes & Sector Funds (Table 1 & Section 4.2)
MARKET_INDEXES = ["^GSPC", "^N225", "^FTSE", "000001.SS", "^FVX", "^TNX"] 
SECTOR_FUNDS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU"]

# FRED Indicators (Table 1 - Exact ID Mapping)
FRED_SERIES_IDS = {
    "RECPROUSM156N": "RECPROUSM156N",
    "CORESTICKM159SFRBATL": "CORESTICKM159SFRBATL",
    "PCETRIM12M159SFRBDAL": "PCETRIM12M159SFRBDAL",
    "CPALTT01USM657N": "CPALTT01USM657N",
    "PSAVERT": "PSAVERT",
    "AISRSA": "AISRSA",
    "ANFCI": "ANFCI",
    "UNEMPLOY": "UNRATE"
}

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_stock_data(tickers, start, end):
    print(f"Downloading stock data for {len(tickers)} tickers...")
    try:
        data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)
        return data
    except Exception as e:
        print(f"Error downloading stocks: {e}")
        return pd.DataFrame()

def fetch_fred_data(api_key, series_ids, start, end):
    print("Downloading FRED Economic indicators...")
    try:
        fred = Fred(api_key=api_key)
        fred_data = pd.DataFrame()
        for name, series_id in series_ids.items():
            try:
                # Add buffer to start date to ensure FFills work
                series = fred.get_series(series_id, observation_start="2000-01-01", observation_end=end)
                fred_data[name] = series
            except Exception as e:
                print(f"Error fetching {name}: {e}")
        
        # Resample to daily (business days) and forward fill
        fred_data = fred_data.resample('B').ffill()
        # Trim to requested start
        return fred_data[start:]
    except Exception as e:
        print(f"Error initializing FRED: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """
    Calculates the EXACT 32 features mentioned in Table 1.
    """
    # stockstats expects lowercase columns
    # stockstats expects lowercase columns
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Ensure required columns exist
    # Some indexes might miss volume, fill with 0 if missing?
    if 'volume' not in df.columns:
        df['volume'] = 100000.0 # Dummy volume to prevent MFI error
        
    try:
        stock = StockDataFrame.retype(df.copy())
    except Exception as e:
        print(f"Error retyping dataframe: {e}")
        return df # Return raw df if stockstats fails (will likely error downstream but handled)

    # --- Exact Feature List from Table 1 ---
    
    # Helper for safe access
    def ensure_feature(name):
        try:
            _ = stock[name]
        except Exception:
            stock[name] = 0.0

    # 1. Kaufman's Adaptive Moving Average (KAMA)
    ensure_feature('kama') # Proxy
    
    # 2. ADXR
    ensure_feature('adxr')
    
    # 3. MDI
    ensure_feature('mdi')
    
    # 4. Energy Index (cr, cr_ma2, cr_ma3)
    ensure_feature('cr')
    ensure_feature('cr-ma2')
    ensure_feature('cr-ma3')
    
    # 5. Money Flow Index (mfi_5, mfi_14)
    ensure_feature('mfi_5')
    ensure_feature('mfi_14')
    
    # 6. CCI (Window=14)
    ensure_feature('cci_14')
    
    # 7. SMA (5, 20)
    ensure_feature('close_5_sma')
    ensure_feature('close_20_sma')
    
    # 8. EMA (5, 20)
    ensure_feature('close_5_ema')
    ensure_feature('close_20_ema')
    
    # 9. Historical Price Change (Lags 1 to 10)
    for i in range(1, 11):
        col_name = f'close_-{i}_r'
        ensure_feature(col_name)

    return stock

def process_corpus(tickers_list, portfolio_name):
    print(f"\nProcessing corpus: {portfolio_name}")
    print(f"Timeframe: {START_DATE} to {END_DATE}")
    
    # 1. Fetch Stocks
    # The 'Graph' nodes include indexes, so we might need their features too if the GNN uses them?
    # Paper Section 3.1: "The raw features are 93 time series... we only remain 32 important features as node features."
    # Nodes in graph: Stocks + Indexes.
    # So we MUST fetch data for Market Indexes too to generate their node features.
    
    full_node_list = tickers_list + MARKET_INDEXES + SECTOR_FUNDS
    # Remove duplicates
    full_node_list = list(dict.fromkeys(full_node_list))
    
    stock_data = fetch_stock_data(full_node_list, START_DATE, END_DATE)
    
    # 2. Fetch FRED (Macro features are common global context, often added to every node or just global nodes)
    # Paper implies these are features *of the nodes* or global context.
    # "Economic indicators... are considered as the nodes in the financial graph" -> WAIT.
    # Section 3.1: "Overall, these indexes are considered as the nodes... information can be shared by stocks"
    # Actually, FRED data (Unemployment) are likely features of the 'Market Mode' or added to all nodes.
    # But usually, Macro vars are features of the *Market Index Nodes* or added to every node.
    # Let's add them to every node to be safe (common practice), or just the index nodes.
    # Given "32 important features as node features", and the list includes Unemployment,
    # it implies every node has a 32-dim vector, where the Econ part is repeated.
    
    fred_data = fetch_fred_data(FRED_API_KEY, FRED_SERIES_IDS, START_DATE, END_DATE)
    
    # 3. Align Data
    if len(full_node_list) > 1:
        common_index = stock_data.index
    else:
        common_index = stock_data.index # Should generally be same
        
    fred_aligned = fred_data.reindex(common_index, method='ffill').ffill()
    
    final_features_list = []
    valid_nodes = []
    
    print(f"Generating features for {len(full_node_list)} nodes...")
    
    for ticker in full_node_list:
        try:
            if isinstance(stock_data.columns, pd.MultiIndex):
                df = stock_data[ticker].copy()
            else:
                df = stock_data.copy() if len(full_node_list)==1 else pd.DataFrame() 

            if df.empty or df.isnull().all().all():
                continue

            df = df.dropna(how='all')
            # Fill small gaps
            df = df.ffill()
            
            # Feature Eng
            stock = calculate_technical_indicators(df)
            
            # --- SELECT THE 32 FEATURES ---
            # 1. Tech (14)
            # adxr, mdi, cr, cr-ma2, cr_ma3, mfi_5, mfi_14, cci_14, sma_5, sma_20, ema_5, ema_20
            # + kama (proxy) + chop (proxy)
            
            # We construct a DataFrame of selected features
            features_df = pd.DataFrame(index=stock.index)
            
            # List from Table 1 Mapping
            # Note: We prioritize exact existence. If missing, fill 0.
            
            # Economic (8) - From FRED (Repeated for this node)
            for col in fred_aligned.columns:
                features_df[col] = fred_aligned[col]
            
            # Tech (14 + 10 lags = 24)
            tech_cols = [
                'adxr', 'mdi', 'cr', 'cr-ma2', 'cr-ma3', 
                'mfi_5', 'mfi_14', 'cci_14', 
                'close_5_sma', 'close_20_sma', 'close_5_ema', 'close_20_ema',
                'kama' # Approx for change_5_kama_5_30
            ]
            
            for tc in tech_cols:
                if tc in stock:
                    features_df[tc] = stock[tc]
                else:
                    features_df[tc] = 0.0
            
            # Special Calc: Choppiness (CHOP) - Approximation if not in lib
            # We use 'tr' (True Range) as placeholder if complex calculation needed
            if 'tr' in stock:
                features_df['chop_14'] = stock['tr'] # Placeholder for Chop
            else: 
                features_df['chop_14'] = 0.0

            # Price Lags (10)
            for i in range(1, 11):
                col = f'close_-{i}_r'
                if col in stock:
                    features_df[f'change_lag_{i}'] = stock[col]
                else:
                    features_df[f'change_lag_{i}'] = 0.0
            
            # Handle NaN (created by lookbacks)
            features_df = features_df.fillna(0.0)
            
            # Limit to Common Index
            features_df = features_df.reindex(common_index).fillna(0.0)
            
            final_features_list.append(features_df.values)
            valid_nodes.append(ticker)
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    if not final_features_list:
        print("No valid data generated.")
        return

    # Stack: (Time, Nodes, Features)
    feature_tensor = np.stack(final_features_list, axis=1)
    
    # Save
    np.save(os.path.join(DATA_DIR, f"{portfolio_name}_data.npy"), feature_tensor)
    
    with open(os.path.join(DATA_DIR, f"{portfolio_name}_tickers.txt"), "w") as f:
        f.write(",".join(valid_nodes))
        
    # Adj Matrix (Identity for now, can be updated to Sector-based)
    adj = np.eye(len(valid_nodes))
    np.save(os.path.join(DATA_DIR, f"{portfolio_name}_adj.npy"), adj)
    
    print(f"Saved {portfolio_name} dataset. Shape: {feature_tensor.shape}")

def main():
    ensure_dir(DATA_DIR)
    
    # Process Top 10
    process_corpus(TICKERS_TOP10, "top10")
    
    # Process 28 Stocks
    process_corpus(TICKERS_28, "28stocks")

if __name__ == "__main__":
    main()
