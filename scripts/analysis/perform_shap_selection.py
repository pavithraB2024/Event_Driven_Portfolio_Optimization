
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from stockstats import StockDataFrame

# --- Configuration ---
INPUT_FILE = 'data/portfolio_data_20.csv'
OUTPUT_DATA_FILE = 'data/portfolio_data_20_shap.csv'
OUTPUT_LIST_FILE = 'data/top20_features.txt'
TARGET_HORIZON = 5 # Days ahead to predict

def generate_features(df_asset):
    """
    Generates ~66 technical indicators using StockStats.
    Input: DataFrame with columns [open, close, high, low, volume]
    """
    # StockStats requires lowercase columns
    df = df_asset.copy()
    df.columns = [c.lower() for c in df.columns]
    
    stock = StockDataFrame.retype(df)
    
    # --- Feature List (Comprehensive) ---
    # We access 'get' to trigger calculation
    
    # 1. Moving Averages
    # sma, ema for 5, 10, 20, 50
    for w in [5, 10, 20, 50]:
        _ = stock[f'close_{w}_sma']
        _ = stock[f'close_{w}_ema']
        
    # 2. MACD
    _ = stock['macd']
    _ = stock['macds'] # signal
    _ = stock['macdh'] # hist
    
    # 3. RSI
    for w in [6, 12, 14, 24]:
        _ = stock[f'rsi_{w}']
        
    # 4. KDJ (Stochastic)
    _ = stock['kdjk']
    _ = stock['kdjd']
    _ = stock['kdjj']
    
    # 5. WR (Williams R)
    for w in [14, 10, 6]:
        _ = stock[f'wr_{w}']
        
    # 6. CCI
    for w in [14, 20]:
        _ = stock[f'cci_{w}']
        
    # 7. Bollinger Bands
    _ = stock['boll']
    _ = stock['boll_ub']
    _ = stock['boll_lb']
    
    # 8. TR, ATR
    _ = stock['tr']
    _ = stock['atr']
    
    # 9. DMA
    _ = stock['dma']
    
    # 10. DMI (PDI, MDI, ADX, DX)
    _ = stock['pdi']
    _ = stock['mdi']
    _ = stock['dx']
    _ = stock['adx']
    _ = stock['adxr']
    
    # 11. Trix
    _ = stock['trix']
    
    # 12. VR (Volume Ratio)
    _ = stock['vr']
    
    # 13. Momentum (Close differences/Rates)
    # Lags/Rates
    for l in [1, 2, 3, 5, 10]:
        _ = stock[f'close_-{l}_r'] # Rate of change lag
    
    # Cleanup: Remove infinite or NaN values created by indicators
    # We convert back to plain DataFrame
    # Note: StockDataFrame is a wrapper, we can just take the dataframe content
    # But it includes the original OHLCV, we might want to drop them for SHAP 
    # if we only want derived features to be ranked.
    # The paper includes 'volume' and 'close' in raw features.
    
    return stock

def main():
    print(f"Loading data from {INPUT_FILE}...")
    # Load raw 20-asset data
    # Format is likely Index=Date, Columns = [AAPL_Open, AAPL_Close..., BTC-USD_Open...] + [CPI, VIX...]
    full_df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    # Identify Assets
    # We look for columns ending in '_Close' to identify tickers
    tickers = [c.split('_')[0] for c in full_df.columns if c.endswith('_Close')]
    tickers = list(set(tickers))
    print(f"Identified {len(tickers)} assets: {tickers}")
    
    # Identify Macro Columns (those not starting with a ticker prefix or common ones)
    # Our get_data script added: RiskFree, VIX, CPI, Unemployment
    macro_cols = ['RiskFree', 'VIX', 'CPI', 'Unemployment']
    macro_df = full_df[macro_cols].copy()
    
    # --- Step 1: Prepare Big Training Set ---
    # We want to find "Universal Features". So we stack data from all assets.
    # X = [Tech_Asset_i + Macro], y = [Target_Asset_i]
    
    X_all = []
    y_all = []
    feature_names = None
    
    print("Generating features for each asset...")
    
    for ticker in tickers:
        # Extract OHLCV for this ticker
        cols = {
            f'{ticker}_Open': 'open',
            f'{ticker}_High': 'high',
            f'{ticker}_Low': 'low',
            f'{ticker}_Close': 'close',
            f'{ticker}_Volume': 'volume'
        }
        
        # Check if all columns exist
        if not all(c in full_df.columns for c in cols.keys()):
            # Maybe some assets miss volume?
            continue
            
        df_asset = full_df[list(cols.keys())].rename(columns=cols).copy()
        
        # 1. Generate Tech Indicators
        df_features = generate_features(df_asset)
        
        # 2. Add Macro (Aligned Index)
        # Macro is same for all assets, but serves as context
        df_features = df_features.join(macro_df, how='left')
        
        # 3. Create Target (from Paper)
        # "Predict whether mean(Close t+1...t+5) > Close t"
        # Rolling forward mean
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=TARGET_HORIZON)
        df_features['close'].rolling(window=indexer).mean().shift(-1) # Shift to look ahead
        # Shift -1 because rolling(forward) includes current? No, FixedForward usually starts current.
        # Actually easier:
        # roll = df['close'].rolling(5).mean().shift(-5) # This is backward rolling shifted back.
        # Let's use strict future lookahead.
        df_features['close'].rolling(window=5).mean().shift(-5)
        # If mean next 5 days > current close => 1
        
        # Wait, simple shift logic:
        # T+1 to T+5.
        # Mean(Close[t+1], ..., Close[t+5])
        # We can compute rolling mean of past 5, then shift back by 5.
        next_5_mean = df_features['close'].rolling(window=5).mean().shift(-5)
        
        target = (next_5_mean > df_features['close']).astype(int)
        
        # 4. Clean up
        # Drop inputs if we don't want them as features (Open, High, Low...)?
        # Paper kept "Price technical...". Usually we keep OHLCV as features too.
        
        # Drop NaNs (from indicators at start and target at end)
        valid_mask = ~df_features.isnull().any(axis=1) & ~target.isnull()
        
        X_asset = df_features[valid_mask]
        y_asset = target[valid_mask]
        
        # Store
        if feature_names is None:
            feature_names = X_asset.columns.tolist()
            
        X_all.append(X_asset.values)
        y_all.append(y_asset.values)

    if not X_all:
        print("Error: No data generated.")
        return
        
    # Stack
    X_train = np.vstack(X_all)
    y_train = np.hstack(y_all)
    
    print(f"Training GBDT on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    
    # --- Step 2: Train XGBoost ---
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        feature_names=feature_names,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # --- Step 3: SHAP Analysis ---
    print("Running SHAP Explainer (this may take a moment)...")
    # TreeExplainer is fast
    explainer = shap.TreeExplainer(model)
    
    # We use a SAMPLE of data for SHAP if dataset is huge (>100k), else full is fine.
    # 20 assets * 3000 days ~ 60k rows. Full is fine.
    shap_values = explainer.shap_values(X_train)
    
    # shap_values for binary classification is often [N, Feats] (log odds)
    
    # Calculate Mean Absolute SHAP value for each feature
    # If binary, shap_values might be list of size 2? XGBoost usually returns raw margin or single vector for binary.
    if isinstance(shap_values, list): # Check if output is list (multiclass)
        shap_sum = np.abs(shap_values[1]).mean(axis=0) # Take positive class
    else:
        shap_sum = np.abs(shap_values).mean(axis=0)
        
    # Create DataFrame of Importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': shap_sum
    })
    
    # Sort
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    
    # Top 30 (Updated from 20)
    top_n = 30
    top_features_df = importance_df.head(top_n)
    print(f"\n--- Top {top_n} SHAP Features ---")
    print(top_features_df)
    
    # Save List
    OUTPUT_LIST_FILE = 'data/top30_features.txt'
    top_features_df['feature'].to_csv(OUTPUT_LIST_FILE, index=False, header=False)
    print(f"\nSaved feature list to {OUTPUT_LIST_FILE}")
    
    # --- Step 4: Save Filtered Dataset ---
    top_features = top_features_df['feature'].tolist()
    
    print("Generating Final Feature Tensor...")
    tensor_list = []
    final_nodes = []
    
    for ticker in tickers:
        cols = {f'{ticker}_Open': 'open', f'{ticker}_High': 'high', f'{ticker}_Low': 'low', f'{ticker}_Close': 'close', f'{ticker}_Volume': 'volume'}
        try:
             df_asset = full_df[list(cols.keys())].rename(columns=cols)
        except:
            continue
            
        df_feat = generate_features(df_asset)
        df_feat = df_feat.join(macro_df, how='left')
        
        # Select Top 30
        # Check if columns exist (SHAP selected them so they must exist in generated set)
        df_selected = df_feat[top_features].copy()
        
        df_selected = df_selected.fillna(0.0)
        tensor_list.append(df_selected.values)
        final_nodes.append(ticker)
        
    feat_tensor = np.stack(tensor_list, axis=1) # (Time, Nodes, 30)
    
    save_npy = 'data/portfolio_data_20_shap_30.npy'
    np.save(save_npy, feat_tensor)
    print(f"Saved SHAP-reduced tensor to {save_npy}. Shape: {feat_tensor.shape}")
    
    # Save Tickers
    with open('data/portfolio_data_20_tickers.txt', 'w') as f:
        f.write(",".join(final_nodes))

if __name__ == '__main__':
    main()
