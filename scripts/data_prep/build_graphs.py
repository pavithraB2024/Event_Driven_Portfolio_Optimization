
import numpy as np
import os
import yfinance as yf

# --- Configuration ---
DATA_DIR = "data"

# --- 1. Static Graph for Dataset A (Sun et al.) ---

def build_static_graph(portfolio_name):
    """
    Builds the Static Financial Graph linking Stocks <-> Sectors <-> Market.
    Reads nodes from {portfolio_name}_tickers.txt
    """
    print(f"\nBuilding Static Heterogeneous Graph for {portfolio_name}...")
    
    # 1. Load Node List
    tickers_path = os.path.join(DATA_DIR, f"{portfolio_name}_tickers.txt")
    with open(tickers_path, 'r') as f:
        nodes = f.read().split(',')
        
    n_nodes = len(nodes)
    node_map = {name: i for i, name in enumerate(nodes)}
    
    adj = np.eye(n_nodes)
    
    # 2. Define High-Level Node Groups
    # Market Indexes (The "Root" Nodes)
    market_nodes = ["^GSPC", "^N225", "^FTSE", "000001.SS", "^FVX", "^TNX"]
    
    # Sector Map
    sector_map = {
        'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'CSCO': 'XLK', 'ORCL': 'XLK', 'IBM': 'XLK', 'INTC': 'XLK', 'ACN': 'XLK',
        'AMZN': 'XLY', 'TSLA': 'XLY', 'HD': 'XLY', 'MCD': 'XLY', 'META': 'XLC', 'NFLX': 'XLC', 'VZ': 'XLC', 'GOOGL': 'XLC', 'GOOG': 'XLC',
        'WMT': 'XLP', 'KO': 'XLP', 'PG': 'XLP', 
        'JPM': 'XLF', 'BAC': 'XLF', 'WFC': 'XLF', 'BRK-B': 'XLF', 
        'JNJ': 'XLV', 'UNH': 'XLV', 'PFE': 'XLV', 'MRK': 'XLV', 'GILD': 'XLV',
        'XOM': 'XLE', 'CVX': 'XLE', 'BP': 'XLE', 'SHEL': 'XLE', 
        'GE': 'XLI', 'CAT': 'XLI', 'BA': 'XLI', 'MMM': 'XLI', 'HON': 'XLI',
        'LIN': 'XLB', 'FCX': 'XLB',
        'NEE': 'XLU', 'DUK': 'XLU'
    }
    
    edge_count = 0
    
    # 3. Create Edges
    for i, node in enumerate(nodes):
        
        # Case A: Node is a Stock
        if node in sector_map:
            # Connect Stock <-> Sector
            sector_etf = sector_map[node]
            if sector_etf in node_map:
                j = node_map[sector_etf]
                # Undirected Edge
                adj[i, j] = 1
                adj[j, i] = 1
                edge_count += 1
                
            # Connect Stock <-> Main Market Index (^GSPC)
            # Paper Fig 5 shows stocks connected to S&P 500
            if "^GSPC" in node_map:
                k = node_map["^GSPC"]
                adj[i, k] = 1
                adj[k, i] = 1
                
        # Case B: Node is a Sector Fund (e.g., XLK)
        # Connect Sector <-> Market Index
        # (Sectors are constituents of the Market)
        if node.startswith('XL'):
            for mkt in market_nodes:
                if mkt in node_map:
                    k = node_map[mkt]
                    adj[i, k] = 1
                    adj[k, i] = 1
                    
        # Case C: Market <-> Market Connectivity?
        # Usually Market nodes might be fully connected to each other to share global context
        # "Graph is connected with the depth of two... S&P 500 has largest node degree"
        if node in market_nodes:
            # Connect to other markets?
            for mkt in market_nodes:
                 if mkt != node and mkt in node_map:
                     k = node_map[mkt]
                     adj[i, k] = 1
    
    print("Graph Construction Complete.")
    print(f"Nodes: {n_nodes}")
    print(f"Total Edges (inc self/symmetric): {np.sum(adj)}")
    
    save_path = os.path.join(DATA_DIR, f"{portfolio_name}_adj.npy")
    np.save(save_path, adj)
    print(f"Saved to {save_path}")

def main():
    # 1. Dataset A (Top 10)
    build_static_graph("top10")
    
    # 2. Dataset C (Multi-Asset 20)
    # Correlation Graph logic remains valid (threshold based)
    # We call the function from previous logic if we kept it, 
    # but I need to make sure I didn't delete the definition.
    # The REPLACE block replaced lines 10-110. I need to keep the `build_correlation_graph` definition.
    # I will re-include it below to be safe.
    
    build_correlation_graph("portfolio_data_20", threshold=0.5)

# --- Re-stating Helper for Dataset C to ensure script validity ---
def build_correlation_graph(portfolio_name, threshold=0.5):
    print(f"\nBuilding Correlation Graph for {portfolio_name}...")
    os.path.join(DATA_DIR, f"{portfolio_name}_shap_30.npy")
    tickers_path = os.path.join(DATA_DIR, f"{portfolio_name}_tickers.txt")
    
    # ... (Logic identical to previous, just ensuring function exists)
    if not os.path.exists(tickers_path): return
    
    with open(tickers_path, 'r') as f:
         valid_tickers = f.read().split(',')
         
    # Quick Correlation via yfinance fetch (best for accuracy)
    print("Fetching close data for correlation...")
    data = yf.download(valid_tickers, start="2012-01-01", end="2022-04-01", progress=False)['Close']
    
    returns = data.pct_change().fillna(0)
    corr = returns.corr().abs()
    
    adj = (corr.values > threshold).astype(int)
    np.fill_diagonal(adj, 1)
    
    save_path = os.path.join(DATA_DIR, f"{portfolio_name}_adj.npy")
    np.save(save_path, adj)
    print(f"Saved correlation graph to {save_path}. Edges: {np.sum(adj)}")

if __name__ == "__main__":
    main()
