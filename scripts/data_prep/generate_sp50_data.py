import os
import sys

# Add repo root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(REPO_ROOT)

from scripts.data_prep.prepare_data import process_corpus

def main():
    DATA_DIR = os.path.join(REPO_ROOT, "data")
    tickers_path = os.path.join(DATA_DIR, "sp50_pit2013_tickers.txt")
    
    if not os.path.exists(tickers_path):
        print(f"Error: Could not find {tickers_path}")
        sys.exit(1)
        
    with open(tickers_path, "r") as f:
        tickers_str = f.read().strip()
        
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    print(f"Loaded {len(tickers)} tickers for sp50_pit2013.")
    
    process_corpus(tickers, "sp50_pit2013")

if __name__ == "__main__":
    main()
