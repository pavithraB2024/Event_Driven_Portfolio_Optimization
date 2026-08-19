"""Dynamic heterogeneous graph construction from correlation and event edges."""
import torch
import numpy as np
import pandas as pd
import os
from types import SimpleNamespace

class DynamicGraphBuilder:
    def __init__(self, data_dir, portfolio_name, news_path, decay_rate=0.1, inject_sentiment=False):
        self.data_dir = data_dir
        self.portfolio_name = portfolio_name
        
        # 3. Ticker Map
        tickers_path = os.path.join(data_dir, f"{portfolio_name}_tickers.txt")
        with open(tickers_path, 'r') as f:
            self.tickers = f.read().strip().split(',')
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}
        
        self.num_stock_nodes = len(self.tickers)
        
        # --- Heterogeneous Node Definition ---
        # Extended Sector Map supporting both 28-stocks and Dataset B (Global-30)
        self.sector_map = {
            # Technology
            "AAPL": "Tech", "CSCO": "Tech", "IBM": "Tech", "INTC": "Tech", "MSFT": "Tech", 
            "ORCL": "Tech", "NVDA": "Tech", "ACN": "Tech",
            # Consumer Discretionary
            "AMZN": "Consumer", "HD": "Consumer", "KO": "Consumer", "WMT": "Consumer",
            "TSLA": "Consumer", "MCD": "Consumer", "NKE": "Consumer", "SBUX": "Consumer",
            # Industrials
            "BA": "Industrial", "CAT": "Industrial", "GE": "Industrial", "HON": "Industrial", 
            "MMM": "Industrial", "UNP": "Industrial", "FDX": "Industrial",
            # Financials
            "BAC": "Finance", "JPM": "Finance", "TD": "Finance", "MFC": "Finance",
            "V": "Finance", "MA": "Finance", "WFC": "Finance", "GS": "Finance", "MS": "Finance",
            # Energy
            "BP": "Energy", "CVX": "Energy", "SHEL": "Energy", "ENB": "Energy",
            "XOM": "Energy", "SLB": "Energy", "COP": "Energy",
            # Healthcare
            "GILD": "Healthcare", "JNJ": "Healthcare", "MRK": "Healthcare", "PFE": "Healthcare",
            "UNH": "Healthcare", "ABBV": "Healthcare", "LLY": "Healthcare", "TMO": "Healthcare",
            # Communication Services
            "VZ": "Comm", "T": "Comm", "GOOGL": "Comm", "GOOG": "Comm", "META": "Comm", "NFLX": "Comm",
            # Consumer Staples
            "PG": "Consumer", "KO": "Consumer", "PEP": "Consumer", "COST": "Consumer", "WMT": "Consumer",
            # Materials
            "LIN": "Materials", "APD": "Materials", "ECL": "Materials", "FCX": "Materials",
            # Utilities
            "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities"
        }
        
        # 1. Load Static Graph (Adjacency)
        adj_path = os.path.join(data_dir, f"{portfolio_name}_adj.npy")
        if not os.path.exists(adj_path):
            # If main adj missing, init zero
            self.static_adj_stock = np.eye(self.num_stock_nodes)
        else:
            self.static_adj_stock = np.load(adj_path)

        # 2. Dynamic Sector Assignment for ETFs and Indexes
        # MUST happen BEFORE sector index mapping to include all sectors
        self._assign_sectors_to_etfs_and_indexes()

        # 3. CRITICAL FIX: Only create sector nodes for sectors that exist in THIS portfolio
        # Don't create sectors for tickers not in our portfolio
        active_sectors = set()
        for ticker in self.tickers:
            if ticker in self.sector_map:
                active_sectors.add(self.sector_map[ticker])
            else:
                # Assign to 'Other' if no mapping
                active_sectors.add('Other')
        
        # Only keep sectors that are actually used by our tickers
        self.sectors = sorted(list(active_sectors))
        self.sector_to_idx = {s: i + self.num_stock_nodes for i, s in enumerate(self.sectors)}
        self.num_sector_nodes = len(self.sectors)
        
        print(f"Graph Builder: Created {self.num_sector_nodes} sector nodes for {portfolio_name}: {self.sectors}")
        
        # Macro Node (1 Global Context Node)
        self.macro_idx = self.num_stock_nodes + self.num_sector_nodes

        # Event Nodes (Buffer)
        self.num_event_nodes = 5
        self.event_start_idx = self.macro_idx + 1

        self.num_nodes = self.event_start_idx + self.num_event_nodes # Total Nodes

        # Event Embeddings (Static Dictionary for now, ideally learned)
        self.event_type_embeddings = {
            "Mergers and Acquisitions": np.random.randn(6),
            "Earnings Report": np.random.randn(6),
            "Stock Movement": np.random.randn(6),
            "Legal and Regulation": np.random.randn(6),
            "Product Launch": np.random.randn(6),
            "Macroeconomic": np.random.randn(6),
            "Analyst Rating": np.random.randn(6),
            "Supply Chain": np.random.randn(6)
        }

        # Build Mixed Static Adjacency [Total, Total]
        self.mixed_adj_template = np.zeros((self.num_nodes, self.num_nodes))

        # (a) Copy Stock-Stock connections
        self.mixed_adj_template[:self.num_stock_nodes, :self.num_stock_nodes] = self.static_adj_stock

        # (b) Add Stock -> Sector Edges (Bidirectional)
        for t, s in self.sector_map.items():
            if t in self.ticker_to_idx:
                t_idx = self.ticker_to_idx[t]
                s_idx = self.sector_to_idx[s]
                self.mixed_adj_template[t_idx, s_idx] = 1.0
                self.mixed_adj_template[s_idx, t_idx] = 1.0

        # (c) Add Macro -> All Edges (Global Influence)
        # Connect Macro to all Sectors (Hierarchy)
        for s_idx in self.sector_to_idx.values():
             self.mixed_adj_template[self.macro_idx, s_idx] = 1.0
             self.mixed_adj_template[s_idx, self.macro_idx] = 1.0

        # Also connect Macro to all Stocks directly
        self.mixed_adj_template[self.macro_idx, :self.num_stock_nodes] = 0.5
        self.mixed_adj_template[:self.num_stock_nodes, self.macro_idx] = 0.5

        # 3. Load Processed News
        if os.path.exists(news_path) and news_path:
            self.news_df = pd.read_csv(news_path)
            if not self.news_df.empty:
                self.news_df['date'] = pd.to_datetime(self.news_df['date'], utc=True)
                # --- Defer news published after 4:00 PM EST to the next day ---
                try:
                    news_eastern = self.news_df['date'].dt.tz_convert('US/Eastern')
                    mask_post_close = news_eastern.dt.hour >= 16
                    self.news_df['adjusted_date'] = self.news_df['date'].copy()
                    self.news_df.loc[mask_post_close, 'adjusted_date'] = self.news_df.loc[mask_post_close, 'date'] + pd.Timedelta(days=1)
                except Exception as e:
                    print(f"[ERROR] Failed to calculate adjusted news dates: {e}")
                    self.news_df['adjusted_date'] = self.news_df['date']
        else:
            if news_path:
                print(f"Warning: News data not found at {news_path}. Using static graph only.")
            else:
                print("Warning: No news path provided. Using static graph only.")
            self.news_df = pd.DataFrame()

        # =====================================================================
        # Solution 2: Temporal Decay Event Memory (Nordansjo et al., 2025)
        # =====================================================================
        # Persistent per-stock sentiment state that decays exponentially.
        # Formula: S_t = S_new + S_{t-1} * exp(-decay_rate)
        # A decay_rate of 0.1 means ~30 trading days half-life.
        # This turns sparse news signals into continuous sentiment waves.
        # =====================================================================
        self.event_memory = np.zeros(self.num_stock_nodes)     # per-stock
        self.sector_memory = np.zeros(self.num_sector_nodes)   # per-sector
        self.decay_rate = decay_rate  # configurable for sensitivity sweep (default=0.1)
        self.inject_sentiment = inject_sentiment
        self.last_date = None
        print(f"Temporal Decay: Initialized with decay_rate={self.decay_rate} "
              f"(half-life ~{int(np.log(2)/self.decay_rate)} trading days)")

    def _assign_sectors_to_etfs_and_indexes(self):
        """
        Dynamically assign sector types to ETFs and market indexes not in the base sector_map.
        This ensures proper graph construction for Dataset B (Global-30).
        """
        # ETF Sector Mappings
        etf_sector_map = {
            # Technology Sector ETFs
            "XLK": "Tech", "QQQ": "Tech",
            # Financial Sector ETFs
            "XLF": "Finance",
            # Energy Sector ETFs
            "XLE": "Energy",
            # Healthcare Sector ETFs
            "XLV": "Healthcare",
            # Industrial Sector ETFs
            "XLI": "Industrial",
            # Consumer Discretionary ETFs
            "XLY": "Consumer",
            # Consumer Staples ETFs
            "XLP": "Consumer",
            # Materials Sector ETFs
            "XLB": "Materials",
            # Utilities Sector ETFs
            "XLU": "Utilities",
            # Broad Market ETFs (assign to 'Market' pseudo-sector)
            "SPY": "Market", "IWM": "Market", "VTI": "Market",
            # Commodity ETFs (assign to 'Commodity' pseudo-sector)
            "GLD": "Commodity", "SLV": "Commodity", "USO": "Commodity", "DBA": "Commodity",
            # Bond/Fixed Income ETFs (assign to 'FixedIncome' pseudo-sector)
            "TLT": "FixedIncome", "IEF": "FixedIncome", "SHY": "FixedIncome", 
            "HYG": "FixedIncome", "LQD": "FixedIncome",
            # Volatility Products (assign to 'Volatility' pseudo-sector)
            "VIXY": "Volatility", "VIX": "Volatility",
            # Currency ETFs (assign to 'Currency' pseudo-sector)
            "UUP": "Currency", "FXE": "Currency", "FXY": "Currency",
            # International Indexes (assign to 'International' pseudo-sector)
            "^N225": "International", "^FTSE": "International", "000001.SS": "International",
            "^HANG": "International", "^KS11": "International",
            # U.S. Market Indexes (assign to 'Market' pseudo-sector)
            "^GSPC": "Market", "^DJI": "Market", "^IXIC": "Market",
            # Treasury Yields (assign to 'FixedIncome' pseudo-sector)
            "^FVX": "FixedIncome", "^TNX": "FixedIncome", "^TYX": "FixedIncome",
        }
        
        # Merge with base sector_map
        self.sector_map.update(etf_sector_map)
        
        # For any remaining tickers without sector assignment, assign to 'Other'
        for ticker in self.tickers:
            if ticker not in self.sector_map:
                self.sector_map[ticker] = "Other"
                print(f"Note: Assigned '{ticker}' to 'Other' sector (no mapping found)")

    def resize_to_match_portfolio(self, num_active_stocks):
        """
        Adjusts the internal graph structure to match the actual number of portfolio assets.
        This is crucial when the ticker definition file includes extra market indexes that are not
        part of the tradable portfolio (e.g., matching '28stocks' from 43 to 28).
        """
        if num_active_stocks == self.num_stock_nodes:
            return

        print(f"GraphBuilder: Resizing from {self.num_stock_nodes} to {num_active_stocks} stocks.")
        
        # 1. Update Tickers & Count
        self.tickers = self.tickers[:num_active_stocks]
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}
        self.num_stock_nodes = num_active_stocks
        
        # 2. Re-calculate Offsets
        # Sector/Macro nodes shift down
        self.sector_to_idx = {s: i + self.num_stock_nodes for i, s in enumerate(self.sectors)}
        self.macro_idx = self.num_stock_nodes + self.num_sector_nodes
        self.max_idx_pre_event = self.macro_idx + 1 # +1 for macro
        self.event_start_idx = self.max_idx_pre_event
        self.num_nodes = self.event_start_idx + self.num_event_nodes

        # 3. Rebuild Adjacency Template
        # Re-initialize
        self.mixed_adj_template = np.zeros((self.num_nodes, self.num_nodes))
        
        # Resize Static Stock Adj
        if self.static_adj_stock.shape[0] > num_active_stocks:
             self.static_adj_stock = self.static_adj_stock[:num_active_stocks, :num_active_stocks]
        elif self.static_adj_stock.shape[0] < num_active_stocks:
             # Should not happen given logic, but pad if needed
             pass
             
        self.mixed_adj_template[:self.num_stock_nodes, :self.num_stock_nodes] = self.static_adj_stock

        # Re-Add Edges
        for t, s in self.sector_map.items():
            if t in self.ticker_to_idx:
                t_idx = self.ticker_to_idx[t]
                s_idx = self.sector_to_idx[s]
                self.mixed_adj_template[t_idx, s_idx] = 1.0
                self.mixed_adj_template[s_idx, t_idx] = 1.0
        
        for s_idx in self.sector_to_idx.values():
             self.mixed_adj_template[self.macro_idx, s_idx] = 1.0
             self.mixed_adj_template[s_idx, self.macro_idx] = 1.0
        
        self.mixed_adj_template[self.macro_idx, :self.num_stock_nodes] = 0.5
        self.mixed_adj_template[:self.num_stock_nodes, self.macro_idx] = 0.5

        # Resize event memory to match new stock count
        self.event_memory = np.zeros(self.num_stock_nodes)
        self.sector_memory = np.zeros(self.num_sector_nodes)

    # ==================================================================
    # Solution 1: Sector-Level Sentiment Aggregation (Yan et al., 2025)
    # ==================================================================
    def _compute_sector_sentiment(self, day_news):
        """
        Computes aggregated sentiment per sector from today's news.
        Stocks with zero coverage inherit sector-level sentiment through
        GNN message passing from their Sector Node.

        Returns:
            sector_sentiment: np.array [num_sector_nodes] with avg sentiment per sector
            stock_sentiment:  np.array [num_stock_nodes]  with per-stock raw sentiment
        """
        stock_sentiment = np.zeros(self.num_stock_nodes)
        sector_sentiment = np.zeros(self.num_sector_nodes)
        sector_counts = {s: 0 for s in self.sectors}
        sector_sums = {s: 0.0 for s in self.sectors}

        if day_news.empty:
            return sector_sentiment, stock_sentiment

        # 1. Aggregate per-stock sentiment from today's news
        for _, row in day_news.iterrows():
            ticker = row.get('entity_a') or row.get('stock')
            sent = row.get('sentiment_scalar', 0.0)
            if pd.isna(sent):
                sent = 0.0

            if ticker and ticker in self.ticker_to_idx:
                idx = self.ticker_to_idx[ticker]
                stock_sentiment[idx] += sent

            # Also accumulate into sector
            if ticker and ticker in self.sector_map:
                sector = self.sector_map[ticker]
                if sector in sector_counts:
                    sector_sums[sector] += sent
                    sector_counts[sector] += 1

        # 2. Average sector sentiment
        for i, s in enumerate(self.sectors):
            if sector_counts[s] > 0:
                sector_sentiment[i] = sector_sums[s] / sector_counts[s]

        return sector_sentiment, stock_sentiment

    # ==================================================================
    # Solution 2: Apply Temporal Decay to Event Memory
    # ==================================================================
    def _update_event_memory(self, stock_sentiment, sector_sentiment):
        """
        Updates the persistent event memory with temporal decay.
        Formula: S_t = S_new + S_{t-1} * exp(-decay_rate)

        This ensures that:
        - Stocks with frequent news have strong, up-to-date signals
        - Stocks with sparse news retain a fading "echo" of past events
        - Stocks with zero news still get signal via correlation imputation (Solution 3)
        """
        decay_factor = np.exp(-self.decay_rate)

        # Decay existing memory and add new signal
        self.event_memory = stock_sentiment + self.event_memory * decay_factor
        self.sector_memory = sector_sentiment + self.sector_memory * decay_factor

        return self.event_memory.copy(), self.sector_memory.copy()

    # ==================================================================
    # Solution 3: Correlation-Based Imputation (Yang, 2023 - TC-MAC)
    # ==================================================================
    def _impute_missing_sentiment(self, decayed_memory):
        """
        For stocks with zero sentiment (no news ever or fully decayed),
        impute from correlated neighbors using the static adjacency matrix.

        Formula: imputed_i = sum_j(adj[i,j] * sentiment_j) / sum_j(adj[i,j])

        This ensures that even stocks with ZERO articles in the raw dataset
        receive meaningful sentiment signals from their price-correlated peers.
        """
        zero_mask = np.abs(decayed_memory) < 1e-8  # stocks with no signal

        if not np.any(zero_mask):
            return decayed_memory  # No imputation needed

        # Use the stock-stock portion of the adjacency matrix
        adj = self.static_adj_stock.copy()
        np.fill_diagonal(adj, 0)  # Don't use self-loop for imputation

        # Weighted average of neighbors' sentiment
        neighbor_sum = adj @ decayed_memory
        degree = adj.sum(axis=1)
        degree[degree == 0] = 1.0  # avoid division by zero

        imputed = neighbor_sum / degree

        # Only fill in where the stock has zero signal
        result = decayed_memory.copy()
        result[zero_mask] = imputed[zero_mask]

        return result

    def get_graph(self, date, market_features, lambda_val=0.5):
        """
        Constructs the heterogeneous graph for a specific date.
        market_features: [Num_Stocks, feat_dim]
        lambda_val: Mixing Factor. 0=Base Only, 1=Event Only.
        Formula: G' = (1 - lambda) * G_base + lambda * G_event

        Enhanced with:
        - Solution 1: Sector-Level Sentiment Aggregation (Yan et al., 2025)
        - Solution 2: Temporal Decay Event Memory (Nordansjo et al., 2025)
        - Solution 3: Correlation-Based Imputation (Yang, 2023)
        """
        # 1. Feature Aggregation (Heterogeneous -> Unified Space)
        # Stock Features
        x_stocks = market_features.clone()
        
        # Ensure x_stocks matches the expected number of active stocks (handle heterogenous inputs)
        if x_stocks.shape[0] > self.num_stock_nodes:
            x_stocks = x_stocks[:self.num_stock_nodes]
            
        feat_dim = x_stocks.shape[1]
        
        # Sector Features (Mean Pooling of constituents)
        x_sectors = torch.zeros((self.num_sector_nodes, feat_dim))
        sector_counts = {s: 0 for s in self.sectors}
        
        for t, s in self.sector_map.items():
            if t in self.ticker_to_idx:
                t_idx = self.ticker_to_idx[t]
                s_idx = self.sector_to_idx[s] - self.num_stock_nodes # relative index
                x_sectors[s_idx] += x_stocks[t_idx]
                sector_counts[s] += 1
                
        for i, s in enumerate(self.sectors):
            if sector_counts[s] > 0:
                x_sectors[i] /= sector_counts[s]
                
        # Macro Feature (Global Mean Pooling for now)
        # Ideally this comes from a dedicated macro data source (VIX, Fed Rate)
        x_macro = torch.mean(x_stocks, dim=0, keepdim=True) # [1, F]
        
        # Event Features [Num_Event, F]
        x_events = torch.zeros((self.num_event_nodes, feat_dim))
        
        # 2. Dynamic Adjacency Construction
        
        # G_base (Static + Sector + Macro)
        # Assuming mixed_adj_template is unweighted binary (0/1) or base weights
        base_adj = self.mixed_adj_template.copy()
        
        # G_event (Transient News Edges)
        event_adj = np.zeros((self.num_nodes, self.num_nodes))
        
        # =====================================================================
        # NEWS PROCESSING: Solutions 1 + 2 + 3
        # =====================================================================
        day_news = pd.DataFrame()

        if not self.news_df.empty:
            target_date = pd.to_datetime(date)
            # Filter news for date using adjusted date to exclude post-4:00 PM EST news
            if 'adjusted_date' in self.news_df.columns:
                day_news = self.news_df[self.news_df['adjusted_date'].dt.date == target_date.date()].copy()
            else:
                day_news = self.news_df[self.news_df['date'].dt.date == target_date.date()].copy()

        # --- Solution 1: Compute sector-level and stock-level sentiment ---
        sector_sentiment, stock_sentiment = self._compute_sector_sentiment(day_news)

        # --- Solution 2: Apply temporal decay to event memory ---
        decayed_stock, decayed_sector = self._update_event_memory(
            stock_sentiment, sector_sentiment
        )

        # --- Solution 3: Impute missing stock sentiment from correlated peers ---
        imputed_stock = self._impute_missing_sentiment(decayed_stock)

        # --- Inject sentiment signals into node features ---
        # Note: We must inject sentiment to support sensitivity sweeps of decay_rate.
        # We clip to [-5, 5] to prevent NaN explosions when decay_rate is extremely small.
        # Guarded by inject_sentiment flag to protect backward compatibility for the main training script.
        if self.inject_sentiment:
            imputed_stock = np.clip(imputed_stock, -5.0, 5.0)
            decayed_sector = np.clip(decayed_sector, -5.0, 5.0)

            if feat_dim > 1:
                for i in range(self.num_stock_nodes):
                    x_stocks[i, -1] = x_stocks[i, -1] + imputed_stock[i]

            # Inject sector sentiment into sector node features (last dim)
            if feat_dim > 1:
                for i in range(self.num_sector_nodes):
                    x_sectors[i, -1] = x_sectors[i, -1] + decayed_sector[i]

        # =====================================================================
        # EVENT NODE CONSTRUCTION (Original Logic - Enhanced)
        # =====================================================================
        if not day_news.empty:
            # Sort by Confidence (Magnitude) or Sentiment Magnitude
            if 'event_confidence' not in day_news.columns:
                 day_news['event_confidence'] = 1.0
            
            day_news['sort_metric'] = day_news['event_confidence'] * day_news['sentiment_scalar'].abs()
            day_news = day_news.sort_values('sort_metric', ascending=False)
            
            # Select Top Events
            top_events = day_news.head(self.num_event_nodes)
            
            # Fill Event Nodes and Edges
            for i, (_, row) in enumerate(top_events.iterrows()):
                ev_node_idx = self.event_start_idx + i
                
                # Features: [Embedding(6), Sentiment(1), Confidence(1), Padding...]
                etype = row.get('event_type', 'Stock Movement')
                emb = self.event_type_embeddings.get(etype, np.zeros(6))
                
                sent = row.get('sentiment_scalar', 0.0)
                conf = row.get('event_confidence', 1.0)
                
                # Vector construction
                ev_vec = np.concatenate([emb, [sent], [conf]])
                
                # Pad/Truncate to feat_dim
                if len(ev_vec) < feat_dim:
                    ev_vec = np.pad(ev_vec, (0, feat_dim - len(ev_vec)), 'constant')
                else:
                    ev_vec = ev_vec[:feat_dim]
                    
                x_events[i] = torch.tensor(ev_vec, dtype=torch.float32)
                
                # Edges: Event <-> Entities
                # Entity A
                if 'entity_a' in row:
                    t_a = row.get('entity_a') or row.get('stock')
                    if t_a in self.ticker_to_idx:
                        idx_a = self.ticker_to_idx[t_a]
                        event_adj[ev_node_idx, idx_a] = 1.0 
                        event_adj[idx_a, ev_node_idx] = 1.0 # Undirected Influence
                        
                # Entity B
                if 'entity_b' in row and pd.notna(row['entity_b']):
                    t_b = row['entity_b']
                    if t_b in self.ticker_to_idx:
                        idx_b = self.ticker_to_idx[t_b]
                        event_adj[ev_node_idx, idx_b] = 1.0
                        event_adj[idx_b, ev_node_idx] = 1.0
            
        # Combine Features
        x_combined = torch.cat([x_stocks, x_sectors, x_macro, x_events], dim=0) # [Total_Nodes, F]

        # 3. Dynamic Mixing Formula
        # Fused = (1 - lambda) * Base + (lambda) * Event
        # Note: If base edge and event edge align, weights sum up.
        # If no event edge, it decays base edge strength? 
        # The prompt says: G' = (1 - lambda) G_base + lambda G_event
        # This implies standard edges are weakened if lambda is high (high event focus).
        
        final_adj = (1.0 - lambda_val) * base_adj + (lambda_val) * event_adj
        
        # 4. Node Types Construction (for HGT)
        # 0: Stock, 1: Sector, 2: Macro, 3: Event
        types_stock = torch.zeros(self.num_stock_nodes, dtype=torch.long)
        types_sector = torch.ones(self.num_sector_nodes, dtype=torch.long)
        types_macro = torch.full((1,), 2, dtype=torch.long)
        types_event = torch.full((self.num_event_nodes,), 3, dtype=torch.long)
        
        node_types = torch.cat([types_stock, types_sector, types_macro, types_event], dim=0) # [Total_Nodes]
        
        # Return Dense Adjacency Matrix + Node Types
        return SimpleNamespace(
            x=x_combined, 
            adj=torch.tensor(final_adj, dtype=torch.float32),
            node_types=node_types
        )
