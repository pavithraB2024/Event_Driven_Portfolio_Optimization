"""
Data Download Module
Downloads financial data from Yahoo Finance and FRED
"""

import pandas as pd
import yfinance as yf
from typing import List, Optional
from datetime import datetime
import os


class DataDownloader:
    """Download financial market data from multiple sources."""

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data downloader.

        Args:
            data_dir: Directory to save raw data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download_asset_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Download asset price data from Yahoo Finance.

        Args:
            tickers: List of ticker symbols (e.g., ['SPY', 'TLT', 'GLD'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data frequency ('1d', '1wk', '1mo')

        Returns:
            DataFrame with adjusted close prices for all assets
        """
        print(f"Downloading {len(tickers)} assets from {start_date} to {end_date}...")

        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=True
        )

        # Extract close prices
        if len(tickers) == 1:
            prices = data[['Close']].copy()
            prices.columns = tickers
        else:
            prices = data['Close'].copy()

        # Save to CSV
        filepath = os.path.join(self.data_dir, f"asset_prices_{interval}.csv")
        prices.to_csv(filepath)
        print(f"Saved to {filepath}")

        return prices

    def download_vix(self, start_date: str, end_date: str) -> pd.Series:
        """
        Download VIX (volatility index) data.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            Series of VIX values
        """
        print("Downloading VIX data...")
        vix_data = yf.download('^VIX', start=start_date, end=end_date, auto_adjust=True)
        vix = vix_data['Close']

        # Save to CSV
        filepath = os.path.join(self.data_dir, "vix.csv")
        vix.to_csv(filepath)
        print(f"Saved to {filepath}")

        return vix

    def download_treasury_rates(self, start_date: str, end_date: str) -> pd.Series:
        """
        Download 10-Year Treasury rates from FRED.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            Series of treasury rates
        """
        try:
            from fredapi import Fred
            import os

            # Requires FRED API key in environment variable
            fred_api_key = os.getenv('FRED_API_KEY')
            if not fred_api_key:
                print("Warning: FRED_API_KEY not found. Using mock data.")
                return self._mock_treasury_rates(start_date, end_date)

            fred = Fred(api_key=fred_api_key)
            rates = fred.get_series('DGS10', start_date, end_date)

            # Save to CSV
            filepath = os.path.join(self.data_dir, "treasury_10y.csv")
            rates.to_csv(filepath)
            print(f"Saved to {filepath}")

            return rates
        except Exception as e:
            print(f"Error downloading FRED data: {e}")
            print("Using mock treasury rates.")
            return self._mock_treasury_rates(start_date, end_date)

    def _mock_treasury_rates(self, start_date: str, end_date: str) -> pd.Series:
        """Create mock treasury rate data for testing."""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        # Mock rates between 1% and 4%
        rates = pd.Series(
            data=2.5 + 0.5 * pd.Series(range(len(date_range))).pct_change().fillna(0).cumsum(),
            index=date_range,
            name='DGS10'
        )
        return rates.clip(lower=0.5, upper=5.0)

    def download_all(
        self,
        asset_tickers: List[str],
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None
    ) -> dict:
        """
        Download all required data for the project.

        Args:
            asset_tickers: List of asset tickers
            start_date: Start date
            end_date: End date (defaults to today)

        Returns:
            Dictionary with all downloaded data
        """
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')

        data = {
            'prices': self.download_asset_data(asset_tickers, start_date, end_date),
            'vix': self.download_vix(start_date, end_date),
            'treasury': self.download_treasury_rates(start_date, end_date)
        }

        print("\nDownload complete!")
        print(f"Assets: {list(data['prices'].columns)}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Total trading days: {len(data['prices'])}")

        return data


if __name__ == "__main__":
    # Example usage
    downloader = DataDownloader()

    # Default asset universe: Stocks, Bonds, Gold, Bitcoin
    tickers = ['SPY', 'TLT', 'GLD', 'BTC-USD']

    data = downloader.download_all(
        asset_tickers=tickers,
        start_date="2010-01-01",
        end_date="2025-01-01"
    )

    print("\nData shapes:")
    for key, value in data.items():
        print(f"{key}: {value.shape}")
