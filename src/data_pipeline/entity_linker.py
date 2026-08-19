"""Fuzzy entity linker mapping news mentions to portfolio ticker symbols."""
import difflib
from typing import List, Optional

class EntityLinker:
    """
    Links text spans (e.g., "Apple Inc.", "Google") to a fixed set of Tickers.
    """
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        # Basic mapping dictionary for common aliases
        self.alias_map = {
            "APPLE": "AAPL", "AMAZON": "AMZN", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL",
            "MICROSOFT": "MSFT", "TESLA": "TSLA", "NVIDIA": "NVDA", "META": "META",
            "FACEBOOK": "META", "NETFLIX": "NFLX", "JP MORGAN": "JPM", "JPMORGAN": "JPM",
            "BANK OF AMERICA": "BAC", "GOLDMAN SACHS": "GS", "INTEL": "INTC", "AMD": "AMD",
            "COCA COLA": "KO", "PEPSI": "PEP", "WALMART": "WMT", "HOME DEPOT": "HD",
            "DISNEY": "DIS", "EXXON": "XOM", "CHEVRON": "CVX", "PFIZER": "PFE",
            "JOHNSON & JOHNSON": "JNJ", "PROCTER & GAMBLE": "PG", "VISA": "V",
            "MASTERCARD": "MA", "UNITEDHEALTH": "UNH"
        }
        
    def link_entity(self, text: str) -> Optional[str]:
        """
        Maps a text span to a Ticker. Returns None if no confident match.
        """
        if not text:
            return None
            
        text_upper = text.upper().strip()
        
        # 1. Direct Ticker Match
        if text_upper in self.tickers:
            return text_upper
            
        # 2. Alias Match
        # Check if any alias key is a substring of the text
        for alias, ticker in self.alias_map.items():
            if alias in text_upper:
                if ticker in self.tickers:
                    return ticker
                    
        # 3. Fuzzy Match against Tickers (Fallback)
        # Be strict to avoid false positives
        matches = difflib.get_close_matches(text_upper, self.tickers, n=1, cutoff=0.8)
        if matches:
            return matches[0]
            
        return None
