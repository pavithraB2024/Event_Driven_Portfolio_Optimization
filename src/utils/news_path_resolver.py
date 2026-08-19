"""
Utility functions for data pipeline scripts.
"""
import os

def get_news_path(data_dir: str, portfolio_name: str, fallback: bool = True) -> str:
    """
    Get the news file path with portfolio-specific naming and fallback support.
    
    Args:
        data_dir: Base data directory path
        portfolio_name: Name of the portfolio (e.g., '28stocks')
        fallback: Whether to try fallback paths if primary not found
        
    Returns:
        Path to the news file
        
    Priority:
        1. portfolio-specific: news_with_events_{portfolio_name}.csv
        2. generic: news_with_events.csv (if fallback=True)
        3. Returns None if not found
    """
    processed_dir = os.path.join(data_dir, "processed")
    
    # Primary: Portfolio-specific path
    primary_path = os.path.join(processed_dir, f"news_with_events_{portfolio_name}.csv")
    
    if os.path.exists(primary_path):
        return primary_path
    
    # Fallback: Generic path
    if fallback:
        generic_path = os.path.join(processed_dir, "news_with_events.csv")
        if os.path.exists(generic_path):
            print(f"Note: Using generic news path: {generic_path}")
            return generic_path
    
    # Not found
    return None


def validate_news_path(news_path: str, portfolio_name: str) -> str:
    """
    Validate and report news file status.
    
    Args:
        news_path: Path to news file (can be None)
        portfolio_name: Name of the portfolio for messaging
        
    Returns:
        Status message
    """
    if news_path is None:
        return f"Warning: News data not found for {portfolio_name}. Using static graph only."
    elif os.path.exists(news_path):
        return f"Loaded news from: {news_path}"
    else:
        return f"Warning: News file does not exist at {news_path}. Using static graph only."
