"""
===============================================================================
ALPACA MARKETS API DATA FETCHER MODULE
===============================================================================

Module Purpose:
    This module handles all interactions with the Alpaca Markets REST API.
    It fetches historical OHLCV (Open, High, Low, Close, Volume) data for
    a given stock symbol over a specified time period.

Key Functions:
    - fetch_stock_data(): Retrieves historical price data from Alpaca API
    - validate_symbol(): Checks if stock symbol is valid
    - format_date(): Ensures proper date formatting for API requests

Data Sources:
    API Provider: Alpaca Markets (https://alpaca.markets)
    Authentication: API Key & Secret Key (free tier available)
    Endpoint: /v1/bars (Stock Bars REST API)

Data Points Returned:
    - Open: Opening price of the day
    - High: Highest price of the day
    - Low: Lowest price of the day
    - Close: Closing price of the day
    - Volume: Number of shares traded
    - Timestamp: Date and time of the bar

Dependencies:
    - alpaca_trade_api: Official Alpaca SDK
    - pandas: Data manipulation and datetime handling
    - pytz: Timezone handling for market hours

Notes:
    - Free tier allows up to 200 requests per minute
    - Historical data available from 2016 onwards
    - Data is adjusted for stock splits and dividends
===============================================================================
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import pytz
from alpaca_trade_api import REST

def fetch_stock_data(symbol, days=365*5):
    """
    Fetch historical stock data from Alpaca Markets API.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'NVDA', 'AAPL', 'TSLA')
        days (int): Number of days of historical data to fetch
                   Default: 5 years (365 * 5 days)
    
    Returns:
        pd.DataFrame: DataFrame with columns [Open, High, Low, Close, Volume]
                     Index is DatetimeIndex with trading dates
    
    Raises:
        ValueError: If API credentials not set or symbol invalid
        Exception: If API request fails
    
    Example:
        >>> df = fetch_stock_data('NVDA', days=252)
        >>> print(df.head())
                         Open   High    Low  Close    Volume
        2024-01-01      100.5  105.0   99.0  104.5   2000000
    
    Workflow:
        1. Validate Alpaca API credentials from environment
        2. Initialize REST client connection
        3. Calculate date range (from_date to to_date)
        4. Request bars data with 1-day timeframe
        5. Convert response to pandas DataFrame
        6. Set timestamp as index
        7. Return cleaned and formatted data
    """
    
    # =========================================================================
    # STEP 1: Validate and retrieve API credentials
    # =========================================================================
    # These should be set as environment variables for security
    api_key = os.getenv('APCA_API_KEY_ID')
    secret_key = os.getenv('APCA_API_SECRET_KEY')
    base_url = 'https://paper-api.alpaca.markets'  # Paper trading (no real money)
    
    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca API credentials not found. "
            "Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables."
        )
    
    # =========================================================================
    # STEP 2: Initialize REST API client
    # =========================================================================
    api = REST(api_key=api_key, secret_key=secret_key, base_url=base_url)
    
    # =========================================================================
    # STEP 3: Calculate date range for historical data
    # =========================================================================
    # End date: today (or last trading day)
    to_date = datetime.now(pytz.timezone('US/Eastern'))
    
    # Start date: calculated from requested number of days
    from_date = to_date - timedelta(days=days)
    
    # Format dates as strings (YYYY-MM-DD) for API
    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')
    
    print(f"[INFO] Fetching data for {symbol} from {from_date_str} to {to_date_str}")
    
    # =========================================================================
    # STEP 4: Request bars (OHLCV data) from Alpaca API
    # =========================================================================
    try:
        # Fetch daily bars (1-day timeframe) - adjust for different frequencies if needed
        bars = api.get_bars(
            symbol,
            timeframe='1D',           # 1-day bars (daily data)
            start=from_date_str,      # Start date
            end=to_date_str,          # End date
            adjustment='all'          # Adjust for splits and dividends
        )
        
        if bars is None or len(bars) == 0:
            raise ValueError(f"No data returned for symbol: {symbol}")
        
        print(f"[INFO] Successfully fetched {len(bars)} trading days")
        
    except Exception as e:
        raise Exception(f"Failed to fetch data from Alpaca API: {str(e)}")
    
    # =========================================================================
    # STEP 5: Convert response to pandas DataFrame
    # =========================================================================
    # Alpaca returns a dictionary-like object; convert to DataFrame
    df = pd.DataFrame({
        'Open': [bar.o for bar in bars],
        'High': [bar.h for bar in bars],
        'Low': [bar.l for bar in bars],
        'Close': [bar.c for bar in bars],
        'Volume': [bar.v for bar in bars],
        'Timestamp': [bar.t for bar in bars]
    })
    
    # =========================================================================
    # STEP 6: Set timestamp as index and ensure proper data types
    # =========================================================================
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df = df.sort_index()  # Ensure chronological order
    
    # Convert data types to float for numerical operations
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
    
    print(f"[SUCCESS] Data fetching complete. Shape: {df.shape}")
    print(f"[INFO] Data range: {df.index[0].date()} to {df.index[-1].date()}")
    
    return df


def validate_symbol(symbol):
    """
    Validate if a stock symbol exists and is tradeable.
    
    Args:
        symbol (str): Stock ticker symbol to validate
    
    Returns:
        bool: True if symbol is valid, False otherwise
    
    Note:
        This is a basic check. Alpaca API will return error if invalid symbol.
    """
    # Basic validation: check if symbol is uppercase alphanumeric
    if not isinstance(symbol, str):
        return False
    
    symbol = symbol.upper().strip()
    if not symbol.isalpha():
        return False
    
    return True


def format_date(date_obj):
    """
    Format datetime object to API-compatible string (YYYY-MM-DD).
    
    Args:
        date_obj (datetime): Python datetime object
    
    Returns:
        str: Formatted date string
    """
    return date_obj.strftime('%Y-%m-%d')
