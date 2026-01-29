"""
===============================================================================
SIMPLE MOVING AVERAGE (SMA) INDICATOR
===============================================================================

Technical Indicator: Simple Moving Average
Definition:
    The Simple Moving Average is the arithmetic mean of the closing price
    over a specified number of trading periods. It is used to smooth out
    short-term price fluctuations and identify longer-term trends.

Mathematical Formula:
    SMA(n) = (Close[t] + Close[t-1] + ... + Close[t-n+1]) / n
    
    Where:
        n = number of periods (typically 20, 50, or 200)
        Close[t] = closing price at time t

Interpretation:
    - SMA above price: Indicates downtrend (bearish signal)
    - SMA below price: Indicates uptrend (bullish signal)
    - Crossovers: SMA 50 crossing above SMA 200 is a bullish signal
    - Support/Resistance: Price tends to bounce off SMA levels

Use Cases in ML:
    - Feature for machine learning models
    - Identifies trend direction
    - Smooths noise in time-series data
    - Normalizes price movement

Common Periods:
    - 20-day SMA: Short-term trend
    - 50-day SMA: Intermediate trend
    - 200-day SMA: Long-term trend

Advantages:
    - Easy to calculate
    - Smooth and responsive
    - Widely used and understood

Limitations:
    - Lagging indicator (responds to price changes with delay)
    - Equally weights all periods (unlike exponential MA)
    - May produce false signals in choppy markets

Dependencies:
    - pandas: For rolling window calculations
    - numpy: For numerical operations
===============================================================================
"""

import pandas as pd
import numpy as np


def calculate_sma(df, period=20, column='Close'):
    """
    Calculate Simple Moving Average for a given stock price series.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock price data
        period (int): Number of trading days for moving average window
                     Default: 20 days (approximately 1 month of trading)
        column (str): Name of column to calculate SMA for
                     Default: 'Close' (closing price)
    
    Returns:
        pd.Series: Series containing SMA values, aligned with original index
    
    Raises:
        ValueError: If period > length of data or period < 1
        KeyError: If column not found in DataFrame
    
    Example:
        >>> df = pd.DataFrame({'Close': [100, 101, 102, 103, ...]})
        >>> sma_20 = calculate_sma(df, period=20)
        >>> print(sma_20)
        0       NaN      (first 19 values are NaN - insufficient history)
        19      101.5    (average of first 20 days)
        20      102.3
        ...
    
    Workflow:
        1. Validate inputs (period and column existence)
        2. Check if period is valid for dataset length
        3. Use pandas rolling window to calculate mean
        4. Return SMA series with NaN for insufficient data points
    """
    
    # =========================================================================
    # Input Validation
    # =========================================================================
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    
    if not isinstance(period, int) or period < 1:
        raise ValueError(f"Period must be positive integer, got {period}")
    
    if period > len(df):
        raise ValueError(
            f"Period ({period}) cannot exceed data length ({len(df)})"
        )
    
    # =========================================================================
    # Calculate SMA using pandas rolling window
    # =========================================================================
    # rolling(period) creates a sliding window of size 'period'
    # mean() calculates the average for each window
    # The first (period-1) values will be NaN due to insufficient data
    sma = df[column].rolling(window=period).mean()
    
    print(f"[INFO] Calculated SMA({period}) on {column} column")
    print(f"[INFO] Total values: {len(sma)}, NaN values: {sma.isna().sum()}")
    
    return sma


def calculate_multiple_sma(df, periods=[20, 50, 200], column='Close'):
    """
    Calculate multiple Simple Moving Averages at once.
    
    This is useful for comparing short, medium, and long-term trends.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock price data
        periods (list): List of periods for SMAs
                       Default: [20, 50, 200] (short, medium, long term)
        column (str): Column name to calculate SMA for
                     Default: 'Close'
    
    Returns:
        pd.DataFrame: DataFrame with columns 'SMA_20', 'SMA_50', 'SMA_200', etc.
    
    Example:
        >>> df = pd.DataFrame({'Close': [..., ..., ...]})
        >>> sma_df = calculate_multiple_sma(df, periods=[20, 50, 200])
        >>> print(sma_df.columns)
        Index(['SMA_20', 'SMA_50', 'SMA_200'], dtype='object')
    
    Workflow:
        1. Iterate through each period
        2. Calculate SMA for that period
        3. Create DataFrame with all SMAs
        4. Return combined dataframe
    """
    
    sma_dict = {}
    
    for period in periods:
        try:
            sma_dict[f'SMA_{period}'] = calculate_sma(df, period, column)
        except ValueError as e:
            print(f"[WARNING] Could not calculate SMA({period}): {str(e)}")
            continue
    
    result_df = pd.DataFrame(sma_dict)
    print(f"[SUCCESS] Calculated {len(sma_dict)} moving averages")
    
    return result_df


def sma_signal(price, sma_value):
    """
    Generate trading signal based on price position relative to SMA.
    
    Args:
        price (float): Current price
        sma_value (float): SMA value
    
    Returns:
        str: Signal type ('BULLISH', 'BEARISH', or 'NEUTRAL')
    
    Logic:
        - Price > SMA: BULLISH (uptrend) - Buy signal
        - Price < SMA: BEARISH (downtrend) - Sell signal
        - Price ≈ SMA: NEUTRAL - Indecisive
    """
    
    if pd.isna(sma_value):
        return 'NEUTRAL'
    
    # Calculate percentage difference
    diff_percent = ((price - sma_value) / sma_value) * 100
    
    # Define thresholds (can be adjusted)
    bullish_threshold = 0.5   # Price > SMA by 0.5%
    bearish_threshold = -0.5  # Price < SMA by 0.5%
    
    if diff_percent > bullish_threshold:
        return 'BULLISH'
    elif diff_percent < bearish_threshold:
        return 'BEARISH'
    else:
        return 'NEUTRAL'
