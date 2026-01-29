"""
===============================================================================
RELATIVE STRENGTH INDEX (RSI) INDICATOR
===============================================================================

Technical Indicator: Relative Strength Index (RSI)
Definition:
    RSI is a momentum oscillator that measures the magnitude of recent price
    changes to evaluate overbought or oversold conditions. It oscillates
    between 0 and 100.

Mathematical Formula:
    RS = Average Gain / Average Loss
    RSI = 100 - (100 / (1 + RS))
    
    Where:
        Average Gain = Sum of upward price changes / n
        Average Loss = Sum of downward price changes / n
        n = number of periods (typically 14)

Step-by-Step Calculation:
    1. Calculate daily price changes (Close[t] - Close[t-1])
    2. Separate gains (positive changes) and losses (negative changes)
    3. Average gains and losses over the period
    4. Calculate Relative Strength (RS) ratio
    5. Convert RS to RSI scale (0-100)

Interpretation:
    RSI < 30: OVERSOLD (potentially undervalued, buy signal)
    RSI 30-70: NEUTRAL (normal market conditions)
    RSI > 70: OVERBOUGHT (potentially overvalued, sell signal)
    
    Note: Thresholds can be adjusted based on market conditions

Use Cases in ML:
    - Feature for predicting reversals
    - Identifies momentum changes
    - Detects extreme price conditions
    - Complements other indicators

Advantages:
    - Oscillates between fixed range (0-100) - easy to interpret
    - Identifies overbought/oversold conditions
    - Good at detecting trend reversals
    - Works well with trend-following strategies

Limitations:
    - Can be overbought/oversold for extended periods
    - Lagging indicator in strong trends
    - Less effective in sideways/ranging markets
    - False signals possible in choppy markets

Dependencies:
    - pandas: For rolling calculations
    - numpy: For numerical operations
===============================================================================
"""

import pandas as pd
import numpy as np


def calculate_rsi(df, period=14, column='Close'):
    """
    Calculate Relative Strength Index (RSI) for stock price data.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock price data
        period (int): Number of periods for RSI calculation
                     Default: 14 days (standard in technical analysis)
        column (str): Column name containing prices
                     Default: 'Close' (closing prices)
    
    Returns:
        pd.Series: Series containing RSI values (0-100 scale)
    
    Raises:
        ValueError: If period < 2 or period > data length
        KeyError: If column not found in DataFrame
    
    Example:
        >>> df = pd.DataFrame({'Close': [100, 101, 99, 102, 103, ...]})
        >>> rsi = calculate_rsi(df, period=14)
        >>> print(rsi)
        0        NaN      (first 14 values are NaN)
        14       45.5     (RSI value)
        15       52.3
        ...
    
    Workflow:
        1. Validate inputs
        2. Calculate daily price changes (delta)
        3. Separate positive (gains) and negative (losses) changes
        4. Calculate average gain and loss over period
        5. Calculate RS (gain/loss ratio)
        6. Convert RS to RSI (0-100 scale)
        7. Return RSI series
    """
    
    # =========================================================================
    # Input Validation
    # =========================================================================
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    
    if not isinstance(period, int) or period < 2:
        raise ValueError(f"Period must be integer >= 2, got {period}")
    
    if period > len(df):
        raise ValueError(
            f"Period ({period}) exceeds data length ({len(df)})"
        )
    
    # =========================================================================
    # STEP 1: Calculate daily price changes
    # =========================================================================
    # delta = Close[t] - Close[t-1]
    delta = df[column].diff()
    
    # =========================================================================
    # STEP 2: Separate gains and losses
    # =========================================================================
    # gains: positive changes (keeping values, replacing negative with 0)
    # losses: absolute value of negative changes (keeping values, replacing positive with 0)
    gains = delta.where(delta > 0, 0)  # Keep gains, set losses to 0
    losses = -delta.where(delta < 0, 0)  # Keep absolute losses, set gains to 0
    
    # =========================================================================
    # STEP 3: Calculate average gain and loss
    # =========================================================================
    # Using rolling mean over the specified period
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()
    
    # =========================================================================
    # STEP 4: Calculate RS (Relative Strength)
    # =========================================================================
    # RS = Average Gain / Average Loss
    # Avoid division by zero
    rs = avg_gain / avg_loss
    
    # =========================================================================
    # STEP 5: Convert to RSI (0-100 scale)
    # =========================================================================
    # RSI = 100 - (100 / (1 + RS))
    rsi = 100 - (100 / (1 + rs))
    
    # Replace infinity values (when avg_loss = 0) with 100
    rsi = rsi.replace([np.inf, -np.inf], 100)
    
    print(f"[INFO] Calculated RSI({period}) on {column} column")
    print(f"[INFO] RSI range: {rsi.min():.2f} to {rsi.max():.2f}")
    print(f"[INFO] NaN values: {rsi.isna().sum()}")
    
    return rsi


def rsi_signal(rsi_value, overbought=70, oversold=30):
    """
    Generate trading signal based on RSI value.
    
    Args:
        rsi_value (float): Current RSI value (0-100)
        overbought (int): Threshold for overbought condition
                         Default: 70 (standard threshold)
        oversold (int): Threshold for oversold condition
                       Default: 30 (standard threshold)
    
    Returns:
        str: Signal type ('OVERBOUGHT', 'OVERSOLD', or 'NEUTRAL')
    
    Logic:
        - RSI > overbought: OVERBOUGHT - sell signal, potential reversal down
        - RSI < oversold: OVERSOLD - buy signal, potential reversal up
        - oversold <= RSI <= overbought: NEUTRAL - no extreme condition
    
    Example:
        >>> rsi_signal(75)   # Returns 'OVERBOUGHT'
        >>> rsi_signal(25)   # Returns 'OVERSOLD'
        >>> rsi_signal(50)   # Returns 'NEUTRAL'
    """
    
    if pd.isna(rsi_value):
        return 'NEUTRAL'
    
    if rsi_value > overbought:
        return 'OVERBOUGHT'
    elif rsi_value < oversold:
        return 'OVERSOLD'
    else:
        return 'NEUTRAL'


def interpret_rsi_extreme(rsi_value, lookback_count=5):
    """
    Provide interpretation of extreme RSI values and their implications.
    
    Args:
        rsi_value (float): Current RSI value
        lookback_count (int): Number of periods to consider for trend
    
    Returns:
        dict: Dictionary with interpretation details
    
    Interpretation Guidelines:
        - RSI 70-100: Overbought conditions, potential pullback
        - RSI 30-0: Oversold conditions, potential bounce
        - Extreme values (>90 or <10): Very strong momentum
    """
    
    if pd.isna(rsi_value):
        return {'status': 'NEUTRAL', 'interpretation': 'Insufficient data'}
    
    if rsi_value > 90:
        return {
            'status': 'EXTREMELY_OVERBOUGHT',
            'interpretation': 'Very strong bullish momentum, reversal likely imminent',
            'action': 'SELL'
        }
    elif rsi_value > 70:
        return {
            'status': 'OVERBOUGHT',
            'interpretation': 'Stock may be overvalued, potential pullback',
            'action': 'SELL_OR_REDUCE'
        }
    elif rsi_value < 10:
        return {
            'status': 'EXTREMELY_OVERSOLD',
            'interpretation': 'Very strong bearish momentum, reversal likely imminent',
            'action': 'BUY'
        }
    elif rsi_value < 30:
        return {
            'status': 'OVERSOLD',
            'interpretation': 'Stock may be undervalued, potential bounce',
            'action': 'BUY'
        }
    else:
        return {
            'status': 'NEUTRAL',
            'interpretation': 'Normal market conditions, no extreme signal',
            'action': 'HOLD'
        }
