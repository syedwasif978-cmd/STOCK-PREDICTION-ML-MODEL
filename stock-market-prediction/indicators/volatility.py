"""
===============================================================================
VOLATILITY INDICATOR
===============================================================================

Technical Concept: Volatility
Definition:
    Volatility measures the degree of price variation of a stock over time.
    It quantifies the dispersion of returns and indicates market uncertainty
    and risk. Higher volatility = greater price swings = higher risk/reward.

Mathematical Formula:
    Historical Volatility (Standard Deviation):
    σ = √[(Σ(Return[t] - Mean Return)²) / n]
    
    Where:
        Return[t] = ln(Close[t] / Close[t-1])  (log return)
        Mean Return = Average of all returns
        n = number of periods
        σ = standard deviation of returns

Alternative Formula (Simplified):
    Volatility = Standard Deviation of Daily Price Changes
    
    This is more intuitive but standard deviation of log returns is more
    statistically sound for financial data.

Interpretation:
    Low Volatility: <10% annualized
        - Stable, predictable prices
        - Lower risk, but also lower potential returns
        - Common in large-cap stocks (e.g., dividend payers)
    
    Medium Volatility: 10-30% annualized
        - Normal market conditions
        - Balanced risk-reward
        - Most typical for mid-cap stocks
    
    High Volatility: >30% annualized
        - Rapid price swings
        - Higher risk, potentially higher rewards
        - Common in growth stocks, tech, small-cap
    
    Extreme Volatility: >50% annualized
        - Highly unpredictable
        - Extreme risk
        - Often indicates market stress or speculative assets

Use Cases in ML:
    - Feature for ML models (volatility clustering)
    - Risk assessment input
    - Options pricing and hedging
    - Portfolio optimization
    - Trade signal confidence weighting

Advantages:
    - Objective quantitative measure of risk
    - Easy to calculate and understand
    - Works across different price levels
    - Useful for risk management

Limitations:
    - Historical volatility may not predict future volatility
    - Assumes returns are normally distributed (not always true)
    - Doesn't capture directional bias
    - Can be distorted by gaps (overnight events)

Dependencies:
    - pandas: For rolling calculations
    - numpy: For mathematical operations
===============================================================================
"""

import pandas as pd
import numpy as np


def calculate_volatility(df, period=20, column='Close', method='log_return'):
    """
    Calculate historical volatility (standard deviation of returns).
    
    Args:
        df (pd.DataFrame): DataFrame containing stock price data
        period (int): Number of trading days for rolling volatility
                     Default: 20 (approximately 1 month)
        column (str): Column name containing prices
                     Default: 'Close' (closing prices)
        method (str): Method for return calculation
                     Options: 'log_return' (default) or 'simple_return'
                     - 'log_return': ln(P[t]/P[t-1]) - preferred for finance
                     - 'simple_return': (P[t]-P[t-1])/P[t-1] - intuitive
    
    Returns:
        pd.Series: Series containing rolling volatility values
    
    Raises:
        ValueError: If period invalid or method unknown
        KeyError: If column not found
    
    Example:
        >>> df = pd.DataFrame({'Close': [100, 101, 99, 102, ...]})
        >>> vol = calculate_volatility(df, period=20)
        >>> print(vol)
        0         NaN      (first 20 values insufficient)
        20        0.0234   (2.34% daily volatility)
        21        0.0251
        ...
    
    Workflow:
        1. Validate inputs
        2. Calculate daily returns (log or simple)
        3. Calculate rolling standard deviation
        4. Return volatility series
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
    
    if method not in ['log_return', 'simple_return']:
        raise ValueError(f"Unknown method: {method}. Use 'log_return' or 'simple_return'")
    
    # =========================================================================
    # STEP 1: Calculate daily returns
    # =========================================================================
    if method == 'log_return':
        # Logarithmic return: ln(P[t] / P[t-1])
        # Preferred in finance because: additive over time, less biased
        returns = np.log(df[column] / df[column].shift(1))
        return_type = "log returns"
    else:
        # Simple return: (P[t] - P[t-1]) / P[t-1]
        # More intuitive: represents percentage change
        returns = df[column].pct_change()
        return_type = "simple returns"
    
    # =========================================================================
    # STEP 2: Calculate rolling standard deviation (volatility)
    # =========================================================================
    # Standard deviation represents the dispersion of returns from mean
    # Higher std dev = wider price swings = higher volatility
    volatility = returns.rolling(window=period).std()
    
    # =========================================================================
    # Remove NaN values info
    # =========================================================================
    nan_count = volatility.isna().sum()
    valid_count = len(volatility) - nan_count
    
    print(f"[INFO] Calculated {period}-day volatility using {return_type}")
    print(f"[INFO] Valid values: {valid_count}, NaN values: {nan_count}")
    print(f"[INFO] Volatility range: {volatility.min():.4f} to {volatility.max():.4f}")
    
    return volatility


def annualize_volatility(daily_volatility, trading_days=252):
    """
    Convert daily volatility to annualized volatility.
    
    Args:
        daily_volatility (float or pd.Series): Daily volatility value(s)
        trading_days (int): Number of trading days per year
                           Default: 252 (standard for stock markets)
    
    Returns:
        float or pd.Series: Annualized volatility
    
    Formula:
        Annualized Vol = Daily Vol × √(Trading Days Per Year)
        
        Example: Daily volatility of 0.02 (2%)
                 → 0.02 × √252 ≈ 0.317 (31.7% annualized)
    
    Explanation:
        Volatility compounds over time according to square root of time.
        This is based on statistical properties of random walks.
    
    Example:
        >>> daily_vol = 0.015  # 1.5% daily
        >>> annual_vol = annualize_volatility(daily_vol)
        >>> print(f"Annualized volatility: {annual_vol:.2%}")
        Annualized volatility: 23.81%
    """
    
    if trading_days <= 0:
        raise ValueError("Trading days must be positive")
    
    annualized = daily_volatility * np.sqrt(trading_days)
    
    return annualized


def volatility_signal(volatility_value, low_threshold=0.01, high_threshold=0.03):
    """
    Generate market condition signal based on volatility level.
    
    Args:
        volatility_value (float): Current volatility (daily or annualized)
        low_threshold (float): Threshold for low volatility
                              Default: 0.01 (1% daily or 15.9% annual)
        high_threshold (float): Threshold for high volatility
                               Default: 0.03 (3% daily or 47.6% annual)
    
    Returns:
        str: Volatility regime ('LOW', 'MEDIUM', or 'HIGH')
    
    Logic:
        - Volatility < low_threshold: LOW (stable market, low risk)
        - low_threshold <= Volatility < high_threshold: MEDIUM (normal)
        - Volatility >= high_threshold: HIGH (turbulent market, high risk)
    
    Use Cases:
        - LOW: Conservative traders prefer this, good for steady portfolios
        - MEDIUM: Typical market condition, balanced risk-reward
        - HIGH: Traders seeking volatile moves, or fear-driven market decline
    
    Example:
        >>> signal = volatility_signal(0.025)  # 2.5% daily
        >>> print(signal)
        'MEDIUM'
    """
    
    if pd.isna(volatility_value) or volatility_value < 0:
        return 'UNKNOWN'
    
    if volatility_value < low_threshold:
        return 'LOW'
    elif volatility_value < high_threshold:
        return 'MEDIUM'
    else:
        return 'HIGH'


def volatility_percentile(current_vol, historical_vols, percentile=False):
    """
    Calculate where current volatility stands relative to historical range.
    
    Args:
        current_vol (float): Current volatility value
        historical_vols (pd.Series or list): Historical volatility values
        percentile (bool): If True, return percentile rank (0-100)
                          If False, return position (0-1)
    
    Returns:
        float: Percentile rank or normalized position
    
    Example:
        >>> vols = [0.01, 0.015, 0.02, 0.025, 0.03]
        >>> current = 0.022
        >>> rank = volatility_percentile(current, vols, percentile=True)
        >>> print(f"Current vol is in {rank:.0f}th percentile")
        Current vol is in 70th percentile
        
    Interpretation:
        - Low percentile (e.g., 20th): Current volatility is low relative to history
        - High percentile (e.g., 80th): Current volatility is high relative to history
    """
    
    # Convert to numpy array if pandas Series
    if isinstance(historical_vols, pd.Series):
        historical_vols = historical_vols.dropna().values
    
    # Count how many historical values are less than current
    lower_count = np.sum(historical_vols < current_vol)
    
    # Calculate percentile (0-100) or position (0-1)
    rank = (lower_count / len(historical_vols)) * 100 if percentile else (lower_count / len(historical_vols))
    
    return rank
