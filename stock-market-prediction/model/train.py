"""
===============================================================================
MODEL TRAINING PIPELINE
===============================================================================

Module Purpose:
    This module orchestrates the complete training workflow:
    1. Data loading and validation
    2. Feature engineering (technical indicators)
    3. Data splitting (train/test)
    4. Model training and evaluation
    5. Model persistence (save)

Data Processing Workflow:
    
    Raw OHLCV Data (from Alpaca)
            ↓
    Calculate Technical Indicators (SMA, RSI, Volatility)
            ↓
    Feature Engineering (combine indicators into feature matrix)
            ↓
    Remove NaN values (alignment and warmup period)
            ↓
    Calculate Target (next-day returns)
            ↓
    Train/Test Split (70% / 30%)
            ↓
    Feature Scaling (StandardScaler)
            ↓
    Model Training (Random Forest)
            ↓
    Model Evaluation (metrics on test set)
            ↓
    Model Persistence (save to disk)

Key Concepts:

    FEATURE ENGINEERING:
        Converting raw data into meaningful features for ML model
        
        Raw: Close prices [100, 101, 99, 102, ...]
        Features:
            - SMA_20: 20-day moving average
            - RSI_14: Relative Strength Index
            - Volatility: Price variation
            - Volume: Trading volume
        
        Benefits: Captures domain knowledge, simplifies learning
    
    TRAIN/TEST SPLIT:
        Divide data into two sets:
        - Training: 70% (used to fit model parameters)
        - Testing: 30% (used to evaluate performance)
        
        Important: Must preserve temporal order for time-series data
        ✓ Correct: Split chronologically (first 70% = train, last 30% = test)
        ✗ Wrong: Random shuffle (causes data leakage)
    
    TARGET VARIABLE:
        What the model predicts: next-day return
        
        Return(t+1) = (Close(t+1) - Close(t)) / Close(t)
        
        Example:
            If Close(t) = 100 and Close(t+1) = 103
            Then Return(t+1) = (103 - 100) / 100 = 0.03 (3%)
        
        Advantages:
            - Normalizes for different price levels
            - Same metric for all stocks
            - Interpretable (percentage return)
    
    FEATURE ALIGNMENT:
        Technical indicators need history (e.g., SMA_20 needs 20 days)
        
        Timestamp  Close  SMA_20  Return
        2024-01-01  100    NaN     NaN     (insufficient history)
        2024-01-02  101    NaN     NaN
        ...
        2024-01-20  120    107.5   0.02    (valid data point)
        
        Solution: Remove rows with NaN values

Dependencies:
    - pandas: Data manipulation
    - numpy: Numerical operations
    - sklearn: train_test_split
    - sys.path: Import from parent modules
===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.sma import calculate_sma
from indicators.rsi import calculate_rsi
from indicators.volatility import calculate_volatility
from model.random_forest import RandomForestModel


def engineer_features(df):
    """
    Create technical indicator features from raw OHLCV data.
    
    This function transforms raw price data into ML-ready features using
    well-established technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with columns [Open, High, Low, Close, Volume]
                          Index should be DatetimeIndex
    
    Returns:
        pd.DataFrame: DataFrame with original OHLCV + engineered features
    
    Features Engineered:
        - SMA_20: 20-day Simple Moving Average
        - SMA_50: 50-day Simple Moving Average
        - SMA_200: 200-day Simple Moving Average
        - RSI_14: 14-day Relative Strength Index
        - Volatility: 20-day volatility
        - Volume_MA: 20-day average trading volume
    
    Workflow:
        1. Calculate each technical indicator
        2. Combine all features into single DataFrame
        3. Add original OHLCV data
        4. Return combined DataFrame
    
    Data Structure:
        Input:
            Timestamp  Open   High    Low    Close  Volume
            2024-01-01 100.0  101.0   99.0   100.5  1000000
        
        Output:
            Timestamp  Open ... Close  SMA_20  SMA_50  RSI_14  Volatility
            2024-01-01 100.0 ... 100.5  NaN     NaN     NaN     NaN
            ...
            2024-01-20 110.0 ... 110.5  105.2   103.1   55.2    0.012
    """
    
    print(f"\n[INFO] Starting feature engineering on {len(df)} records")
    
    # Create a copy to avoid modifying original
    features_df = df.copy()
    
    # =========================================================================
    # STEP 1: Calculate Simple Moving Averages (trend indicators)
    # =========================================================================
    print("[INFO] Calculating SMA indicators...")
    sma_20 = calculate_sma(df, period=20, column='Close')
    sma_50 = calculate_sma(df, period=50, column='Close')
    sma_200 = calculate_sma(df, period=200, column='Close')
    
    features_df['SMA_20'] = sma_20
    features_df['SMA_50'] = sma_50
    features_df['SMA_200'] = sma_200
    
    # =========================================================================
    # STEP 2: Calculate RSI (momentum indicator)
    # =========================================================================
    print("[INFO] Calculating RSI indicator...")
    rsi_14 = calculate_rsi(df, period=14, column='Close')
    features_df['RSI_14'] = rsi_14
    
    # =========================================================================
    # STEP 3: Calculate Volatility (risk indicator)
    # =========================================================================
    print("[INFO] Calculating Volatility indicator...")
    volatility = calculate_volatility(df, period=20, column='Close')
    features_df['Volatility'] = volatility
    
    # =========================================================================
    # STEP 4: Calculate Volume Moving Average (liquidity)
    # =========================================================================
    print("[INFO] Calculating Volume MA...")
    volume_ma = df['Volume'].rolling(window=20).mean()
    features_df['Volume_MA'] = volume_ma
    
    # =========================================================================
    # Summary
    # =========================================================================
    nan_count = features_df.isna().sum().sum()
    print(f"[INFO] Feature engineering complete")
    print(f"       - Total NaN values: {nan_count}")
    print(f"       - Features created: {features_df.shape[1] - df.shape[1]}")
    
    return features_df


def prepare_data(df, test_size=0.3, sequence_offset=1):
    """
    Prepare data for model training and testing.
    
    This function:
    1. Engineers features from raw data
    2. Calculates target variable (next-day return)
    3. Removes rows with NaN values
    4. Splits into train/test sets
    5. Separates features (X) from target (y)
    
    Args:
        df (pd.DataFrame): Raw OHLCV data from API
        test_size (float): Fraction of data for testing. Default: 0.3 (30%)
        sequence_offset (int): Days ahead for return calculation. Default: 1
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, dates)
    
    Example:
        >>> X_train, X_test, y_train, y_test, features, dates = prepare_data(df)
        >>> print(f"Training samples: {len(X_train)}")
        Training samples: 875
        >>> print(f"Testing samples: {len(X_test)}")
        Testing samples: 375
    
    Workflow:
        1. Engineer features (SMA, RSI, Volatility, etc.)
        2. Calculate target: next-day return
        3. Remove rows with NaN values
        4. Split into train (70%) and test (30%)
        5. Extract feature matrix and target vector
        6. Return all components for training
    """
    
    print("\n" + "="*70)
    print("[INFO] PREPARING DATA FOR MODEL TRAINING")
    print("="*70)
    
    # =========================================================================
    # STEP 1: Engineer features
    # =========================================================================
    featured_df = engineer_features(df)
    
    # =========================================================================
    # STEP 2: Calculate target variable (next-day return)
    # =========================================================================
    # Return(t+1) = (Close(t+1) - Close(t)) / Close(t)
    # We shift backwards by 1 to align with features at time t
    featured_df['Return'] = (featured_df['Close'].shift(-sequence_offset) / 
                             featured_df['Close']) - 1
    
    print(f"\n[INFO] Target variable (next-day return) calculated")
    print(f"       Formula: Return(t+1) = (Close(t+1) - Close(t)) / Close(t)")
    print(f"       Return range: {featured_df['Return'].min():.4f} to {featured_df['Return'].max():.4f}")
    
    # =========================================================================
    # STEP 3: Select features for model (exclude OHLCV and target)
    # =========================================================================
    feature_columns = [col for col in featured_df.columns 
                      if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Return']]
    
    print(f"\n[INFO] Selected {len(feature_columns)} features for model:")
    for i, col in enumerate(feature_columns, 1):
        print(f"       {i}. {col}")
    
    # =========================================================================
    # STEP 4: Remove rows with NaN values
    # =========================================================================
    # Features need historical data (warmup period)
    # Targets are NaN at the end (can't predict beyond data)
    before_clean = len(featured_df)
    featured_df_clean = featured_df.dropna()
    after_clean = len(featured_df_clean)
    removed = before_clean - after_clean
    
    print(f"\n[INFO] Data cleaning (remove NaN):")
    print(f"       - Before: {before_clean} rows")
    print(f"       - After:  {after_clean} rows")
    print(f"       - Removed: {removed} rows ({removed/before_clean*100:.1f}%)")
    
    if after_clean < 100:
        raise ValueError(
            f"Insufficient data after cleaning ({after_clean} rows). "
            "Need at least 100 data points."
        )
    
    # =========================================================================
    # STEP 5: Prepare feature matrix (X) and target vector (y)
    # =========================================================================
    X = featured_df_clean[feature_columns].values
    y = featured_df_clean['Return'].values
    dates = featured_df_clean.index
    
    print(f"\n[INFO] Feature matrix prepared:")
    print(f"       - Shape: {X.shape} (samples × features)")
    print(f"       - Target shape: {y.shape}")
    
    # =========================================================================
    # STEP 6: Split into train/test sets (chronological split)
    # =========================================================================
    # For time-series data, we split chronologically, not randomly
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    
    print(f"\n[INFO] Train/Test split (chronological):")
    print(f"       - Training:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"       - Testing:    {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"       - Train date range: {train_dates[0].date()} to {train_dates[-1].date()}")
    print(f"       - Test date range:  {test_dates[0].date()} to {test_dates[-1].date()}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n[SUCCESS] Data preparation complete")
    
    return X_train, X_test, y_train, y_test, feature_columns, (train_dates, test_dates)


def train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names):
    """
    Train Random Forest model and evaluate on test data.
    
    Args:
        X_train (np.ndarray): Training features (n_samples, n_features)
        X_test (np.ndarray): Test features
        y_train (np.ndarray): Training targets (returns)
        y_test (np.ndarray): Test targets
        feature_names (list): Names of features for importance analysis
    
    Returns:
        RandomForestModel: Trained model instance
    
    Workflow:
        1. Initialize model with hyperparameters
        2. Train on training data
        3. Evaluate on test data
        4. Display feature importance
        5. Return trained model
    """
    
    print("\n" + "="*70)
    print("[INFO] TRAINING RANDOM FOREST MODEL")
    print("="*70)
    
    # =========================================================================
    # STEP 1: Initialize model
    # =========================================================================
    model = RandomForestModel(
        n_estimators=100,      # Number of decision trees
        max_depth=15,          # Maximum tree depth (prevents overfitting)
        min_samples_split=5,   # Minimum samples to split a node
        min_samples_leaf=2,    # Minimum samples at leaf
        random_state=42,       # Reproducibility
        n_jobs=-1              # Use all CPU cores
    )
    
    # =========================================================================
    # STEP 2: Train model
    # =========================================================================
    model.train(X_train, y_train, scale_features=True)
    
    # =========================================================================
    # STEP 3: Evaluate on test data
    # =========================================================================
    test_metrics = model.evaluate(X_test, y_test, scale_features=True)
    
    # =========================================================================
    # STEP 4: Feature importance analysis
    # =========================================================================
    print("\n" + "-"*70)
    importance_df = model.get_feature_importance(feature_names, top_n=10)
    
    # =========================================================================
    # Summary and recommendations
    # =========================================================================
    print(f"\n[INFO] Model Training Summary:")
    print(f"       - Training R²: {model.training_history['r2']:.4f}")
    print(f"       - Test R²:     {test_metrics['r2']:.4f}")
    
    if test_metrics['r2'] < 0:
        print(f"\n       ⚠️  Warning: Negative R² indicates poor model performance")
        print(f"           The model performs worse than predicting the mean.")
    elif test_metrics['r2'] < 0.1:
        print(f"\n       ℹ️  Info: Low R² is typical for stock market predictions")
        print(f"           Stock returns are inherently noisy and unpredictable.")
    else:
        print(f"\n       ✓ Good: R² indicates the model explains variance in returns")
    
    return model
