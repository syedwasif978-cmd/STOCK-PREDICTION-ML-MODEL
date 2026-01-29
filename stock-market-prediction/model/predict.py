"""
===============================================================================
MODEL PREDICTION MODULE
===============================================================================

Module Purpose:
    This module handles making predictions with trained models.
    
    Functions:
    - predict_next_day(): Predict next trading day return for a stock
    - price_from_return(): Convert predicted return to predicted price
    - generate_price_curve(): Create predicted price series for visualization
    - forecast_future(): Generate multi-day forecasts

Prediction Workflow:

    Trained Model + Latest Features
            ↓
    Extract latest technical indicators (SMA, RSI, etc.)
            ↓
    Scale features using fitted scaler
            ↓
    Feed to Random Forest model
            ↓
    Get predicted next-day return
            ↓
    Convert return to price (current_price × (1 + return))
            ↓
    Display prediction with confidence interval
    
Important Notes:

    RETURN TO PRICE CONVERSION:
        Prediction from model: Return (e.g., 0.02 = 2%)
        
        Predicted Price = Current Price × (1 + Return)
        
        Example:
            Current Close: $100
            Predicted Return: 0.03 (3%)
            Predicted Close: $100 × 1.03 = $103
    
    TEMPORAL LIMITATIONS:
        The model predicts 1-day ahead based on historical patterns.
        
        It CANNOT:
        - Account for news events or earnings
        - Predict based on future information
        - Handle regime changes (market structure shifts)
        - Predict cryptocurrency or exotic assets reliably
        
        These require more sophisticated models (LSTM, attention, etc.)
    
    CONFIDENCE LEVELS:
        The model provides point predictions (single value).
        True confidence requires ensemble methods or Bayesian approaches.
        
        For now, use historical prediction error as uncertainty metric:
        - Confidence ≈ based on test set prediction error

Dependencies:
    - pandas: Data manipulation
    - numpy: Numerical operations
    - model.random_forest: RandomForestModel class
===============================================================================
"""

import pandas as pd
import numpy as np
from model.random_forest import RandomForestModel


def predict_next_day(model, current_features):
    """
    Predict next trading day return for a stock.
    
    Args:
        model (RandomForestModel): Trained model
        current_features (np.ndarray): Latest feature values
                                      Shape: (1, n_features)
                                      Features: [SMA_20, SMA_50, RSI_14, ...]
    
    Returns:
        float: Predicted next-day return (-1.0 to +1.0, typically -0.10 to +0.10)
    
    Example:
        >>> predicted_return = predict_next_day(model, features)
        >>> print(f"Predicted return: {predicted_return:.4f} ({predicted_return*100:.2f}%)")
        Predicted return: 0.0234 (2.34%)
    
    Workflow:
        1. Validate features shape
        2. Reshape to (1, n_features) if needed
        3. Call model.predict()
        4. Extract scalar value
        5. Return prediction
    """
    
    # Ensure features are 2D array (1 sample, n features)
    if isinstance(current_features, (list, tuple)):
        current_features = np.array(current_features).reshape(1, -1)
    elif current_features.ndim == 1:
        current_features = current_features.reshape(1, -1)
    
    # Get prediction
    prediction = model.predict(current_features, scale_features=True)
    
    return prediction[0]


def price_from_return(current_price, predicted_return):
    """
    Convert predicted return to predicted closing price.
    
    Args:
        current_price (float): Current (or last known) closing price
        predicted_return (float): Predicted return (e.g., 0.02 for 2%)
    
    Returns:
        float: Predicted closing price for next day
    
    Formula:
        Predicted Price = Current Price × (1 + Return)
    
    Example:
        >>> current = 100.0
        >>> ret = 0.05  # 5% return
        >>> pred_price = price_from_return(current, ret)
        >>> print(f"Current: ${current:.2f}, Predicted: ${pred_price:.2f}")
        Current: $100.00, Predicted: $105.00
    
    Notes:
        - Return should be in decimal format (-1.0 to +1.0)
        - Larger returns (>0.50 or <-0.50) are unusual and may indicate model error
        - Use with caution for very high/low return predictions
    """
    
    if current_price <= 0:
        raise ValueError(f"Current price must be positive, got {current_price}")
    
    predicted_price = current_price * (1 + predicted_return)
    
    return predicted_price


def generate_price_curve(actual_returns, current_price, predicted_returns):
    """
    Generate predicted price curve from series of predicted returns.
    
    Args:
        actual_returns (np.ndarray): Series of actual returns from test data
        current_price (float): Starting price (last historical close)
        predicted_returns (np.ndarray): Series of predicted returns
    
    Returns:
        tuple: (actual_prices, predicted_prices, dates)
    
    Algorithm:
        For each day:
            Price = Previous Price × (1 + Return)
    
    Example:
        >>> actual_ret = np.array([0.01, -0.02, 0.03, 0.01])
        >>> pred_ret = np.array([0.015, -0.015, 0.025, 0.005])
        >>> actual_prices, pred_prices, dates = generate_price_curve(
        ...     actual_ret, 100.0, pred_ret
        ... )
        >>> print(actual_prices)
        [100.0, 101.0, 98.98, 101.95, ...]
    
    Workflow:
        1. Start with current_price
        2. For each return value:
            Next Price = Previous Price × (1 + Return)
        3. Build array of prices
        4. Align with dates (if provided)
        5. Return price curves
    """
    
    # Validate inputs
    if len(actual_returns) != len(predicted_returns):
        raise ValueError("Actual and predicted returns must have same length")
    
    # Initialize price arrays
    actual_prices = np.zeros(len(actual_returns) + 1)
    predicted_prices = np.zeros(len(predicted_returns) + 1)
    
    actual_prices[0] = current_price
    predicted_prices[0] = current_price
    
    # Calculate cumulative prices
    for i, (actual_ret, pred_ret) in enumerate(zip(actual_returns, predicted_returns)):
        actual_prices[i + 1] = actual_prices[i] * (1 + actual_ret)
        predicted_prices[i + 1] = predicted_prices[i] * (1 + pred_ret)
    
    return actual_prices, predicted_prices


def calculate_prediction_accuracy(y_test, y_pred, metric='rmse'):
    """
    Calculate accuracy metrics for predictions.
    
    Args:
        y_test (np.ndarray): Actual returns
        y_pred (np.ndarray): Predicted returns
        metric (str): Metric to calculate
                     Options: 'rmse', 'mae', 'mape', 'direction_accuracy'
    
    Returns:
        float: Accuracy metric value
    
    Metrics:
        
        RMSE (Root Mean Squared Error):
            √(Σ(actual - predicted)² / n)
            - Penalizes large errors
            - Same units as returns
        
        MAE (Mean Absolute Error):
            Σ|actual - predicted| / n
            - Average absolute deviation
            - Less sensitive to outliers
        
        MAPE (Mean Absolute Percentage Error):
            Σ|actual - predicted| / |actual| / n
            - Percentage error (scale-independent)
            - Problematic when actual ≈ 0
        
        Direction Accuracy:
            % of predictions with correct sign (up/down)
            - Useful for buy/sell signals
            - Simple interpretation
    
    Example:
        >>> actual = np.array([0.02, -0.01, 0.03])
        >>> pred = np.array([0.018, -0.015, 0.028])
        >>> rmse = calculate_prediction_accuracy(actual, pred, 'rmse')
        >>> print(f"RMSE: {rmse:.6f}")
        RMSE: 0.002345
    """
    
    if len(y_test) != len(y_pred):
        raise ValueError("y_test and y_pred must have same length")
    
    if metric == 'rmse':
        return np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    elif metric == 'mae':
        return np.mean(np.abs(y_test - y_pred))
    
    elif metric == 'mape':
        # Avoid division by zero
        non_zero = np.abs(y_test) > 0.00001
        if not np.any(non_zero):
            return np.nan
        return np.mean(np.abs((y_test[non_zero] - y_pred[non_zero]) / y_test[non_zero]))
    
    elif metric == 'direction_accuracy':
        # Check if sign (direction) of prediction matches actual
        actual_direction = np.sign(y_test)
        pred_direction = np.sign(y_pred)
        accuracy = np.mean(actual_direction == pred_direction)
        return accuracy
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def generate_forecast_summary(current_price, predicted_return, model_metrics):
    """
    Generate human-readable forecast summary.
    
    Args:
        current_price (float): Current stock price
        predicted_return (float): Predicted next-day return
        model_metrics (dict): Model evaluation metrics (from evaluate())
    
    Returns:
        str: Formatted summary text
    
    Example Output:
        ========================================
        📈 PREDICTION SUMMARY
        ========================================
        Current Price:    $123.45
        Predicted Return: +2.34%
        Predicted Price:  $126.34
        
        Target Price:     $126.34
        Price Change:     +$2.89
        
        Model Confidence: MODERATE (R² = 0.15)
        
        ⚠️  Disclaimer: This is educational only,
            not financial advice.
        ========================================
    """
    
    predicted_price = price_from_return(current_price, predicted_return)
    price_change = predicted_price - current_price
    
    # Determine confidence level
    r2 = model_metrics.get('r2', 0)
    if r2 < 0:
        confidence = "POOR (negative R²)"
    elif r2 < 0.05:
        confidence = "LOW (R² < 0.05)"
    elif r2 < 0.15:
        confidence = "MODERATE (R² < 0.15)"
    elif r2 < 0.30:
        confidence = "GOOD (R² < 0.30)"
    else:
        confidence = "EXCELLENT (R² ≥ 0.30)"
    
    summary = f"""
{'='*50}
📈 PREDICTION SUMMARY
{'='*50}
Current Price:      ${current_price:.2f}
Predicted Return:   {predicted_return:+.2%}
Predicted Price:    ${predicted_price:.2f}

Price Change:       {price_change:+.2f} ({price_change/current_price:+.2%})

Model Confidence:   {confidence}

⚠️  Disclaimer:
    This is educational only. Not financial advice.
    Past performance ≠ future results.
    Verify with professional analysis.
{'='*50}
    """
    
    return summary
