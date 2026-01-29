"""
===============================================================================
VISUALIZATION MODULE - PLOTTING RESULTS
===============================================================================

Module Purpose:
    Generate publication-quality plots comparing actual vs predicted prices.
    Visualization helps understand model performance and prediction accuracy.

Visualization Types:

    1. PRICE CURVE PLOT:
       Shows actual vs predicted closing prices over time
       - X-axis: Time (trading days)
       - Y-axis: Stock price
       - Blue line: Actual historical prices
       - Red line: Model predictions
       - Helps identify: lag, systematic bias, turning points
    
    2. RETURN DISTRIBUTION:
       Histogram of actual vs predicted returns
       - Shows if predictions match actual distribution
       - Identifies systematic over/under-prediction
    
    3. RESIDUAL PLOT:
       Actual return - Predicted return over time
       - Helps identify patterns in prediction errors
       - Random residuals indicate good model
       - Correlated residuals indicate model weakness
    
    4. SCATTER PLOT:
       Actual vs Predicted (scatter points)
       - Each point represents one day
       - Points on y=x line indicate perfect predictions
       - Distance from line = prediction error

Why Visualization Matters for ML:

    ✓ Visual inspection catches patterns missing in metrics
    ✓ Helps identify systematic biases
    ✓ Shows when model fails (e.g., during market crashes)
    ✓ Communicates results to non-technical stakeholders
    ✓ Detects overfitting vs underfitting
    ✓ Identifies data quality issues

Dependencies:
    - matplotlib: Plot generation
    - numpy: Numerical operations
    - pandas: Data manipulation
===============================================================================
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime
import os


def plot_actual_vs_predicted(
    actual_prices,
    predicted_prices,
    dates=None,
    stock_symbol='STOCK',
    model_name='Random Forest',
    save_path=None,
    figsize=(14, 7)
):
    """
    Plot actual vs predicted stock prices over time.
    
    This is the primary visualization showing model performance visually.
    
    Args:
        actual_prices (np.ndarray or list): Actual historical prices
        predicted_prices (np.ndarray or list): Model predicted prices
        dates (pd.DatetimeIndex or list): Date labels for x-axis
                                          If None, uses sequential indices
        stock_symbol (str): Stock ticker for title. Default: 'STOCK'
        model_name (str): Model name for legend. Default: 'Random Forest'
        save_path (str): Path to save figure. If None, doesn't save
        figsize (tuple): Figure size (width, height). Default: (14, 7)
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    
    Visual Elements:
        - Blue line: Actual prices (ground truth)
        - Red line: Predicted prices (model output)
        - Shaded area: Prediction error zone
        - Grid: For easy reading
        - Labels: Clear axis and title labels
        - Legend: Identifies both lines
    
    Workflow:
        1. Create figure and axes
        2. Plot actual prices (blue line)
        3. Plot predicted prices (red line)
        4. Add shaded region for error
        5. Format axes (labels, legend, grid)
        6. Save if path provided
        7. Return figure handles
    
    Example:
        >>> fig, ax = plot_actual_vs_predicted(
        ...     actual_prices=np.array([100, 101, 102, ...]),
        ...     predicted_prices=np.array([100, 100.5, 101.8, ...]),
        ...     dates=pd.date_range('2024-01-01', periods=250),
        ...     stock_symbol='NVDA',
        ...     save_path='nvda_prediction.png'
        ... )
        >>> plt.show()
    """
    
    # Convert to numpy arrays if needed
    actual_prices = np.asarray(actual_prices)
    predicted_prices = np.asarray(predicted_prices)
    
    # Create x-axis values
    if dates is None:
        x_values = np.arange(len(actual_prices))
        x_label = 'Trading Days'
    else:
        x_values = pd.to_datetime(dates) if not isinstance(dates, pd.DatetimeIndex) else dates
        x_label = 'Date'
    
    # =========================================================================
    # Create figure and plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual prices (ground truth)
    ax.plot(
        x_values,
        actual_prices,
        label='Actual Price',
        color='#1f77b4',  # Blue
        linewidth=2.5,
        marker='o',
        markersize=3,
        alpha=0.8,
        zorder=2
    )
    
    # Plot predicted prices (model output)
    ax.plot(
        x_values,
        predicted_prices,
        label=f'{model_name} Prediction',
        color='#d62728',  # Red
        linewidth=2.5,
        marker='s',
        markersize=3,
        alpha=0.8,
        linestyle='--',
        zorder=2
    )
    
    # Add shaded region for prediction error
    ax.fill_between(
        range(len(actual_prices)),
        actual_prices,
        predicted_prices,
        alpha=0.15,
        color='gray',
        label='Prediction Error',
        zorder=1
    )
    
    # =========================================================================
    # Format axes
    # =========================================================================
    
    # Title
    ax.set_title(
        f'{stock_symbol} Stock Price: Actual vs Predicted\n{model_name}',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    # Labels
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    
    # Format x-axis for dates
    if isinstance(x_values, pd.DatetimeIndex):
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(
        loc='best',
        fontsize=11,
        framealpha=0.95,
        edgecolor='black'
    )
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # =========================================================================
    # Save if path provided
    # =========================================================================
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Plot saved to {save_path}")
    
    print(f"[INFO] Price comparison plot created for {stock_symbol}")
    
    return fig, ax


def plot_returns_comparison(
    actual_returns,
    predicted_returns,
    stock_symbol='STOCK',
    save_path=None,
    figsize=(14, 7)
):
    """
    Create side-by-side comparison of actual vs predicted returns.
    
    Args:
        actual_returns (np.ndarray): Series of actual daily returns
        predicted_returns (np.ndarray): Series of predicted daily returns
        stock_symbol (str): Stock ticker. Default: 'STOCK'
        save_path (str): Path to save figure
        figsize (tuple): Figure size
    
    Returns:
        tuple: (fig, axes) matplotlib figure and axes array
    
    Workflow:
        1. Create 2x2 subplot grid
        2. Plot 1: Actual returns histogram
        3. Plot 2: Predicted returns histogram
        4. Plot 3: Return comparison over time
        5. Plot 4: Scatter plot (actual vs predicted)
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f'{stock_symbol} Returns Analysis: Actual vs Predicted',
        fontsize=16,
        fontweight='bold'
    )
    
    # =========================================================================
    # Plot 1: Actual returns distribution
    # =========================================================================
    ax = axes[0, 0]
    ax.hist(
        actual_returns * 100,  # Convert to percentage
        bins=40,
        color='#1f77b4',
        alpha=0.7,
        edgecolor='black'
    )
    ax.set_title('Actual Returns Distribution', fontweight='bold')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    ax.axvline(actual_returns.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {actual_returns.mean()*100:.3f}%')
    ax.legend()
    
    # =========================================================================
    # Plot 2: Predicted returns distribution
    # =========================================================================
    ax = axes[0, 1]
    ax.hist(
        predicted_returns * 100,  # Convert to percentage
        bins=40,
        color='#d62728',
        alpha=0.7,
        edgecolor='black'
    )
    ax.set_title('Predicted Returns Distribution', fontweight='bold')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    ax.axvline(predicted_returns.mean() * 100, color='blue', linestyle='--', linewidth=2, label=f'Mean: {predicted_returns.mean()*100:.3f}%')
    ax.legend()
    
    # =========================================================================
    # Plot 3: Returns over time
    # =========================================================================
    ax = axes[1, 0]
    x_vals = np.arange(len(actual_returns))
    ax.plot(x_vals, actual_returns * 100, label='Actual', color='#1f77b4', linewidth=1.5, alpha=0.8)
    ax.plot(x_vals, predicted_returns * 100, label='Predicted', color='#d62728', linewidth=1.5, alpha=0.8, linestyle='--')
    ax.set_title('Returns Over Time', fontweight='bold')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    # =========================================================================
    # Plot 4: Scatter plot (actual vs predicted)
    # =========================================================================
    ax = axes[1, 1]
    ax.scatter(actual_returns * 100, predicted_returns * 100, alpha=0.5, s=30, color='#1f77b4')
    
    # Add perfect prediction line
    min_val = min(actual_returns.min(), predicted_returns.min()) * 100
    max_val = max(actual_returns.max(), predicted_returns.max()) * 100
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_title('Actual vs Predicted Returns', fontweight='bold')
    ax.set_xlabel('Actual Return (%)')
    ax.set_ylabel('Predicted Return (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Returns comparison plot saved to {save_path}")
    
    print(f"[INFO] Returns comparison plot created for {stock_symbol}")
    
    return fig, axes


def plot_residuals(
    actual_returns,
    predicted_returns,
    dates=None,
    stock_symbol='STOCK',
    save_path=None,
    figsize=(14, 6)
):
    """
    Plot prediction errors (residuals) over time.
    
    Args:
        actual_returns (np.ndarray): Actual returns
        predicted_returns (np.ndarray): Predicted returns
        dates (pd.DatetimeIndex): Date labels
        stock_symbol (str): Stock ticker
        save_path (str): Path to save figure
        figsize (tuple): Figure size
    
    Returns:
        tuple: (fig, ax) figure and axes
    
    Interpretation:
        - Random scatter around zero = good model
        - Systematic patterns = model weakness
        - Large outliers = model fails on extreme days
    """
    
    # Calculate residuals
    residuals = actual_returns - predicted_returns
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(
        f'{stock_symbol} Prediction Residuals (Errors)',
        fontsize=14,
        fontweight='bold'
    )
    
    # =========================================================================
    # Plot 1: Residuals over time
    # =========================================================================
    if dates is None:
        x_values = np.arange(len(residuals))
    else:
        x_values = pd.to_datetime(dates) if not isinstance(dates, pd.DatetimeIndex) else dates
    
    colors = ['#d62728' if r > 0 else '#1f77b4' for r in residuals]
    ax1.bar(range(len(residuals)), residuals * 100, color=colors, alpha=0.6)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.set_title('Residuals Over Time', fontweight='bold')
    ax1.set_ylabel('Residual (%)')
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 2: Residual distribution
    # =========================================================================
    ax2.hist(residuals * 100, bins=40, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax2.axvline(residuals.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean()*100:.4f}%')
    ax2.axvline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Residual Distribution', fontweight='bold')
    ax2.set_xlabel('Residual (%)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Residuals plot saved to {save_path}")
    
    return fig, (ax1, ax2)


def create_summary_report(
    stock_symbol,
    model_metrics,
    feature_importance_df=None,
    save_path=None
):
    """
    Create a text-based summary report of model performance.
    
    Args:
        stock_symbol (str): Stock ticker
        model_metrics (dict): Metrics from model.evaluate()
        feature_importance_df (pd.DataFrame): Feature importance data
        save_path (str): Path to save report
    
    Returns:
        str: Formatted report text
    """
    
    report = f"""
{'='*70}
📊 MODEL PERFORMANCE REPORT
{'='*70}

Stock Symbol: {stock_symbol}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
EVALUATION METRICS
{'='*70}

Mean Squared Error (MSE):        {model_metrics['mse']:.8f}
Root Mean Squared Error (RMSE):  {model_metrics['rmse']:.8f}
Mean Absolute Error (MAE):       {model_metrics['mae']:.8f}
R-squared (R²):                  {model_metrics['r2']:.6f}

Test Set Size: {model_metrics.get('n_samples', 'N/A')} samples

{'='*70}
INTERPRETATION GUIDE
{'='*70}

R² Score:
  - Range: 0 to 1 (negative values indicate worse than mean)
  - Your Score: {model_metrics['r2']:.4f}
  - Interpretation: Model explains {model_metrics['r2']*100:.2f}% of variance
  
  - R² < 0.00: Poor (worse than predicting mean)
  - 0.00 < R² < 0.10: Weak (typical for stock returns)
  - 0.10 < R² < 0.30: Moderate (good for market prediction)
  - 0.30 < R² < 0.50: Strong
  - R² > 0.50: Excellent (rare for stock returns)

RMSE:
  - Average absolute prediction error in return units
  - Your RMSE: {model_metrics['rmse']:.6f} ({model_metrics['rmse']*100:.4f}%)
  
MAE:
  - Mean of absolute errors (less sensitive to outliers)
  - Your MAE: {model_metrics['mae']:.6f} ({model_metrics['mae']*100:.4f}%)

"""
    
    # Add feature importance if available
    if feature_importance_df is not None and len(feature_importance_df) > 0:
        report += f"""{'='*70}
TOP IMPORTANT FEATURES
{'='*70}

"""
        for idx, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
            report += f"{idx:2d}. {row['Feature']:20s}: {row['Importance']:.6f} ({row['Importance']*100:.2f}%)\n"
    
    report += f"""
{'='*70}
DISCLAIMER
{'='*70}

This analysis is for EDUCATIONAL purposes only.
Not financial advice. Past performance ≠ future results.

Always verify with professional financial advisors.

{'='*70}
    """
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"[SUCCESS] Report saved to {save_path}")
    
    return report
