"""
End-to-end test script (synthetic data)
Generates synthetic OHLCV data, runs feature engineering, trains model,
creates plots and report, and saves outputs.
"""

print('[TEST] e2e_run.py starting')

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Ensure project root is on sys.path for imports when running tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use non-interactive backend for matplotlib to avoid GUI blocking during tests
import matplotlib
matplotlib.use('Agg')

from model.train import prepare_data, train_and_evaluate_model
from model.predict import generate_price_curve, calculate_prediction_accuracy, generate_forecast_summary
from visualization.plot_results import plot_actual_vs_predicted, plot_returns_comparison, plot_residuals, create_summary_report


def generate_synthetic_data(days=500, start_price=100.0, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq='B')  # business days
    # Simulate small daily returns
    daily_returns = np.random.normal(loc=0.0005, scale=0.02, size=days)
    prices = start_price * np.cumprod(1 + daily_returns)
    volume = np.random.randint(100000, 5000000, size=days)
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.001, size=days)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, size=days))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, size=days))),
        'Close': prices,
        'Volume': volume
    }, index=dates)
    return df


def run_e2e():
    out_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(out_dir, exist_ok=True)

    print('[TEST] Generating synthetic data...')
    df = generate_synthetic_data(days=520)
    print(f"[TEST] Generated {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")

    print('[TEST] Preparing data...')
    X_train, X_test, y_train, y_test, feature_names, dates = prepare_data(df, test_size=0.3, sequence_offset=1)

    print('[TEST] Training model...')
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names)

    print('[TEST] Predicting on test set...')
    y_pred = model.predict(X_test, scale_features=True)

    rmse = calculate_prediction_accuracy(y_test, y_pred, metric='rmse')
    mae = calculate_prediction_accuracy(y_test, y_pred, metric='mae')
    dir_acc = calculate_prediction_accuracy(y_test, y_pred, metric='direction_accuracy')

    print(f"[RESULT] RMSE: {rmse:.6f}, MAE: {mae:.6f}, Direction Acc: {dir_acc*100:.2f}%")

    last_price = df['Close'].iloc[len(df) - len(y_test) - 1]
    actual_prices, predicted_prices = generate_price_curve(y_test, last_price, y_pred)

    print('[TEST] Creating visualizations...')
    # Use the full test dates series so x/y lengths match
    plot_actual_vs_predicted(actual_prices[1:], predicted_prices[1:], dates=dates[1], stock_symbol='SYNTH', save_path=os.path.join(out_dir, 'synth_prediction.png'))
    plot_returns_comparison(y_test, y_pred, stock_symbol='SYNTH', save_path=os.path.join(out_dir, 'synth_returns.png'))
    plot_residuals(y_test, y_pred, dates=dates[1], stock_symbol='SYNTH', save_path=os.path.join(out_dir, 'synth_residuals.png'))

    print('[TEST] Generating report...')
    importance_df = model.get_feature_importance(feature_names, top_n=10)
    report = create_summary_report('SYNTH', model.training_history, importance_df, save_path=os.path.join(out_dir, 'synth_report.txt'))
    print(report)

    print('[TEST] Saving predictions...')
    predictions_df = pd.DataFrame({
        'Date': dates[1],
        'Actual_Return': y_test,
        'Predicted_Return': y_pred,
        'Error': y_test - y_pred
    })
    predictions_df.to_csv(os.path.join(out_dir, 'synth_predictions.csv'), index=False)

    print('[SUCCESS] End-to-end synthetic test completed. Outputs in ./output/')


if __name__ == '__main__':
    run_e2e()
