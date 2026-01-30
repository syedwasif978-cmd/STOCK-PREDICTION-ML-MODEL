
import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.alpaca_fetch import fetch_stock_data, validate_symbol
from model.train import prepare_data, train_and_evaluate_model
from model.predict import (
    generate_price_curve,
    calculate_prediction_accuracy,
    generate_forecast_summary
)
from visualization.plot_results import (
    plot_actual_vs_predicted,
    plot_returns_comparison,
    plot_residuals,
    
    create_summary_report
)


def cli_main():
    """
    CLI entry point for the stock market analysis system.

    This function runs the interactive terminal workflow (prompts via input()).
    Use the `--cli` flag when running the script if you want the terminal interface.
    """
    
    print("\n" + "="*70)
    print("📈 STOCK MARKET ANALYSIS & PREDICTION SYSTEM 📈")
    print("="*70)
    print("Machine Learning-Based Next-Day Return Prediction")
    print("="*70)
    
    # =========================================================================
    # STEP 1: Get stock symbol from user
    # =========================================================================
    while True:
        stock_symbol = input("\n📍 Enter stock symbol (e.g., NVDA, AAPL, TSLA): ").strip().upper()
        
        if not stock_symbol:
            print("[ERROR] Please enter a valid stock symbol")
            continue
        
        if validate_symbol(stock_symbol):
            break
        else:
            print(f"[ERROR] Invalid symbol: {stock_symbol}. Use uppercase letters only.")
    
    print(f"\n✓ Selected: {stock_symbol}")
    
    # =========================================================================
    # STEP 2: Fetch historical data from Alpaca API
    # =========================================================================
    print(f"\n[INFO] Fetching historical data for {stock_symbol}...")
    
    try:
        # Fetch 5 years of data (1250+ trading days)
        df = fetch_stock_data(stock_symbol, days=365*5)
        
        if df.empty or len(df) < 250:
            print(f"\n[ERROR] Insufficient data: {len(df)} trading days found")
            print("[ERROR] Need at least 250 trading days for meaningful analysis")
            return
        
        print(f"\n✓ Data retrieved successfully: {len(df)} trading days")
        print(f"✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to fetch data: {str(e)}")
        print("[HINT] Check your Alpaca API credentials:")
        print("       - Windows: set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
        print("       - Linux/Mac: export APCA_API_KEY_ID and APCA_API_SECRET_KEY")
        return
    
    # =========================================================================
    # STEP 3: Prepare data (feature engineering, train/test split)
    # =========================================================================
    print(f"\n[INFO] Preparing data for model training...")
    
    try:
        X_train, X_test, y_train, y_test, feature_names, dates = prepare_data(
            df,
            test_size=0.3,    # 70% train, 30% test
            sequence_offset=1  # Predict 1 day ahead
        )
        
    except Exception as e:
        print(f"\n[ERROR] Data preparation failed: {str(e)}")
        return
    
    # =========================================================================
    # STEP 4: Train Random Forest model
    # =========================================================================
    print(f"\n[INFO] Training Random Forest model...")
    
    try:
        model = train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names)
        
    except Exception as e:
        print(f"\n[ERROR] Model training failed: {str(e)}")
        return
    
    # =========================================================================
    # STEP 5: Generate predictions and evaluate
    # =========================================================================
    print(f"\n[INFO] Generating predictions...")
    
    # Get predictions on test set
    y_pred = model.predict(X_test, scale_features=True)
    
    # Calculate accuracy metrics
    rmse = calculate_prediction_accuracy(y_test, y_pred, metric='rmse')
    mae = calculate_prediction_accuracy(y_test, y_pred, metric='mae')
    direction_accuracy = calculate_prediction_accuracy(y_test, y_pred, metric='direction_accuracy')
    
    print(f"\n[INFO] Prediction Accuracy Metrics:")
    print(f"       - RMSE: {rmse:.6f} ({rmse*100:.4f}%)")
    print(f"       - MAE:  {mae:.6f} ({mae*100:.4f}%)")
    print(f"       - Direction Accuracy: {direction_accuracy*100:.2f}%")
    print(f"         (Correctly predicts if price goes up or down)")
    
    # =========================================================================
    # STEP 6: Generate price curves for visualization
    # =========================================================================
    print(f"\n[INFO] Generating price curves...")
    
    # Get last training price as reference
    last_price = df['Close'].iloc[len(df) - len(y_test) - 1]
    
    actual_prices, predicted_prices = generate_price_curve(
        y_test,
        last_price,
        y_pred
    )
    
    # =========================================================================
    # STEP 7: Create visualizations
    # =========================================================================
    print(f"\n[INFO] Creating visualizations...")
    
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Actual vs Predicted prices
    fig1, ax1 = plot_actual_vs_predicted(
        actual_prices[1:],  # Skip first point (reference price)
        predicted_prices[1:],
        dates=dates[1][1:],  # Use test dates
        stock_symbol=stock_symbol,
        model_name='Random Forest',
        save_path=os.path.join(output_dir, f'{stock_symbol}_prediction.png')
    )
    
    # Plot 2: Returns comparison
    fig2, axes2 = plot_returns_comparison(
        y_test,
        y_pred,
        stock_symbol=stock_symbol,
        save_path=os.path.join(output_dir, f'{stock_symbol}_returns.png')
    )
    
    # Plot 3: Residual analysis
    fig3, axes3 = plot_residuals(
        y_test,
        y_pred,
        dates=dates[1],
        stock_symbol=stock_symbol,
        save_path=os.path.join(output_dir, f'{stock_symbol}_residuals.png')
    )
    
    # =========================================================================
    # STEP 8: Create performance report
    # =========================================================================
    print(f"\n[INFO] Generating performance report...")
    
    # Get feature importance
    importance_df = model.get_feature_importance(feature_names, top_n=15)
    
    # Create report
    report = create_summary_report(
        stock_symbol,
        model.training_history,
        importance_df,
        save_path=os.path.join(output_dir, f'{stock_symbol}_report.txt')
    )
    
    print(report)
    
    # =========================================================================
    # STEP 9: Predict next day and forecast
    # =========================================================================
    print(f"\n[INFO] Making next-day prediction...")
    
    # Get latest features (last row of test set)
    latest_features = X_test[-1:] * 1.0  # Make a copy
    
    # Normalize using the model's scaler
    latest_features_scaled = model.scaler.transform(latest_features)
    
    # Predict next day
    next_day_return = model.predict(latest_features, scale_features=True)[0]
    next_day_price = df['Close'].iloc[-1] * (1 + next_day_return)
    
    # Create forecast summary
    forecast_summary = generate_forecast_summary(
        df['Close'].iloc[-1],
        next_day_return,
        model.training_history
    )
    
    print(forecast_summary)
    
    # =========================================================================
    # STEP 10: Save predictions to CSV
    # =========================================================================
    print(f"\n[INFO] Saving predictions to CSV...")
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'Date': dates[1],
        'Actual_Price': actual_prices[1:],
        'Predicted_Price': predicted_prices[1:],
        'Actual_Return': y_test,
        'Predicted_Return': y_pred,
        'Error': y_test - y_pred
    })
    
    predictions_df.to_csv(
        os.path.join(output_dir, f'{stock_symbol}_predictions.csv'),
        index=False
    )
    
    print(f"[SUCCESS] Predictions saved")
    
    # =========================================================================
    # STEP 11: Summary and next steps
    # =========================================================================
    print(f"\n" + "="*70)
    print("[SUCCESS] ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n📂 Output Files (in '{output_dir}/' directory):")
    print(f"   1. {stock_symbol}_prediction.png - Price prediction plot")
    print(f"   2. {stock_symbol}_returns.png - Returns analysis")
    print(f"   3. {stock_symbol}_residuals.png - Prediction errors")
    print(f"   4. {stock_symbol}_report.txt - Performance report")
    print(f"   5. {stock_symbol}_predictions.csv - Detailed predictions")
    
    print(f"\n📊 Key Metrics:")
    print(f"   - Test Set R²: {model.training_history['r2']:.4f}")
    print(f"   - Prediction RMSE: {rmse*100:.4f}%")
    print(f"   - Direction Accuracy: {direction_accuracy*100:.2f}%")
    
    print(f"\n🔮 Next Day Prediction ({stock_symbol}):")
    print(f"   - Current Price: ${df['Close'].iloc[-1]:.2f}")
    print(f"   - Predicted Return: {next_day_return:+.2%}")
    print(f"   - Target Price: ${next_day_price:.2f}")
    
    print(f"\n⚠️  Important Disclaimers:")
    print(f"   - This is EDUCATIONAL material only")
    print(f"   - NOT financial or investment advice")
    print(f"   - Past performance ≠ future results")
    print(f"   - Always consult financial professionals")
    print(f"   - Stock market predictions are inherently uncertain")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Review the generated visualizations")
    print(f"   2. Analyze feature importance to understand model decisions")
    print(f"   3. Validate predictions with real-world data")
    print(f"   4. Consider ensemble methods for improved accuracy")
    print(f"   5. Explore LSTM/Transformer models for sequential data")
    
    print(f"\n" + "="*70)
    print("Thank you for using Stock Market Analysis & Prediction System!")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Default: launch GUI. Use '--cli' or 'cli' to run the terminal interface.
    try:
        if len(sys.argv) > 1 and sys.argv[1].lower() in ("--cli", "cli"):
            cli_main()
        else:
            from gui_app import main as gui_main
            gui_main()
    except Exception as e:
        # If GUI fails (missing deps, headless environment), fall back to CLI
        print(f"[WARN] GUI launch failed ({e}). Falling back to CLI interface.")
        cli_main()
