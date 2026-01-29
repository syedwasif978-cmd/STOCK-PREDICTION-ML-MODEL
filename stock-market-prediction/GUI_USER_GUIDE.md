# Stock Market Prediction GUI - User Guide

## Overview

The Stock Market Prediction GUI is a professional desktop application that provides:

- **Real-time Stock Search** with autocomplete suggestions
- **Live Market Data** fetching from Alpaca Markets API
- **ML Predictions** using Random Forest models
- **Interactive Visualizations** showing price trends and error analysis
- **Performance Metrics** with detailed interpretation

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install pillow  # For image handling
```

### 2. Configure API Credentials

#### Windows:
```bash
setx APCA_API_KEY_ID "your_api_key_here"
setx APCA_API_SECRET_KEY "your_secret_key_here"
```

#### Linux/Mac:
```bash
export APCA_API_KEY_ID="your_api_key_here"
export APCA_API_SECRET_KEY="your_secret_key_here"
```

**Get API credentials from:** https://alpaca.markets (free paper trading account)

## Running the Application

```bash
python run_gui.py
```

Or directly:

```bash
python gui_app.py
```

## GUI Components

### 1. Search Bar (Top Section)

**Features:**
- Enter any stock symbol (NVDA, AAPL, TSLA, etc.)
- Press Enter or click "Fetch & Predict" button
- Autocomplete shows matching stocks as you type

**Popular Stocks (Quick Select):**
- Click any button: AAPL, MSFT, GOOGL, TSLA, AMZN, NVDA
- Instantly loads and predicts for that stock

**Search Suggestions:**
- Shows 10 matching stocks from a database of 25+ popular companies
- Click any suggestion to select and analyze

**Status Indicator:**
- Green: Ready
- Orange: Processing
- Red: Error occurred

### 2. Metrics & Information Panel (Left)

**Displays:**

| Metric | What it means |
|--------|--------------|
| MSE | Mean Squared Error - average error magnitude |
| RMSE | Root Mean Squared Error - in percentage terms |
| MAE | Mean Absolute Error - typical prediction error |
| R² Score | Variance explained (0-1, higher is better) |

**Interpretation Guide:**

```
R² Score > 0.7  →  ✓ Good model fit (reliable)
R² Score > 0.5  →  ~ Moderate fit (reasonable)
R² Score < 0.5  →  ✗ Poor fit (use with caution)
```

**Top Features:**
- Shows 6 most important technical indicators
- Helps understand what influences predictions

### 3. Price Prediction Chart (Middle)

**Two Subplots:**

1. **Top: Return Prediction**
   - Blue line: Actual daily returns from market
   - Orange dashed line: Model predictions
   - Gray shaded area: Prediction error zone

2. **Bottom: Error Distribution**
   - Histogram of prediction errors
   - Red line: Average error
   - Shows consistency of predictions

**Interpretation:**
- Overlapping lines = better predictions
- Wide error distribution = less reliable
- Centered distribution = unbiased model

### 4. Prediction Details Panel (Right)

**Contains:**

1. **Next-Day Forecast**
   - Current prediction quality assessment
   - Model accuracy percentage
   - Average error margin

2. **Market Analysis Basis**
   - Lists the 5 technical indicators used
   - Explains data input to the model

3. **Last Updated Time**
   - Timestamp of prediction generation

4. **Disclaimer**
   - Educational purpose only
   - NOT financial advice
   - Use for analysis, not trading decisions

## How It Works

### Data Flow:

```
User Input (Stock Symbol)
        ↓
Search & Validation
        ↓
Fetch 5 Years of Market Data (Alpaca API)
        ↓
Calculate Technical Indicators
  • SMA (20, 50, 200-day)
  • RSI (14-day momentum)
  • Volatility (20-day)
  • Volume moving average
        ↓
Prepare Data (Clean, Split 70/30, Scale)
        ↓
Train Random Forest Model (100 trees)
        ↓
Generate Predictions on Test Data
        ↓
Calculate Performance Metrics
        ↓
Create Visualizations & Reports
        ↓
Display Results in GUI
```

### Prediction Process:

1. **Data Fetching** - Gets historical OHLCV data from Alpaca Markets
2. **Feature Engineering** - Calculates 6 technical indicators
3. **Data Preparation** - Cleans data, removes NaN values, splits into train/test
4. **Model Training** - Fits Random Forest to 70% of historical data
5. **Prediction** - Predicts next-day returns on 30% test data
6. **Evaluation** - Calculates metrics (MSE, RMSE, MAE, R²)
7. **Visualization** - Creates charts showing actual vs predicted

## Technical Indicators Explained

### 1. SMA (Simple Moving Average)
- **Periods:** 20, 50, 200 days
- **Meaning:** Average price over time period
- **Use:** Identify trends (uptrend, downtrend, consolidation)

### 2. RSI (Relative Strength Index)
- **Period:** 14 days
- **Range:** 0-100
- **Meaning:** 
  - < 30: Oversold (potential buy)
  - 30-70: Neutral
  - > 70: Overbought (potential sell)

### 3. Volatility
- **Period:** 20 days
- **Meaning:** How much price fluctuates
- **Use:** Risk assessment

### 4. Volume Moving Average
- **Period:** 20 days
- **Meaning:** Average trading volume
- **Use:** Confirm price movements

## Model Explanation

### Random Forest Regressor
- **Type:** Ensemble machine learning model
- **Trees:** 100 decision trees
- **Max Depth:** 15 levels per tree
- **Algorithm:** 
  1. Create random samples of training data (bootstrap)
  2. Build decision tree for each sample
  3. Average predictions from all trees
  4. Results in robust predictions

### Why Random Forest?
- ✓ Handles non-linear relationships
- ✓ Robust to outliers
- ✓ Provides feature importance
- ✓ No normalization needed for decision trees
- ✓ Fast predictions

## Performance Metrics Guide

### MSE (Mean Squared Error)
```
Formula: MSE = Σ(actual - predicted)² / n
Meaning: Average squared error
Lower is better
Units: Return percentage²
```

### RMSE (Root Mean Squared Error)
```
Formula: RMSE = √MSE
Meaning: Square root of average squared error
Interpretable in original units
Example: RMSE = 0.05 → Average error ~5% return
```

### MAE (Mean Absolute Error)
```
Formula: MAE = Σ|actual - predicted| / n
Meaning: Average absolute error
Most interpretable metric
Example: MAE = 0.03 → Average error ~3% return
```

### R² Score
```
Formula: R² = 1 - (SS_res / SS_tot)
Meaning: Proportion of variance explained by model
Range: 0 to 1 (can be negative for poor fits)
Interpretation:
  0.9: Excellent (90% variance explained)
  0.7: Good (70% variance explained)
  0.5: Moderate (50% variance explained)
  0.3: Poor (30% variance explained)
  0.0: Model no better than mean
```

## Example Workflows

### Workflow 1: Quick Analysis

1. Click "NVDA" button
2. Wait for processing (2-5 minutes)
3. Check R² score in left panel
4. Look at chart in middle
5. Read prediction details on right

### Workflow 2: Compare Multiple Stocks

1. Enter "AAPL", click Fetch & Predict
2. Note R² score and MAE
3. Enter "MSFT", click Fetch & Predict
4. Compare which has better R²
5. Use better model for predictions

### Workflow 3: Analyze Technical Setup

1. Fetch stock data
2. Check top features in metrics panel
3. See which indicators matter most
4. Read interpretation in prediction panel
5. Consider technical context

## Troubleshooting

### "API credentials not found"
**Solution:**
```bash
# Set environment variables correctly
setx APCA_API_KEY_ID "your_key"
setx APCA_API_SECRET_KEY "your_secret"

# Then close and reopen terminal
# Or restart computer
```

### "Insufficient data for symbol"
**Cause:** Stock is too new or delisted
**Solution:** Try major stocks (AAPL, MSFT, NVDA)

### "Processing takes too long"
**Normal:** First run takes 2-5 minutes
**Reason:** Fetching 5 years of data and training model
**Note:** Subsequent runs may be faster with caching

### "Visualization not showing"
**Check:** 
- Matplotlib installed: `pip install matplotlib`
- Pillow installed: `pip install pillow`
- Proper Python version: Python 3.7+

### "Search suggestions not updating"
**Solution:** Click in search box and type again

## Advanced Tips

### 1. Interpret Prediction Quality
- R² > 0.7: Trust predictions more
- High MAE: Predictions less accurate
- Compare RMSE to stock volatility

### 2. Feature Importance Analysis
- First features are most influential
- Understand what moves the stock
- Use for fundamental analysis

### 3. Cross-Validate Results
- Test multiple stocks
- Compare R² scores
- Look for patterns

### 4. Understand Limitations
- Model trained on historical data
- Past performance ≠ future results
- Market can change (events, news)
- Black swan events unpredictable

## Important Disclaimers

⚠️ **EDUCATIONAL PURPOSE ONLY**

This system is designed to teach machine learning concepts with real market data.

❌ **NOT FINANCIAL ADVICE**
- Do not use for actual trading decisions
- Consult with financial professionals
- Markets are inherently unpredictable

⚠️ **HISTORICAL DATA LIMITATIONS**
- Model learns from past
- Future may differ completely
- New events can cause unexpected moves

## Key Concepts

### Stock Market Basics
- **Return:** Daily price change percentage
- **OHLCV:** Open, High, Low, Close, Volume
- **Volatility:** How much price fluctuates
- **Volume:** Number of shares traded

### Machine Learning
- **Training:** Model learns patterns
- **Testing:** Evaluate on unseen data
- **Overfitting:** Model learns noise, not patterns
- **Regularization:** Prevent overfitting

### Time Series
- **Sequential:** Data order matters
- **Trend:** Long-term direction
- **Seasonality:** Regular patterns
- **Stationarity:** Statistical properties consistent

## System Requirements

- **Python:** 3.7 or higher
- **Memory:** 4 GB RAM minimum
- **Storage:** 1 GB free space
- **Internet:** For API data fetching
- **OS:** Windows, macOS, Linux

## Getting Help

### Check Logs
- Look at terminal output
- Error messages indicate problems
- Traceback shows exact location

### Common Issues
- API Key: Double-check spelling
- Internet: Ensure connection active
- Dependencies: Verify all installed
- Python Version: Update if too old

### Debug Mode
Run with verbose output:
```bash
python -u gui_app.py
```

## Future Enhancements

Planned features:
- [ ] Save predictions to database
- [ ] Export reports to PDF
- [ ] Multiple timeframe analysis
- [ ] Sentiment analysis integration
- [ ] Backtesting system
- [ ] Real-time alerts
- [ ] Portfolio comparison

## References

### Libraries Used
- **Pandas:** Data manipulation
- **NumPy:** Numerical computing
- **Scikit-learn:** Machine learning
- **Matplotlib:** Visualization
- **Tkinter:** GUI framework
- **Alpaca API:** Market data

### Learning Resources
- Scikit-learn Documentation: https://scikit-learn.org
- Pandas Tutorial: https://pandas.pydata.org
- Random Forest: https://en.wikipedia.org/wiki/Random_forest
- Technical Analysis: https://www.investopedia.com

## Support

For issues or questions:
1. Check this guide
2. Review error messages
3. Check GitHub issues
4. Read code comments

---

**Happy Predicting! 📈**

Remember: Always do your own research and consult financial professionals before making investment decisions.
