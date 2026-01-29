# Quick Reference Guide

## Stock Market Prediction System - Quick Start

### 🚀 5-Minute Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
# Windows:
set APCA_API_KEY_ID=your_key
set APCA_API_SECRET_KEY=your_secret

# Linux/Mac:
export APCA_API_KEY_ID=your_key
export APCA_API_SECRET_KEY=your_secret

# 3. Run the program
python main.py

# 4. Enter stock symbol
NVDA

# 5. Wait 2-5 minutes
# Check output/ folder for results
```

---

## 📊 Key Metrics Explained

### R² Score (Most Important)
```
R² = 0.00 → Model is useless (explains 0% of variance)
R² = 0.10 → Weak but typical for stocks (10%)
R² = 0.20 → Good performance (20%)
R² = 0.50 → Excellent (50%) - rare in finance
```

### RMSE (Root Mean Squared Error)
```
RMSE = average prediction error
RMSE = 0.015 → Average error is 1.5% per day
Lower is better
```

### MAE (Mean Absolute Error)
```
MAE = average absolute deviation
Less sensitive to outliers than RMSE
Good for practical trading
```

---

## 📈 Technical Indicators

### SMA (Simple Moving Average)
```
Price = average of last N closing prices
SMA_20: 20-day average (trend)
SMA_50: 50-day average (intermediate)
SMA_200: 200-day average (long-term)
```

### RSI (Relative Strength Index, 0-100)
```
RSI < 30:   OVERSOLD (buy signal)
30-70:      NEUTRAL (normal)
RSI > 70:   OVERBOUGHT (sell signal)
```

### Volatility
```
σ = standard deviation of returns
Low volatility: <10% annual (stable)
Medium: 10-30% (normal)
High: >30% (choppy, risky)
```

---

## 📁 File Structure

```
stock-market-prediction/
├── main.py                 ← Run this file
├── README.md              ← Full documentation
├── SETUP_GUIDE.py         ← Installation guide
├── requirements.txt       ← Dependencies
├── api/alpaca_fetch.py   ← Data fetching
├── indicators/            ← SMA, RSI, Volatility
├── model/                 ← Random Forest, training, prediction
├── visualization/         ← Plotting and reports
├── output/               ← Generated results
│   ├── NVDA_prediction.png
│   ├── NVDA_returns.png
│   ├── NVDA_residuals.png
│   ├── NVDA_report.txt
│   └── NVDA_predictions.csv
└── data/                 ← Historical prices (if saved)
```

---

## 🔧 Common Tasks

### Change Stock Symbol
```python
# In main.py, modify:
stock_symbol = "AAPL"  # Instead of user input
```

### Adjust Model Parameters
```python
# In model/train.py, modify:
model = RandomForestModel(
    n_estimators=200,  # More trees = better but slower
    max_depth=20,      # Deeper = complex, may overfit
    ...
)
```

### Change Train/Test Split
```python
# In model/train.py:
X_train, X_test, ... = prepare_data(df, test_size=0.2)
# 0.2 = 20% test, 80% train (default is 0.3)
```

### Add New Feature
```python
# In model/train.py, in engineer_features():
new_feature = df['Close'].rolling(30).mean()
features_df['New_Feature'] = new_feature
```

---

## ⚠️ Common Errors & Fixes

### "Alpaca API credentials not found"
```bash
# Set environment variables:
Windows: setx APCA_API_KEY_ID "your_key"
Linux/Mac: export APCA_API_KEY_ID="your_key"

# Then restart terminal/IDE
```

### "ModuleNotFoundError: pandas"
```bash
# Install dependencies:
pip install -r requirements.txt
```

### "Invalid symbol: UNKNOWN"
```bash
# Use valid stock symbols:
NVDA, AAPL, TSLA, MSFT, GOOG, AMZN
# From https://www.nasdaq.com
```

### "Insufficient data"
```bash
# Use major stocks with 5+ years data:
- Large cap (NVDA, AAPL, MSFT)
- ETFs (SPY, QQQ, VTI)
- Not penny stocks or recently listed
```

---

## 📊 Understanding Output

### NVDA_prediction.png
```
Blue line = Actual prices (what really happened)
Red dashed = Predictions (what model thought)
Overlap = Good prediction
Divergence = Model lag or error
```

### NVDA_report.txt
```
R² = 0.1234
→ Model explains 12.34% of price movements
→ Typical for stock prediction (good!)
→ Don't expect 100% accuracy
```

### NVDA_predictions.csv
```
Columns:
- Date: Trading day
- Actual_Price: Real price
- Predicted_Price: What model predicted
- Actual_Return: Real % change
- Predicted_Return: Predicted % change
- Error: Actual - Predicted
```

---

## 🎓 Learning Path

1. **Run the system** (main.py)
   - Get comfortable with basic workflow
   - Generate your first predictions

2. **Read the code comments**
   - Each function is heavily commented
   - Understand what each step does
   - Learn the ML concepts

3. **Study the README**
   - Section 5: Technical Indicators
   - Section 6: Machine Learning Model
   - Section 7: Model Training

4. **Modify and experiment**
   - Try different stock symbols
   - Change model parameters
   - Add new features

5. **Validate your understanding**
   - Explain each indicator
   - Understand Random Forest
   - Know the evaluation metrics

---

## 🔗 Useful Links

**Documentation:**
- README.md - Full documentation
- SETUP_GUIDE.py - Installation guide
- Code comments - Detailed explanations

**APIs & Resources:**
- Alpaca Markets: https://alpaca.markets
- Technical Indicators: https://www.investopedia.com
- Scikit-learn: https://scikit-learn.org

**Learning:**
- Wikipedia: Random Forest, Technical Analysis
- Investopedia: Stock market basics
- Khan Academy: Machine learning concepts

---

## 📞 Troubleshooting Checklist

- [ ] Python 3.7+ installed? `python --version`
- [ ] Dependencies installed? `pip list | grep pandas`
- [ ] Alpaca account created? https://alpaca.markets
- [ ] API keys generated? Account → API Keys
- [ ] Environment variables set? `echo $APCA_API_KEY_ID`
- [ ] Project directory correct? `ls` shows main.py
- [ ] Virtual environment activated? `(venv)` prefix visible?
- [ ] Internet connection working?
- [ ] Stock symbol valid? (NVDA, AAPL, TSLA)
- [ ] output/ directory exists?

---

## 🎯 Success Checklist

After running main.py, you should have:

- [ ] ✅ Model training completed (shows R² score)
- [ ] ✅ NVDA_prediction.png created
- [ ] ✅ NVDA_returns.png created
- [ ] ✅ NVDA_residuals.png created
- [ ] ✅ NVDA_report.txt created
- [ ] ✅ NVDA_predictions.csv created
- [ ] ✅ Next-day prediction displayed
- [ ] ✅ Feature importance shown
- [ ] ✅ All plots visible and sensible
- [ ] ✅ Report with metrics readable

---

## 🚀 Next Steps

1. ✅ Run system successfully (you are here)
2. Run on different stocks (AAPL, TSLA, etc.)
3. Compare results across stocks
4. Study model predictions vs actual
5. Modify hyperparameters and see effects
6. Add new technical indicators
7. Try ensemble of models
8. Explore LSTM/RNN variants
9. Add backtesting framework
10. Create web dashboard

---

**Remember:** This system is for EDUCATIONAL purposes only. NOT for real trading without validation!

Good luck! 🚀📊
