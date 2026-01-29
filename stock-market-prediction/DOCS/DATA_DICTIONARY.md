# Data Dictionary & Schema Reference

## Complete Reference for All Data Structures

---

## 📊 Raw Market Data (from Alpaca API)

### OHLCV DataFrame Structure

```
DataFrame Shape: (1250, 5)
Index: DatetimeIndex (trading dates only)

Columns:
┌─────────┬──────────┬─────────────────────────────────────┐
│ Column  │ Type     │ Description                         │
├─────────┼──────────┼─────────────────────────────────────┤
│ Open    │ float64  │ Opening price at 9:30 AM ET         │
│ High    │ float64  │ Highest price during trading day    │
│ Low     │ float64  │ Lowest price during trading day     │
│ Close   │ float64  │ Closing price at 4:00 PM ET         │
│ Volume  │ float64  │ Number of shares traded (millions)  │
└─────────┴──────────┴─────────────────────────────────────┘

Example Data:
               Open    High     Low   Close       Volume
2024-01-01   125.00  126.50  124.75  126.25   2,500,000.0
2024-01-02   126.30  128.00  126.00  127.50   3,200,000.0
2024-01-03   127.45  128.50  126.80  127.20   2,800,000.0
...

Data Ranges:
• Prices: $50 - $500+ (varies by stock)
• Volume: 500K - 50M shares per day
• Date Range: 2020-present (5 years)
• Trading Days: ~252 per year = 1,250+ total
```

---

## 🔧 Feature Engineering Output

### Extended DataFrame with Indicators

```
DataFrame Shape: (1200, 11)  [after removing NaN]
Index: DatetimeIndex

New Columns Added:
┌──────────────┬──────────┬─────────────────────────────────┐
│ Column       │ Type     │ Description                     │
├──────────────┼──────────┼─────────────────────────────────┤
│ SMA_20       │ float64  │ 20-day Simple Moving Average    │
│ SMA_50       │ float64  │ 50-day Simple Moving Average    │
│ SMA_200      │ float64  │ 200-day Simple Moving Average   │
│ RSI_14       │ float64  │ 14-day Relative Strength Index  │
│ Volatility   │ float64  │ 20-day standard deviation       │
│ Volume_MA    │ float64  │ 20-day volume moving average    │
│ Return       │ float64  │ Next-day return (TARGET)        │
└──────────────┴──────────┴─────────────────────────────────┘

Data Ranges:
• SMA_20:     Same range as Close price
• SMA_50:     Same range as Close price
• SMA_200:    Same range as Close price
• RSI_14:     0 - 100 (scale)
• Volatility: 0.005 - 0.050 (daily, ~0.5% - 5%)
• Volume_MA:  500K - 50M (similar to Volume)
• Return:     -0.15 to +0.15 (-15% to +15% per day)

Example Data:
              SMA_20  SMA_50  RSI_14  Volatility  Return
2024-01-20   107.34  105.12    55.2       0.012   0.0234
2024-01-21   107.89  105.45    58.1       0.014   0.0198
2024-01-22   108.12  105.78    60.5       0.016  -0.0045
...

NaN Handling:
• First 200 rows: NaN (warmup for indicators)
• Last 1 row: NaN (no future price for target)
• Dropped during prepare_data() function
```

---

## 🎯 Model Input Features (X)

### Feature Matrix for Random Forest

```
Array Shape: (1000, 6)  [after NaN removal and splitting]
Type: numpy.ndarray or pd.DataFrame

Features (in order):
┌──────────────┬──────────┬─────────────────────────────────┐
│ Index │ Name       │ Type     │ Range & Meaning             │
├───────┼──────────┬──────────┼─────────────────────────────┤
│ [0]   │ SMA_20     │ float64  │ Price level (~100-200)      │
│ [1]   │ SMA_50     │ float64  │ Price level (~100-200)      │
│ [2]   │ SMA_200    │ float64  │ Price level (~100-200)      │
│ [3]   │ RSI_14     │ float64  │ 0-100 (momentum)            │
│ [4]   │ Volatility │ float64  │ 0.005-0.050 (risk)          │
│ [5]   │ Volume_MA  │ float64  │ 500K-50M (liquidity)        │
└───────┴──────────┴──────────┴─────────────────────────────┘

Feature Properties:
• All numerical (no categorical)
• Different scales (feature scaling applied)
• All continuous (no discrete)
• Positive values only
• No missing values (NaN removed)

Example Row (single sample):
[107.34, 105.12, 101.50, 55.2, 0.012, 2500000.0]

After StandardScaler:
[-0.321, -0.456, -0.789, 0.234, -0.123, 0.567]
(μ=0, σ=1 for each feature)
```

---

## 🎯 Model Target Variable (y)

### Prediction Target: Next-Day Return

```
Array Shape: (1000,)  [same length as X]
Type: numpy.ndarray (1D)

Values: Daily Returns
┌──────────────────────────────────────────────┐
│ Definition:                                  │
│ Return(t+1) = (Close(t+1) - Close(t)) /     │
│               Close(t)                       │
└──────────────────────────────────────────────┘

Example Calculations:
┌────────────┬────────┬────────────┬────────────┐
│ Date       │ Close  │ Next Close │ Return     │
├────────────┼────────┼────────────┼────────────┤
│ 2024-01-01 │ 100.00 │ 103.00     │ +0.0300    │
│ 2024-01-02 │ 103.00 │ 101.50     │ -0.0146    │
│ 2024-01-03 │ 101.50 │ 104.00     │ +0.0246    │
└────────────┴────────┴────────────┴────────────┘

Range:
• Typical: -0.10 to +0.10 (-10% to +10%)
• Extreme: -0.30 to +0.30 (rare crash/rally days)
• Mean: ~0.0005 (tiny positive bias)
• Std Dev: ~0.02 (2% typical daily volatility)

Properties:
• Continuous (not discrete)
• Can be negative (down days)
• Can be positive (up days)
• Mean ≈ 0 (random walk property)
• Normally distributed (approximately)
```

---

## 📈 Model Output: Predictions

### Predicted Returns (y_pred)

```
Array Shape: (300,)  [test set size, 30% of data]
Type: numpy.ndarray
Range: -0.15 to +0.15

Example Predictions:
Test Sample  Actual Return  Predicted Return  Error
      1      +0.0234        +0.0211            -0.0023
      2      -0.0045        +0.0012            +0.0057
      3      +0.0198        +0.0185            -0.0013
      4      +0.0120        +0.0142            +0.0022
     ...      ...            ...                ...

Conversion to Price:
Predicted_Price = Current_Price × (1 + Predicted_Return)

Example:
Current_Price = $125.50
Predicted_Return = +0.0234 (2.34%)
Predicted_Price = $125.50 × 1.0234 = $128.43

Error Metrics:
┌─────────────────────────────────────┐
│ RMSE = 0.01536 (1.536% avg error)  │
│ MAE  = 0.01234 (1.234% avg error)  │
│ R²   = 0.1234 (explains 12.34%)    │
└─────────────────────────────────────┘
```

---

## 📊 Evaluation Metrics Output

### Model Performance Dictionary

```python
model_metrics = {
    'mse': 0.00023567,          # Mean Squared Error
    'rmse': 0.01535634,         # Root Mean Squared Error
    'mae': 0.01234567,          # Mean Absolute Error
    'r2': 0.12345678,           # R-squared coefficient
    'n_samples': 300            # Number of test samples
}

Feature Importance Dictionary:
┌─────────────────────────────────────┐
│ 'SMA_20':     0.3456  (34.56%)      │
│ 'RSI_14':     0.2134  (21.34%)      │
│ 'Volatility': 0.1876  (18.76%)      │
│ 'SMA_50':     0.1234  (12.34%)      │
│ 'SMA_200':    0.0900  (9.00%)       │
│ 'Volume_MA':  0.0400  (4.00%)       │
└─────────────────────────────────────┘
Total: 100.00%
```

---

## 📁 CSV Output Format

### Predictions CSV (NVDA_predictions.csv)

```
Date,Actual_Price,Predicted_Price,Actual_Return,Predicted_Return,Error
2024-01-15,125.45,125.32,0.0234,0.0211,0.0023
2024-01-16,127.89,127.21,0.0195,0.0192,-0.0003
2024-01-17,126.34,127.45,-0.0121,0.0098,-0.0219
2024-01-18,130.21,130.45,0.0307,0.0318,0.0011
...

Columns:
┌──────────────────┬──────────┬────────────────────────────┐
│ Column           │ Type     │ Description                │
├──────────────────┼──────────┼────────────────────────────┤
│ Date             │ datetime │ Trading date               │
│ Actual_Price     │ float    │ Real closing price         │
│ Predicted_Price  │ float    │ Model's predicted price    │
│ Actual_Return    │ float    │ Real % change              │
│ Predicted_Return │ float    │ Predicted % change         │
│ Error            │ float    │ Actual - Predicted         │
└──────────────────┴──────────┴────────────────────────────┘

Statistics:
• Total rows: 300 (test set size)
• Date range: Depends on training data
• Prices: $50-$500+ (varies by stock)
• Returns: -15% to +15% typical
```

---

## 📄 Report File Format

### NVDA_report.txt

```
======================================================================
📊 MODEL PERFORMANCE REPORT
======================================================================

Stock Symbol: NVDA
Generated: 2024-01-15 14:30:45

======================================================================
EVALUATION METRICS
======================================================================

Mean Squared Error (MSE):        0.00045678
Root Mean Squared Error (RMSE):  0.02136234
Mean Absolute Error (MAE):       0.01567892
R-squared (R²):                  0.1234

Test Set Size: 300 samples

======================================================================
TOP IMPORTANT FEATURES
======================================================================

1. SMA_20      : 0.3456 (34.56%)
2. RSI_14      : 0.2134 (21.34%)
3. Volatility  : 0.1876 (18.76%)
4. SMA_50      : 0.1234 (12.34%)
5. Volume_MA   : 0.1300 (13.00%)
```

---

## 🔄 Data Shape Changes Through Pipeline

### Transformation Summary

```
Raw Data (from API)
    ↓
Shape: (1250, 5)  [1250 trading days, 5 columns: OHLCV]
    ↓
Feature Engineering
    ↓
Shape: (1250, 11)  [Added 6 features: SMAs, RSI, Vol, Vol_MA, Return]
    ↓
NaN Removal (warmup period)
    ↓
Shape: (1200, 11)  [First 50 rows removed due to warmup]
    ↓
Train/Test Split (70/30)
    ↓
Training Set: (840, 6)   [Features only, no OHLCV]
Testing Set: (360, 6)    [Features only, no OHLCV]
    ↓
Feature Scaling (StandardScaler)
    ↓
Training Set: (840, 6)   [Features normalized μ=0, σ=1]
Testing Set: (360, 6)    [Features normalized using train stats]
    ↓
Model Training on Training Set
Model Evaluation on Testing Set
    ↓
Predictions: (360,)  [One predicted return per test sample]
```

---

## 🎯 Memory Usage Estimate

```
Raw DataFrame:
1250 rows × 5 columns × 8 bytes (float64) = ~50 KB

Featured DataFrame:
1250 rows × 11 columns × 8 bytes = ~110 KB

Arrays (after processing):
X_train: 840 × 6 × 8 = ~40 KB
X_test: 360 × 6 × 8 = ~17 KB
y_train: 840 × 8 = ~7 KB
y_test: 360 × 8 = ~3 KB

Trained Model:
100 trees × ~100 KB per tree = ~10 MB

Total Memory Usage: ~10-15 MB (very small!)
```

---

## 🔍 Data Quality Checks

### Validation Rules Applied

```
Input Validation:
✓ Symbol exists and is tradeable
✓ Data contains no duplicates
✓ Prices are positive numbers
✓ Dates are in chronological order
✓ Volume is non-negative
✓ OHLC ordering: Low <= Close <= High

Feature Validation:
✓ All indicators calculated successfully
✓ No infinite or NaN values in features
✓ Feature values within expected ranges
✓ All features have variance (not constant)

Data Alignment:
✓ Features and target same length
✓ No data leakage (test data not in training)
✓ Chronological train/test split
✓ No missing values in final dataset

Model Assumptions:
✓ Target variable is numeric
✓ Features are numeric
✓ Sample independence (mostly true)
✓ Feature-target relationship exists
```

---

## 📊 Typical Value Ranges

### Expected Values for Each Feature

```
Feature        │ Min    │ Typical │ Max
───────────────┼────────┼─────────┼─────────
SMA_20         │ $50    │ $100    │ $500+
SMA_50         │ $50    │ $100    │ $500+
SMA_200        │ $50    │ $100    │ $500+
RSI_14         │ 0      │ 50      │ 100
Volatility     │ 0.005  │ 0.020   │ 0.100
Volume_MA      │ 500K   │ 5M      │ 50M
───────────────┼────────┼─────────┼─────────
Actual_Return  │ -0.30  │ 0.00    │ +0.30
Predicted_Return│ -0.20 │ 0.00    │ +0.20
```

---

## 🎓 Understanding Metric Relationships

### How Metrics Relate to Each Other

```
MSE ───→ Square root ───→ RMSE
  (larger errors penalized more)   (same units as target)

RMSE ═══════════════════════════════╗
  ↓                                 ║
All Residuals Considered      Overall Accuracy
  ↓                                 ║
MAE ───→ Less outlier sensitive ────╝

═══════════════════════════════════════════════════════════

R² ───→ Proportion of variance explained
   0% = Model is useless
  10% = Typical for stocks (good!)
  50% = Excellent
 100% = Perfect prediction

═══════════════════════════════════════════════════════════

Feature Importance ───→ Which features matter most
                        (Sum to 100%)
```

---

**End of Data Dictionary**

For more details, see README.md and inline code comments.
