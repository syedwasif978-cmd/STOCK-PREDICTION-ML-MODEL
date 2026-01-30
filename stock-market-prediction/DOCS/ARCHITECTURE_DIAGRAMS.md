# Architecture Diagrams

## 1. ACTOR USE CASE DIAGRAM

```
                          ┌─────────────────────────────────────┐
                          │   STOCK MARKET PREDICTION SYSTEM     │
                          └─────────────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
            ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
            │   END USER   │      │   ALPACA API │      │   SYSTEM     │
            │  (Analyst)   │      │  (Market     │      │  (ML Engine) │
            │              │      │   Data)      │      │              │
            └──────────────┘      └──────────────┘      └──────────────┘
                    │                      │                      │
                    │                      │                      │
        ┌───────────┴──────────┐           │                      │
        │                      │           │                      │
        ▼                      ▼           ▼                      ▼
    ┌─────────────┐    ┌──────────────┐  ┌──────────────┐    ┌──────────────┐
    │  UC-1       │    │  UC-2        │  │  UC-3        │    │  UC-4        │
    │  Input      │    │  Fetch Data  │  │  Train Model │    │  Generate    │
    │  Stock      │    │  from API    │  │              │    │  Prediction  │
    │  Symbol     │    │              │  │              │    │              │
    └─────────────┘    └──────────────┘  └──────────────┘    └──────────────┘
        │                      │                │                      │
        ▼                      ▼                ▼                      ▼
    ┌─────────────┐    ┌──────────────┐  ┌──────────────┐    ┌──────────────┐
    │  UC-5       │    │  UC-6        │  │  UC-7        │    │  UC-8        │
    │  Feature    │    │  Prepare     │  │  Evaluate    │    │  Visualize   │
    │  Engineering│    │  Data        │  │  Results     │    │  Results     │
    │             │    │  (Clean,     │  │              │    │              │
    │             │    │   Split,     │  │              │    │              │
    │             │    │   Scale)     │  │              │    │              │
    └─────────────┘    └──────────────┘  └──────────────┘    └──────────────┘
        │                      │                │                      │
        ▼                      ▼                ▼                      ▼
    ┌─────────────┐    ┌──────────────┐  ┌──────────────┐    ┌──────────────┐
    │  Calculate  │    │  Generate    │  │  Metrics     │    │  Display     │
    │  Indicators │    │  Features:   │  │  Calculation │    │  Charts &    │
    │  (SMA,RSI,  │    │  • SMA 20    │  │  (MSE, RMSE, │    │  Reports     │
    │   Volatility)    │  • SMA 50    │  │   MAE, R²)   │    │              │
    │             │    │  • SMA 200   │  │              │    │  Output:     │
    │             │    │  • RSI       │  │  Feature     │    │  • PNG plots │
    │             │    │  • Volatility│  │  Importance  │    │  • CSV data  │
    │             │    │  • Volume_MA │  │              │    │  • TXT report│
    └─────────────┘    └──────────────┘  └──────────────┘    └──────────────┘
```

### Use Cases Detailed Breakdown:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      USE CASE DESCRIPTIONS                             │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│ UC-1: INPUT STOCK SYMBOL                                             │
│       Actor: End User                                                │
│       Description: User enters stock ticker (NVDA, AAPL, etc.)      │
│       Precondition: System initialized, API credentials available   │
│       Postcondition: Valid symbol stored for processing             │
│                                                                        │
│ UC-2: FETCH DATA FROM ALPACA API                                    │
│       Actor: Alpaca API / System                                    │
│       Description: Retrieve 5 years of OHLCV data                   │
│       Precondition: Valid symbol, API keys configured               │
│       Postcondition: DataFrame with 1,250+ trading days             │
│                                                                        │
│ UC-3: TRAIN MACHINE LEARNING MODEL                                  │
│       Actor: ML Engine / System                                     │
│       Description: Fit Random Forest on training data               │
│       Precondition: Features engineered, data prepared              │
│       Postcondition: Trained model with weights saved               │
│                                                                        │
│ UC-4: GENERATE PREDICTIONS                                          │
│       Actor: ML Engine / System                                     │
│       Description: Predict next-day returns for test set            │
│       Precondition: Model trained, features available               │
│       Postcondition: Prediction array with return estimates         │
│                                                                        │
│ UC-5: FEATURE ENGINEERING                                           │
│       Actor: System / ML Engine                                     │
│       Description: Calculate SMA, RSI, Volatility indicators        │
│       Precondition: Raw OHLCV data available                        │
│       Postcondition: DataFrame with 6 new feature columns           │
│                                                                        │
│ UC-6: DATA PREPARATION                                              │
│       Actor: System / ML Engine                                     │
│       Description: Clean data, split train/test, scale features    │
│       Precondition: Features calculated                             │
│       Postcondition: X_train, X_test, y_train, y_test ready        │
│                                                                        │
│ UC-7: MODEL EVALUATION                                              │
│       Actor: System / ML Engine                                     │
│       Description: Calculate performance metrics                    │
│       Precondition: Predictions generated, actual values available  │
│       Postcondition: MSE, RMSE, MAE, R² values computed             │
│                                                                        │
│ UC-8: VISUALIZE & REPORT                                            │
│       Actor: System / End User                                      │
│       Description: Generate charts, plots, and text reports         │
│       Precondition: Predictions and metrics available               │
│       Postcondition: PNG plots, CSV data, TXT report saved          │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. SWIMLANE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STOCK MARKET PREDICTION WORKFLOW                     │
│                         (Swimlane Diagram)                              │
├────────────────┬────────────────┬────────────────┬─────────────────────┤
│   DATA LAYER   │  FEATURE ENG.  │  MODEL LAYER   │ VISUALIZATION LAYER │
│ (API Module)   │  (Indicators)  │   (ML Model)   │  (Output Results)   │
├────────────────┼────────────────┼────────────────┼─────────────────────┤
│                │                │                │                     │
│  1. USER INPUT │                │                │                     │
│     ↓          │                │                │                     │
│  Enter symbol  │                │                │                     │
│  (NVDA)        │                │                │                     │
│     │          │                │                │                     │
│     ▼          │                │                │                     │
│  2. VALIDATE   │                │                │                     │
│     │          │                │                │                     │
│     ▼          │                │                │                     │
│  3. CONNECT    │                │                │                     │
│     TO ALPACA  │                │                │                     │
│     API        │                │                │                     │
│     │          │                │                │                     │
│     ▼          │                │                │                     │
│  4. FETCH DATA │                │                │                     │
│     │          │                │                │                     │
│     ▼ (OHLCV)  │                │                │                     │
│  5 YEARS DATA  │                │                │                     │
│  1,250+ ROWS   │                │                │                     │
│     │          │                │                │                     │
│     └──────────┼────────────────┼────────────────┼─────────────────────┤
│                │ FEATURE CALC.  │                │                     │
│                │     ↓          │                │                     │
│                │  Calculate SMA │                │                     │
│                │  (20,50,200)   │                │                     │
│                │     ↓          │                │                     │
│                │  Calculate RSI │                │                     │
│                │  (14-day)      │                │                     │
│                │     ↓          │                │                     │
│                │  Calculate     │                │                     │
│                │  Volatility    │                │                     │
│                │  (20-day)      │                │                     │
│                │     ↓          │                │                     │
│                │  Calculate     │                │                     │
│                │  Volume MA     │                │                     │
│                │  (20-day)      │                │                     │
│                │     ↓          │                │                     │
│                │  Calculate     │                │                     │
│                │  TARGET VAR.   │                │                     │
│                │  (Next-day ret)│                │                     │
│                │     ↓          │                │                     │
│                │  FEATURE       │                │                     │
│                │  MATRIX:       │                │                     │
│                │  [SMA20, SMA50,│                │                     │
│                │   SMA200, RSI, │                │                     │
│                │   Volatility,  │                │                     │
│                │   Volume_MA]   │                │                     │
│                │     │          │                │                     │
│                └─────┼──────────┼────────────────┼─────────────────────┤
│                      │          │ DATA PREP      │                     │
│                      │          │    ↓          │                     │
│                      │          │ Remove NaN     │                     │
│                      │          │    ↓          │                     │
│                      │          │ Train/Test     │                     │
│                      │          │ Split (70/30)  │                     │
│                      │          │    ↓          │                     │
│                      │          │ Feature        │                     │
│                      │          │ Scaling        │                     │
│                      │          │ (StandardScaler)                     │
│                      │          │    ↓          │                     │
│                      │          │                │                     │
│                      │          │ TRAINING SET   │                     │
│                      │          │ X_train, y_train                     │
│                      │          │ (70% data)     │                     │
│                      │          │    │          │                     │
│                      │          │    ▼          │                     │
│                      │          │ MODEL TRAIN    │                     │
│                      │          │    ↓          │                     │
│                      │          │ Random Forest  │                     │
│                      │          │ (100 trees)    │                     │
│                      │          │    │          │                     │
│                      │          │    ▼          │                     │
│                      │          │ TRAINED MODEL  │                     │
│                      │          │    │          │                     │
│                      │          │    ▼          │                     │
│                      │          │                │                     │
│                      │          │ TEST SET       │                     │
│                      │          │ X_test, y_test │                     │
│                      │          │ (30% data)     │                     │
│                      │          │    │          │                     │
│                      │          │    ▼          │                     │
│                      │          │ PREDICTION     │                     │
│                      │          │    ↓          │                     │
│                      │          │ y_pred array   │                     │
│                      │          │    │          │                     │
│                      │          │    ▼          │                     │
│                      │          │ EVALUATION     │                     │
│                      │          │    ↓          │                     │
│                      │          │ Calc Metrics   │                     │
│                      │          │    │          │                     │
│                      │          │    ▼          │                     │
│                      │          │ MSE, RMSE,    │                     │
│                      │          │ MAE, R²       │                     │
│                      │          │    │          │                     │
│                      │          │    ▼          │                     │
│                      │          │ Feature       │                     │
│                      │          │ Importance    │                     │
│                      │          │    │          │                     │
│                      └──────────┼────┼──────────┼─────────────────────┤
│                                 │    │          │ VISUALIZATION      │
│                                 │    │          │     ↓              │
│                                 │    │          │ Price Curves Plot  │
│                                 │    │          │ (Actual vs Pred)   │
│                                 │    │          │     ↓              │
│                                 │    │          │ Returns Dist Plot  │
│                                 │    │          │ (4 subplots)       │
│                                 │    │          │     ↓              │
│                                 │    │          │ Residual Analysis  │
│                                 │    │          │     ↓              │
│                                 │    │          │ Feature Importance │
│                                 │    │          │ Chart              │
│                                 │    │          │     ↓              │
│                                 │    │          │ TEXT REPORT        │
│                                 │    │          │ • Metrics summary  │
│                                 │    │          │ • Feature ranking  │
│                                 │    │          │ • Interpretation   │
│                                 │    │          │     ↓              │
│                                 │    │          │ CSV EXPORT         │
│                                 │    │          │ • Date/Price data  │
│                                 │    │          │ • Predictions      │
│                                 │    │          │ • Errors           │
│                                 │    │          │     ↓              │
│                                 │    │          │ OUTPUT FILES:      │
│                                 │    │          │ • chart_*.png      │
│                                 │    │          │ • report.txt       │
│                                 │    │          │ • predictions.csv  │
│                                 │    │          │     │              │
│                                 │    │          │     ▼              │
│                                 │    │          │ DISPLAY TO USER    │
│                                 │    │          │                    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. PROCESS FLOW DIAGRAM

```
START
  │
  ▼
┌──────────────────────────┐
│ USER PROVIDES STOCK      │
│ SYMBOL (e.g., NVDA)      │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ API LAYER                │
│ api/alpaca_fetch.py      │
│                          │
│ • Validate symbol        │
│ • Connect to API         │
│ • Fetch OHLCV data       │
│ • Return DataFrame       │
│   (1,250+ rows)          │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ FEATURE ENGINEERING      │
│ indicators/ module       │
│                          │
│ • Calculate SMA (3×)     │
│ • Calculate RSI          │
│ • Calculate Volatility   │
│ • Calculate Volume_MA    │
│ • Calculate Target (y)   │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ DATA PREPARATION         │
│ model/train.py           │
│                          │
│ • Remove NaN rows        │
│ • Train/Test Split (70/30)
│ • Feature Scaling        │
│   (StandardScaler)       │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ MODEL TRAINING           │
│ model/random_forest.py   │
│                          │
│ • Initialize RF (100)    │
│ • Fit on train data      │
│ • Store model in memory  │
│ • Extract weights        │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ MAKE PREDICTIONS         │
│ model/predict.py         │
│                          │
│ • Predict on test set    │
│ • Get next-day returns   │
│ • Calculate accuracy     │
│ • Error analysis         │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ MODEL EVALUATION         │
│ Calculate metrics:       │
│                          │
│ • MSE (Mean Squared Error)
│ • RMSE (Root MSE)        │
│ • MAE (Mean Abs Error)   │
│ • R² (Variance explained)│
│ • Feature importance     │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ VISUALIZATION            │
│ visualization/plot_*.py  │
│                          │
│ • Price curves plot      │
│ • Returns distribution   │
│ • Residual analysis      │
│ • Feature importance     │
│ • Export PNG (300 DPI)   │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ REPORTING                │
│ Generate output files:   │
│                          │
│ • Text report (.txt)     │
│ • CSV predictions (.csv) │
│ • PNG visualizations     │
│ • Save to output/ dir    │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ DISPLAY RESULTS TO USER  │
│                          │
│ • Print summary metrics  │
│ • Show file locations    │
│ • Display next-day pred. │
│ • Interpretation guide   │
└────────┬─────────────────┘
         │
         ▼
        END
```

---

## 4. SYSTEM ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE LAYERS                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  PRESENTATION LAYER (main.py)                              │  │
│  │  • User Interface                                           │  │
│  │  • Input/Output Handling                                   │  │
│  │  • Workflow Orchestration                                  │  │
│  │  • Result Display                                          │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  BUSINESS LOGIC LAYER                                      │  │
│  │                                                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │  │
│  │  │ API Layer    │  │ Feature Eng. │  │ Model Layer  │     │  │
│  │  │              │  │              │  │              │     │  │
│  │  │ • Fetch data │  │ • SMA        │  │ • RF Train   │     │  │
│  │  │ • Validate   │  │ • RSI        │  │ • Predict    │     │  │
│  │  │ • Format     │  │ • Volatility │  │ • Evaluate   │     │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘     │  │
│  │                                                             │  │
│  │  ┌──────────────┐  ┌──────────────────────────────────┐   │  │
│  │  │ Data Prep    │  │ Visualization & Reporting        │   │  │
│  │  │              │  │                                  │   │  │
│  │  │ • Clean data │  │ • Plot results                   │   │  │
│  │  │ • Split data │  │ • Generate reports               │   │  │
│  │  │ • Scale      │  │ • Export CSV                     │   │  │
│  │  └──────────────┘  └──────────────────────────────────┘   │  │
│  │                                                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  DATA LAYER                                                 │  │
│  │                                                             │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │  │
│  │  │ Input Data  │  │ Processed    │  │ Output Results   │  │  │
│  │  │             │  │ Data         │  │                  │  │  │
│  │  │ • Symbol    │  │              │  │ • PNG plots      │  │  │
│  │  │ • API key   │  │ • DataFrame  │  │ • CSV data       │  │  │
│  │  │             │  │ • Arrays     │  │ • Text report    │  │  │
│  │  │             │  │ • Scalars    │  │ • Metrics        │  │  │
│  │  └─────────────┘  └──────────────┘  └──────────────────┘  │  │
│  │                                                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  EXTERNAL SERVICES                                          │  │
│  │                                                             │  │
│  │  ┌──────────────────┐        ┌──────────────────────────┐ │  │
│  │  │ Alpaca API       │        │ Matplotlib / Libraries   │ │  │
│  │  │ (Market Data)    │        │ (Processing & Viz)       │ │  │
│  │  │                  │        │                          │ │  │
│  │  │ • OHLCV data     │        │ • Pandas (data wrangling)│ │  │
│  │  │ • 5-year history │        │ • NumPy (numerics)       │ │  │
│  │  │ • Daily bars     │        │ • Scikit-learn (ML)      │ │  │
│  │  └──────────────────┘        │ • Matplotlib (plots)     │ │  │
│  │                              └──────────────────────────┘ │  │
│  │                                                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. DATA FLOW DIAGRAM

```
INPUT
  │
  ▼ STOCK SYMBOL (NVDA)
  │
  ├─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  ┌──────────────────────────────────────────────────────────┐  │
  │  │ ALPACA API LAYER (api/alpaca_fetch.py)                  │  │
  │  └───────────────────┬──────────────────────────────────────┘  │
  │                      │                                          │
  │                      ▼ FETCH DATA                              │
  │                 DataFrame:                                     │
  │            [Date, Open, High, Low, Close, Volume]             │
  │            1,250+ rows (5-year daily data)                    │
  │                      │                                          │
  └──────────────────────┼──────────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │ FEATURE ENGINEERING LAYER (indicators/)                   │ │
  │  └─────────────────┬──────────────────────────────────────────┘ │
  │                    │                                             │
  │      ┌─────────────┼─────────────┬──────────────┬──────────┐   │
  │      │             │             │              │          │   │
  │      ▼             ▼             ▼              ▼          ▼   │
  │   ┌─────┐     ┌──────┐     ┌──────────┐   ┌──────────┐  ┌───┐ │
  │   │ SMA │     │ RSI  │     │Volatility│   │Volume_MA │  │ y │ │
  │   │     │     │      │     │          │   │          │  │   │ │
  │   │20,50│     │ 14   │     │ 20-day   │   │ 20-day   │  │Ret│ │
  │   │200  │     │period│     │ std dev  │   │ average  │  │urn│ │
  │   └─────┘     └──────┘     └──────────┘   └──────────┘  └───┘ │
  │      │             │             │              │          │   │
  │      └─────────────┼─────────────┴──────────────┴──────────┘   │
  │                    │                                             │
  │                    ▼ FEATURES CREATED                            │
  │            6 New Columns Added:                                 │
  │        [SMA_20, SMA_50, SMA_200, RSI_14,                        │
  │         Volatility, Volume_MA, Next_Day_Return]                 │
  │                    │                                             │
  └────────────────────┼──────────────────────────────────────────────┘
                       │
                       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │ DATA PREPARATION LAYER (model/train.py)                  │ │
  │  └─────────────────┬──────────────────────────────────────────┘ │
  │                    │                                             │
  │                    ├─ Remove NaN rows (warmup period)           │
  │                    │  (~200 rows removed)                        │
  │                    │                                             │
  │                    ├─ Create Feature Matrix X                   │
  │                    │  (1,000 rows × 6 columns)                  │
  │                    │                                             │
  │                    ├─ Create Target Vector y                    │
  │                    │  (1,000 next-day returns)                  │
  │                    │                                             │
  │                    ├─ Train/Test Split (70/30)                  │
  │                    │  ├─ X_train: 700 × 6                       │
  │                    │  └─ X_test:  300 × 6                       │
  │                    │  ├─ y_train: 700 values                    │
  │                    │  └─ y_test:  300 values                    │
  │                    │                                             │
  │                    ├─ Feature Scaling (StandardScaler)          │
  │                    │  ├─ μ = 0                                  │
  │                    │  └─ σ = 1                                  │
  │                    │                                             │
  │                    ▼                                             │
  │            Scaled Matrices Ready for ML                         │
  │                    │                                             │
  └────────────────────┼──────────────────────────────────────────────┘
                       │
                       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │ MACHINE LEARNING LAYER (model/random_forest.py)          │ │
  │  └─────────────────┬──────────────────────────────────────────┘ │
  │                    │                                             │
  │                    ├─ RandomForestRegressor(n_estimators=100)   │
  │                    │  ├─ 100 decision trees                     │
  │                    │  ├─ max_depth=15                           │
  │                    │  └─ min_samples_split=5                    │
  │                    │                                             │
  │                    ├─ Fit Model on Training Data                │
  │                    │  ├─ X_train: 700 × 6                       │
  │                    │  └─ y_train: 700 values                    │
  │                    │                                             │
  │                    ├─ Trained Model in Memory                   │
  │                    │  ├─ 100 fitted trees                       │
  │                    │  ├─ Feature weights                        │
  │                    │  └─ Bootstrap samples                      │
  │                    │                                             │
  │                    ├─ Predict on Test Set                       │
  │                    │  ├─ X_test: 300 × 6                        │
  │                    │  ▼ y_pred: 300 predictions                 │
  │                    │                                             │
  │                    ├─ Evaluate Model                            │
  │                    │  ├─ Compare y_pred vs y_test               │
  │                    │  ├─ Calculate metrics                      │
  │                    │  └─ Compute importance                     │
  │                    │                                             │
  │                    ▼                                             │
  │        Metrics, Predictions, Feature Importance                 │
  │                    │                                             │
  └────────────────────┼──────────────────────────────────────────────┘
                       │
                       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │ EVALUATION LAYER (model/predict.py)                       │ │
  │  └─────────────────┬──────────────────────────────────────────┘ │
  │                    │                                             │
  │      ┌─────────────┼─────────────┬──────────────┬─────────────┐ │
  │      │             │             │              │             │ │
  │      ▼             ▼             ▼              ▼             ▼ │
  │   ┌─────┐     ┌──────┐     ┌──────────┐   ┌──────────┐  ┌────┐│
  │   │ MSE │     │RMSE  │     │   MAE    │   │   R²     │  │ Dir││
  │   │     │     │      │     │          │   │          │  │Acc ││
  │   │Sq.Er│     │Sq.Rt │     │Avg Error │   │ Var Exp  │  │uracy
  │   │ror  │     │Error │     │Magnitude │   │ Fraction │  │     ││
  │   └─────┘     └──────┘     └──────────┘   └──────────┘  └────┘│
  │      │             │             │              │             │ │
  │      └─────────────┼─────────────┴──────────────┴─────────────┘ │
  │                    │                                             │
  │                    ├─ Feature Importance Rankings               │
  │                    │  (Which features matter most)              │
  │                    │                                             │
  │                    ├─ Error Analysis                            │
  │                    │  (Where model made mistakes)               │
  │                    │                                             │
  │                    ▼                                             │
  │        Performance Metrics + Insights                           │
  │                    │                                             │
  └────────────────────┼──────────────────────────────────────────────┘
                       │
                       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │ VISUALIZATION LAYER (visualization/plot_results.py)       │ │
  │  └─────────────────┬──────────────────────────────────────────┘ │
  │                    │                                             │
  │      ┌─────────────┼─────────────┬──────────────┬─────────────┐ │
  │      │             │             │              │             │ │
  │      ▼             ▼             ▼              ▼             ▼ │
  │   ┌─────────┐ ┌──────────┐ ┌──────────┐  ┌──────────┐  ┌────┐│
  │   │Price    │ │ Returns  │ │Residual  │  │ Feature  │  │ Txt││
  │   │Curves   │ │Distrib.  │ │Analysis  │  │Importance   │Repo││
  │   │         │ │(4-subplot)           │  │Chart     │  │ rt ││
  │   │Actual vs│ │          │          │  │          │  │    ││
  │   │Predicted│ │Actual    │          │  │Top 10    │  │    ││
  │   │         │ │Predicted │          │  │Features  │  │    ││
  │   └─────────┘ │Scatter   │          │  │          │  └────┘│
  │      │        │          │          │  │          │         │ │
  │      │        └──────────┘          │  │          │         │ │
  │      │             │                │  │          │         │ │
  │      │             │                │  └──────────┘         │ │
  │      │             │                │         │             │ │
  │      │             │                │         │             │ │
  │      └─────────────┼────────────────┴─────────┼─────────────┘ │
  │                    │                          │                 │
  │            PNG Output (300 DPI)          TXT Output             │
  │                    │                          │                 │
  └────────────────────┼──────────────────────────┼──────────────────┘
                       │                          │
                       ▼                          ▼
  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  ┌──────────────────────────────────────────────────┐ │
  │  │ OUTPUT LAYER (Files Created in output/ dir)     │ │
  │  └──────────────────────────────────────────────────┘ │
  │                                                        │
  │  ├─ NVDA_prediction.png                             │
  │  │  └─ Price curves (actual vs predicted)           │
  │  │                                                   │
  │  ├─ NVDA_returns.png                                │
  │  │  └─ 4-subplot returns analysis                   │
  │  │                                                   │
  │  ├─ NVDA_residuals.png                              │
  │  │  └─ Error distribution analysis                  │
  │  │                                                   │
  │  ├─ NVDA_report.txt                                 │
  │  │  └─ Metrics, interpretation, top features        │
  │  │                                                   │
  │  └─ NVDA_predictions.csv                            │
  │     └─ Date, Actual, Predicted, Error data          │
  │                                                        │
  └────────────────────────────────────────────────────────┘
                       │
                       ▼
                OUTPUT
```

---

## 6. MODULE DEPENDENCY DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│                   main.py                                   │
│              (Entry Point)                                  │
│                                                             │
│  Imports:                                                  │
│  • api.alpaca_fetch                                        │
│  • model.train                                             │
│  • model.predict                                           │
│  • visualization.plot_results                              │
│                                                             │
└──────┬──────────────┬──────────────┬──────────────┬────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
    ┌──────┐     ┌──────┐      ┌──────┐      ┌──────┐
    │ api/ │     │model/│      │indic.│      │visual│
    └──────┘     └──────┘      └──────┘      └──────┘
       │              │              │              │
       ├─ alpaca_     ├─ train.py    ├─ sma.py     ├─ plot_
       │  fetch.py    │              │              │  results
       │              ├─ random_     ├─ rsi.py     │  .py
       │              │  forest.py   │
       │              │              ├─ volatility
       │              ├─ predict.py  │  .py
       │              │
       │              └─ Imports:    └─ Dependencies:
       │                 • indicators  • pandas
       │                 • pandas      • numpy
       │                 • numpy
       │                 • sklearn     └─ Dependencies:
       │                                • pandas
       │                                • numpy
       └─ Dependencies:
          • alpaca-trade-api
          • pandas
          • pytz
          • datetime
          • os

LEGEND:
  ┌────────┐     = Module/Package
  │ file   │
  └────────┘

  Dependencies shown at bottom of each module
```

---

## 7. CLASS DIAGRAM (RandomForestModel)

```
┌────────────────────────────────────────────────────────────┐
│         RandomForestModel (model/random_forest.py)         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ATTRIBUTES:                                              │
│  ───────────                                              │
│  • model: RandomForestRegressor                           │
│  • scaler: StandardScaler                                 │
│  • feature_names: list                                    │
│  • metrics: dict                                          │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  METHODS:                                                 │
│  ────────                                                 │
│                                                            │
│  __init__(n_estimators=100,                              │
│           max_depth=15,                                   │
│           min_samples_split=5,                            │
│           min_samples_leaf=2)                             │
│  → Initialize model with parameters                       │
│                                                            │
│  train(X_train, y_train, scale_features=True)             │
│  → Fit model on training data                             │
│  → Return: metrics dict                                   │
│                                                            │
│  predict(X_test, scale_features=True)                     │
│  → Generate predictions for test data                     │
│  → Return: numpy array of predictions                     │
│                                                            │
│  evaluate(X_test, y_test, scale_features=True)            │
│  → Calculate performance metrics                          │
│  → Return: metrics dict (MSE, RMSE, MAE, R²)             │
│                                                            │
│  get_feature_importance(feature_names, top_n=10)          │
│  → Get importance of each feature                         │
│  → Return: DataFrame with rankings                        │
│                                                            │
│  save_model(filepath)                                     │
│  → Persist model to disk (pickle)                         │
│  → File format: .pkl                                      │
│                                                            │
│  load_model(filepath) [STATIC]                            │
│  → Load model from disk                                   │
│  → Return: RandomForestModel instance                     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 8. ERROR HANDLING FLOW DIAGRAM

```
USER INPUT (Stock Symbol)
       │
       ▼
   ┌──────────────────────┐
   │ Validate Input       │
   │ • Check length       │
   │ • Check alphanumeric │
   └──────┬───────────────┘
          │
      ERROR? ───── NO ─────┐
     /     \              │
   YES      │              ▼
    │       └──────────┐  ┌──────────────────────┐
    │                  │  │ Connect to Alpaca    │
    ▼                  │  │ API                  │
┌──────────┐           │  └──────┬───────────────┘
│ Print    │           │         │
│Error Msg │           │     ERROR? ───── NO ─────┐
└──────────┘           │    /     \              │
                    RETRY? │      │              ▼
                     /   \  │      └──────────┐  ┌──────────────────────┐
                    /     \ │                │  │ Fetch Data           │
                  YES      NO               │  └──────┬───────────────┘
                  │        │                │         │
                  │        ▼                │     ERROR? ───── NO ─────┐
                  │   ┌──────────────┐     │    /     \              │
                  │   │ Exit Program │     │  YES      │              ▼
                  │   └──────────────┘     │   │      └──────────────┐
                  │                       │   │                    │
                  └──────────┬────────────┘   │  ┌──────────────────────┐
                             │                │  │ Process Data         │
                             ▼                │  │ • Feature Eng.       │
                        ┌──────────┐          │  │ • Data Prep          │
                        │ Try Again│          │  │ • Model Training     │
                        └──────────┘          │  │ • Predictions        │
                                              │  └──────┬───────────────┘
                                              │         │
                                              │     ERROR? ───── NO ─────┐
                                              │    /     \              │
                                              │  YES      │              ▼
                                              │   │      └──────────────┐
                                              │   │                    │
                                              │   ▼                    │
                                              │ ┌──────────────────────┐
                                              │ │ Log Error Details    │
                                              │ │ Print User Message   │
                                              │ └──────────────────────┘
                                              │         │
                                              │         ▼
                                              │  ┌──────────────────────┐
                                              │  │ Generate Output      │
                                              │  │ Files                │
                                              │  └──────┬───────────────┘
                                              │         │
                                              │         ▼
                                              │  ┌──────────────────────┐
                                              │  │ Display Results      │
                                              │  │ To User              │
                                              │  └──────────────────────┘
                                              │         │
                                              └─────────┘
                                                    │
                                                    ▼
                                               SUCCESS ✓
```

---

This comprehensive architecture documentation provides visual representations of:
- **Actor Use Cases** - What users and systems do
- **Swimlane Diagrams** - Workflow across different layers.
- **Process Flow** - Step-by-step execution
- **System Architecture** - Layered design
- **Data Flow** - How data transforms through system
- **Module Dependencies** - Imports and relationships
- **Class Diagrams** - RandomForestModel structure
- **Error Handling** - Exception management flow

All diagrams use code formatting (ASCII art) for easy inclusion in documentation and version control systems.
