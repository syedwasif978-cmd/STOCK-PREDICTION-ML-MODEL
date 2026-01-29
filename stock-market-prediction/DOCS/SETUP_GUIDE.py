"""
===============================================================================
SETUP GUIDE & CONFIGURATION
===============================================================================

Complete step-by-step guide to set up and run the Stock Market
Analysis & Prediction System.

This guide covers:
1. Installation
2. API Configuration
3. Running the System
4. Troubleshooting
5. Example Usage

===============================================================================
SECTION 1: INSTALLATION
===============================================================================

Step 1.1: Verify Python Installation
─────────────────────────────────────

Windows Command Prompt:
    C:\> python --version
    Python 3.9.0
    ✓ Should be 3.7 or higher

Linux/Mac Terminal:
    $ python3 --version
    Python 3.9.0
    ✓ Should be 3.7 or higher

If Python is not installed, download from https://www.python.org/


Step 1.2: Navigate to Project Directory
───────────────────────────────────────

Windows:
    C:\Users\YourName\> cd path\to\stock-market-prediction

Linux/Mac:
    $ cd path/to/stock-market-prediction


Step 1.3: Create Virtual Environment (Optional but Recommended)
──────────────────────────────────────────────────────────────

Windows:
    C:\project\> python -m venv venv
    C:\project\> venv\Scripts\activate
    
    After activation:
    (venv) C:\project\>  ← Note the (venv) prefix

Linux/Mac:
    $ python3 -m venv venv
    $ source venv/bin/activate
    
    After activation:
    (venv) $ ← Note the (venv) prefix

Why virtual environment?
    ✓ Isolates project dependencies
    ✓ Prevents conflicts with other Python projects
    ✓ Easy cleanup (just delete venv folder)
    ✓ Professional best practice


Step 1.4: Install Required Libraries
────────────────────────────────────

Option A: Using requirements.txt (Recommended)
    (venv) C:\project\> pip install -r requirements.txt

Option B: Individual installation
    (venv) C:\project\> pip install pandas numpy scikit-learn matplotlib alpaca-trade-api pytz

Option C: Upgrade pip first, then install
    (venv) C:\project\> python -m pip install --upgrade pip
    (venv) C:\project\> pip install -r requirements.txt

Expected installation time: 2-5 minutes

Output should show:
    Successfully installed pandas-1.3.5 numpy-1.21.0 ...
    
If you see errors:
    - Make sure you're in virtual environment (check for (venv) prefix)
    - Try: pip install --upgrade pip
    - Check internet connection


Step 1.5: Verify Installation
─────────────────────────────

Run Python and import libraries:
    (venv) C:\project\> python
    >>> import pandas as pd
    >>> import numpy as np
    >>> import sklearn
    >>> import matplotlib
    >>> from alpaca_trade_api import REST
    >>> print("✓ All libraries installed successfully!")
    >>> exit()

===============================================================================
SECTION 2: ALPACA API CONFIGURATION
===============================================================================

Step 2.1: Create Alpaca Account
──────────────────────────────

1. Go to https://alpaca.markets
2. Click "Sign Up" (top right)
3. Fill in registration form:
    - Email address
    - Password
    - Username
4. Verify email address
5. Login to dashboard

Time: 5 minutes
Cost: FREE


Step 2.2: Generate API Keys
──────────────────────────

1. Login to https://app.alpaca.markets
2. Navigate to: Account → API Keys
3. Click "Create New Key"
4. Choose:
    - Key Type: "Full Access"
    - Key Name: "Stock Prediction System" (any name)
5. Copy the API Key and Secret Key
6. **Store these securely** - never share!

⚠️ Important:
    - Anyone with these keys can trade using your account
    - Use "Paper Trading" mode (no real money risk)
    - Regenerate keys if compromised
    - Don't commit to version control


Step 2.3: Set Environment Variables
───────────────────────────────────

Windows Command Prompt (cmd.exe):
────────────────────────────────

    C:\> setx APCA_API_KEY_ID "your_api_key_here"
    C:\> setx APCA_API_SECRET_KEY "your_secret_key_here"
    
    (Restart Command Prompt for changes to take effect)
    
    Verify:
    C:\> echo %APCA_API_KEY_ID%
    your_api_key_here ✓


Windows PowerShell:
──────────────────

    PS> [Environment]::SetEnvironmentVariable("APCA_API_KEY_ID", "your_api_key_here", "User")
    PS> [Environment]::SetEnvironmentVariable("APCA_API_SECRET_KEY", "your_secret_key_here", "User")
    
    Verify (restart PowerShell first):
    PS> $env:APCA_API_KEY_ID
    your_api_key_here ✓


Linux/Mac Terminal (bash/zsh):
──────────────────────────────

    Temporary (only current session):
    $ export APCA_API_KEY_ID="your_api_key_here"
    $ export APCA_API_SECRET_KEY="your_secret_key_here"
    
    Verify:
    $ echo $APCA_API_KEY_ID
    your_api_key_here ✓
    
    Permanent (add to ~/.bashrc or ~/.zshrc):
    $ nano ~/.bashrc
    
    Add these lines at the end:
    export APCA_API_KEY_ID="your_api_key_here"
    export APCA_API_SECRET_KEY="your_secret_key_here"
    
    Save (Ctrl+O, Enter, Ctrl+X)
    
    Reload:
    $ source ~/.bashrc
    
    Verify:
    $ echo $APCA_API_KEY_ID


Step 2.4: Verify API Connection
───────────────────────────────

Test if credentials are correctly set:

    (venv) C:\project\> python
    >>> from alpaca_trade_api import REST
    >>> api = REST()
    >>> account = api.get_account()
    >>> print(account.status)
    active ✓
    >>> exit()

If you see an error:
    - Check API keys are correct
    - Verify environment variables are set
    - Restart your terminal/IDE
    - Check internet connection

===============================================================================
SECTION 3: RUNNING THE SYSTEM
===============================================================================

Step 3.1: Start the Program
──────────────────────────

From project directory:

    (venv) C:\project\> python main.py

Or for Windows without venv:

    C:\project\> python main.py

Expected output:
    
    ======================================================================
    📈 STOCK MARKET ANALYSIS & PREDICTION SYSTEM 📈
    ======================================================================
    
    📍 Enter stock symbol (e.g., NVDA, AAPL, TSLA): 


Step 3.2: Enter Stock Symbol
───────────────────────────

Examples of valid symbols:
    - NVDA (NVIDIA)
    - AAPL (Apple)
    - TSLA (Tesla)
    - MSFT (Microsoft)
    - GOOG (Alphabet/Google)
    - AMZN (Amazon)
    - META (Meta/Facebook)
    - SPY (S&P 500 ETF)
    - QQQ (Nasdaq ETF)

Enter (uppercase or lowercase):
    📍 Enter stock symbol: NVDA
    
    ✓ Selected: NVDA


Step 3.3: Wait for Processing
────────────────────────────

The system will:
    1. Fetch data from Alpaca (30 seconds)
    2. Calculate technical indicators (10 seconds)
    3. Prepare features (10 seconds)
    4. Train Random Forest model (30 seconds)
    5. Generate predictions (5 seconds)
    6. Create visualizations (20 seconds)
    7. Generate report (5 seconds)
    
    Total time: 2-5 minutes

During processing, you'll see:
    [INFO] Fetching data for NVDA...
    [SUCCESS] Data retrieved: 1250 trading days
    [INFO] Calculating indicators...
    [INFO] Preparing data...
    [INFO] Training Random Forest...
    [SUCCESS] Training complete
    ...


Step 3.4: Review Results
───────────────────────

Output files created in 'output/' directory:

    ✓ NVDA_prediction.png      ← Main visualization
    ✓ NVDA_returns.png         ← Return analysis
    ✓ NVDA_residuals.png       ← Error analysis
    ✓ NVDA_report.txt          ← Performance report
    ✓ NVDA_predictions.csv     ← Detailed predictions

Open PNG files with any image viewer.
Open TXT and CSV files with text editor.

===============================================================================
SECTION 4: EXAMPLE WALKTHROUGH
===============================================================================

Complete Example: Analyzing NVDA Stock
──────────────────────────────────────

Step 1: Run the program
    C:\project\> python main.py

Step 2: Input stock symbol
    📍 Enter stock symbol: NVDA

Step 3: Program processes (shows progress)
    [INFO] Fetching data for NVDA from 2019-01-10 to 2024-01-09
    [SUCCESS] Successfully fetched 1250 trading days
    [INFO] Starting feature engineering on 1250 records
    [SUCCESS] Calculated SMA(20), SMA(50), SMA(200)
    [SUCCESS] Calculated RSI(14)
    [SUCCESS] Calculated 20-day volatility
    [SUCCESS] Feature engineering complete
    ...

Step 4: Model training results
    [SUCCESS] Model training complete
             - MSE:  0.000456
             - RMSE: 0.021362
             - MAE:  0.015678
             - R²:   0.1234

Interpretation:
    - R² = 0.1234 means model explains 12.34% of variance
    - RMSE = 2.14% average prediction error
    - NVDA: Decent prediction (typical for stock returns)

Step 5: Feature importance
    [INFO] Top 10 Most Important Features:
          1. SMA_20                : 0.3456 (34.56%)
          2. RSI_14                : 0.2134 (21.34%)
          3. Volatility            : 0.1876 (18.76%)
          4. SMA_50                : 0.1234 (12.34%)
          5. Volume_MA             : 0.1300 (13.00%)

Interpretation:
    - Trend (SMA) is most important (34%)
    - Momentum (RSI) is significant (21%)
    - Risk (Volatility) matters (19%)

Step 6: Next-day prediction
    ========================================
    📈 PREDICTION SUMMARY
    ========================================
    Current Price:    $125.45
    Predicted Return: +2.34%
    Predicted Price:  $128.37
    
    Price Change:     +$2.92
    
    Model Confidence: MODERATE (R² = 0.12)
    ========================================

Interpretation:
    - Current NVDA: $125.45
    - Model predicts: +2.34% return
    - Target price: $128.37
    - Confidence: Moderate (explains 12% of variance)

Step 7: Review output files
    open output/NVDA_prediction.png    ← See the plot
    open output/NVDA_report.txt        ← Read detailed report
    open output/NVDA_predictions.csv   ← Analyze all predictions

===============================================================================
SECTION 5: TROUBLESHOOTING
===============================================================================

Problem: "Alpaca API credentials not found"
──────────────────────────────────────────
Cause: Environment variables not set or not recognized
Solution:
    1. Verify keys are set:
       Windows: echo %APCA_API_KEY_ID%
       Linux/Mac: echo $APCA_API_KEY_ID
    2. Restart your IDE or terminal
    3. Re-set environment variables
    4. Use absolute path to python executable

    
Problem: "ModuleNotFoundError: No module named 'alpaca_trade_api'"
────────────────────────────────────────────────────────────────
Cause: Dependencies not installed
Solution:
    1. Check virtual environment is activated: (venv) prefix visible?
    2. Reinstall: pip install -r requirements.txt
    3. Verify installation: pip list | grep alpaca
    4. Try: pip install alpaca-trade-api --upgrade


Problem: "Invalid symbol: XYZ"
─────────────────────────────
Cause: Symbol doesn't exist or is invalid
Solution:
    1. Use valid stock symbols (NVDA, AAPL, TSLA, etc.)
    2. Check https://www.nasdaq.com for valid symbols
    3. Use only letters (no numbers or special characters)
    4. Common mistake: lowercase (use NVDA, not nvda)


Problem: "Insufficient data after cleaning"
────────────────────────────────────────────
Cause: Stock doesn't have enough historical data
Solution:
    1. Use major stocks: NVDA, AAPL, MSFT, GOOG, TSLA
    2. Avoid penny stocks or newly listed companies
    3. Need at least 5 years = 1250+ trading days


Problem: Plots not displaying or saving
───────────────────────────────────────
Cause: Output directory doesn't exist or permission issue
Solution:
    1. Create output directory: mkdir output
    2. Check write permissions: ls -la output (Linux/Mac)
    3. Try running from different directory
    4. Use absolute path: python C:\full\path\to\main.py


Problem: "Connection timeout" or "Network error"
────────────────────────────────────────────────
Cause: API server unreachable
Solution:
    1. Check internet connection
    2. Try again in a few minutes
    3. Check Alpaca status: https://status.alpaca.markets
    4. Use VPN if blocked by firewall


Problem: Low R² Score (< 0.05)
──────────────────────────────
This is NORMAL for stock market prediction!
Cause: Stock returns are inherently random and noisy
Explanation:
    - Market prices reflect all known information
    - Remaining movements are largely unpredictable
    - Experts get R² around 0.1-0.2
    - R² of 0.05-0.15 is GOOD for stocks
Not a problem - it's expected behavior!

===============================================================================
SECTION 6: NEXT STEPS
===============================================================================

After successful run:

1. Review Visualizations
   - Examine NVDA_prediction.png
   - Look for prediction lag or systematic bias
   - Compare different stocks

2. Study the Code
   - Read comments in main.py
   - Understand indicator calculations
   - Learn Random Forest structure

3. Experiment
   - Try different stocks
   - Modify model hyperparameters
   - Add new features

4. Validate Results
   - Compare predictions to actual outcomes
   - Track prediction accuracy over time
   - Backtest on historical data

5. Enhance the System
   - Add LSTM model
   - Include sentiment analysis
   - Add risk management

===============================================================================
"""

if __name__ == "__main__":
    print(__doc__)
