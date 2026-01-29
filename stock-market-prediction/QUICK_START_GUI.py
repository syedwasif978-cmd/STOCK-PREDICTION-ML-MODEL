#!/usr/bin/env python3
"""
QUICK START - Stock Market Prediction GUI

This is a summary of everything you need to get started in 5 minutes.
"""

import sys
import os

print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║        📈 STOCK MARKET PREDICTION SYSTEM - QUICK START 🚀         ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

✨ YOU NOW HAVE A PROFESSIONAL GUI APPLICATION!

Features:
  ✓ Real-time stock search with autocomplete
  ✓ Live market data from Alpaca API (ANY stock)
  ✓ ML predictions with visualizations
  ✓ Interactive 3-panel interface
  ✓ No predefined stocks - works with real data

════════════════════════════════════════════════════════════════════════

🚀 QUICK START (5 MINUTES):

1️⃣  INSTALL DEPENDENCIES
   
   Windows:
   ├─ Open Command Prompt
   ├─ Run: pip install -r requirements.txt
   └─ Wait for completion (2 minutes)

   Linux/Mac:
   ├─ Open Terminal
   ├─ Run: pip3 install -r requirements.txt
   └─ Wait for completion

   Expected output: Successfully installed pandas, numpy, scikit-learn...

2️⃣  GET API CREDENTIALS (FREE)

   Visit: https://alpaca.markets
   
   Steps:
   ├─ Click "Sign up" (free paper trading account)
   ├─ Create account
   ├─ Go to Dashboard
   ├─ Copy API Key and Secret Key
   └─ Keep them safe!

3️⃣  SET ENVIRONMENT VARIABLES

   Windows (Command Prompt):
   ├─ setx APCA_API_KEY_ID "YOUR_API_KEY_HERE"
   ├─ setx APCA_API_SECRET_KEY "YOUR_SECRET_KEY_HERE"
   ├─ Close and reopen Command Prompt
   └─ Verify: echo %APCA_API_KEY_ID%

   Linux/Mac (Terminal):
   ├─ export APCA_API_KEY_ID="YOUR_API_KEY_HERE"
   ├─ export APCA_API_SECRET_KEY="YOUR_SECRET_KEY_HERE"
   └─ Verify: echo $APCA_API_KEY_ID

4️⃣  LAUNCH THE GUI APPLICATION

   Windows:
   └─ python run_gui.py

   Linux/Mac:
   └─ python3 run_gui.py

   Expected: GUI window opens with search bar

5️⃣  TRY IT OUT

   Option A - Quick Test:
   ├─ Click "AAPL" button
   ├─ Wait 2-5 minutes
   └─ See prediction chart and metrics!

   Option B - Custom Stock:
   ├─ Type "NVDA" in search box
   ├─ Press Enter
   ├─ Wait for processing
   └─ View results!

════════════════════════════════════════════════════════════════════════

📊 WHAT YOU'LL SEE:

Left Panel:          Middle Panel:           Right Panel:
┌─────────────────┐  ┌──────────────────┐   ┌─────────────────┐
│ Performance     │  │  Price Prediction    │ Prediction      │
│ Metrics:        │  │      Chart       │   │ Details:        │
│                 │  │                  │   │                 │
│ • MSE           │  │ (actual vs pred) │   │ • Model Quality │
│ • RMSE          │  │ • Blue: actual   │   │ • Feature Info  │
│ • MAE           │  │ • Red: predicted │   │ • Disclaimer    │
│ • R² Score      │  │ • Shows error    │   │                 │
│                 │  │   distribution   │   │                 │
│ Top Features:   │  │                  │   │ Time Stamp:     │
│ 1. SMA_20       │  │                  │   │ Last Updated    │
│ 2. RSI          │  │                  │   │                 │
│ 3. Volatility   │  │                  │   │                 │
└─────────────────┘  └──────────────────┘   └─────────────────┘

════════════════════════════════════════════════════════════════════════

🔍 SEARCH & PREDICT WORKFLOW:

User Types "NVD"
       ↓
Suggestions appear:
  • NVDA - NVIDIA Corporation
  • NVRO - Navro Inc.
       ↓
User clicks "NVDA"
       ↓
Status shows: "Fetching market data..." (Orange)
       ↓
System fetches 5 years of OHLCV data
       ↓
Technical indicators calculated (SMA, RSI, etc.)
       ↓
Model trained on historical data
       ↓
Predictions generated
       ↓
Status shows: "✓ Ready" (Green)
       ↓
Charts and metrics display!

════════════════════════════════════════════════════════════════════════

📈 UNDERSTANDING THE RESULTS:

R² Score (Most Important):
├─ 0.9+  → Excellent ✓ (trust predictions)
├─ 0.7+  → Good ✓ (reasonable predictions)
├─ 0.5+  → Moderate ~ (use with caution)
└─ <0.5  → Poor ✗ (unreliable)

MAE (Mean Absolute Error):
├─ 0.03  → Average 3% error (good)
├─ 0.05  → Average 5% error (okay)
└─ 0.10+ → Average 10% error (poor)

Features (Ranked by Importance):
├─ SMA_20: 0.25  → Trend matters 25%
├─ RSI: 0.20     → Momentum matters 20%
└─ Others...     → Each contributes something

════════════════════════════════════════════════════════════════════════

💡 PRO TIPS:

1. Compare Multiple Stocks
   • Test AAPL (usually has high R²)
   • Test MSFT (good predictions)
   • Test TSLA (more volatile, harder)
   • Compare their R² scores

2. Understand the Chart
   • If blue & red lines overlap → good predictions
   • If lines diverge → model struggles
   • Look at error histogram (should be bell-shaped)

3. Feature Importance
   • Top feature = most influential
   • Helps understand what drives predictions
   • Validate with technical analysis

4. Time Your Analysis
   • After market close better
   • Less volatile data
   • More complete daily information

════════════════════════════════════════════════════════════════════════

⚠️  IMPORTANT REMINDERS:

❌ DO NOT USE FOR REAL TRADING
   • This is educational only
   • Not financial advice
   • Past performance ≠ future results
   • Always consult financial professionals

⚠️  LIMITATIONS:
   • Trained on historical data
   • Can't predict black swan events
   • Markets change unexpectedly
   • Model may become outdated

✅ GOOD FOR:
   • Learning machine learning
   • Understanding technical indicators
   • Analyzing historical patterns
   • Educational projects
   • Understanding model behavior

════════════════════════════════════════════════════════════════════════

🆘 TROUBLESHOOTING:

Problem: "ModuleNotFoundError: No module named 'tkinter'"
Solution: tkinter should be built-in Python
          Try: python -m pip install tk

Problem: "API credentials not found"
Solution: Check you set env vars correctly
          Restart your terminal
          On Windows, restart computer

Problem: "Insufficient data for symbol"
Solution: Symbol too new or doesn't exist
          Try AAPL, MSFT, NVDA instead

Problem: "Processing is very slow"
Normal:  First run takes 2-5 minutes
Reason:  Fetching 5 years + training model
Tip:     Let it complete, be patient!

Problem: "Chart doesn't show"
Solution: pip install --upgrade matplotlib
          pip install --upgrade pillow

════════════════════════════════════════════════════════════════════════

📚 LEARN MORE:

GUI User Guide:
  → Read: GUI_USER_GUIDE.md

Project README:
  → Read: README.md

Implementation Details:
  → Read: GUI_IMPLEMENTATION_SUMMARY.txt

Architecture:
  → Read: ARCHITECTURE_DIAGRAMS.md

════════════════════════════════════════════════════════════════════════

🎯 NEXT STEPS:

1. Install & Run:
   ✓ pip install -r requirements.txt
   ✓ Set API credentials
   ✓ python run_gui.py

2. Explore:
   ✓ Click popular stock buttons
   ✓ Try custom stocks
   ✓ Watch predictions generate
   ✓ Read the metrics

3. Learn:
   ✓ Read GUI_USER_GUIDE.md
   ✓ Understand technical indicators
   ✓ Learn about Random Forest
   ✓ Study the code

4. Experiment:
   ✓ Try different stocks
   ✓ Compare R² scores
   ✓ Analyze features
   ✓ Validate predictions

════════════════════════════════════════════════════════════════════════

🎉 YOU'RE ALL SET!

Your stock market prediction system is:
✓ Installed
✓ Configured
✓ Ready to use
✓ Professional GUI
✓ Real-time search
✓ Live market data
✓ ML predictions

HAPPY PREDICTING! 📈

Questions? Check:
• GUI_USER_GUIDE.md (comprehensive)
• README.md (full documentation)
• Code comments (implementation details)

════════════════════════════════════════════════════════════════════════
""")

if __name__ == '__main__':
    print("\n✨ Run the following to start:")
    print("   python run_gui.py")
