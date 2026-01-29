"""
Stock Market Prediction - GUI Launcher
======================================

Start the graphical user interface application.

Usage:
    python run_gui.py

Features:
    • Real-time stock search with autocomplete
    • Live market data from Alpaca API
    • ML predictions with visualizations
    • Professional desktop application
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Check API credentials
api_key = os.getenv('APCA_API_KEY_ID')
api_secret = os.getenv('APCA_API_SECRET_KEY')

if not api_key or not api_secret:
    print("⚠️  WARNING: API credentials not found!")
    print("\nPlease set environment variables:")
    print("  Windows:")
    print("    setx APCA_API_KEY_ID \"your_api_key\"")
    print("    setx APCA_API_SECRET_KEY \"your_secret_key\"")
    print("\n  Linux/Mac:")
    print("    export APCA_API_KEY_ID='your_api_key'")
    print("    export APCA_API_SECRET_KEY='your_secret_key'")
    print("\nGet API credentials from: https://alpaca.markets")
    print("\nContinuing anyway...\n")

# Launch GUI
try:
    from gui_app import main
    print("🚀 Launching Stock Market Prediction GUI...")
    print("=" * 50)
    print("Ready for real-time stock analysis!")
    print("=" * 50)
    main()
except ImportError as e:
    print(f"❌ Error importing GUI modules: {e}")
    print("Please make sure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting GUI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
