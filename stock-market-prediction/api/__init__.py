"""
API Module - Market Data Integration
====================================

This module handles communication with Alpaca Markets REST API.

Functions:
    - fetch_stock_data(): Retrieve historical OHLCV data
    - validate_symbol(): Check if stock symbol is valid
    - format_date(): Format datetime objects for API

API Provider: Alpaca Markets (https://alpaca.markets)
Data: Historical daily OHLC data with volume (5+ years available)
"""
