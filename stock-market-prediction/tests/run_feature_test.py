import traceback

from api.alpaca_fetch import fetch_stock_data
from indicators.sma import calculate_multiple_sma
from indicators.rsi import calculate_rsi
from indicators.volatility import calculate_volatility
import pandas as pd

try:
    df = fetch_stock_data('AAPL', days=30)
    print('Fetched rows:', len(df))
    sma_all = calculate_multiple_sma(df, periods=[20,50,200], column='Close')
    print('SMA columns:', sma_all.columns.tolist())
    # Assign as in GUI code
    if 'SMA_20' in sma_all.columns:
        df['SMA_20'] = sma_all['SMA_20']
    else:
        df['SMA_20'] = pd.Series(index=df.index, dtype=float)

    if 'SMA_50' in sma_all.columns:
        df['SMA_50'] = sma_all['SMA_50']
    else:
        df['SMA_50'] = pd.Series(index=df.index, dtype=float)

    if 'SMA_200' in sma_all.columns:
        df['SMA_200'] = sma_all['SMA_200']
    else:
        df['SMA_200'] = pd.Series(index=df.index, dtype=float)

    df['RSI'] = calculate_rsi(df, period=14, column='Close')
    df['Volatility'] = calculate_volatility(df, period=20, column='Close')
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Next_Day_Return'] = df['Close'].pct_change().shift(-1)

    print('Final columns preview:', df.columns.tolist())
    print(df.tail())
except Exception:
    traceback.print_exc()
