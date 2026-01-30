from api.alpaca_fetch import fetch_stock_data
import traceback

if __name__ == '__main__':
    try:
        df = fetch_stock_data('AAPL', days=7)
        print(df.tail())
    except Exception as e:
        traceback.print_exc()
        print('\n--- DONE ---')
