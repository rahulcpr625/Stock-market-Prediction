import yfinance as yf
import datetime as dt

def test_stock(symbol):
    print(f"Testing stock symbol: {symbol}")
    try:
        # Define the start and end dates for stock data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.now()
        
        # Download stock data
        df = yf.download(symbol, start=start, end=end)
        
        if df.empty:
            print(f"No data found for {symbol}")
        else:
            print(f"Data found for {symbol}")
            print(f"Number of records: {len(df)}")
            print("\nFirst few records:")
            print(df.head())
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")

# Test with different stock symbols
test_stock('POWERGRID.NS')  # Indian stock
test_stock('AAPL')          # US stock
test_stock('RELIANCE.NS')   # Another Indian stock 