import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta

def test_stock_symbol(symbol, max_retries=3):
    print(f"\nTesting symbol: {symbol}")
    for attempt in range(max_retries):
        try:
            # Use a shorter date range for testing
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Just test last 30 days
            
            # First try to get stock info
            stock = yf.Ticker(symbol)
            info = stock.info
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                print(f"✓ Symbol is valid (Current price: {info['regularMarketPrice']})")
            else:
                print("✗ Symbol appears invalid (No current price)")
                return False
            
            # Try to download some data
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if df.empty:
                print("✗ No historical data available")
                return False
            
            print(f"✓ Historical data available (Last {len(df)} days)")
            print(f"Latest close price: {df['Close'][-1]:.2f}")
            return True
            
        except Exception as e:
            wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
            if 'Rate limit' in str(e):
                print(f"Rate limit hit, waiting {wait_time} seconds before retry... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"✗ Error: {str(e)}")
                return False
    
    print("✗ Failed after all retries")
    return False

# Test a smaller set of popular symbols first
test_symbols = [
    'RELIANCE.NS',  # Reliance Industries (India)
    'HDFCBANK.NS',  # HDFC Bank (India)
    'INFY.NS',      # Infosys (India)
    'AAPL',         # Apple (US)
    'MSFT'          # Microsoft (US)
]

print("Testing Stock Symbols...")
working_symbols = []
for symbol in test_symbols:
    if test_stock_symbol(symbol):
        working_symbols.append(symbol)
    time.sleep(10)  # Add longer delay between symbols

print("\n=== Summary of Working Symbols ===")
for symbol in working_symbols:
    print(f"- {symbol}")

# Save working symbols to a file
with open('working_symbols.txt', 'w') as f:
    f.write("=== Working Stock Symbols ===\n\n")
    for symbol in working_symbols:
        f.write(f"{symbol}\n") 