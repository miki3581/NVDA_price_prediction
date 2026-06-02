import yfinance
import pandas as pd

def load_data(ticker="NVDA", start_date="2023-01-01", end_date="2026-01-01"):
    """
    Loads historical stock data for a given ticker.
    """
    data = yfinance.download(ticker, period = '3y', interval = '1d')["Close"]
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date']) # Ensure 'Date' column is datetime objects
    data.set_index('Date', inplace=True) # Set 'Date' as the index
    data.sort_index(inplace=True) # Sort data by date

    return data
