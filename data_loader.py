import yfinance

# Loading historical data
def load_data(ticker="NVDA"):

    data = yfinance.download(ticker, period='3y', interval='1d')
    
    # Extracting 'Close' price
    df = data['Close'].copy()
    df.columns = ['Price']
    
    # Cleaning up the data
    df.dropna(inplace=True)
    df.sort_index(inplace=True) 

    print(f"Downloaded {ticker} data from {df.index.min()} to {df.index.max()}")
    return df

