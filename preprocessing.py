import data_loader
import numpy as np


df = data_loader.load_data()
print(df.head())

# Calculating rolling mean and volatility, similar to calculating team/driver form.
def calculate_rolling_features(df, window=5):

    df_fe = df.copy()

    # Calculating daily log returns (analogous to daily points)
    df_fe["LogReturn"] = np.log(df_fe["Price"] / df_fe["Price"].shift(1))

    # Calculating rolling average price (Trend)
    df_fe["RollingMeanPrice"] = df_fe["Price"].rolling(window=window).mean()

    # Calculating rolling volatility (Risk/Form consistency)
    df_fe["RollingVolatility"] = df_fe["LogReturn"].rolling(window=window).std()

    # Lag features (Previous day performance)
    df_fe["Lag1_Price"] = df_fe["Price"].shift(1)
    df_fe["Lag2_Price"] = df_fe["Price"].shift(2)
    df_fe["Lag1_Return"] = df_fe["LogReturn"].shift(1)
    df_fe.dropna(inplace=True)


    return df_fe

# Calculation Relative Strength Index (RSI)
def calculate_rsi(df, period=14):

    df_fe = df.copy()

    # Calculating price deltas
    delta = df_fe["Price"].diff()
    # Separating gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Calculating RS and RSI
    rs = gain / loss
    df_fe["RSI"] = 100 - (100 / (1 + rs))

    return df_fe

# Putting it all together
def engineer_all_features(df):

    df_features = calculate_rolling_features(df)
    df_rsi = calculate_rsi(df_features)
    
    # Dropping NaNs created by rolling windows and shifts
    df_rsi.dropna(inplace=True)
    
    return df_rsi

df = engineer_all_features(df)
print(df.head(10))