import numpy as np

# Calculating rolling mean and volatility, similar to calculating team/driver form.
def calculate_rolling_features(df, window=5):

    df_fe = df.copy()

    # Calculating daily log returns (analogous to daily points)
    df_fe["LogReturn"] = np.log(df_fe["Price"] / df_fe["Price"].shift(1))

    # Calculating rolling average price (Trend)
    df_fe["RollingMeanReturn"] = df_fe["LogReturn"].rolling(window=window).mean()

    # Calculating rolling volatility (Risk/Form consistency)
    df_fe["RollingVolatility"] = df_fe["LogReturn"].rolling(window=window).std()

    # Lag features (Previous day performance)
    df_fe["Lag1_Return"] = df_fe["LogReturn"].shift(1)
    df_fe["Lag2_Return"] = df_fe["LogReturn"].shift(2)
    df_fe["Lag3_Return"] = df_fe["LogReturn"].shift(3)

    df_fe.dropna(inplace=True)


    return df_fe

# Calculation Relative Strength Index (RSI)
def calculate_rsi(df, period=14):

    df_fe = df.copy()

    # Calculating price deltas
    delta = df_fe["LogReturn"]

    # Separating gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Calculating RS and RSI
    rs = gain / loss
    df_fe["RSI"] = 100 - (100 / (1 + rs))

    return df_fe

# Creating target variable, LogReturn because Price is Non-stationary
def target_5day_avg(df):
    
    df_target = df.copy()

   # Shift LogReturn backward to get future returns
    df_target['r_1'] = df_target['LogReturn'].shift(-1)
    df_target['r_2'] = df_target['LogReturn'].shift(-2)
    df_target['r_3'] = df_target['LogReturn'].shift(-3)
    df_target['r_4'] = df_target['LogReturn'].shift(-4)
    df_target['r_5'] = df_target['LogReturn'].shift(-5)
    
    # Calculate the mean of these 5 future returns
    df_target['Avg Log Return'] = df_target[['r_1', 'r_2', 'r_3', 'r_4', 'r_5']].mean(axis=1)
    
    # Drop the temporary columns
    df_target.drop(columns=['r_1', 'r_2', 'r_3', 'r_4', 'r_5'], inplace=True)
    
    # Drop NaNs (last 5 rows will be NaN due to future look-ahead)
    df_target.dropna(inplace=True)
    
    return df_target

# Putting it all together
def engineer_all_features(df):

    df_features = calculate_rolling_features(df)

    df_rsi = calculate_rsi(df_features)
    
    # Dropping NaNs created by rolling windows and shifts
    df_rsi.dropna(inplace=True)
    
    df_final = target_5day_avg(df_rsi)
    
    feature_cols = ['RollingMeanReturn', 'RollingVolatility', 'Lag1_Return', 'Lag2_Return', 'Lag3_Return', 'RSI']
    X = df_final[feature_cols]
    y = df_final['Avg Log Return']
    prices = df_final['Price']
    
    return X, y, prices, df_final.index
