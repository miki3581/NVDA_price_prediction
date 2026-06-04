import preprocessing
from statsmodels.tsa.stattools import adfuller, kpss

# Checking stationarity
def stationarity(series, name = 'Series'):

    # Augmented Dickey-Fuller test
    adf_stat, adf_pval, _, _, _, _ = adfuller(series.dropna(), autolag = 'AIC')
    print(f"\nADF Test: {adf_stat}, p-value: {adf_pval}")

    #KPSS test
    kpss_stat, kpss_pval, _, _ = kpss(series.dropna(), regression = 'c', nlags = 'auto')
    print(f"\nKPSS Test: {kpss_stat}, p-value: {kpss_pval}")

    if adf_pval < 0.05:
        print(f"\nResult: {name} is Stationary (ADF p < 0.05)")
    else:
        print(f"\nResult: {name} is Non-Stationary (ADF p >= 0.05)")


# Testing for Price
stationarity(preprocessing.df["Price"], 'Price')

# Testing for LogRet
stationarity(preprocessing.df['LogReturn'], 'LogReturn')