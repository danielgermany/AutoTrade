import yfinance as yf
import pandas as pd

# Download data
dataF = yf.download("MNQ=F", start="2024-10-04", end="2025-05-10", interval="1h")
dataF.dropna(inplace=True)

#Support and Resistance

#df1 is the dataframe used
#candle is the index

#Checks the high and lows of the candles from the dataframe

def support(df1, candle, candles_before, candles_after):
    for i in range(1-candles_before+1, candle+1):
        if df1.low[i]>df1.low[i - 1]:
            return 0
    for i in range(candle+1,candle+candles_after+1):
        if df1.low[i]<df1.low[i - 1]:
            return 0
    return 1

def resistance (df1, candle, candles_before, candles_after):
    for i in range(candle-candles_before+1, candle+1):
        if df1.high[i]<df1.high[i-1]:
            return 0
    for i in range(candle+1,candle+candles_after+1):
        if df1.high[i]>df1.high[i-1]:
            return 0
    return 1
