import yfinance as yf
import pandas as pd

# Download data
dataF = yf.download("MNQ=F", start="2024-10-04", end="2025-05-10", interval="1h")
dataF.dropna(inplace=True)

# Fix multi-level column names (Price/Ticker -> just Price)
dataF.columns = dataF.columns.get_level_values(0)

print(dataF)
#Support and Resistance

#df1 is the dataframe used
#candle is the index of the row we are looking at

#Checks the high and lows of the candles from the dataframe

def support(df1, candle, candles_before, candles_after):
    if candle - candles_before < 0 or candle + candles_after >= len(df1):
        return 0

    for i in range(candle - candles_before + 1, candle + 1):
        # Sanity check here
            #print(f"[SUPPORT BEFORE] i={i}, i-1={i-1}, Low[i]={df1['Low'].iloc[i]}, Low[i-1]={df1['Low'].iloc[i - 1]}")
        if df1['Low'].iloc[i] > df1['Low'].iloc[i - 1]:
            return 0

    for i in range(candle + 1, candle + candles_after + 1):
        # Sanity check here
            #print(f"[SUPPORT AFTER] i={i}, i-1={i-1}, Low[i]={df1['Low'].iloc[i]}, Low[i-1]={df1['Low'].iloc[i - 1]}")
        if df1['Low'].iloc[i] < df1['Low'].iloc[i - 1]:
            return 0

    return 1


def resistance(df1, candle, candles_before, candles_after):
    if candle - candles_before < 0 or candle + candles_after >= len(df1):
        return 0

    for i in range(candle - candles_before + 1, candle + 1):
            #print(f"[RESISTANCE BEFORE] i={i}, i-1={i-1}, High[i]={df1['High'].iloc[i]}, High[i-1]={df1['High'].iloc[i - 1]}")
        if df1['High'].iloc[i] < df1['High'].iloc[i - 1]:
            return 0

    for i in range(candle + 1, candle + candles_after + 1):
            #print(f"[RESISTANCE AFTER] i={i}, i-1={i-1}, High[i]={df1['High'].iloc[i]}, High[i-1]={df1['High'].iloc[i - 1]}")
        if df1['High'].iloc[i] > df1['High'].iloc[i - 1]:
            return 0

    return 1


#Support and resistance levels
sr = []
n1=3
n2=2
for row in range(len(dataF)):
    if support(dataF, row, n1, n2):
        sr.append((row, dataF['Low'].iloc[row], 1))
    if resistance(dataF, row, n1, n2):
        sr.append((row, dataF['High'].iloc[row], 2))

