import yfinance as yf
import pandas as pd

# Download 1-hour EUR/USD data
dataF = yf.download("EURUSD=X", start="2024-10-04", end="2025-05-10", interval="1h")

def signal_generator(df):
    open_ = float(df['Open'].iloc[-1])
    close = float(df['Close'].iloc[-1])
    previous_open = float(df['Open'].iloc[-2])
    previous_close = float(df['Close'].iloc[-2])

    # Bearish Pattern
    if close < previous_open and previous_open <= open_:
        return 1

    # Bullish Pattern
    elif open_ <= previous_close and close > previous_open:
        return 2

    else:
        return 0

# Initialize signal list
signal = [0]  # First row has no previous data

# Loop from second row onward
for i in range(1, len(dataF)):
    df_slice = dataF.iloc[i-1:i+1]
    signal.append(signal_generator(df_slice))

dataF["signal"] = signal
print(dataF.signal.value_counts())
