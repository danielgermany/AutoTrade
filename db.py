from config import *
import yfinance as yf
import pandas as pd

# Download data
dataF = yf.download(ticker, start=start_date, end=end_date, interval=interval)
dataF.dropna(inplace=True)

# Fix multi-level column names (Price/Ticker -> just Price)
dataF.columns = dataF.columns.get_level_values(0)

print(dataF)