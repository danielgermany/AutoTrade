from candles_advanced import CandleSeries   # your file
from plot_candles_matlib import plot_candles
from csv_to_df import load_ohlcv_csv

path = "nq_data_1m.csv"

df, report = load_ohlcv_csv(path, tz="America/New_York", run_cleaner=True)

# Downstream fast ingest:

cs = CandleSeries(df, assume_clean=True)  # safe because cleaner set dtypes/invariants

#If you only have a list of Candle objects:

plot_candles(cs, last_n=100, title="List[Candle] - Last 100")


