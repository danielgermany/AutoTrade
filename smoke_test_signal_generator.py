import pandas as pd
from signal_generator import IndicatorConfig, generate_signals


# Replace with your path
csv_path = 'data/nq_1m.csv' # columns: time/open/high/low/close/volume or similar


raw = pd.read_csv(csv_path)
# Basic column cleanup (auto‑infers aliases like o,h,l,c,vol,time,date)
# If your file already has canonical names, this is not needed.
raw.columns = [c.lower().strip() for c in raw.columns]
if 'datetime' not in raw.columns and 'time' in raw.columns:
    raw = raw.rename(columns={'time':'datetime'})
# If a unix timestamp exists, convert:
# raw['datetime'] = pd.to_datetime(raw['datetime'], unit='s', utc=True).tz_convert('America/New_York')


indicators = [
IndicatorConfig('rsi', {'length':14, 'buy_below':30, 'sell_above':70}, 1.0),
IndicatorConfig('macd', {'fast':12, 'slow':26, 'signal_len':9}, 1.0),
IndicatorConfig('vwap', {}, 1.0),
]
per, comp = generate_signals(raw, indicators, aggregation='majority')
print(comp.head())
print(comp['signal'].value_counts(dropna=False))