# smoke_test_signal_generator.py
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from signal_generator import IndicatorConfig, generate_signals


# --- synthetic OHLCV ---
rs = np.random.default_rng(0)
N = 600
# Smooth-ish base signal with small noise → produces MACD crosses and RSI dynamics
base = 100 + 2*np.sin(2*np.pi*np.arange(N)/50)
close = base + rs.normal(0, 0.25, N)
open_ = np.r_[close[0], close[:-1]]
high = np.maximum(open_, close) + 0.4
low = np.minimum(open_, close) - 0.4
vol = rs.integers(900, 1100, N)


# Optional datetime index (1‑minute bars)
idx = pd.date_range('2024-01-02 09:30', periods=N, freq='T')
df = pd.DataFrame({'open':open_, 'high':high, 'low':low, 'close':close, 'volume':vol}, index=idx)


indicators = [
IndicatorConfig('rsi', {'length':14, 'buy_below':30, 'sell_above':70}, weight=1.0),
IndicatorConfig('macd', {'fast':12, 'slow':26, 'signal_len':9}, weight=1.0),
IndicatorConfig('supertrend', {'length':10, 'multiplier':3.0}, weight=1.0),
]


per_ind, composite = generate_signals(df, indicators, aggregation='weighted', deadband=0.1)


print('\n=== Per‑indicator (tail) ===')
print(per_ind.filter(like='signal_').tail())
print('\n=== Composite (tail) ===')
print(composite.tail())


# Simple assertions (won't raise if OK)
assert {'signal','score','reason'}.issubset(set(composite.columns))
assert per_ind.shape[0] == composite.shape[0] == N
assert per_ind.filter(like='signal_').abs().max().max() <= 1 # ternary in {−1,0,+1}

# Append to the smoke script to visualize signals on price

ax = df['close'].plot(figsize=(10,4), title='Close with composite signals')
# Plot long/short markers (naïve visualization)
long_ix = composite['signal'] == 1
short_ix = composite['signal'] == -1
ax.plot(df.index[long_ix], df['close'][long_ix], marker='^', linestyle='None', markersize=5)
ax.plot(df.index[short_ix], df['close'][short_ix], marker='v', linestyle='None', markersize=5)
plt.show()

print('\nSmoke test passed!')