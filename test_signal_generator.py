import numpy as np, pandas as pd
from signal_generator import IndicatorConfig, generate_signals

def make_df(n=400,seed=42):
    rs = np.random.default_rng(seed)
    base = 100 + 2*np.sin(2*np.pi*np.arange(n)/50)
    close = base + rs.normal(0, 0.3, n)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    vol = rs.integers(800, 1200, n)
    idx = pd.date_range('2024-01-02 09:30', periods=n, freq='T')
    return pd.DataFrame({'open':open_, 'high':high, 'low':low, 'close':close, 'volume':vol}, index=idx)

def test_shapes_and_columns():
    df = make_df(300)
    inds = [
        IndicatorConfig('rsi', {'length':14,'buy_below':30,'sell_above':70}, 1.0),
        IndicatorConfig('macd', {'fast':12,'slow':26,'signal_len':9}, 1.0),
        IndicatorConfig('supertrend', {'length':10,'multiplier':3.0}, 1.0),
    ]
    per, comp = generate_signals(df, inds, aggregation='weighted', deadband=0.1)
    assert per.shape[0] == comp.shape[0] == len(df)
    assert {'signal','score','reason'}.issubset(set(comp.columns))
    sigcols = [c for c in per.columns if c.startswith('signal_')]
    assert len(sigcols) == len(inds)
    assert per[sigcols].abs().max().max() <= 1

def test_deadband_behaviour():
    df = make_df(200)
    # Two opposing voters most of the time → near zero weighted score
    inds = [
        IndicatorConfig('roc', {'length':5, 'buy_above':0.0, 'sell_below':0.0}, 1.0),
        IndicatorConfig('vwap', {}, 1.0),
    ]
    per, comp = generate_signals(df, inds, aggregation='weighted', deadband=0.2)
    # Ensure a decent fraction of zeros from deadband
    frac_zero = (comp['signal'] == 0).mean()
    assert frac_zero > 0.15

def test_rsi_extremes_monotonic_series():
    # Strictly increasing prices → RSI should trend high, mostly short signals
    n = 120
    close = pd.Series(np.linspace(100, 110, n))
    df = pd.DataFrame({
        'open': close.shift(1).fillna(close.iloc[0]),
        'high': close + 0.5,
        'low': close - 0.5,
        'close': close,
        'volume': 1000
    })
    per, comp = generate_signals(df, [IndicatorConfig('rsi', {'length':14,'buy_below':30,'sell_above':70}, 1.0)], aggregation='majority')
    # After warmup, RSI should be > 60 most of the time → many −1 signals
    later = per['signal_rsi'].iloc[60:]
    frac_short = (later == -1).mean()
    assert frac_short > 0.5

def test_column_inference_aliases():
    # Provide aliased columns (O/H/L/C/Vol/Time) and ensure no crash
    n = 150
    rs = np.random.default_rng(7)
    c = 100 + rs.normal(0, 0.2, n).cumsum()/10
    df = pd.DataFrame({
        'O': np.r_[c[0], c[:-1]],
        'H': c + 0.5,
        'L': c - 0.5,
        'C': c,
        'Vol': rs.integers(500, 1500, n),
        'Time': pd.date_range('2024-02-01', periods=n, freq='T')
    })
    inds = [IndicatorConfig('macd', {'fast':12,'slow':26,'signal_len':9}, 1.0)]
    per, comp = generate_signals(df, inds, aggregation='majority')
    assert per.shape[0] == n and comp.shape[0] == n