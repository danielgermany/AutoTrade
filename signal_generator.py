"""
signal_generator.py
===================
Readable, well-documented signal generator for OHLC[V] data.

Goals
-----
- Prioritize clarity of names and structure over micro-optimizations.
- Keep indicator math explicit and easy to follow.
- Provide simple, policy-based signals for each indicator.
- Aggregate per-bar votes into a single composite signal.

Inputs
------
- DataFrame with columns that *look like* open, high, low, close[, volume, datetime].
  (Case-insensitive and short aliases like o,h,l,c,vol are accepted.)
- A list of `IndicatorConfig` entries specifying which indicators to compute
  and any parameter overrides.

Outputs
-------
- `per_indicator`: DataFrame with each indicator's signal and helpful values.
- `composite`:     DataFrame with the final one-number decision per bar.

Tip
---
Keep this module independent of regime logic for readability.
Use `regime.py` to classify the market regime and decide which indicator
configs to pass in (or how to weight them) for a given period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ArrayLike = Union[np.ndarray, pd.Series]
SeriesLike = Union[np.ndarray, pd.Series]


# ---------------------------------------------------------------------------
# Column inference and adapters
# ---------------------------------------------------------------------------

def infer_price_columns(data: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Map to canonical names: open, high, low, close, volume, datetime.

    We accept common aliases (case-insensitive): o, h, l, c, vol, v, time, timestamp, date.
    """
    lower_map = {col.lower(): col for col in data.columns}

    def find(*candidates: str) -> Optional[str]:
        for name in candidates:
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        return None

    return {
        "open":     find("open", "o"),
        "high":     find("high", "h"),
        "low":      find("low", "l"),
        "close":    find("close", "c"),
        "volume":   find("volume", "vol", "v"),
        "datetime": find("datetime", "time", "timestamp", "date"),
    }


def candles_to_dataframe(candles: Iterable[Any]) -> pd.DataFrame:
    """
    Convert an iterable of Candle-like objects (e.g., from candles_advanced.py)
    into a pandas DataFrame with the canonical columns.
    """
    rows: List[Dict[str, Any]] = []

    def fetch(obj: Any, names: Sequence[str], default: Any = np.nan) -> Any:
        # attribute access
        for nm in names:
            if hasattr(obj, nm):
                return getattr(obj, nm)
        # dict-like access
        if isinstance(obj, Mapping):
            for nm in names:
                if nm in obj:
                    return obj[nm]
        return default

    for c in candles:
        rows.append(
            {
                "open":     fetch(c, ("open", "o", "Open", "O")),
                "high":     fetch(c, ("high", "h", "High", "H")),
                "low":      fetch(c, ("low", "l", "Low", "L")),
                "close":    fetch(c, ("close", "c", "Close", "C")),
                "volume":   fetch(c, ("volume", "vol", "v", "Volume", "Vol", "V"), np.nan),
                "datetime": fetch(c, ("datetime", "time", "timestamp", "Date", "date"), np.nan),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Small numeric helpers (intentionally straightforward)
# ---------------------------------------------------------------------------

def as_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(copy=False)
    if isinstance(x, np.ndarray):
        return x
    raise TypeError("Expected Series or ndarray.")


def series_like(reference: ArrayLike, values: np.ndarray, name: Optional[str] = None) -> ArrayLike:
    if isinstance(reference, pd.Series):
        return pd.Series(values, index=reference.index, name=name or reference.name)
    return values


def shift_array(values: np.ndarray, periods: int) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    out[:] = np.nan
    if periods == 0:
        out[:] = values
        return out
    if periods > 0:
        out[periods:] = values[:-periods]
    else:
        out[:periods] = values[-periods:]
    return out


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return np.full_like(values, np.nan, dtype=float)
    result = np.empty_like(values, dtype=float)
    result[:] = np.nan
    csum = np.cumsum(np.insert(values, 0, 0.0))
    idx = np.arange(window, len(values) + 1)
    result[window - 1 :] = (csum[idx] - csum[:-window]) / window
    return result


def rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return np.full_like(values, np.nan, dtype=float)
    from collections import deque

    out = np.full_like(values, np.nan, dtype=float)
    dq: "deque[Tuple[float,int]]" = deque()
    for i, val in enumerate(values):
        while dq and dq[-1][0] <= val:
            dq.pop()
        dq.append((val, i))
        while dq and dq[0][1] <= i - window:
            dq.popleft()
        if i >= window - 1:
            out[i] = dq[0][0]
    return out


def rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    return -rolling_max(-values, window)


def ema_vector(values: np.ndarray, length: int) -> np.ndarray:
    if length <= 0 or values.size == 0:
        return np.full_like(values, np.nan, dtype=float)
    result = np.empty_like(values, dtype=float)
    result[:] = np.nan
    alpha = 2.0 / (length + 1.0)

    if len(values) >= length:
        seed = np.nanmean(values[:length])
        result[length - 1] = seed
        prev = seed
        for i in range(length, len(values)):
            prev = alpha * values[i] + (1 - alpha) * prev
            result[i] = prev
    else:
        prev = values[0]
        result[0] = prev
        for i in range(1, len(values)):
            prev = alpha * values[i] + (1 - alpha) * prev
            result[i] = prev
    return result


def rma_vector(values: np.ndarray, length: int) -> np.ndarray:
    """Wilder-style EMA (alpha = 1/length)."""
    if length <= 0 or values.size == 0:
        return np.full_like(values, np.nan, dtype=float)
    result = np.empty_like(values, dtype=float)
    result[:] = np.nan
    alpha = 1.0 / length

    if len(values) >= length:
        seed = np.nanmean(values[:length])
        result[length - 1] = seed
        prev = seed
        for i in range(length, len(values)):
            prev = alpha * values[i] + (1 - alpha) * prev
            result[i] = prev
    else:
        prev = values[0]
        result[0] = prev
        for i in range(1, len(values)):
            prev = alpha * values[i] + (1 - alpha) * prev
            result[i] = prev
    return result


def safe_div(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    denom_safe = denom.copy()
    denom_safe[np.isclose(denom_safe, 0)] = np.nan
    return numer / denom_safe


# ---------------------------------------------------------------------------
# Indicator implementations (clear names)
# ---------------------------------------------------------------------------

def compute_sma(close: ArrayLike, length: int = 20) -> ArrayLike:
    close_np = as_numpy(close).astype(float)
    result = rolling_mean(close_np, length)
    return series_like(close, result, name="sma")


def compute_ema(close: ArrayLike, length: int = 20) -> ArrayLike:
    close_np = as_numpy(close).astype(float)
    result = ema_vector(close_np, length)
    return series_like(close, result, name="ema")


def compute_rsi(close: ArrayLike, length: int = 14) -> ArrayLike:
    close_np = as_numpy(close).astype(float)
    delta = np.diff(close_np, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = rma_vector(gain, length)
    avg_loss = rma_vector(loss, length)
    rs = safe_div(avg_gain, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return series_like(close, rsi, name="rsi")


def compute_rate_of_change(close: ArrayLike, length: int = 10) -> ArrayLike:
    close_np = as_numpy(close).astype(float)
    result = np.full_like(close_np, np.nan, dtype=float)
    if length > 0 and len(close_np) > length:
        prev = shift_array(close_np, length)
        result = safe_div(close_np - prev, np.abs(prev)) * 100.0
    return series_like(close, result, name="roc")


def compute_true_range(high: ArrayLike, low: ArrayLike, close: ArrayLike) -> ArrayLike:
    high_np = as_numpy(high).astype(float)
    low_np = as_numpy(low).astype(float)
    close_np = as_numpy(close).astype(float)
    prev_close = shift_array(close_np, 1)
    tr = np.maximum(high_np - low_np, np.maximum(np.abs(high_np - prev_close), np.abs(low_np - prev_close)))
    return series_like(close, tr, name="true_range")


def compute_atr(high: ArrayLike, low: ArrayLike, close: ArrayLike, length: int = 14) -> ArrayLike:
    tr = as_numpy(compute_true_range(high, low, close)).astype(float)
    result = rma_vector(tr, length)
    return series_like(close, result, name="atr")


def compute_macd(close: ArrayLike, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    close_np = as_numpy(close).astype(float)
    fast_ema = ema_vector(close_np, fast)
    slow_ema = ema_vector(close_np, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema_vector(macd_line, signal)
    histogram = macd_line - signal_line
    return (
        series_like(close, macd_line, name="macd_line"),
        series_like(close, signal_line, name="macd_signal"),
        series_like(close, histogram, name="macd_hist"),
    )


def compute_bollinger_bands(close: ArrayLike, length: int = 20, stdev_mult: float = 2.0) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    close_np = as_numpy(close).astype(float)
    basis = rolling_mean(close_np, length)
    stdev = np.empty_like(close_np, dtype=float)
    stdev[:] = np.nan
    if length > 1 and len(close_np) >= length:
        for i in range(length - 1, len(close_np)):
            window = close_np[i - length + 1 : i + 1]
            stdev[i] = np.nanstd(window, ddof=0)
    upper = basis + stdev_mult * stdev
    lower = basis - stdev_mult * stdev
    width = upper - lower
    return (
        series_like(close, basis, name="bb_basis"),
        series_like(close, upper, name="bb_upper"),
        series_like(close, lower, name="bb_lower"),
        series_like(close, width, name="bb_width"),
    )


def compute_keltner_channels(high: ArrayLike, low: ArrayLike, close: ArrayLike, length: int = 20, atr_mult: float = 2.0) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    basis = compute_ema(close, length=length)
    atr_values = compute_atr(high, low, close, length=length)
    upper = as_numpy(basis) + atr_mult * as_numpy(atr_values)
    lower = as_numpy(basis) - atr_mult * as_numpy(atr_values)
    return basis, series_like(close, upper, name="kc_upper"), series_like(close, lower, name="kc_lower")


def compute_dmi_adx(high: ArrayLike, low: ArrayLike, close: ArrayLike, length: int = 14) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    high_np = as_numpy(high).astype(float)
    low_np = as_numpy(low).astype(float)
    close_np = as_numpy(close).astype(float)

    up_move = high_np - shift_array(high_np, 1)
    down_move = shift_array(low_np, 1) - low_np
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = as_numpy(compute_true_range(high_np, low_np, close_np)).astype(float)
    plus_di = 100.0 * rma_vector(plus_dm, length) / np.clip(rma_vector(tr, length), 1e-12, np.inf)
    minus_di = 100.0 * rma_vector(minus_dm, length) / np.clip(rma_vector(tr, length), 1e-12, np.inf)
    dx = 100.0 * np.abs(plus_di - minus_di) / np.clip((plus_di + minus_di), 1e-12, np.inf)
    adx = rma_vector(dx, length)

    return series_like(close, plus_di, name="plus_di"), series_like(close, minus_di, name="minus_di"), series_like(close, adx, name="adx")


def compute_stochastic(high: ArrayLike, low: ArrayLike, close: ArrayLike, k_period: int = 14, d_period: int = 3, k_smoothing: int = 3) -> Tuple[ArrayLike, ArrayLike]:
    high_np = as_numpy(high).astype(float)
    low_np = as_numpy(low).astype(float)
    close_np = as_numpy(close).astype(float)

    lowest_low = rolling_min(low_np, k_period)
    highest_high = rolling_max(high_np, k_period)
    raw_k = 100.0 * safe_div(close_np - lowest_low, highest_high - lowest_low)
    smooth_k = rolling_mean(raw_k, k_smoothing) if k_smoothing > 1 else raw_k
    smooth_d = rolling_mean(smooth_k, d_period) if d_period > 1 else smooth_k
    return series_like(close, smooth_k, name="stoch_k"), series_like(close, smooth_d, name="stoch_d")


def compute_supertrend(high: ArrayLike, low: ArrayLike, close: ArrayLike, length: int = 10, multiplier: float = 3.0) -> Tuple[ArrayLike, ArrayLike]:
    """
    Returns (supertrend_line, direction), where direction is +1 for up, -1 for down.
    """
    high_np = as_numpy(high).astype(float)
    low_np = as_numpy(low).astype(float)
    close_np = as_numpy(close).astype(float)
    n = len(close_np)

    atr_values = as_numpy(compute_atr(high_np, low_np, close_np, length=length)).astype(float)
    hl2 = (high_np + low_np) / 2.0
    upper_basic = hl2 + multiplier * atr_values
    lower_basic = hl2 - multiplier * atr_values

    upper = upper_basic.copy()
    lower = lower_basic.copy()
    for i in range(1, n):
        upper[i] = min(upper_basic[i], upper[i - 1]) if not np.isnan(upper[i - 1]) else upper_basic[i]
        lower[i] = max(lower_basic[i], lower[i - 1]) if not np.isnan(lower[i - 1]) else lower_basic[i]

    trend_line = np.full(n, np.nan, dtype=float)
    direction = np.full(n, np.nan, dtype=float)
    direction[0] = 1.0
    for i in range(1, n):
        prev_line = trend_line[i - 1]
        prev_dir = direction[i - 1] if not np.isnan(direction[i - 1]) else 1.0

        if np.isnan(prev_line):
            # initialize
            trend_line[i] = upper[i] if close_np[i] <= upper[i] else lower[i]
            direction[i] = -1.0 if close_np[i] <= upper[i] else 1.0
            continue

        if close_np[i] > prev_line:
            direction[i] = 1.0
            trend_line[i] = max(lower[i], prev_line)
        else:
            direction[i] = -1.0
            trend_line[i] = min(upper[i], prev_line)

    return series_like(close, trend_line, name="supertrend"), series_like(close, direction, name="supertrend_dir")


def compute_donchian_channels(high: ArrayLike, low: ArrayLike, length: int = 20) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    upper = rolling_max(as_numpy(high).astype(float), length)
    lower = rolling_min(as_numpy(low).astype(float), length)
    middle = (upper + lower) / 2.0
    return series_like(high, upper, name="donchian_upper"), series_like(low, lower, name="donchian_lower"), series_like(high, middle, name="donchian_mid")


def compute_ichimoku(high: ArrayLike, low: ArrayLike, close: ArrayLike, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> Dict[str, ArrayLike]:
    high_np = as_numpy(high).astype(float)
    low_np = as_numpy(low).astype(float)
    close_np = as_numpy(close).astype(float)

    tenkan_hi = rolling_max(high_np, tenkan)
    tenkan_lo = rolling_min(low_np, tenkan)
    kijun_hi = rolling_max(high_np, kijun)
    kijun_lo = rolling_min(low_np, kijun)

    conversion = (tenkan_hi + tenkan_lo) / 2.0
    base = (kijun_hi + kijun_lo) / 2.0
    span_a = (conversion + base) / 2.0
    span_b = (rolling_max(high_np, senkou_b) + rolling_min(low_np, senkou_b)) / 2.0
    lagging = shift_array(close_np, -kijun)

    return {
        "conversion": series_like(close, conversion, name="ichimoku_conversion"),
        "base":       series_like(close, base, name="ichimoku_base"),
        "span_a":     series_like(close, span_a, name="ichimoku_span_a"),
        "span_b":     series_like(close, span_b, name="ichimoku_span_b"),
        "lagging":    series_like(close, lagging, name="ichimoku_lagging"),
    }


def compute_obv(close: ArrayLike, volume: ArrayLike) -> ArrayLike:
    close_np = as_numpy(close).astype(float)
    volume_np = as_numpy(volume).astype(float)
    delta = np.diff(close_np, prepend=np.nan)
    sign = np.where(delta > 0, 1.0, np.where(delta < 0, -1.0, 0.0))
    signed_volume = np.where(np.isnan(sign), 0.0, sign * volume_np)
    obv = np.cumsum(np.nan_to_num(signed_volume, nan=0.0))
    return series_like(close, obv, name="obv")


def compute_mfi(high: ArrayLike, low: ArrayLike, close: ArrayLike, volume: ArrayLike, length: int = 14) -> ArrayLike:
    high_np = as_numpy(high).astype(float)
    low_np = as_numpy(low).astype(float)
    close_np = as_numpy(close).astype(float)
    volume_np = as_numpy(volume).astype(float)

    typical_price = (high_np + low_np + close_np) / 3.0
    raw_money_flow = typical_price * volume_np
    prev_typical = shift_array(typical_price, 1)

    positive_flow = np.where(typical_price > prev_typical, raw_money_flow, 0.0)
    negative_flow = np.where(typical_price < prev_typical, raw_money_flow, 0.0)

    pos_sum = rolling_mean(positive_flow, length) * length
    neg_sum = rolling_mean(negative_flow, length) * length

    money_ratio = safe_div(pos_sum, neg_sum)
    mfi = 100.0 - (100.0 / (1.0 + money_ratio))
    return series_like(close, mfi, name="mfi")


def compute_vwap(high: ArrayLike, low: ArrayLike, close: ArrayLike, volume: ArrayLike, session_resets: Optional[ArrayLike] = None) -> ArrayLike:
    high_np = as_numpy(high).astype(float)
    low_np = as_numpy(low).astype(float)
    close_np = as_numpy(close).astype(float)
    volume_np = as_numpy(volume).astype(float)
    typical_price = (high_np + low_np + close_np) / 3.0
    numerator = typical_price * volume_np

    if session_resets is None:
        cum_numer = np.cumsum(numerator)
        cum_denom = np.cumsum(volume_np)
        vwap = safe_div(cum_numer, cum_denom)
        return series_like(close, vwap, name="vwap")

    reset_mask = as_numpy(session_resets).astype(bool)
    out = np.empty_like(close_np, dtype=float)
    out[:] = np.nan
    run_numer = 0.0
    run_denom = 0.0
    for i in range(len(close_np)):
        if i == 0 or reset_mask[i]:
            run_numer = numerator[i]
            run_denom = volume_np[i]
        else:
            run_numer += numerator[i]
            run_denom += volume_np[i]
        out[i] = run_numer / run_denom if run_denom != 0 else np.nan
    return series_like(close, out, name="vwap")


# ---------------------------------------------------------------------------
# Policy functions (turn indicator values into +1/0/-1 signals)
# ---------------------------------------------------------------------------

def rsi_policy(close: ArrayLike, *, length: int = 14, buy_below: float = 30.0, sell_above: float = 70.0) -> Dict[str, SeriesLike]:
    rsi_values = compute_rsi(close, length=length)
    rsi_np = as_numpy(rsi_values).astype(float)
    signal = np.where(rsi_np <= buy_below, 1, np.where(rsi_np >= sell_above, -1, 0))
    return {"signal": series_like(close, signal, name="signal_rsi"), "rsi": rsi_values}


def macd_policy(close: ArrayLike, *, fast: int = 12, slow: int = 26, signal_len: int = 9) -> Dict[str, SeriesLike]:
    macd_line, macd_signal, macd_hist = compute_macd(close, fast=fast, slow=slow, signal=signal_len)
    line_np = as_numpy(macd_line)
    sig_np = as_numpy(macd_signal)
    crossed_up = (line_np > sig_np) & (shift_array(line_np - sig_np, 1) <= 0)
    crossed_down = (line_np < sig_np) & (shift_array(line_np - sig_np, 1) >= 0)
    signal = np.where(crossed_up, 1, np.where(crossed_down, -1, 0)).astype(float)
    return {
        "signal": series_like(close, signal, name="signal_macd"),
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
    }


def bollinger_policy(close: ArrayLike, *, length: int = 20, stdev_mult: float = 2.0, mode: str = "mean_reversion") -> Dict[str, SeriesLike]:
    basis, upper, lower, width = compute_bollinger_bands(close, length=length, stdev_mult=stdev_mult)
    c_np = as_numpy(close)
    u_np = as_numpy(upper)
    l_np = as_numpy(lower)

    if mode.lower() == "trend":
        signal = np.where(c_np > u_np, 1, np.where(c_np < l_np, -1, 0))
    else:
        signal = np.where(c_np <= l_np, 1, np.where(c_np >= u_np, -1, 0))

    return {
        "signal": series_like(close, signal, name="signal_bollinger"),
        "bb_basis": basis,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": width,
    }


def keltner_policy(high: ArrayLike, low: ArrayLike, close: ArrayLike, *, length: int = 20, atr_mult: float = 2.0) -> Dict[str, SeriesLike]:
    basis, upper, lower = compute_keltner_channels(high, low, close, length=length, atr_mult=atr_mult)
    c_np = as_numpy(close)
    u_np = as_numpy(upper)
    l_np = as_numpy(lower)
    signal = np.where(c_np > u_np, 1, np.where(c_np < l_np, -1, 0))
    return {"signal": series_like(close, signal, name="signal_keltner"), "kc_basis": basis, "kc_upper": upper, "kc_lower": lower}


def dmi_adx_policy(high: ArrayLike, low: ArrayLike, close: ArrayLike, *, length: int = 14, adx_min: float = 20.0) -> Dict[str, SeriesLike]:
    plus_di, minus_di, adx_values = compute_dmi_adx(high, low, close, length=length)
    p = as_numpy(plus_di)
    m = as_numpy(minus_di)
    a = as_numpy(adx_values)
    long_ok = (p > m) & (a >= adx_min)
    short_ok = (m > p) & (a >= adx_min)
    signal = np.where(long_ok, 1, np.where(short_ok, -1, 0)).astype(float)
    return {"signal": series_like(close, signal, name="signal_dmi_adx"), "plus_di": plus_di, "minus_di": minus_di, "adx": adx_values}


def stochastic_policy(high: ArrayLike, low: ArrayLike, close: ArrayLike, *, k: int = 14, d: int = 3, smooth_k: int = 3, overbought: float = 80.0, oversold: float = 20.0) -> Dict[str, SeriesLike]:
    k_line, d_line = compute_stochastic(high, low, close, k_period=k, d_period=d, k_smoothing=smooth_k)
    K = as_numpy(k_line)
    D = as_numpy(d_line)
    up_cross = (K > D) & (shift_array(K - D, 1) <= 0) & (K < oversold)
    down_cross = (K < D) & (shift_array(K - D, 1) >= 0) & (K > overbought)
    signal = np.where(up_cross, 1, np.where(down_cross, -1, 0)).astype(float)
    return {"signal": series_like(close, signal, name="signal_stochastic"), "stoch_k": k_line, "stoch_d": d_line}


def supertrend_policy(high: ArrayLike, low: ArrayLike, close: ArrayLike, *, length: int = 10, multiplier: float = 3.0) -> Dict[str, SeriesLike]:
    st_line, st_dir = compute_supertrend(high, low, close, length=length, multiplier=multiplier)
    c = as_numpy(close)
    st = as_numpy(st_line)
    d = as_numpy(st_dir)
    signal = np.where((c > st) & (d > 0), 1, np.where((c < st) & (d < 0), -1, 0)).astype(float)
    return {"signal": series_like(close, signal, name="signal_supertrend"), "supertrend": st_line, "supertrend_dir": st_dir}


def donchian_policy(high: ArrayLike, low: ArrayLike, close: ArrayLike, *, length: int = 20) -> Dict[str, SeriesLike]:
    upper, lower, mid = compute_donchian_channels(high, low, length=length)
    c = as_numpy(close)
    u = as_numpy(upper)
    l = as_numpy(lower)
    signal = np.where(c > u, 1, np.where(c < l, -1, 0)).astype(float)
    return {"signal": series_like(close, signal, name="signal_donchian"), "donchian_upper": upper, "donchian_lower": lower, "donchian_mid": mid}


def ichimoku_policy(high: ArrayLike, low: ArrayLike, close: ArrayLike, *, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> Dict[str, SeriesLike]:
    ichi = compute_ichimoku(high, low, close, tenkan=tenkan, kijun=kijun, senkou_b=senkou_b)
    c = as_numpy(close)
    span_a = as_numpy(ichi["span_a"])
    span_b = as_numpy(ichi["span_b"])
    conv = as_numpy(ichi["conversion"])
    base = as_numpy(ichi["base"])

    above_cloud = c > np.maximum(span_a, span_b)
    below_cloud = c < np.minimum(span_a, span_b)
    tk_cross_up = (conv > base) & (shift_array(conv - base, 1) <= 0)
    tk_cross_down = (conv < base) & (shift_array(conv - base, 1) >= 0)

    signal = np.where(above_cloud & tk_cross_up, 1, np.where(below_cloud & tk_cross_down, -1, 0)).astype(float)
    out = {"signal": series_like(close, signal, name="signal_ichimoku")}
    out.update(ichi)
    return out


def vwap_policy(high: ArrayLike, low: ArrayLike, close: ArrayLike, volume: ArrayLike, *, session_resets: Optional[ArrayLike] = None) -> Dict[str, SeriesLike]:
    vwap_values = compute_vwap(high, low, close, volume, session_resets=session_resets)
    signal = np.where(as_numpy(close) > as_numpy(vwap_values), 1, np.where(as_numpy(close) < as_numpy(vwap_values), -1, 0)).astype(float)
    return {"signal": series_like(close, signal, name="signal_vwap"), "vwap": vwap_values}


def roc_policy(close: ArrayLike, *, length: int = 10, buy_above: float = 0.0, sell_below: float = 0.0) -> Dict[str, SeriesLike]:
    roc_values = compute_rate_of_change(close, length=length)
    r = as_numpy(roc_values)
    signal = np.where(r > buy_above, 1, np.where(r < sell_below, -1, 0)).astype(float)
    return {"signal": series_like(close, signal, name="signal_roc"), "roc": roc_values}


def obv_policy(close: ArrayLike, volume: ArrayLike, *, ema_length: int = 20) -> Dict[str, SeriesLike]:
    obv_values = as_numpy(compute_obv(close, volume)).astype(float)
    obv_ema = ema_vector(obv_values, ema_length)
    signal = np.where(obv_values > obv_ema, 1, np.where(obv_values < obv_ema, -1, 0)).astype(float)
    return {
        "signal": series_like(close, signal, name="signal_obv"),
        "obv": series_like(close, obv_values, name="obv"),
        "obv_ema": series_like(close, obv_ema, name="obv_ema"),
    }


def mfi_policy(high: ArrayLike, low: ArrayLike, close: ArrayLike, volume: ArrayLike, *, length: int = 14, overbought: float = 80.0, oversold: float = 20.0) -> Dict[str, SeriesLike]:
    mfi_values = compute_mfi(high, low, close, volume, length=length)
    m = as_numpy(mfi_values)
    signal = np.where(m <= oversold, 1, np.where(m >= overbought, -1, 0)).astype(float)
    return {"signal": series_like(close, signal, name="signal_mfi"), "mfi": mfi_values}


# Registry of available indicator policies
POLICY_REGISTRY = {
    "rsi":        rsi_policy,
    "macd":       macd_policy,
    "bollinger":  bollinger_policy,
    "keltner":    keltner_policy,
    "dmi_adx":    dmi_adx_policy,
    "stochastic": stochastic_policy,
    "supertrend": supertrend_policy,
    "donchian":   donchian_policy,
    "ichimoku":   ichimoku_policy,
    "vwap":       vwap_policy,
    "roc":        roc_policy,
    "obv":        obv_policy,
    "mfi":        mfi_policy,
}


# ---------------------------------------------------------------------------
# Configuration and CSV defaults
# ---------------------------------------------------------------------------

@dataclass
class IndicatorConfig:
    """
    Defines one indicator's participation in the voting ensemble.
    - `name`: key in POLICY_REGISTRY (e.g., "rsi", "macd").
    - `params`: dictionary of parameter overrides for the policy.
    - `weight`: vote weight used by "weighted" aggregation.
    """
    name: str
    params: Dict[str, Any]
    weight: float = 1.0


def parse_params_string(text: Optional[str]) -> Dict[str, Any]:
    """
    Parse a string like "length=14, buy_below=30; sell_above=70" into a dict.
    Values are coerced to int/float/bool when possible.
    """
    if text is None:
        return {}
    s = str(text).strip()
    if not s:
        return {}
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    out: Dict[str, Any] = {}
    for p in parts:
        if "=" in p:
            key, value = [x.strip() for x in p.split("=", 1)]
            val_lower = value.lower()
            if val_lower in ("true", "false"):
                out[key] = (val_lower == "true")
                continue
            try:
                if "." in value or "e" in val_lower:
                    out[key] = float(value)
                else:
                    out[key] = int(value)
            except ValueError:
                out[key] = value
        else:
            out[p] = True
    return out


def load_default_params_from_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Expect a CSV with columns like: Key, Params
    Returns: {indicator_name: {param_key: value, ...}, ...}
    """
    df = pd.read_csv(csv_path)
    key_col = "Key" if "Key" in df.columns else df.columns[0]
    params_col = "Params" if "Params" in df.columns else None
    defaults: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        name = str(row[key_col]).strip().lower()
        defaults[name] = parse_params_string(row[params_col]) if params_col else {}
    return defaults


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def weighted_vote(signals: List[float], weights: List[float]) -> Tuple[float, float]:
    """
    Return (final_signal, score). Score in [-1, +1] is the normalized sum.
    """
    if not signals:
        return 0.0, 0.0
    numerator = float(np.dot(signals, weights))
    denominator = float(np.sum(np.abs(weights))) or 1.0
    score = numerator / denominator
    final_signal = float(np.sign(score)) if abs(score) > 1e-12 else 0.0
    return final_signal, score


def majority_vote(signals: List[float]) -> Tuple[float, float]:
    """
    Return (final_signal, score). Score is the median of non-zero votes.
    """
    votes = [s for s in signals if s != 0]
    if len(votes) == 0:
        return 0.0, 0.0
    med = float(np.median(votes))
    final = float(np.sign(med)) if abs(med) > 1e-12 else 0.0
    return final, med


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_signals(
    data: Union[pd.DataFrame, Iterable[Any]],
    indicators: Sequence[IndicatorConfig],
    *,
    defaults_csv: Optional[str] = None,
    aggregation: str = "weighted",          # "weighted" or "majority"
    deadband: float = 0.10,                  # only used for "weighted" to avoid flip-flops
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute indicator signals and aggregate into a single composite per bar.

    Parameters
    ----------
    data : DataFrame or iterable of Candle-like
        Must contain open, high, low, close; volume optional.
    indicators : list[IndicatorConfig]
        Which indicator policies to compute, with parameters and weights.
    defaults_csv : str, optional
        If provided, loads default params per indicator and merges with overrides.
    aggregation : {"weighted", "majority"}
        How to fuse votes into one decision.
    deadband : float
        For "weighted" aggregation, treat small scores (|score| < deadband) as 0.

    Returns
    -------
    per_indicator : DataFrame
        Columns like: signal_rsi, rsi, signal_macd, macd_line, ...
    composite : DataFrame
        Columns: signal, score, reason (semicolon-joined voters that fired).
    """
    # Adapt input to DataFrame
    if isinstance(data, pd.DataFrame):
        df_in = data.copy()
    else:
        df_in = candles_to_dataframe(data)

    cols = infer_price_columns(df_in)
    required = ["open", "high", "low", "close"]
    for name in required:
        if cols[name] is None:
            raise ValueError(f"Missing required column for '{name}'. Available: {list(df_in.columns)}")

    open_prices = df_in[cols["open"]]
    high_prices = df_in[cols["high"]]
    low_prices = df_in[cols["low"]]
    close_prices = df_in[cols["close"]]
    volume_series = df_in[cols["volume"]] if cols["volume"] is not None else pd.Series(np.nan, index=df_in.index)

    # Merge CSV defaults (if any) with user-provided params
    default_params = load_default_params_from_csv(defaults_csv) if defaults_csv else {}

    # Compute policy outputs
    per_indicator_columns: Dict[str, pd.Series] = {}
    registry_names = set(POLICY_REGISTRY.keys())

    # Keep metadata for aggregation
    indicator_meta: List[Dict[str, Any]] = []

    for cfg in indicators:
        policy_name = cfg.name.lower()
        if policy_name not in registry_names:
            raise KeyError(f"Unknown indicator '{policy_name}'. Available: {sorted(registry_names)}")

        # Merge defaults with overrides
        params = dict(default_params.get(policy_name, {}))
        params.update(cfg.params or {})

        # Call with explicit named parameters for readability
        if policy_name == "rsi":
            out = rsi_policy(close_prices, **params)
        elif policy_name == "macd":
            out = macd_policy(close_prices, **params)
        elif policy_name == "bollinger":
            out = bollinger_policy(close_prices, **params)
        elif policy_name == "keltner":
            out = keltner_policy(high_prices, low_prices, close_prices, **params)
        elif policy_name == "dmi_adx":
            out = dmi_adx_policy(high_prices, low_prices, close_prices, **params)
        elif policy_name == "stochastic":
            out = stochastic_policy(high_prices, low_prices, close_prices, **params)
        elif policy_name == "supertrend":
            out = supertrend_policy(high_prices, low_prices, close_prices, **params)
        elif policy_name == "donchian":
            out = donchian_policy(high_prices, low_prices, close_prices, **params)
        elif policy_name == "ichimoku":
            out = ichimoku_policy(high_prices, low_prices, close_prices, **params)
        elif policy_name == "vwap":
            out = vwap_policy(high_prices, low_prices, close_prices, volume_series, **params)
        elif policy_name == "roc":
            out = roc_policy(close_prices, **params)
        elif policy_name == "obv":
            out = obv_policy(close_prices, volume_series, **params)
        elif policy_name == "mfi":
            out = mfi_policy(high_prices, low_prices, close_prices, volume_series, **params)
        else:
            raise RuntimeError(f"Policy dispatch missing for '{policy_name}'.")

        # Collect columns; ensure there's a signal column
        if "signal" not in out:
            raise RuntimeError(f"Policy '{policy_name}' did not return a 'signal' key.")
        per_indicator_columns[f"signal_{policy_name}"] = out["signal"].astype(float)
        for key, val in out.items():
            if key == "signal":
                continue
            per_indicator_columns[f"{policy_name}_{key}"] = val

        indicator_meta.append({"name": policy_name, "weight": float(cfg.weight)})

    per_indicator = pd.DataFrame(per_indicator_columns, index=df_in.index)

    # Aggregate votes per bar
    final_signals: List[float] = []
    final_scores: List[float] = []
    reasons: List[str] = []

    for row_idx in range(len(per_indicator)):
        votes_this_bar: List[float] = []
        weights_this_bar: List[float] = []
        reason_parts: List[str] = []

        for meta in indicator_meta:
            name = meta["name"]
            vote = float(per_indicator.loc[per_indicator.index[row_idx], f"signal_{name}"])
            if np.isnan(vote):
                continue
            votes_this_bar.append(vote)
            weights_this_bar.append(meta["weight"])
            if vote != 0.0:
                reason_parts.append(f"{name}={int(vote)}")

        if not votes_this_bar:
            final_signals.append(0.0)
            final_scores.append(0.0)
            reasons.append("no_votes")
            continue

        if aggregation == "majority":
            sig, score = majority_vote(votes_this_bar)
        else:
            sig, score = weighted_vote(votes_this_bar, weights_this_bar)
            # optional deadband for stability
            if abs(score) < deadband:
                sig = 0.0

        final_signals.append(sig)
        final_scores.append(score)
        reasons.append("; ".join(reason_parts) if reason_parts else "all_neutral")

    composite = pd.DataFrame(
        {"signal": final_signals, "score": final_scores, "reason": reasons},
        index=df_in.index,
    )
    if cols["datetime"] is not None:
        composite.insert(0, "datetime", df_in[cols["datetime"]])

    return per_indicator, composite
