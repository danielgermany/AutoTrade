"""
Matplotlib candlestick plotter for AutoTrade (non‑recursive, robust).

- Accepts: pandas.DataFrame, list/iterable of Candle objects, or CandleSeries.
- Normalizes to columns: ['Datetime','Open','High','Low','Close','Volume'].
- Provides select_candles() and plot_candles().

Why this version?
-----------------
The previous helper `_to_dataframe()` called itself recursively after trying
to "re‑wrap" the input. If the adapter didn’t change the type (e.g., handed
back the same object), it caused infinite recursion. This version uses a
**single, iterative normalization pass** with explicit type checks and
NEVER recurses.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple
import itertools
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------
# Type helpers / duck-typing
# ------------------------------
def _is_dataframe(x: Any) -> bool:
    return isinstance(x, pd.DataFrame)


def _is_candleseries(x: Any) -> bool:
    # Duck-type against the upgraded CandleSeries in candles_advanced.py
    return all(hasattr(x, a) for a in ("timestamps", "open", "high", "low", "close", "volume"))


def _is_candle_like(x: Any) -> bool:
    return all(hasattr(x, a) for a in ("timestamp", "open", "high", "low", "close", "volume"))


# ------------------------------
# Normalization: ANY -> DataFrame
# ------------------------------
_REQUIRED = ("Datetime", "Open", "High", "Low", "Close", "Volume")


def _normalize_df_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common variations to the required schema and coerce dtypes."""
    out = df.copy()
    # Try common synonyms, case-sensitive first; then case-insensitive
    rename_map: dict = {}
    synonyms = {
        "Datetime": ("Datetime", "datetime", "DateTime", "date", "Date", "timestamp", "Timestamp", "time", "Time"),
        "Open":     ("Open", "open", "o"),
        "High":     ("High", "high", "h"),
        "Low":      ("Low", "low", "l"),
        "Close":    ("Close", "close", "c"),
        "Volume":   ("Volume", "volume", "vol", "Vol", "Qty", "quantity"),
    }
    for tgt, cands in synonyms.items():
        if tgt in out.columns:
            continue
        for c in cands:
            if c in out.columns:
                rename_map[c] = tgt
                break
        # second pass CI
        if tgt not in out.columns and tgt not in rename_map.values():
            lower_to_orig = {c.lower(): c for c in out.columns}
            for c in cands:
                if c.lower() in lower_to_orig:
                    rename_map[lower_to_orig[c.lower()]] = tgt
                    break
    if rename_map:
        out = out.rename(columns=rename_map)

    # Keep only required if present, in order
    keep = [c for c in _REQUIRED if c in out.columns]
    out = out[keep]

    # Dtypes
    if "Datetime" in out.columns and not pd.api.types.is_datetime64_any_dtype(out["Datetime"]):
        out["Datetime"] = pd.to_datetime(out["Datetime"], errors="coerce", utc=False)

    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _from_candleseries(cs: Any) -> pd.DataFrame:
    # Expect arrays of equal length
    n = len(cs.open)
    data = {
        "Datetime": pd.to_datetime(list(cs.timestamps), errors="coerce"),
        "Open":     np.asarray(cs.open, dtype=float),
        "High":     np.asarray(cs.high, dtype=float),
        "Low":      np.asarray(cs.low, dtype=float),
        "Close":    np.asarray(cs.close, dtype=float),
        "Volume":   np.asarray(cs.volume, dtype=float),
    }
    df = pd.DataFrame(data)
    return _normalize_df_cols(df)


def _from_candle_iter(candles: Iterable[Any]) -> pd.DataFrame:
    # Peek without consuming using itertools.tee
    it1, it2 = itertools.tee(iter(candles), 2)
    try:
        first = next(it1)
    except StopIteration:
        # Empty: return empty canonical frame
        return pd.DataFrame(columns=_REQUIRED)

    # If not candle-like, try to infer mapping/tuple forms
    if not _is_candle_like(first):
        # Mapping with keys?
        if isinstance(first, Mapping):
            # Build from list of mappings and then normalize keys
            df = pd.DataFrame(list(itertools.chain([first], it1)))
            return _normalize_df_cols(df)

        # Tuple/list zip?
        if isinstance(first, (tuple, list)) and len(first) >= 6:
            rows = list(itertools.chain([first], it1))
            df = pd.DataFrame(rows, columns=list(_REQUIRED)[:len(first)])
            return _normalize_df_cols(df)

        raise TypeError("Iterable provided but elements are not Candle-like or mappable to OHLCV.")

    # Candle-like path
    rows = [first] + list(it1)
    df = pd.DataFrame({
        "Datetime": [getattr(c, "timestamp") for c in rows],
        "Open":     [float(getattr(c, "open")) for c in rows],
        "High":     [float(getattr(c, "high")) for c in rows],
        "Low":      [float(getattr(c, "low")) for c in rows],
        "Close":    [float(getattr(c, "close")) for c in rows],
        "Volume":   [float(getattr(c, "volume")) for c in rows],
    })
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    return df


def normalize_to_dataframe(source: Any) -> pd.DataFrame:
    """One-shot adapter: NEVER recurses. Raises TypeError on unsupported input."""
    if _is_dataframe(source):
        return _normalize_df_cols(source)

    if _is_candleseries(source):
        return _from_candleseries(source)

    # If it's an iterable (list of Candle or list of dict/tuple)
    if isinstance(source, Iterable) and not isinstance(source, (str, bytes, dict)):
        return _from_candle_iter(source)

    # Dict of arrays?
    if isinstance(source, Mapping):
        return _normalize_df_cols(pd.DataFrame(source))

    raise TypeError(
        "Unsupported source type for candlestick plotting. "
        "Pass a pandas.DataFrame, CandleSeries, or iterable of Candle-like objects."
    )


# ------------------------------
# Selection helpers
# ------------------------------
def select_candles(source: Any,
                   *,
                   start: Optional[pd.Timestamp] = None,
                   end: Optional[pd.Timestamp] = None,
                   last_n: Optional[int] = None) -> pd.DataFrame:
    """
    Convert `source` to a DataFrame and slice by [start, end] or tail last_n.
    """
    df = normalize_to_dataframe(source)

    if "Datetime" not in df.columns:
        raise ValueError("Normalized data lacks 'Datetime' column.")

    # Drop rows with NaT time or NaNs in price
    df = df.dropna(subset=["Datetime"]).copy()
    if start is not None:
        start = pd.to_datetime(start)
        df = df.loc[df["Datetime"] >= start]
    if end is not None:
        end = pd.to_datetime(end)
        df = df.loc[df["Datetime"] <= end]

    df = df.sort_values("Datetime", kind="stable").reset_index(drop=True)

    if last_n is not None:
        df = df.tail(int(last_n)).reset_index(drop=True)

    return df


# ------------------------------
# Plotting
# ------------------------------
def plot_candles(source: Any,
                 *,
                 start: Optional[pd.Timestamp] = None,
                 end: Optional[pd.Timestamp] = None,
                 last_n: Optional[int] = None,
                 title: Optional[str] = None,
                 width: float = 0.6) -> pd.DataFrame:
    """
    Render a simple candlestick chart via matplotlib and return the sliced df.
    """
    df = select_candles(source, start=start, end=end, last_n=last_n)
    if df.empty:
        raise ValueError("No candles to plot after selection.")

    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    t = df["Datetime"]

    n = len(df)

    fig, ax = plt.subplots(figsize=(12, 6))

    xs = np.arange(n, dtype=float)
    half_w = width / 2.0

    # Wick lines
    for i in range(n):
        ax.vlines(xs[i], l[i], h[i])

    # Bodies
    for i in range(n):
        open_i, close_i = o[i], c[i]
        y = min(open_i, close_i)
        height = abs(close_i - open_i)
        if height == 0:  # doji -> draw a very thin body
            height = (h[i] - l[i]) * 0.001
        rect = plt.Rectangle((xs[i] - half_w, y), width, height, fill=True, alpha=0.6)
        ax.add_patch(rect)

    # X ticks: show a handful of timestamps
    max_ticks = 12
    if n <= max_ticks:
        ax.set_xticks(xs)
        ax.set_xticklabels([ts.strftime("%Y-%m-%d %H:%M") for ts in t])
    else:
        step = math.ceil(n / max_ticks)
        idxs = list(range(0, n, step))
        ax.set_xticks([xs[i] for i in idxs])
        ax.set_xticklabels([t.iloc[i].strftime("%Y-%m-%d %H:%M") for i in idxs], rotation=30, ha="right")

    ax.set_xlim(-0.5, n - 0.5)
    ymin = float(np.nanmin(l)); ymax = float(np.nanmax(h))
    ypad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
    ax.set_ylim(ymin - ypad, ymax + ypad)

    if title:
        ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    return df
