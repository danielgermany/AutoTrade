
# ohlcv_cleaner.py
# -----------------------------------------------------------------------------
# Purpose
#   Centralize all data cleaning & validation for OHLCV DataFrames so that
#   downstream consumers (e.g., CandleSeries) can skip redundant checks in
#   performance-critical code paths.
#
# Contract (post-clean invariants)
#   1) Columns present: ["Datetime","Open","High","Low","Close","Volume"]
#   2) Dtypes:
#        Datetime -> pandas datetime64[ns] (tz-naive, already in desired tz)
#        Open/High/Low/Close/Volume -> float64
#   3) No NaNs
#   4) Strictly increasing Datetime (sorted, unique)
#   5) OHLC invariants: Low <= min(Open, Close) and High >= max(Open, Close)
#   6) Volume >= 0
#
# Usage
#   from ohlcv_cleaner import clean_ohlcv, validate_ohlcv, REQUIRED_COLS
#   clean_df, report = clean_ohlcv(raw_df, tz="America/New_York")
#   validate_ohlcv(clean_df)  # raises if any invariant is violated
#
#   # Optional: persist a "clean stamp" (hash-based) for fast trust checks
#   #   (This avoids re-running validations in hot paths.)
#   from ohlcv_cleaner import stamp_clean, is_stamped_clean
#   stamp_clean(clean_df)
#   assert is_stamped_clean(clean_df)
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from zoneinfo import ZoneInfo

REQUIRED_COLS: Tuple[str, ...] = ("Datetime","Open","High","Low","Close","Volume")
META_FLAG = "ohlcv_clean"
META_SIG  = "ohlcv_signature"
META_TZ   = "ohlcv_tz"

@dataclass
class CleanReport:
    rows_in: int
    rows_out: int
    dropped_na: int
    dropped_dupes: int
    dropped_ohlc_invariant: int
    clipped_volume: int
    sorted: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "dropped_na": self.dropped_na,
            "dropped_dupes": self.dropped_dupes,
            "dropped_ohlc_invariant": self.dropped_ohlc_invariant,
            "clipped_volume": self.clipped_volume,
            "sorted": self.sorted,
        }

def _ensure_required_cols(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def _coerce_dtypes(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    # Datetime to datetime64[ns], localized to tz then made tz-naive in that tz
    s = pd.to_datetime(df["Datetime"], errors="coerce", utc=False)
    if tz is not None:
        # If series is tz-aware, convert then drop tz; else localize then drop
        if s.dt.tz is not None:
            s = s.dt.tz_convert(ZoneInfo(tz)).dt.tz_localize(None)
        else:
            # Interpret naive timestamps as already in tz, just ensure dtype
            # (no actual clock-time conversion here)
            s = s.astype("datetime64[ns]")
    else:
        # Ensure naive dtype
        if s.dt.tz is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            s = s.astype("datetime64[ns]")
    out = df.copy()
    out["Datetime"] = s

    # Numeric columns to float64
    for c in ("Open","High","Low","Close","Volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out

def _compute_signature(df: pd.DataFrame) -> str:
    # Stable hash over required columns (ignores index)
    h = hash_pandas_object(df[list(REQUIRED_COLS)], index=False)
    return str(hash(h.values.tobytes()))

def stamp_clean(df: pd.DataFrame, tz: Optional[str] = None) -> None:
    df.attrs[META_FLAG] = True
    df.attrs[META_SIG]  = _compute_signature(df)
    df.attrs[META_TZ]   = tz or ""

def is_stamped_clean(df: pd.DataFrame) -> bool:
    try:
        if not df.attrs.get(META_FLAG, False):
            return False
        sig = df.attrs.get(META_SIG, "")
        tz  = df.attrs.get(META_TZ, "")
        # Recompute and compare
        return sig == _compute_signature(df) and isinstance(tz, str)
    except Exception:
        return False

def clean_ohlcv(df: pd.DataFrame, tz: Optional[str] = None, clip_negative_volume: bool = True) -> tuple[pd.DataFrame, CleanReport]:
    """
    Clean and validate an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame containing at least REQUIRED_COLS.
    tz : Optional[str]
        Desired timezone context for Datetime. Output will be tz-naive but interpreted in this tz.
    clip_negative_volume : bool
        If True, negative volumes are clipped to 0 (and counted). If False, rows with negative
        volume are dropped.

    Returns
    -------
    (clean_df, report) : Tuple[pd.DataFrame, CleanReport]
        Cleaned DataFrame satisfying the contract above and a report with counts.
    """
    rows_in = len(df)
    _ensure_required_cols(df)

    # 1) Coerce dtypes
    x = _coerce_dtypes(df, tz=tz)

    # 2) Drop rows with NA in required columns
    before = len(x)
    x = x.dropna(subset=list(REQUIRED_COLS))
    dropped_na = before - len(x)

    # 3) Sort by Datetime and drop duplicates (keep first)
    #    Note: duplicates can arise from provider merges or re-downloads.
    x = x.sort_values("Datetime", kind="mergesort")  # stable sort
    sorted_flag = True
    before = len(x)
    x = x.drop_duplicates(subset=["Datetime"], keep="first")
    dropped_dupes = before - len(x)

    # 4) Enforce OHLC invariants; drop violating rows
    o = x["Open"]; h = x["High"]; l = x["Low"]; c = x["Close"]
    ok = (l <= np.minimum(o, c)) & (h >= np.maximum(o, c))
    dropped_ohlc_invariant = int((~ok).sum())
    if dropped_ohlc_invariant:
        x = x[ok]

    # 5) Non-negative volume
    if clip_negative_volume:
        neg_mask = x["Volume"] < 0
        clipped_volume = int(neg_mask.sum())
        if clipped_volume:
            x.loc[neg_mask, "Volume"] = 0.0
    else:
        before = len(x)
        x = x[x["Volume"] >= 0]
        clipped_volume = before - len(x)

    # 6) Final assert: strictly increasing Datetime (already sorted & deduped)
    #    We don't re-raise, but ensure property.
    #    If necessary, re-sort; dedupe already done.
    if not x["Datetime"].is_monotonic_increasing:
        x = x.sort_values("Datetime", kind="mergesort")

    # 7) Reset index for tidy downstream consumption
    x = x.reset_index(drop=True)

    report = CleanReport(
        rows_in=rows_in,
        rows_out=len(x),
        dropped_na=dropped_na,
        dropped_dupes=dropped_dupes,
        dropped_ohlc_invariant=dropped_ohlc_invariant,
        clipped_volume=clipped_volume,
        sorted=sorted_flag,
    )

    # 8) Stamp with a "clean" marker and signature for fast trust checks
    stamp_clean(x, tz=tz)

    return x, report

def validate_ohlcv(df: pd.DataFrame) -> None:
    """
    Raise AssertionError if df is not "clean by contract".
    This is O(n) but vectorized and fast; meant to be used in CI or offline validation,
    not in hot runtime paths.
    """
    _ensure_required_cols(df)

    assert pd.api.types.is_datetime64_ns_dtype(df["Datetime"]), "Datetime must be datetime64[ns] (tz-naive)."
    for c in ("Open","High","Low","Close","Volume"):
        assert pd.api.types.is_float_dtype(df[c]), f"{c} must be float dtype."
        assert not df[c].isna().any(), f"{c} contains NaNs."

    assert not df["Datetime"].isna().any(), "Datetime contains NaNs."
    assert df["Datetime"].is_monotonic_increasing, "Datetime must be strictly increasing."
    assert df["Datetime"].is_unique, "Datetime contains duplicates."

    o = df["Open"]; h = df["High"]; l = df["Low"]; c = df["Close"]
    ok = (l <= np.minimum(o, c)) & (h >= np.maximum(o, c))
    assert ok.all(), "OHLC invariant violated in one or more rows."
    assert (df["Volume"] >= 0).all(), "Volume must be non-negative."
