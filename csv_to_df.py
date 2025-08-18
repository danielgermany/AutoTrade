# csv_to_df.py
# ------------------------------------------------------------
# Robust CSV -> OHLCV DataFrame loader
# - No fragile parse_dates in read_csv (prevents "Missing column" error)
# - Auto-detects separate Date/Time columns and combines into "Datetime"
# - Case-insensitive column matching & renaming to
#   ["Datetime","Open","High","Low","Close","Volume"]
# - Optional cleaning step using ohlcv_cleaner.clean_ohlcv (lazy import)
# - Safe for downstream CandleSeries(..., assume_clean=True)

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple
import pandas as pd

# Expected output schema (case and order matter)
TARGET = ["Datetime", "Open", "High", "Low", "Close", "Volume"]

# Common synonyms (case-insensitive) for auto-mapping
DEFAULT_SYNONYMS: Dict[str, Iterable[str]] = {
    "Datetime": ("datetime", "timestamp", "ts", "time", "date"),
    "Open":     ("open", "o"),
    "High":     ("high", "h"),
    "Low":      ("low", "l"),
    "Close":    ("close", "c", "adj close", "adj_close"),
    "Volume":   ("volume", "vol", "qty", "quantity"),
}


def _build_name_maps(cols):
    lower_to_orig = {c.lower(): c for c in cols}
    orig_to_lower = {v: k for k, v in lower_to_orig.items()}
    return lower_to_orig, orig_to_lower


def _resolve_price_columns(df: pd.DataFrame,
                           synonyms: Dict[str, Iterable[str]]) -> Dict[str, str]:
    lower_to_orig, _ = _build_name_maps(df.columns)
    ren: Dict[str, str] = {}
    for target in ("Open","High","Low","Close","Volume"):
        found_src = None
        for cand in synonyms[target]:
            src = lower_to_orig.get(cand)
            if src is not None:
                found_src = src
                break
        if not found_src:
            raise ValueError(f"Missing required column for '{target}'. "
                             f"Looked for any of: {list(synonyms[target])}. "
                             f"Found: {list(df.columns)}")
        ren[found_src] = target
    return ren


def _make_datetime_series(df: pd.DataFrame,
                          synonyms: Dict[str, Iterable[str]],
                          date_format: Optional[str]) -> pd.Series:
    lower_to_orig, _ = _build_name_maps(df.columns)

    # Prefer a single datetime/timestamp column if present (avoid 'date'/'time' split parts)
    for cand in synonyms["Datetime"]:
        if cand in ("date", "time"):
            continue
        src = lower_to_orig.get(cand)
        if src is not None:
            s = pd.to_datetime(df[src], format=date_format, errors="coerce", utc=False)
            if s.notna().any():
                return s

    # Combine Date + Time if both exist
    date_col = lower_to_orig.get("date")
    time_col = lower_to_orig.get("time")
    if date_col is not None and time_col is not None:
        combined = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        s = pd.to_datetime(combined, format=date_format, errors="coerce", utc=False)
        return s

    # Only Date present
    if date_col is not None:
        s = pd.to_datetime(df[date_col], format=date_format, errors="coerce", utc=False)
        return s

    # Last resort: try a 'timestamp' or 'time' alone
    time_like = lower_to_orig.get("timestamp") or lower_to_orig.get("time")
    if time_like is not None:
        s = pd.to_datetime(df[time_like], format=date_format, errors="coerce", utc=False)
        return s

    raise ValueError("Could not resolve a Datetime column. Provide a single 'Datetime/Timestamp' "
                     "column or both 'Date' and 'Time'.")


def load_ohlcv_csv(
    path: str,
    *,
    tz: str = "America/New_York",
    delimiter: Optional[str] = None,      # None => pandas default/comma
    date_format: Optional[str] = None,    # e.g. "%Y-%m-%d %H:%M:%S"
    synonyms: Optional[Dict[str, Iterable[str]]] = None,
    run_cleaner: bool = True,
    memory_efficient: bool = False,
) -> Tuple[pd.DataFrame, Optional[object]]:
    """Load CSV -> standardized OHLCV DataFrame (and optional CleanReport).

    Returns:
        (df_or_clean_df, report_or_None)
    """
    syn = {k: tuple(v) for k, v in (synonyms or DEFAULT_SYNONYMS).items()}

    # Read without parse_dates to avoid pandas 'Missing column provided to parse_dates' errors
    engine = "pyarrow"
    try:
        import pyarrow  # noqa: F401
    except Exception:
        engine = "c"

    read_csv_kwargs = dict(
        sep=delimiter if delimiter else ",",
        engine=engine,
        low_memory=not memory_efficient,
    )

    if memory_efficient:
        chunks = pd.read_csv(path, chunksize=250_000, **read_csv_kwargs)
        df = pd.concat(chunks, axis=0, ignore_index=True)
    else:
        df = pd.read_csv(path, **read_csv_kwargs)

    # Rename price columns first (original -> target)
    price_ren = _resolve_price_columns(df, syn)
    df = df.rename(columns=price_ren)

    # Build Datetime series from best available source(s)
    dt_series = _make_datetime_series(df, syn, date_format)

    # Assign if exists; otherwise insert as first column
    if "Datetime" in df.columns:
        df["Datetime"] = dt_series
        # Ensure Datetime is first column
        cols = ["Datetime"] + [c for c in df.columns if c != "Datetime"]
        df = df[cols]
    else:
        df.insert(0, "Datetime", dt_series)

    # Keep only TARGET columns in canonical order
    df = df[[c for c in TARGET if c in df.columns]]

    # Coerce dtypes
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional cleaning (lazy import to avoid circular imports)
    report = None
    if run_cleaner:
        try:
            from ohlcv_cleaner import clean_ohlcv  # lazy import
        except Exception:
            clean_ohlcv = None
        if callable(clean_ohlcv):
            df, report = clean_ohlcv(df, tz=tz)

    # Minimal normalization if not cleaned
    if report is None:
        df = df.sort_values("Datetime", kind="stable").reset_index(drop=True)

    return df, report


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    path = "nq_data_1m.csv"
    df, report = load_ohlcv_csv(path, tz="America/New_York", run_cleaner=True)
    print(df.head(3))
    print(df.dtypes)
    if report is not None:
        try:
            print(report.as_dict())
        except Exception:
            print(report)