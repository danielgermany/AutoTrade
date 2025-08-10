# candles_advanced.py
# -----------------------------------------------------------------------------
# A production‑grade Candle / CandleSeries implementation with:
#
#   • Configurable sessions (NY AM/PM, London, Asia, Off by default)
#   • Robust timestamp parsing with optional timezone normalization
#   • __slots__ on Candle to reduce memory and speed attribute access
#   • Fast DataFrame ingestion (vectorized NumPy arrays)
#   • Optional O(1) range min/max via Sparse Table (RMQ) after O(n log n) build
#   • Strict monotonic timestamp validation (configurable) to prevent silent bugs
#   • Ergonomic API: iteration, slicing, and DataFrame exporters for analysis
#   • Invariants (assert‑style validations) for OHLC correctness
#
# WHY THESE UPGRADES MATTER (compared to the simple version):
#   • Robust timestamps: you can pass strings, datetime, pandas.Timestamp, etc.,
#     and choose a timezone for normalization. This avoids brittle string splitting.
#   • Configurable sessions: no more hard‑coded hour checks; pass your own ranges.
#   • Speed: we hold OHLCV in NumPy arrays for vectorized highs/lows; optional
#     Sparse Tables enable O(1) query time after a one‑time O(n log n) build.
#   • Safety: strictly increasing timestamps (if enabled) catch out‑of‑order data
#     early; OHLC invariants catch malformed bars (e.g., low > open).
#   • API quality: works as a Pythonic container with __len__, __iter__, slicing,
#     and utilities to export levels/flags to DataFrames.
#
# NOTE: The upgrades are implemented without changing your domain semantics:
#       session labels, level‑tracking flow, and method names/returns remain
#       consistent with your earlier expectations — they’re just safer & faster.
# -----------------------------------------------------------------------------

from __future__ import annotations  # allows forward references in type hints

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union
from datetime import datetime, timezone

# zoneinfo (stdlib, py>=3.9) for timezone conversions; optional fallback used if missing
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover (only hit on very old Python versions)
    ZoneInfo = None  # type: ignore

import numpy as np  # NumPy for fast vectorized numeric ops

# pandas is optional for construction and exporters — we guard imports so this file can be imported without pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:  # pragma: no cover
    HAS_PANDAS = False


# ============================== Session configuration ==============================

@dataclass(frozen=True)
class SessionRange:
    """
    Inclusive minute‑of‑day range [start_min, end_min], where:
      - 0   == 00:00
      - 1439 == 23:59

    Using minutes since midnight yields simple, readable range checks and avoids
    fragile hour/minute branching. This also makes custom sessions trivial to define.
    """
    start_min: int
    end_min: int


@dataclass(frozen=True)
class SessionConfig:
    """
    Defines trading sessions by name and one or more minute‑of‑day ranges.

    'ranges' maps session name -> tuple of SessionRange blocks.
    'fallback' is used if a minute‑of‑day matches none of the named ranges.

    The default matches your previous logic:
      - Asia:   18:00–23:59 and 00:00–01:59
      - London: 02:00–06:59
      - NY AM:  07:00–10:59
      - NY PM:  11:00–15:59
      - Off:    16:00–17:59  (anything else falls back to Off)
    """
    ranges: Mapping[str, Tuple[SessionRange, ...]]
    fallback: str = "Off"

    @staticmethod
    def default() -> "SessionConfig":
        """
        Provide a ready‑to‑use session config that mirrors the original session rules.
        """
        return SessionConfig(
            ranges={
                "Asia":   (SessionRange(18*60, 23*60 + 59), SessionRange(0, 1*60 + 59)),
                "London": (SessionRange(2*60, 6*60 + 59),),
                "NY AM":  (SessionRange(7*60, 10*60 + 59),),
                "NY PM":  (SessionRange(11*60, 15*60 + 59),),
                "Off":    (SessionRange(16*60, 17*60 + 59),),
            },
            fallback="Off",
        )


# =================================== Utilities ====================================

def _ensure_datetime(ts: Any, tz: Optional[str] = None) -> datetime:
    """
    Convert a timestamp‑like value into a datetime. If tz is provided:
      • If ZoneInfo is available:
          - Naive datetimes are assumed UTC and converted to the given tz
          - Aware datetimes are converted to the given tz
      • If ZoneInfo is not available:
          - We leave tz handling as‑is (best‑effort).

    Accepted inputs: string, datetime, pandas.Timestamp, numpy.datetime64, etc.

    We accept many string shapes (ISO‑like forms) and normalize:
      - Replace 'T' with ' ' (ISO to space)
      - Drop a trailing 'Z' or '+HH:MM' for parsing
      - If time has no seconds, append ':00'
      - If no time given, add ' 00:00:00'

    WHY: This avoids brittle 'split' logic, prevents crashes on mixed types, and
    keeps session/day‑rollover logic consistent regardless of input source.
    """
    if isinstance(ts, datetime):
        dt = ts
    elif HAS_PANDAS and isinstance(ts, pd.Timestamp):  # type: ignore[attr-defined]
        dt = ts.to_pydatetime()
    else:
        s = str(ts).replace('T', ' ')
        # Strip common timezone suffix patterns (we’ll re‑localize using tz)
        if s.endswith('Z'):
            s = s[:-1]
        if '+' in s:
            s = s.split('+', 1)[0]
        # If only date present, add midnight
        if ' ' not in s:
            s = s + ' 00:00:00'
        # If missing seconds, add ':00'
        date_part, time_part = s.split(' ')
        if time_part.count(':') == 1:
            s = f"{date_part} {time_part}:00"
        # Parse to naive datetime (no tz)
        dt = datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

    # Normalize timezone if requested
    if tz:
        if ZoneInfo is not None:
            # If naive, interpret as UTC and convert to target tz for a consistent baseline
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz))
            else:
                dt = dt.astimezone(ZoneInfo(tz))
        else:
            # No ZoneInfo — keep as‑is (you could plug in pytz if desired)
            pass
    else:
        # If aware, convert to UTC and drop tz to keep everything naive (consistent comparisons)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

    return dt


def _minutes_since_midnight(dt: datetime) -> int:
    """
    Compute minutes since midnight for a datetime. Used for session lookup.
    """
    return dt.hour * 60 + dt.minute


def _classify_session(dt: datetime, config: SessionConfig) -> str:
    """
    Map a datetime to a session name by scanning configured ranges.
    We check named sessions in the order provided by 'config.ranges'.
    The first match wins. If nothing matches, we return 'config.fallback'.
    """
    m = _minutes_since_midnight(dt)
    for name, ranges in config.ranges.items():
        for r in ranges:
            if r.start_min <= m <= r.end_min:
                return name
    return config.fallback


# ================================== Core classes ==================================

class Candle:
    """
    A single OHLCV bar with a precomputed session label.

    UPGRADES:
      • __slots__ reduces per‑object memory overhead and speeds attribute access.
      • Fields normalized to float for numeric stability/consistency.
      • 'timestamp' is a datetime (naive or tz‑adjusted per series config).

    NOTE: Keep this object lightweight — heavy analytics happen in CandleSeries
    using NumPy arrays for speed.
    """
    __slots__ = ("timestamp", "open", "high", "low", "close", "volume", "session")

    def __init__(self, close: float, open_: float, high: float, low: float,
                 volume: float, timestamp: Any, session: str):
        # Store normalized types for consistency and speed in hot paths
        self.timestamp = timestamp  # datetime, already normalized by the series
        self.open = float(open_)
        self.high = float(high)
        self.low = float(low)
        self.close = float(close)
        self.volume = float(volume)
        self.session = session  # precomputed session name string

    # --- Convenience analytics on a single bar (unchanged semantics) ---

    def body(self) -> float:
        """Absolute body size: |close - open|."""
        return abs(self.close - self.open)

    def is_bullish(self) -> bool:
        """True iff close > open."""
        return self.close > self.open

    def is_bearish(self) -> bool:
        """True iff close < open."""
        return self.close < self.open

    def wick_top(self) -> float:
        """Upper wick length: high - max(open, close)."""
        return self.high - max(self.open, self.close)

    def wick_bottom(self) -> float:
        """Lower wick length: min(open, close) - low."""
        return min(self.open, self.close) - self.low

    def __repr__(self) -> str:
        """Readable string for debugging/logging."""
        ts = self.timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(self.timestamp, datetime) else str(self.timestamp)
        return (f"Candle({ts}, Open:{self.open}, Close:{self.close}, "
                f"High:{self.high}, Low:{self.low}, Volume:{self.volume})")


class RangeMinMaxSparseTable:
    """
    Static RMQ (Range Min/Max Query) structure with O(1) query time after O(n log n) build.

    WHY:
      • If you perform many high/low queries on fixed data windows, this provides
        constant‑time answers vs scanning slices (O(window)).
      • Build once, query many times — great for backtesting on large series.

    IMPLEMENTATION NOTES:
      • We hold two Sparse Tables: one for mins and one for maxes.
      • 'logs' array gives fast lookup of floor(log2(length)) for any range.
      • Queries are inclusive on both ends: [l, r].
    """
    def __init__(self, arr: np.ndarray):
        n = int(arr.size)
        self.n = n
        self.k = int(np.floor(np.log2(max(1, n))))  # max power needed

        # Precompute logs[i] = floor(log2(i)) for 1..n
        self.logs = np.zeros(n + 1, dtype=np.int32)
        for i in range(2, n + 1):
            self.logs[i] = self.logs[i // 2] + 1

        # Allocate tables: (k+1) x n; level 0 is the base array
        self.st_min = np.empty((self.k + 1, n), dtype=arr.dtype)
        self.st_max = np.empty((self.k + 1, n), dtype=arr.dtype)
        self.st_min[0] = arr
        self.st_max[0] = arr

        # Build up levels: at level j, each slot spans length 2^j
        j = 1
        while j <= self.k:
            span = 1 << (j - 1)  # half‑span from the previous level
            # Combine adjacent intervals from previous level to build j
            left_min = self.st_min[j - 1, :n - span]
            right_min = self.st_min[j - 1, span:n]
            left_max = self.st_max[j - 1, :n - span]
            right_max = self.st_max[j - 1, span:n]
            self.st_min[j, :n - span] = np.minimum(left_min, right_min)
            self.st_max[j, :n - span] = np.maximum(left_max, right_max)
            # For the tail where full span is not available, carry previous values
            self.st_min[j, n - span:n] = self.st_min[j - 1, n - span:n]
            self.st_max[j, n - span:n] = self.st_max[j - 1, n - span:n]
            j += 1

    def query_min(self, l: int, r: int) -> float:
        """
        Inclusive min on [l, r]. Complexity: O(1).
        """
        if l > r:  # normalize order
            l, r = r, l
        j = self.logs[r - l + 1]             # largest power of two fitting the range
        # Combine two intervals covering [l, r]: [l, l+2^j-1] and [r-2^j+1, r]
        return float(min(self.st_min[j, l], self.st_min[j, r - (1 << j) + 1]))

    def query_max(self, l: int, r: int) -> float:
        """
        Inclusive max on [l, r]. Complexity: O(1).
        """
        if l > r:
            l, r = r, l
        j = self.logs[r - l + 1]
        return float(max(self.st_max[j, l], self.st_max[j, r - (1 << j) + 1]))


class CandleSeries:
    """
    Container for Candle objects with high‑performance array backings and session logic.

    CONSTRUCTION PATH (fast & safe):
      1) Validate required columns and pick a timestamp column (or use index)
      2) Normalize timestamps to datetime (optionally to a target timezone)
      3) Copy OHLCV into NumPy arrays (float)
      4) Validate OHLC invariants (low ≤ min(open,close) ≤ max(open,close) ≤ high)
      5) Vectorize session classification using minute‑of‑day
      6) Build lightweight Candle objects for ergonomic iteration/debugging
      7) Optionally prebuild Sparse Tables for O(1) range min/max queries

    KEY PARAMETERS:
      • timestamp_col: column name with times (defaults to 'Datetime'; falls back to index if absent)
      • session_config: range map; defaults to the same sessions as before
      • tz: timezone string (e.g., 'America/New_York') or None to keep naive/UTC
      • enforce_strict_order: require strictly increasing timestamps (catches subtle bugs)
      • build_sparse_tables: build RMQ structures for O(1) min/max on fixed data
    """
    REQUIRED_COLS = ('Open', 'High', 'Low', 'Close', 'Volume')

    def __init__(
        self,
        df: "pd.DataFrame",
        timestamp_col: str = "Datetime",
        session_config: Optional[SessionConfig] = None,
        tz: Optional[str] = None,
        enforce_strict_order: bool = True,
        build_sparse_tables: bool = False
    ) -> None:
        if not HAS_PANDAS:
            raise ImportError("pandas is required to construct CandleSeries from a DataFrame.")

        # 1) Validate presence of required OHLCV columns
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        # Keep session rules configurable (upgrade)
        self.session_config = session_config or SessionConfig.default()
        self.tz = tz
        self.enforce_strict_order = enforce_strict_order

        # 2) Normalize frame to ensure a timestamp column exists
        #    If 'timestamp_col' exists: copy only needed columns (cheap + explicit).
        #    Else: try to materialize index as that column name.
        if timestamp_col in df.columns:
            work = df[[timestamp_col, 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        else:
            # Reset index and ensure the column exists (rename index to timestamp_col if necessary)
            tmp = df.reset_index()
            if timestamp_col not in tmp.columns:
                # If the original index name wasn’t the desired timestamp_col, rename it
                tmp = tmp.rename(columns={tmp.columns[0]: timestamp_col})
            work = tmp[[timestamp_col, 'Open', 'High', 'Low', 'Close', 'Volume']]

        # 3) Convert timestamps robustly to datetimes (optionally normalized to tz)
        timestamps = np.empty(len(work), dtype=object)
        for i, v in enumerate(work[timestamp_col].values.tolist()):
            timestamps[i] = _ensure_datetime(v, tz=self.tz)

        # 4) Enforce strictly increasing timestamps (optional but recommended)
        if self.enforce_strict_order and len(timestamps) > 1:
            for i in range(1, len(timestamps)):
                if timestamps[i] <= timestamps[i-1]:
                    # This catches both duplicates and out‑of‑order times — common backtest pitfalls
                    raise ValueError(
                        f"Timestamps must be strictly increasing. "
                        f"Offender at index {i}: {timestamps[i]} <= {timestamps[i-1]}"
                    )

        # Persist normalized arrays for fast vectorized analytics
        self.timestamps: np.ndarray = timestamps  # dtype=object datetime
        self.open = work['Open'].to_numpy(dtype=float, copy=True)
        self.high = work['High'].to_numpy(dtype=float, copy=True)
        self.low = work['Low'].to_numpy(dtype=float, copy=True)
        self.close = work['Close'].to_numpy(dtype=float, copy=True)
        self.volume = work['Volume'].to_numpy(dtype=float, copy=True)

        # 5) OHLC invariants (defensive programming): prevent silent bad data
        if not (self.low <= np.minimum(self.open, self.close)).all():
            raise ValueError("Invariant violated: low must be <= min(open, close).")
        if not (self.high >= np.maximum(self.open, self.close)).all():
            raise ValueError("Invariant violated: high must be >= max(open, close).")

        # 6) Vectorized session classification:
        #    We compute minutes since midnight for each bar and match ranges in order.
        minutes = np.array([_minutes_since_midnight(t) for t in self.timestamps], dtype=np.int32)
        session_names = list(self.session_config.ranges.keys())
        sess_id = np.full(len(minutes), -1, dtype=np.int16)
        # Assign by first‑match policy, preserving the order in config.ranges
        for sid, name in enumerate(session_names):
            for r in self.session_config.ranges[name]:
                mask = (minutes >= r.start_min) & (minutes <= r.end_min)
                sess_id[mask & (sess_id == -1)] = sid

        # Fallback assignment for any mins not matched
        fb_id = session_names.index(self.session_config.fallback) if self.session_config.fallback in session_names else -1
        if fb_id != -1:
            sess_id[sess_id == -1] = fb_id

        self.session_names: List[str] = session_names
        self.session_id: np.ndarray = sess_id
        self._session_of: List[str] = [session_names[i] if i >= 0 else self.session_config.fallback for i in sess_id]

        # 7) Build lightweight Candle objects for ergonomic iteration/debugging
        #    (All analytics should prefer the NumPy arrays above.)
        self.candles: List[Candle] = []
        # Python lists have no reserve(), conditional check stays no‑op but documents intent
        self.candles.reserve(len(work)) if hasattr(self.candles, "reserve") else None
        for i in range(len(work)):
            self.candles.append(Candle(
                close=self.close[i],
                open_=self.open[i],
                high=self.high[i],
                low=self.low[i],
                volume=self.volume[i],
                timestamp=self.timestamps[i],
                session=self._session_of[i],
            ))

        # --- Per‑day/session tracking state (key levels, flags) ---
        # Use the explicit order of non‑fallback sessions for daily rollups:
        self._sessions_order = tuple([name for name in session_names if name != self.session_config.fallback])
        self._rolling_levels: Dict[str, Dict[str, Optional[float]]] = {}
        self._final_levels: Dict[str, Dict[str, Optional[float]]] = {}
        self._active_session: Optional[str] = None
        self._current_day_anchor: Optional[str] = None
        self._level_hit_flags: List[Dict[str, Any]] = []
        self._daily_levels: Dict[str, Dict[str, Optional[float]]] = {}

        # Initialize all per‑day structures (bug‑resistant default state)
        self.reset_daily_levels()

        # 8) Optional Range Min/Max Sparse Tables for O(1) queries
        self._rmq_high: Optional[RangeMinMaxSparseTable] = None
        self._rmq_low: Optional[RangeMinMaxSparseTable] = None
        if build_sparse_tables and len(self.high) > 0:
            self._rmq_high = RangeMinMaxSparseTable(self.high)
            self._rmq_low = RangeMinMaxSparseTable(self.low)

    # ----------------------------- Pythonic container API -----------------------------

    def __len__(self) -> int:
        """Number of candles in the series (O(1))."""
        return len(self.candles)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Candle, List[Candle]]:
        """
        Indexing support:
          • int  -> single Candle
          • slice -> list of Candle (shallow view; arrays remain authoritative)
        """
        if isinstance(idx, slice):
            return self.candles[idx]
        return self.candles[idx]

    def __iter__(self):
        """Iterate over Candle objects (ergonomic; arrays are faster when batching)."""
        return iter(self.candles)

    def __repr__(self) -> str:
        """Human‑readable summary showing count and time span."""
        if len(self.candles) == 0:
            return "Candles(0 candles)"
        first_ts = self.candles[0].timestamp
        last_ts = self.candles[-1].timestamp
        f = first_ts.strftime('%Y-%m-%d %H:%M:%S') if isinstance(first_ts, datetime) else str(first_ts)
        l = last_ts.strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_ts, datetime) else str(last_ts)
        return f"Candles({len(self.candles)} candles from {f} to {l})"

    def get_candles(self) -> List[Candle]:
        """Return the underlying list of Candle objects."""
        return self.candles

    # --------------------------- Range queries (fast paths) ---------------------------

    def _validate_window(self, start_index: int, end_index: int) -> Tuple[int, int]:
        """
        Normalize and validate a range window [start_index, end_index], inclusive.
        Ensures non‑empty series, in‑bounds indices, and start ≤ end.
        """
        n = len(self.candles)
        if n == 0:
            raise ValueError("Empty series.")
        if start_index < 0 or end_index >= n:
            raise ValueError("Invalid start or end index")
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        return start_index, end_index

    def get_high_low(self, start_index: int, end_index: int) -> Tuple[float, float]:
        """
        Return (highest_high, lowest_low) over the inclusive window [start_index, end_index].

        PERFORMANCE:
          • If Sparse Tables were built, queries are O(1) time.
          • Otherwise, uses NumPy vectorized max/min over the window slice.
        """
        start_index, end_index = self._validate_window(start_index, end_index)
        if self._rmq_high is not None and self._rmq_low is not None:
            return self._rmq_high.query_max(start_index, end_index), self._rmq_low.query_min(start_index, end_index)
        window_high = self.high[start_index:end_index+1]
        window_low = self.low[start_index:end_index+1]
        return float(np.max(window_high)), float(np.min(window_low))

    def get_session_high_low(self, session: str, start_index: int = 0, end_index: Optional[int] = None) -> Tuple[float, float]:
        """
        Return (highest_high, lowest_low) for candles in a given 'session' within [start_index, end_index].

        IMPLEMENTATION:
          • Build a boolean mask for the specified window, filter by that session id,
            and use NumPy max/min on the filtered subarrays.
        """
        if end_index is None:
            end_index = len(self.candles) - 1
        start_index, end_index = self._validate_window(start_index, end_index)

        # Find numeric session id (raises a clear error if the name is unknown)
        try:
            sess_idx = self.session_names.index(session)
        except ValueError:
            raise ValueError(f"Unknown session '{session}'. Known: {self.session_names}")

        # Mask down to the window, then to the session
        mask = (self.session_id[start_index:end_index+1] == sess_idx)
        if not np.any(mask):
            raise ValueError("No matching candles found in given range.")

        sub_high = self.high[start_index:end_index+1][mask]
        sub_low = self.low[start_index:end_index+1][mask]
        return float(np.max(sub_high)), float(np.min(sub_low))

    # ---------------------- Daily/session levels & hit‑flag logic ---------------------

    def reset_daily_levels(self):
        """
        Reset all per‑day/session tracking structures.

        WHY: Called at construction and at each detected trading‑day rollover to:
          • reset rolling per‑session high/low
          • clear finalized levels from prior sessions
          • clear flags
          • reset active session and anchor
          • initialize _daily_levels (fixes a bug in some earlier versions)
        """
        self._rolling_levels = {s: {"high": None, "low": None} for s in self._sessions_order}
        self._final_levels = {}
        self._active_session = None
        self._current_day_anchor = None
        self._level_hit_flags = []
        self._daily_levels = {s: {"high": None, "low": None} for s in self._sessions_order}

    def _anchor_of(self, dt: datetime, rollover_hour: int) -> str:
        """
        Compute the trading‑day anchor label. We use the simple calendar date string.

        NOTE: If you want a session‑based/trading‑session rollover (e.g., 18:00),
        you can modify this to account for 'rollover_hour'; by default we keep the
        anchor to the calendar date for predictability and parity with prior behavior.
        """
        return dt.strftime('%Y-%m-%d')

    def is_new_trading_day(self, prev_anchor: Optional[str], curr_dt: datetime, rollover_hour: int = 18) -> bool:
        """
        Detect if a new trading day has started.

        CURRENT STRATEGY:
          • We anchor by calendar date. If date string changed, it's a new day.
          • If you prefer a clock‑based boundary (e.g., 18:00 NY), adapt _anchor_of().
        """
        if prev_anchor is None:
            return True
        return curr_dt.strftime('%Y-%m-%d') != prev_anchor

    def _finalize_session_if_needed(self, new_session: str):
        """
        If the session changes, freeze the previous session’s rolling high/low into _final_levels
        (only once) and switch the active session.

        WHY: Downstream logic needs finalized levels for prior sessions to detect “touches”.
        """
        prev = self._active_session
        if prev is not None and prev != new_session:
            rl = self._rolling_levels.get(prev)
            if rl and rl["high"] is not None and rl["low"] is not None and prev not in self._final_levels:
                self._final_levels[prev] = {"high": rl["high"], "low": rl["low"]}
        self._active_session = new_session

    def _update_rolling_levels(self, session: str, high: float, low: float):
        """
        Update per‑session rolling high/low with the current candle’s values.
        """
        cell = self._rolling_levels.get(session)
        if cell is None:
            return
        if cell["high"] is None or high > cell["high"]:
            cell["high"] = high
        if cell["low"] is None or low < cell["low"]:
            cell["low"] = low

    def _check_hits_vs_final_levels(self, day_label: Optional[str], curr_session: str,
                                    c_index: int, c_timestamp: Any, c_high: float, c_low: float):
        """
        If this candle’s price range [c_low, c_high] touches any prior‑session finalized level,
        append an event dict to _level_hit_flags.

        TOUCH DEFINITION:
          • A level L is “touched” if c_low ≤ L ≤ c_high (inclusive touches allowed).
          • We skip the current session — only prior sessions’ finalized levels produce flags.
        """
        for sess, lv in self._final_levels.items():
            if sess == curr_session:
                continue
            fh = lv.get("high")
            fl = lv.get("low")

            if fl is not None and c_low <= fl <= c_high:
                self._level_hit_flags.append({
                    "day": day_label,
                    "touched_session": sess,
                    "which": "low",
                    "level": fl,
                    "by_session": curr_session,
                    "at_index": c_index,
                    "timestamp": c_timestamp,
                })
            if fh is not None and c_low <= fh <= c_high:
                self._level_hit_flags.append({
                    "day": day_label,
                    "touched_session": sess,
                    "which": "high",
                    "level": fh,
                    "by_session": curr_session,
                    "at_index": c_index,
                    "timestamp": c_timestamp,
                })

    def update_key_levels_for_candle(self, c: Candle, c_index: int, rollover_hour: int = 18):
        """
        Stream‑update per‑session key levels and touch flags for a single new candle 'c'.

        ORDER OF OPERATIONS:
          1) New trading day? -> reset all daily structures and set the anchor date
          2) If session changed -> finalize previous session’s levels once
          3) Update rolling levels for the current session
          4) Detect touches against prior sessions’ finalized levels and record flags

        RETURNS a snapshot dict:
          {
            "rolling": {session: {"high": float|None, "low": float|None}, ...},
            "final":   {session: {"high": float, "low": float}, ...},  # prior finalized
            "flags_count": int
          }
        """
        # Ensure internal dicts are initialized (defensive)
        if self._rolling_levels is None or self._daily_levels is None:
            self.reset_daily_levels()

        # 1) New trading day based on calendar anchor?
        if self.is_new_trading_day(self._current_day_anchor, c.timestamp, rollover_hour):
            self.reset_daily_levels()
            self._current_day_anchor = self._anchor_of(c.timestamp, rollover_hour)

        # 2) Finalize prior session if we’ve transitioned
        self._finalize_session_if_needed(c.session)

        # 3) Update rolling highs/lows
        self._update_rolling_levels(c.session, c.high, c.low)

        # Maintain _daily_levels as well (day‑level aggregate by session)
        cell = self._daily_levels.get(c.session)
        if cell is not None:
            if cell["high"] is None or c.high > cell["high"]:
                cell["high"] = c.high
            if cell["low"] is None or c.low < cell["low"]:
                cell["low"] = c.low

        # 4) Touch checks vs prior finalized sessions
        self._check_hits_vs_final_levels(
            day_label=self._current_day_anchor,
            curr_session=c.session,
            c_index=c_index,
            c_timestamp=c.timestamp,
            c_high=c.high,
            c_low=c.low
        )

        return {
            "rolling": self._rolling_levels,
            "final": self._final_levels,
            "flags_count": len(self._level_hit_flags),
        }

    def consume_level_hit_flags(self) -> List[Dict[str, Any]]:
        """
        Return and CLEAR accumulated 'level hit' flags.

        Each flag has:
          {
            "day": "YYYY-MM-DD" | None,
            "touched_session": "Asia" | "London" | "NY AM" | "NY PM",
            "which": "high" | "low",
            "level": float,
            "by_session": str,        # session of the candle that made the touch
            "at_index": int,          # index of the candle in the series
            "timestamp": datetime,    # candle timestamp
          }
        """
        out = self._level_hit_flags
        self._level_hit_flags = []
        return out

    def get_today_final_levels(self) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Return a shallow copy of finalized levels for sessions already completed today.
        """
        return dict(self._final_levels)

    def compute_daily_session_levels(self, rollover_hour: int = 18) -> List[Dict[str, Any]]:
        """
        Batch‑compute per‑day per‑session highs/lows over the entire series.

        FLOW:
          • Resets daily state and walks the series linearly.
          • On day change, flush current day to 'daily_levels_history', then reset.
          • For each candle, update that day’s per‑session high/low.
          • After the loop, flush the last day.

        RETURNS:
          [
            { "day": "YYYY-MM-DD",
              "levels": {
                "Asia":  {"high": float|None, "low": float|None},
                "London":{"high": ...}, "NY AM": {...}, "NY PM": {...}
              }
            },
            ...
          ]
        """
        self.reset_daily_levels()
        self.daily_levels_history: List[Dict[str, Any]] = []
        self._current_day_anchor = None

        for i, c in enumerate(self.candles):
            if self.is_new_trading_day(self._current_day_anchor, c.timestamp, rollover_hour):
                # If we were already collecting a day, flush it before resetting
                if self._current_day_anchor is not None:
                    self.daily_levels_history.append({
                        "day": self._current_day_anchor,
                        "levels": {
                            s: {"high": v["high"], "low": v["low"]}
                            for s, v in self._daily_levels.items()
                        }
                    })
                self.reset_daily_levels()
                self._current_day_anchor = self._anchor_of(c.timestamp, rollover_hour)

            # Update daily per‑session levels
            s = c.session
            cell = self._daily_levels.get(s)
            if cell is not None:
                if cell["high"] is None or c.high > cell["high"]:
                    cell["high"] = c.high
                if cell["low"] is None or c.low < cell["low"]:
                    cell["low"] = c.low

        # Flush the final day
        if self._current_day_anchor is not None:
            self.daily_levels_history.append({
                "day": self._current_day_anchor,
                "levels": {
                    s: {"high": v["high"], "low": v["low"]}
                    for s, v in self._daily_levels.items()
                }
            })

        return self.daily_levels_history

    # --------------------------------- Exporters -------------------------------------

    def flags_to_dataframe(self) -> "pd.DataFrame":
        """
        Export accumulated hit flags to a pandas DataFrame:
        columns: ["day","touched_session","which","level","by_session","at_index","timestamp"].

        NOTE: Requires pandas. Returns an empty frame with columns if no flags.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for DataFrame export.")
        if not self._level_hit_flags:
            return pd.DataFrame(columns=["day","touched_session","which","level","by_session","at_index","timestamp"])
        return pd.DataFrame(self._level_hit_flags)

    def daily_levels_to_dataframe(self) -> "pd.DataFrame":
        """
        Export self.daily_levels_history (from compute_daily_session_levels) to a DataFrame:
        columns: ["day","session","high","low"].
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for DataFrame export.")
        rows = []
        for rec in getattr(self, "daily_levels_history", []):
            day = rec["day"]
            for sess, lv in rec["levels"].items():
                rows.append({
                    "day": day,
                    "session": sess,
                    "high": lv["high"],
                    "low": lv["low"],
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["day","session","high","low"])
