# candles_advanced.py
# -----------------------------------------------------------------------------
# Production-grade Candle / CandleSeries with:
#   • Configurable sessions (Asia, London, NY AM, NY PM, Off)
#   • Robust timestamp parsing + optional timezone normalization
#   • __slots__ on Candle to reduce memory and speed attribute access
#   • Fast NumPy array backings for OHLCV
#   • Optional RangeMin/Max Sparse Tables (RMQ) for O(1) high/low queries
#   • Strict monotonic timestamp validation (optional)
#   • Per-session daily high/low tracking + “prior session level touch” flags
#
# PERFORMANCE NOTE
#   Session classification is a single-pass state machine enforcing:
#   Asia → London → NY AM → NY PM → (Off) → Asia.
#   For each bar we check at most “same or next” session; we fall back to Off
#   (or a one-time full scan) only when necessary. Fewer comparisons; faster.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from datetime import datetime, timezone
from zoneinfo import ZoneInfo  # Python 3.13 standard library

import numpy as np
import pandas as pd


# ============================== Session configuration ==============================

@dataclass(frozen=True)
class SessionRange:
    """
    Inclusive minute-of-day interval [start_min, end_min].

    Convention:
      • 0   == 00:00
      • 1439 == 23:59

    Using minutes since midnight makes membership checks trivial, handles ranges
    that cross midnight by splitting into two blocks, and avoids brittle hour logic.
    """
    start_min: int
    end_min: int


@dataclass(frozen=True)
class SessionConfig:
    """
    Defines a mapping: session name -> tuple of SessionRange blocks.

    Attributes:
        ranges   : Mapping[str, Tuple[SessionRange, ...]]
        fallback : Name of the fallback session if no ranges match (default: "Off")

    Defaults match common intraday blocks used earlier in this project:
        Asia   : 18:00–23:59 and 00:00–01:59
        London : 02:00–06:59
        NY AM  : 07:00–10:59
        NY PM  : 11:00–15:59
        Off    : 16:00–17:59
    """
    ranges: Mapping[str, Tuple[SessionRange, ...]]
    fallback: str = "Off"

    @staticmethod
    def default() -> "SessionConfig":
        """Provide the default session mapping described above."""
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
    Convert various timestamp-like inputs to a Python datetime.

    Accepted inputs:
      • datetime
      • pandas.Timestamp
      • strings like "YYYY-MM-DD HH:MM[:SS][Z|+HH:MM]"

    Timezone behavior:
      • If tz is provided:
          - Naive input is assumed UTC then converted to tz
          - Aware input is converted to tz
      • If tz is None:
          - Aware input is converted to UTC, then tzinfo is dropped (naive UTC)
          - Naive input is left as-is
    """
    if isinstance(ts, datetime):
        dt = ts
    elif isinstance(ts, pd.Timestamp):
        dt = ts.to_pydatetime()
    else:
        s = str(ts).replace('T', ' ')
        if s.endswith('Z'):
            s = s[:-1]
        if '+' in s:
            s = s.split('+', 1)[0]
        if ' ' not in s:
            s += ' 00:00:00'
        date_part, time_part = s.split(' ')
        if time_part.count(':') == 1:
            s = f"{date_part} {time_part}:00"
        dt = datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

    if tz:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz))
        else:
            dt = dt.astimezone(ZoneInfo(tz))
    else:
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

    return dt


def _minutes_since_midnight(dt: datetime) -> int:
    """Return minutes since midnight for session range checks."""
    return dt.hour * 60 + dt.minute


def _classify_session(dt: datetime, config: SessionConfig) -> str:
    """
    Legacy: scan all configured ranges, return the first match; else fallback.

    Kept for compatibility / direct calls. CandleSeries uses a faster state machine.
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
    A lightweight OHLCV bar with precomputed session label.

    Implementation notes:
      • __slots__ minimizes per-instance memory and speeds attribute access
      • Values are normalized to float for consistent math and vectorized ops
      • Heavy analytics belong in CandleSeries (array-based), not here
    """
    __slots__ = ("timestamp", "open", "high", "low", "close", "volume", "session")

    def __init__(self, close: float, open_: float, high: float, low: float,
                 volume: float, timestamp: Any, session: str):
        self.timestamp = timestamp  # already normalized to datetime upstream
        self.open = float(open_)
        self.high = float(high)
        self.low = float(low)
        self.close = float(close)
        self.volume = float(volume)
        self.session = session

    def body(self) -> float: return abs(self.close - self.open)
    def is_bullish(self) -> bool: return self.close > self.open
    def is_bearish(self) -> bool: return self.close < self.open
    def wick_top(self) -> float: return self.high - max(self.open, self.close)
    def wick_bottom(self) -> float: return min(self.open, self.close) - self.low

    def __repr__(self) -> str:
        ts = self.timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(self.timestamp, datetime) else str(self.timestamp)
        return (f"Candle({ts}, Open:{self.open}, Close:{self.close}, "
                f"High:{self.high}, Low:{self.low}, Volume:{self.volume})")


class RangeMinMaxSparseTable:
    """
    Static RMQ (Range Min/Max Query) with O(1) queries after O(n log n) build.
    """
    def __init__(self, arr: np.ndarray):
        n = int(arr.size)
        self.n = n
        self.k = int(np.floor(np.log2(max(1, n))))

        self.logs = np.zeros(n + 1, dtype=np.int32)
        for i in range(2, n + 1):
            self.logs[i] = self.logs[i // 2] + 1

        self.st_min = np.empty((self.k + 1, n), dtype=arr.dtype)
        self.st_max = np.empty((self.k + 1, n), dtype=arr.dtype)
        self.st_min[0] = arr
        self.st_max[0] = arr

        j = 1
        while j <= self.k:
            span = 1 << (j - 1)
            left_min = self.st_min[j - 1, :n - span]
            right_min = self.st_min[j - 1, span:n]
            left_max = self.st_max[j - 1, :n - span]
            right_max = self.st_max[j - 1, span:n]
            self.st_min[j, :n - span] = np.minimum(left_min, right_min)
            self.st_max[j, :n - span] = np.maximum(left_max, right_max)
            self.st_min[j, n - span:n] = self.st_min[j - 1, n - span:n]
            self.st_max[j, n - span:n] = self.st_max[j - 1, n - span:n]
            j += 1

    def query_min(self, l: int, r: int) -> float:
        if l > r: l, r = r, l
        j = self.logs[r - l + 1]
        return float(min(self.st_min[j, l], self.st_min[j, r - (1 << j) + 1]))

    def query_max(self, l: int, r: int) -> float:
        if l > r: l, r = r, l
        j = self.logs[r - l + 1]
        return float(max(self.st_max[j, l], self.st_max[j, r - (1 << j) + 1]))


class CandleSeries:
    """
    Candle container with:
      • Fast array backings (OHLCV, timestamps, session ids)
      • State-machine session classification (same→next only)
      • Per-session daily high/low tracking and “prior session touch” flags
      • Optional RMQ structures for constant-time high/low queries

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV with columns ["Datetime","Open","High","Low","Close","Volume"].
    timestamp_col : str
        Name of the time column (default: "Datetime"). If absent, index is materialized.
    session_config : SessionConfig | None
        Session definition; defaults to SessionConfig.default().
    tz : str | None
        If provided and assume_clean=False, timestamps are normalized into this tz.
    enforce_strict_order : bool
        Enforce strictly increasing timestamps (only when assume_clean=False).
    build_sparse_tables : bool
        Build O(1) RMQ tables for highs/lows (optional).
    assume_clean : bool
        If True, skip dtype coercion, strict-order and OHLC invariant checks.
        Use only if the DataFrame has been cleaned & stamped by your ETL.
    """
    REQUIRED_COLS = ('Open', 'High', 'Low', 'Close', 'Volume')

    def __init__(
        self,
        df: "pd.DataFrame",
        timestamp_col: str = "Datetime",
        session_config: Optional[SessionConfig] = None,
        tz: Optional[str] = None,
        enforce_strict_order: bool = True,
        build_sparse_tables: bool = False,
        assume_clean: bool = False,            # <-- NEW
    ) -> None:
        self.session_config = session_config or SessionConfig.default()
        self.tz = tz
        self.enforce_strict_order = enforce_strict_order

        # --- Ensure we have the timestamp column; otherwise materialize index
        if timestamp_col in df.columns:
            work = df[[timestamp_col, 'Open', 'High', 'Low', 'Close', 'Volume']]
        else:
            tmp = df.reset_index()
            if timestamp_col not in tmp.columns:
                tmp = tmp.rename(columns={tmp.columns[0]: timestamp_col})
            work = tmp[[timestamp_col, 'Open', 'High', 'Low', 'Close', 'Volume']]

        # ======================= FAST PATH: assume_clean =======================
        if assume_clean:
            # We trust upstream cleaner: no coercions, no invariant/monotonic scans.
            # Timestamps expected tz-naive, already in desired tz context.
            self.timestamps = np.array(work[timestamp_col].dt.to_pydatetime(), dtype=object)
            self.open   = work['Open'].to_numpy(dtype=float, copy=False)
            self.high   = work['High'].to_numpy(dtype=float, copy=False)
            self.low    = work['Low'].to_numpy(dtype=float, copy=False)
            self.close  = work['Close'].to_numpy(dtype=float, copy=False)
            self.volume = work['Volume'].to_numpy(dtype=float, copy=False)
        # ======================= SAFE PATH: validate and coerce ================
        else:
            # Basic column presence guard (cheap but catches obvious issues)
            missing = [c for c in self.REQUIRED_COLS if c not in work.columns]
            if missing:
                raise ValueError(f"DataFrame missing required columns: {missing}")

            # Normalize timestamps robustly
            timestamps = np.empty(len(work), dtype=object)
            for i, v in enumerate(work[timestamp_col].values.tolist()):
                timestamps[i] = _ensure_datetime(v, tz=self.tz)

            # Enforce strictly increasing times (optional)
            if self.enforce_strict_order and len(timestamps) > 1:
                for i in range(1, len(timestamps)):
                    if timestamps[i] <= timestamps[i-1]:
                        raise ValueError(
                            f"Timestamps must be strictly increasing. "
                            f"Offender at index {i}: {timestamps[i]} <= {timestamps[i-1]}"
                        )

            self.timestamps = timestamps
            self.open   = work['Open'].to_numpy(dtype=float, copy=True)
            self.high   = work['High'].to_numpy(dtype=float, copy=True)
            self.low    = work['Low'].to_numpy(dtype=float, copy=True)
            self.close  = work['Close'].to_numpy(dtype=float, copy=True)
            self.volume = work['Volume'].to_numpy(dtype=float, copy=True)

            # OHLC invariants (defensive)
            if not (self.low <= np.minimum(self.open, self.close)).all():
                raise ValueError("Invariant violated: low must be <= min(open, close).")
            if not (self.high >= np.maximum(self.open, self.close)).all():
                raise ValueError("Invariant violated: high must be >= max(open, close).")

        # --- Session classification: single-pass state machine (Asia→London→NY AM→NY PM)
        minutes = np.array([_minutes_since_midnight(t) for t in self.timestamps], dtype=np.int32)
        fixed_order = ("Asia", "London", "NY AM", "NY PM")

        session_names = list(self.session_config.ranges.keys())
        missing_required = [s for s in fixed_order if s not in session_names]
        if missing_required:
            raise ValueError(f"SessionConfig must contain {fixed_order}; missing: {missing_required}")

        fb_name = self.session_config.fallback
        fb_id = session_names.index(fb_name) if fb_name in session_names else -1

        def _in_session(m: int, sess: str) -> bool:
            for r in self.session_config.ranges[sess]:
                if r.start_min <= m <= r.end_min:
                    return True
            return False

        def _full_scan(m: int) -> int:
            for sid, name in enumerate(session_names):
                for r in self.session_config.ranges[name]:
                    if r.start_min <= m <= r.end_min:
                        return sid
            return -1

        idx_of = {name: i for i, name in enumerate(session_names)}
        order_idx = {name: fixed_order.index(name) for name in fixed_order}

        sess_id = np.full(len(minutes), -1, dtype=np.int16)

        if len(minutes) > 0:
            m0 = int(minutes[0])
            sid0 = _full_scan(m0)
            if sid0 == -1 and fb_id != -1:
                sid0 = fb_id
            sess_id[0] = sid0

        for i in range(1, len(minutes)):
            m = int(minutes[i])
            prev_sid = int(sess_id[i - 1])

            if prev_sid < 0:
                sid = _full_scan(m)
                if sid == -1 and fb_id != -1:
                    sid = fb_id
                sess_id[i] = sid
                continue

            prev_name = session_names[prev_sid]
            if prev_name in order_idx:
                next_name = fixed_order[(order_idx[prev_name] + 1) % len(fixed_order)]
                if _in_session(m, prev_name):
                    sess_id[i] = idx_of[prev_name]
                elif _in_session(m, next_name):
                    sess_id[i] = idx_of[next_name]
                else:
                    if fb_id != -1 and _in_session(m, fb_name):
                        sess_id[i] = fb_id
                    else:
                        sid = _full_scan(m)
                        if sid == -1 and fb_id != -1:
                            sid = fb_id
                        sess_id[i] = sid
            else:
                sid = _full_scan(m)
                if sid == -1 and fb_id != -1:
                    sid = fb_id
                sess_id[i] = sid

        self.session_names: List[str] = session_names
        self.session_id: np.ndarray = sess_id
        self._session_of: List[str] = [session_names[i] if i >= 0 else self.session_config.fallback for i in sess_id]

        # Build ergonomic Candle objects (arrays remain authoritative)
        self.candles: List[Candle] = []
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

        # Per-day/session tracking state
        self._sessions_order = tuple([name for name in session_names if name != self.session_config.fallback])
        self._rolling_levels: Dict[str, Dict[str, Optional[float]]] = {}
        self._final_levels: Dict[str, Dict[str, Optional[float]]] = {}
        self._active_session: Optional[str] = None
        self._current_day_anchor: Optional[str] = None
        self._level_hit_flags: List[Dict[str, Any]] = []
        self._daily_levels: Dict[str, Dict[str, Optional[float]]] = {}
        self.reset_daily_levels()

        # Optional RMQ build
        self._rmq_high: Optional[RangeMinMaxSparseTable] = None
        self._rmq_low: Optional[RangeMinMaxSparseTable] = None
        if build_sparse_tables and len(self.high) > 0:
            self._rmq_high = RangeMinMaxSparseTable(self.high)
            self._rmq_low = RangeMinMaxSparseTable(self.low)

    # ----------------------------- Pythonic container API -----------------------------

    def __len__(self) -> int: return len(self.candles)
    def __getitem__(self, idx: Union[int, slice]) -> Union[Candle, List[Candle]]: return self.candles[idx]
    def __iter__(self): return iter(self.candles)

    def __repr__(self) -> str:
        if len(self.candles) == 0:
            return "Candles(0 candles)"
        first_ts = self.candles[0].timestamp
        last_ts = self.candles[-1].timestamp
        f = first_ts.strftime('%Y-%m-%d %H:%M:%S') if isinstance(first_ts, datetime) else str(first_ts)
        l = last_ts.strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_ts, datetime) else str(last_ts)
        return f"Candles({len(self.candles)} candles from {f} to {l})"

    def get_candles(self) -> List[Candle]:
        return self.candles

    # --------------------------- Range queries (fast paths) ---------------------------

    def _validate_window(self, start_index: int, end_index: int) -> Tuple[int, int]:
        n = len(self.candles)
        if n == 0: raise ValueError("Empty series.")
        if start_index < 0 or end_index >= n: raise ValueError("Invalid start or end index")
        if start_index > end_index: start_index, end_index = end_index, start_index
        return start_index, end_index

    def get_high_low(self, start_index: int, end_index: int) -> Tuple[float, float]:
        start_index, end_index = self._validate_window(start_index, end_index)
        if self._rmq_high is not None and self._rmq_low is not None:
            return self._rmq_high.query_max(start_index, end_index), self._rmq_low.query_min(start_index, end_index)
        window_high = self.high[start_index:end_index+1]
        window_low = self.low[start_index:end_index+1]
        return float(np.max(window_high)), float(np.min(window_low))

    def get_session_high_low(self, session: str, start_index: int = 0, end_index: Optional[int] = None) -> Tuple[float, float]:
        if end_index is None:
            end_index = len(self.candles) - 1
        start_index, end_index = self._validate_window(start_index, end_index)
        try:
            sess_idx = self.session_names.index(session)
        except ValueError:
            raise ValueError(f"Unknown session '{session}'. Known: {self.session_names}")
        mask = (self.session_id[start_index:end_index+1] == sess_idx)
        if not np.any(mask):
            raise ValueError("No matching candles found in given range.")
        sub_high = self.high[start_index:end_index+1][mask]
        sub_low = self.low[start_index:end_index+1][mask]
        return float(np.max(sub_high)), float(np.min(sub_low))

    # ---------------------- Daily/session levels & hit-flag logic ---------------------

    def reset_daily_levels(self):
        self._rolling_levels = {s: {"high": None, "low": None} for s in self._sessions_order}
        self._final_levels = {}
        self._active_session = None
        self._current_day_anchor = None
        self._level_hit_flags = []
        self._daily_levels = {s: {"high": None, "low": None} for s in self._sessions_order}

    def _anchor_of(self, dt: datetime, rollover_hour: int) -> str:
        return dt.strftime('%Y-%m-%d')

    def is_new_trading_day(self, prev_anchor: Optional[str], curr_dt: datetime, rollover_hour: int = 18) -> bool:
        if prev_anchor is None:
            return True
        return curr_dt.strftime('%Y-%m-%d') != prev_anchor

    def _finalize_session_if_needed(self, new_session: str):
        prev = self._active_session
        if prev is not None and prev != new_session:
            rl = self._rolling_levels.get(prev)
            if rl and rl["high"] is not None and rl["low"] is not None and prev not in self._final_levels:
                self._final_levels[prev] = {"high": rl["high"], "low": rl["low"]}
        self._active_session = new_session

    @staticmethod
    def _update_high_low(cell: Optional[Dict[str, Optional[float]]], high: float, low: float) -> None:
        if cell is None:
            return
        if cell["high"] is None or high > cell["high"]:
            cell["high"] = high
        if cell["low"] is None or low < cell["low"]:
            cell["low"] = low

    def _update_rolling_levels(self, session: str, high: float, low: float):
        self._update_high_low(self._rolling_levels.get(session), high, low)

    def _check_hits_vs_final_levels(self, day_label: Optional[str], curr_session: str,
                                    c_index: int, c_timestamp: Any, c_high: float, c_low: float):
        for sess, lv in self._final_levels.items():
            if sess == curr_session:
                continue
        # touch lows
            fh = lv.get("high"); fl = lv.get("low")
            if fl is not None and c_low <= fl <= c_high:
                self._level_hit_flags.append({
                    "day": day_label, "touched_session": sess, "which": "low",
                    "level": fl, "by_session": curr_session, "at_index": c_index, "timestamp": c_timestamp,
                })
            if fh is not None and c_low <= fh <= c_high:
                self._level_hit_flags.append({
                    "day": day_label, "touched_session": sess, "which": "high",
                    "level": fh, "by_session": curr_session, "at_index": c_index, "timestamp": c_timestamp,
                })

    def update_key_levels_for_candle(self, c: Candle, c_index: int, rollover_hour: int = 18):
        if self._rolling_levels is None or self._daily_levels is None:
            self.reset_daily_levels()

        if self.is_new_trading_day(self._current_day_anchor, c.timestamp, rollover_hour):
            self.reset_daily_levels()
            self._current_day_anchor = self._anchor_of(c.timestamp, rollover_hour)

        self._finalize_session_if_needed(c.session)
        self._update_rolling_levels(c.session, c.high, c.low)
        self._update_high_low(self._daily_levels.get(c.session), c.high, c.low)

        self._check_hits_vs_final_levels(
            day_label=self._current_day_anchor,
            curr_session=c.session,
            c_index=c_index,
            c_timestamp=c.timestamp,
            c_high=c.high,
            c_low=c.low
        )

        return {"rolling": self._rolling_levels, "final": self._final_levels, "flags_count": len(self._level_hit_flags)}

    def consume_level_hit_flags(self) -> List[Dict[str, Any]]:
        out = self._level_hit_flags
        self._level_hit_flags = []
        return out

    def get_today_final_levels(self) -> Dict[str, Dict[str, Optional[float]]]:
        return dict(self._final_levels)

    def compute_daily_session_levels(self, rollover_hour: int = 18) -> List[Dict[str, Any]]:
        self.reset_daily_levels()
        self.daily_levels_history: List[Dict[str, Any]] = []
        self._current_day_anchor = None

        for i, c in enumerate(self.candles):
            if self.is_new_trading_day(self._current_day_anchor, c.timestamp, rollover_hour):
                if self._current_day_anchor is not None:
                    self.daily_levels_history.append({
                        "day": self._current_day_anchor,
                        "levels": {s: {"high": v["high"], "low": v["low"]} for s, v in self._daily_levels.items()}
                    })
                self.reset_daily_levels()
                self._current_day_anchor = self._anchor_of(c.timestamp, rollover_hour)

            self._update_high_low(self._daily_levels.get(c.session), c.high, c.low)

        if self._current_day_anchor is not None:
            self.daily_levels_history.append({
                "day": self._current_day_anchor,
                "levels": {s: {"high": v["high"], "low": v["low"]} for s, v in self._daily_levels.items()}
            })

        return self.daily_levels_history

    # --------------------------------- Exporters -------------------------------------

    def flags_to_dataframe(self) -> "pd.DataFrame":
        if not self._level_hit_flags:
            return pd.DataFrame(columns=["day","touched_session","which","level","by_session","at_index","timestamp"])
        return pd.DataFrame(self._level_hit_flags)

    def daily_levels_to_dataframe(self) -> "pd.DataFrame":
        rows = []
        for rec in getattr(self, "daily_levels_history", []):
            day = rec["day"]
            for sess, lv in rec["levels"].items():
                rows.append({"day": day, "session": sess, "high": lv["high"], "low": lv["low"]})
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["day","session","high","low"])
