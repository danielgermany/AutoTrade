
"""
test_candles_advanced.py
-------------------------------------------------------------------------------
A comprehensive, heavily commented unit test suite for:

  • ohlcv_cleaner.py  – data cleaning, validation, and "clean stamp" logic
  • candles_advanced.py – Candle, CandleSeries, RMQ, session logic & exporters

How to run (from the repo root or this file's directory):
    python -m unittest -v test_candles_advanced.py

Design notes:
  - Uses only Python stdlib "unittest" plus numpy/pandas (already project deps).
  - Generates synthetic OHLCV data with realistic minute bars across multiple
    days/sessions. Also constructs intentionally "dirty" datasets to test cleaning.
  - Exercises both SAFE and FAST paths:
        assume_clean=False (validation & coercion)
        assume_clean=True  (trusts cleaner / fast ingest)
  - Validates state-machine session classification, high/low queries (with/without
    RMQ), daily session levels + "prior session level touch" flags, and exporters.
  - Includes optional integration test with a real CSV if available locally.
"""

from __future__ import annotations

import os
import sys
import math
import unittest
from datetime import datetime, timedelta
from typing import Tuple, List

import numpy as np
import pandas as pd

# Ensure imports work whether tests are run from repo root or this file's folder
HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    from ohlcv_cleaner import clean_ohlcv, validate_ohlcv, stamp_clean, is_stamped_clean, REQUIRED_COLS
    from candles_advanced import (
        Candle, CandleSeries, SessionConfig, SessionRange,
        _classify_session, RangeMinMaxSparseTable
    )
except Exception as e:
    raise RuntimeError(
        "Failed to import modules. Ensure candles_advanced.py and ohlcv_cleaner.py "
        "are on the Python path or in the same directory as this test file.\n"
        f"Import error: {e}"
    )


# ----------------------------- Synthetic data helpers -----------------------------

def _make_minute_range(start: datetime, minutes: int) -> List[datetime]:
    """Build a list of datetime stamps at 1‑minute resolution."""
    return [start + timedelta(minutes=i) for i in range(minutes)]


def make_synthetic_ohlcv(
    start: datetime,
    minutes: int,
    start_price: float = 10000.0,
    drift_per_min: float = 0.25,
    noise_amp: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a clean OHLCV DataFrame with realistic minute bars.
    - "Datetime" is tz‑naive and strictly increasing.
    - OHLC is consistent (Low <= min(Open, Close) <= High).
    - Volume is non‑negative.
    """
    rng = np.random.default_rng(seed)
    times = _make_minute_range(start, minutes)
    # Create a smooth-ish path with drift + noise for close-to-close
    steps = drift_per_min + rng.normal(0.0, noise_amp, size=minutes)
    close = np.cumsum(steps) + start_price
    open_ = np.concatenate(([start_price], close[:-1]))
    # High/low around open/close with small wicks
    wick_up = np.abs(rng.normal(0.4, 0.2, size=minutes))
    wick_dn = np.abs(rng.normal(0.4, 0.2, size=minutes))
    high = np.maximum(open_, close) + wick_up
    low = np.minimum(open_, close) - wick_dn
    volume = np.abs(rng.integers(100, 5000, size=minutes)).astype(float)

    df = pd.DataFrame({
        "Datetime": pd.to_datetime(times, utc=False),  # tz‑naive, desired for cleaner
        "Open": open_.astype(float),
        "High": high.astype(float),
        "Low": low.astype(float),
        "Close": close.astype(float),
        "Volume": volume,
    })
    return df


def make_dirty_ohlcv(base: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a "dirty" frame from a clean one:
      - Insert NaNs in a few places
      - Duplicate one timestamp
      - Break OHLC invariant in one row
      - Add a negative volume in one row
      - Shuffle some rows to break monotonic order
    """
    x = base.copy()

    # 1) Inject NaNs (Open & Volume) in two separate rows
    if len(x) > 10:
        x.loc[3, "Open"] = np.nan
        x.loc[7, "Volume"] = np.nan

    # 2) Duplicate a timestamp (row 5 duplicates row 4)
    if len(x) > 6:
        x.loc[5, "Datetime"] = x.loc[4, "Datetime"]

    # 3) Break OHLC invariant in one row: set Low above both Open and Close
    if len(x) > 12:
        o, c = x.loc[12, "Open"], x.loc[12, "Close"]
        x.loc[12, "Low"] = max(o, c) + 2.0  # invalid

    # 4) Negative volume
    if len(x) > 15:
        x.loc[15, "Volume"] = -100.0

    # 5) Shuffle a slice to break monotonic order
    if len(x) > 25:
        swap_slice = x.iloc[20:25].iloc[::-1].copy()
        x.iloc[20:25] = swap_slice.values

    return x


# ------------------------------ Test: Cleaner module -------------------------------

class TestOHLCVCleaner(unittest.TestCase):
    """Tests for clean_ohlcv, validate_ohlcv, and clean stamp utilities."""

    def setUp(self):
        # 3 days of 1‑min bars to ensure multiple session cycles (4320 minutes)
        start = datetime(2025, 6, 23, 0, 0, 0)
        self.clean_df = make_synthetic_ohlcv(start, minutes=3 * 24 * 60, seed=123)
        self.dirty_df = make_dirty_ohlcv(self.clean_df)

    def test_clean_ohlcv_reports_and_contract(self):
        # Clean the intentionally dirty frame; ask cleaner to "clip" negative volume to 0
        out, report = clean_ohlcv(self.dirty_df, tz="America/New_York", clip_negative_volume=True)

        # 1) Post‑clean invariants are enforced (no NaNs, strictly increasing, etc.)
        validate_ohlcv(out)  # should not raise

        # 2) Cleaner reports should reflect the manipulations we made
        #    We don't hard‑code exact counts (fragile), but assert they're >= expected.
        self.assertGreaterEqual(report.dropped_na, 2, "Should drop at least the two NA rows injected.")
        self.assertGreaterEqual(report.dropped_dupes, 1, "Should drop at least one duplicate timestamp.")
        self.assertGreaterEqual(report.dropped_ohlc_invariant, 1, "Should drop at least one OHLC violation.")
        self.assertGreaterEqual(report.clipped_volume, 1, "Should have clipped at least one negative volume to 0.")

        # 3) Check required columns & dtypes
        for col in REQUIRED_COLS:
            self.assertIn(col, out.columns)
        self.assertTrue(pd.api.types.is_datetime64_ns_dtype(out["Datetime"]))
        for c in ("Open","High","Low","Close","Volume"):
            self.assertTrue(pd.api.types.is_float_dtype(out[c]))

    def test_clean_stamp_round_trip(self):
        out, _ = clean_ohlcv(self.clean_df, tz="America/New_York")
        # Stamp clean and check signature/flag
        stamp_clean(out, tz="America/New_York")
        self.assertTrue(is_stamped_clean(out), "Clean stamp should validate immediately.")

        # Mutate a value -> signature mismatch -> not stamped clean anymore
        out2 = out.copy()
        out2.loc[10, "Close"] += 1.0
        self.assertFalse(is_stamped_clean(out2), "Mutating the frame should invalidate the stamp.")


# --------------------------- Test: RMQ (sparse table) ------------------------------

class TestRangeMinMaxSparseTable(unittest.TestCase):
    """Verify O(1) range min/max queries after O(n log n) build."""

    def test_rmq_correctness(self):
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)
        rmq = RangeMinMaxSparseTable(arr)
        # Query a few ranges, including reversed order (l > r handled internally)
        self.assertEqual(rmq.query_min(0, 1), 1.0)
        self.assertEqual(rmq.query_max(0, 1), 3.0)
        self.assertEqual(rmq.query_min(2, 5), 1.0)
        self.assertEqual(rmq.query_max(2, 5), 9.0)
        self.assertEqual(rmq.query_min(7, 3), 1.0)  # reversed indices
        self.assertEqual(rmq.query_max(7, 3), 9.0)


# ----------------------------- Test: Candle & series -------------------------------

class TestCandleAndSeriesCore(unittest.TestCase):
    """Covers Candle methods, constructor fast vs safe paths, and __repr__."""

    @classmethod
    def setUpClass(cls):
        start = datetime(2025, 6, 23, 0, 0, 0)
        cls.base_clean = make_synthetic_ohlcv(start, minutes=48 * 60, seed=999)  # 2 days
        cls.cleaned, _ = clean_ohlcv(cls.base_clean, tz="America/New_York")

    def test_candle_methods(self):
        c = Candle(close=101.0, open_=100.0, high=102.0, low=99.0, volume=1000.0,
                   timestamp=datetime(2025, 6, 23, 7, 30, 0), session="NY AM")
        self.assertAlmostEqual(c.body(), 1.0)
        self.assertTrue(c.is_bullish())
        self.assertFalse(c.is_bearish())
        self.assertAlmostEqual(c.wick_top(), 1.0)   # 102 - max(100,101) = 1
        self.assertAlmostEqual(c.wick_bottom(), 1.0)  # min(100,101) - 99 = 1
        self.assertIn("Candle(", repr(c))

    def test_series_init_fast_assume_clean(self):
        # Using the "assume_clean=True" fast path: timestamps must be datetime64[ns]
        s = CandleSeries(self.cleaned, assume_clean=True, build_sparse_tables=True)
        self.assertGreater(len(s), 0)
        # Sparse tables must be present
        self.assertIsNotNone(s._rmq_high)
        self.assertIsNotNone(s._rmq_low)
        # __repr__ should show range
        r = repr(s)
        self.assertIn("Candles(", r)
        self.assertIn("from", r)
        self.assertIn("to", r)

    def test_series_init_safe_strict_order_violation(self):
        # Create a descending time order to trigger strict order check
        bad = self.cleaned.copy()
        bad = bad.iloc[::-1].reset_index(drop=True)
        with self.assertRaises(ValueError):
            _ = CandleSeries(bad, assume_clean=False, enforce_strict_order=True)

    def test_get_high_low_with_and_without_rmq(self):
        s = CandleSeries(self.cleaned, assume_clean=True, build_sparse_tables=False)
        hi1, lo1 = s.get_high_low(10, 200)  # vectorized scan
        s2 = CandleSeries(self.cleaned, assume_clean=True, build_sparse_tables=True)
        hi2, lo2 = s2.get_high_low(10, 200)  # O(1) RMQ
        self.assertAlmostEqual(hi1, hi2, places=10)
        self.assertAlmostEqual(lo1, lo2, places=10)

    def test_get_session_high_low_and_errors(self):
        s = CandleSeries(self.cleaned, assume_clean=True)
        # Valid session name (default config)
        for sess in ("Asia", "London", "NY AM", "NY PM", "Off"):
            hi, lo = s.get_session_high_low(sess, 0, len(s) - 1)
            self.assertTrue(hi >= lo)
        # Unknown session raises
        with self.assertRaises(ValueError):
            _ = s.get_session_high_low("Mars Open", 0, len(s) - 1)


# -------------------- Test: Session classification (state machine) ------------------

class TestSessionClassification(unittest.TestCase):
    """Ensure sessions follow Asia → London → NY AM → NY PM → Off in time."""

    @classmethod
    def setUpClass(cls):
        start = datetime(2025, 6, 23, 0, 0, 0)
        df = make_synthetic_ohlcv(start, minutes=2 * 24 * 60, seed=111)
        cls.cleaned, _ = clean_ohlcv(df, tz="America/New_York")
        cls.series = CandleSeries(cls.cleaned, assume_clean=True)

    def _find_index_at_time(self, hour: int, minute: int) -> int:
        """Find the first candle index at a specific (hour, minute) in any day."""
        for i, t in enumerate(self.series.timestamps):
            if t.hour == hour and t.minute == minute:
                return i
        raise AssertionError("Requested time not found in synthetic data.")

    def test_specific_times_map_to_expected_sessions(self):
        # Defaults from SessionConfig.default():
        #   Asia:   18:00–01:59
        #   London: 02:00–06:59
        #   NY AM:  07:00–10:59
        #   NY PM:  11:00–15:59
        #   Off:    16:00–17:59
        checks = [
            ((0, 30), "Asia"),   # 00:30
            ((2, 0), "London"),  # 02:00
            ((7, 0), "NY AM"),   # 07:00
            ((12, 0), "NY PM"),  # 12:00
            ((16, 30), "Off"),   # 16:30
            ((18, 5), "Asia"),   # 18:05
            ((23, 59), "Asia"),  # 23:59
        ]
        for (h, m), expected in checks:
            idx = self._find_index_at_time(h, m)
            self.assertEqual(self.series.candles[idx].session, expected, f"{h:02d}:{m:02d} should be {expected}")

    def test_legacy_classifier_agrees_on_random_samples(self):
        rng = np.random.default_rng(7)
        cfg = SessionConfig.default()
        # Compare state-machine label vs legacy function on random timestamps
        for _ in range(50):
            i = int(rng.integers(0, len(self.series)))
            dt = self.series.timestamps[i]
            sm_label = self.series.candles[i].session
            legacy_label = _classify_session(dt, cfg)
            self.assertEqual(sm_label, legacy_label)


# ---------------------- Test: Daily levels + "touch" flags logic -------------------

class TestDailyLevelsAndFlags(unittest.TestCase):
    """
    Validates:
      • compute_daily_session_levels (per-day, per-session high/low)
      • update_key_levels_for_candle / consume_level_hit_flags
      • Flag behavior when a later session touches a prior session's high/low
    """

    def test_flags_when_prior_session_level_is_touched(self):
        # Build a tiny, hand-crafted frame to control levels precisely.
        # Day 1: Asia high=101, low=99; London prints a candle that touches Asia high.
        rows = []
        base = datetime(2025, 6, 23, 18, 0, 0)  # 18:00 (Asia)
        # Asia candles
        rows.append((base,       100, 101,  99, 100, 1000.0))  # range includes 101/99
        rows.append((base+timedelta(minutes=1), 100, 100.5, 99.5, 100.2, 1200.0))
        # London candle that passes through prior Asia high (101)
        lon = datetime(2025, 6, 24, 2, 0, 0)
        rows.append((lon,        100.6, 101.2, 100.4, 100.8, 800.0))
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=["Datetime","Open","High","Low","Close","Volume"]).astype({
            "Open":"float64","High":"float64","Low":"float64","Close":"float64","Volume":"float64"
        })
        df["Datetime"] = pd.to_datetime(df["Datetime"])

        # Clean (idempotent here) and build series
        clean, _ = clean_ohlcv(df, tz="America/New_York")
        s = CandleSeries(clean, assume_clean=True)

        # Stream candles through update_key_levels_for_candle, capturing flags
        for i, c in enumerate(s.candles):
            s.update_key_levels_for_candle(c, i, rollover_hour=18)

        flags = s.consume_level_hit_flags()
        # Expect at least one flag: London touching Asia high=101
        hits = [f for f in flags if f["touched_session"] == "Asia" and f["which"] == "high"]
        self.assertGreaterEqual(len(hits), 1, "London should produce a 'touched Asia high' flag.")
        # Sanity: ensure the recorded level is very close to 101
        self.assertTrue(any(abs(h["level"] - 101.0) < 1e-9 for h in hits))

    def test_compute_daily_session_levels_output_shape(self):
        start = datetime(2025, 6, 23, 0, 0, 0)
        df = make_synthetic_ohlcv(start, minutes=48 * 60, seed=321)  # 2 days
        clean, _ = clean_ohlcv(df, tz="America/New_York")
        s = CandleSeries(clean, assume_clean=True)

        hist = s.compute_daily_session_levels(rollover_hour=18)
        # Should have >= 2 day records
        self.assertGreaterEqual(len(hist), 2)

        # Convert to DataFrame via exporter
        out = s.daily_levels_to_dataframe()
        # Expect rows for each session per recorded day
        expected_sessions = {"Asia", "London", "NY AM", "NY PM"}
        self.assertTrue(set(out["session"].unique()).issubset(expected_sessions))
        self.assertIn("day", out.columns)
        self.assertIn("high", out.columns)
        self.assertIn("low", out.columns)


# -------------------------------- Test: Exporters ----------------------------------

class TestExporters(unittest.TestCase):
    """flags_to_dataframe and daily_levels_to_dataframe basic behavior."""

    def test_flags_dataframe_empty_and_nonempty(self):
        # Initially, no flags
        start = datetime(2025, 6, 23, 0, 0, 0)
        clean, _ = clean_ohlcv(make_synthetic_ohlcv(start, minutes=180), tz="America/New_York")
        s = CandleSeries(clean, assume_clean=True)
        empty = s.flags_to_dataframe()
        self.assertEqual(len(empty), 0)
        # Generate one simple touch by constructing two sessions with overlap in a later candle
        tiny = pd.DataFrame({
            "Datetime": pd.to_datetime([
                datetime(2025,6,23,18,0,0),  # Asia
                datetime(2025,6,23,18,1,0),
                datetime(2025,6,24,2,0,0),   # London
            ]),
            "Open":   [100.0, 100.0, 100.5],
            "High":   [101.0, 100.5, 101.1],  # London touches 101.0
            "Low":    [ 99.0,  99.5, 100.2 ],
            "Close":  [100.0, 100.2, 100.6],
            "Volume": [1000.0, 1200.0, 900.0],
        })
        clean2, _ = clean_ohlcv(tiny, tz="America/New_York")
        s2 = CandleSeries(clean2, assume_clean=True)
        for i, c in enumerate(s2.candles):
            s2.update_key_levels_for_candle(c, i)

        df_flags = s2.flags_to_dataframe()
        self.assertGreaterEqual(len(df_flags), 1)
        self.assertTrue(set(["day","touched_session","which","level","by_session","at_index","timestamp"]).issubset(df_flags.columns))


# ----------------------- Optional integration test (CSV) ---------------------------

class TestOptionalIntegration(unittest.TestCase):
    """
    If a real CSV (nq_data_1m.csv) is present in the working directory, verify that
    it can be cleaned and ingested without errors. This test is skipped otherwise.
    """
    @unittest.skipUnless(os.path.exists(os.path.join(HERE, "nq_data_1m.csv")), "nq_data_1m.csv not found; skipping integration test.")
    def test_real_csv_ingests(self):
        path = os.path.join(HERE, "nq_data_1m.csv")
        raw = pd.read_csv(path)
        # Expect but don't require Datetime column naming; rename if needed
        if "Datetime" not in raw.columns and "DateTime" in raw.columns:
            raw = raw.rename(columns={"DateTime":"Datetime"})
        clean, _ = clean_ohlcv(raw, tz="America/New_York")
        _ = CandleSeries(clean, assume_clean=True, build_sparse_tables=True)  # ensure no exceptions


if __name__ == "__main__":
    unittest.main(verbosity=2)
