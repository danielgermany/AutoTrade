# Candles.py
# -----------------------------------------------------------------------------
# This module defines:
#   1) Candle:       a single OHLCV bar with a simple session classifier.
#   2) CandleSeries: a wrapper around a pandas DataFrame providing convenience
#                    access to a list of Candle objects plus utilities for:
#                       - high/low queries over ranges or sessions
#                       - intraday session level tracking/reset on rollovers
#                       - per-day finalized session levels and "level hits"
#
# NOTE: The code below is UNCHANGED in logic and behavior. Only comments were
#       added to explain the intent, assumptions, and complexity of methods.
# -----------------------------------------------------------------------------


# Define a single candlestick object representing one time interval of market data
class Candle:
    def __init__(self, close, open_, high, low, volume, timestamp):
        """
        Initialize a Candle object representing OHLCV market data at a specific time.

        Args:
            close (float): Closing price of the candle.
            open_ (float): Opening price of the candle. (named 'open_' to avoid conflict with Python keyword 'open')
            high (float): Highest price during the candle's time period.
            low (float): Lowest price during the candle's time period.
            volume (float): Volume traded during the time period.
            timestamp (any): Timestamp for the candle (can be a datetime or string).

        Side-effect:
            - Also computes and stores a 'session' label by calling self.session_of(timestamp).

        Assumptions / Caveats:
            - session_of() below expects a *string* timestamp of form "YYYY-MM-DD HH:MM:SS".
              If a non-string (e.g., pandas.Timestamp) is passed, time-based parsing will
              need to be compatible with str.split(). In the current code, this is assumed.
        """
        self.timestamp = timestamp
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.session = self.session_of(timestamp)  # derive trading "session" tag (Asia/London/NY AM/NY PM/Off)

    def session_of(self,time):
        """
        Classify the candle's timestamp into a coarse trading session based on NY local time.

        Expected timestamp format:
            "YYYY-MM-DD HH:MM:SS" so that .split() logic works.

        Session buckets:
            - "NY AM": 07:00 - 10:59
            - "NY PM": 11:00 - 15:59
            - "London": 02:00 - 06:59
            - "Asia": 18:00 - 23:59 OR 00:00 - 01:59
            - "Off": 16:00 - 17:59 (i.e., non-primary trading hours in this scheme)

        Returns:
            str: One of {"NY AM", "NY PM", "London", "Asia", "Off"}.
        """
        # Split into date and time (assumes a single space separator)
        date_part, time_part = time.split()
        # Split "HH:MM:SS" and coerce hour/minute to integers
        hour, minute, _ = time_part.split(':')
        hour = int(hour)
        minute = int(minute)

        # NY AM: 07:00 - 10:59 (inclusive of any minute in 7:00..10:59)
        if (hour == 7 and minute >= 0) or (7 < hour < 11) or (hour == 10 and minute < 60):
            return "NY AM"
        # NY PM: 11:00 - 15:59
        if 11 <= hour < 16:
            return "NY PM"
        # London: 02:00 - 06:59
        if 2 <= hour < 7:
            return "London"
        # Asia: 18:00 - 23:59 OR 00:00 - 01:59
        if hour >= 18 or hour < 2:
            return "Asia"
        # Off: 16:00 - 17:59 (outside the main sessions above)
        return "Off"


    def body(self):
        """
        Calculate the size of the candle's body (absolute difference between open and close).

        Returns:
            float: Absolute size of the candle body.
        """
        return abs(self.close - self.open)

    def is_bullish(self):
        """
        Determine if the candle is bullish (closing price higher than opening price).

        Returns:
            bool: True if bullish, else False.
        """
        return self.close > self.open

    def is_bearish(self):
        """
        Determine if the candle is bearish (closing price lower than opening price).

        Returns:
            bool: True if bearish, else False.
        """
        return self.close < self.open

    def wick_top(self):
        """
        Calculate the size of the upper wick (high minus the higher of open/close).

        Returns:
            float: Length of the top wick.
        """
        return self.high - max(self.open, self.close)

    def wick_bottom(self):
        """
        Calculate the size of the lower wick (lower of open/close minus low).

        Returns:
            float: Length of the bottom wick.
        """
        return min(self.open, self.close) - self.low

    def __repr__(self):
        """
        Return a string representation of the Candle object, useful for debugging.

        Returns:
            str: Human-readable summary of the candle.
        """
        return (f"Candle({self.timestamp}, Open:{self.open}, Close:{self.close}, "
                f"High:{self.high}, Low:{self.low}, Volume:{self.volume})")


# Define a class to manage and operate on a list of Candle objects
class CandleSeries:
    def __init__(self, df):
        """
        Initializes the CandleSeries object with a DataFrame of OHLCV data.

        Args:
            df (pandas.DataFrame): A DataFrame containing at least the columns:
                ['Open', 'High', 'Low', 'Close', 'Volume'] and optionally 'Datetime'.

        Behavior:
            - Resets the index so that 'Datetime' is a column (if it had been the index).
            - Iterates rows to construct a parallel Python list of Candle objects.

        Complexity:
            - Iterating df.iterrows() is O(N) in the number of rows. For large datasets,
              vectorization or itertuples() may be faster, but iterrows() is clearer.
        """
        # Reset the index so that 'Datetime' becomes a column (if it was the index)
        self.df = df.reset_index()

        # Convert each row in the DataFrame into a Candle object and store in a list
        self.candles = [self.row_to_candle(row) for _, row in self.df.iterrows()]

        # Optional dict for external stats aggregation if desired (not used below)
        self.session_stats = {}  # e.g., {"Asia": {"high": ..., "low": ...}, ...}

    def row_to_candle(self, row):
        """
        Converts a single row of the DataFrame into a Candle object.

        Args:
            row (pandas.Series): A single row of OHLCV data.

        Returns:
            Candle: A Candle object created from the row data.

        Robustness:
            - Uses 'Datetime' column if present; otherwise falls back to row.name.
            - Assumes row has the required numeric columns.
        """
        return Candle(
            # Use 'Datetime' if it exists, otherwise fall back to the row index (row.name)
            timestamp = row['Datetime'] if 'Datetime' in row else row.name,
            open_     = row['Open'],    # Opening price
            high      = row['High'],    # Highest price
            low       = row['Low'],     # Lowest price
            close     = row['Close'],   # Closing price
            volume    = row['Volume'],  # Volume traded
        )

    def __getitem__(self, idx):
        """
        Enables indexing into the Candles object like a list.

        Args:
            idx (int): The index of the candle to retrieve.

        Returns:
            Candle: The Candle object at the given index.

        Complexity:
            - O(1), list indexing.
        """
        return self.candles[idx]

    def __len__(self):
        """
        Returns the number of Candle objects stored.

        Returns:
            int: The total number of candles. O(1).
        """
        return len(self.candles)

    def slice(self, start, end):
        """
        Returns a slice of the candle list.

        Args:
            start (int): Starting index.
            end (int): Ending index (non-inclusive).

        Returns:
            list[Candle]: A sublist of Candle objects from start to end.

        Notes:
            - Mirrors Python slicing semantics with end exclusive.
        """
        return self.candles[start:end]

    def _calc_high_low_in_range(self, start_index, end_index, session=None):
        """
        Internal helper to calculate highest high and lowest low for a given
        index range, optionally filtering by session.

        Parameters
        ----------
        start_index : int
        end_index   : int
        session     : str or None
            If given, only candles matching this session are considered.

        Returns
        -------
        (float, float): (highest_high, lowest_low)

        Raises
        ------
        ValueError: If the index window is invalid or no candles match 'session'.

        Complexity:
            - O(K) where K = end_index - start_index + 1 (linear scan of the window).
        """
        # Validate bounds and ordering
        if start_index < 0 or end_index >= len(self.candles) or start_index > end_index:
            raise ValueError("Invalid start or end index")

        found = False          # tracks whether at least one candidate candle was considered
        highest_high = None    # will store running max
        lowest_low = None      # will store running min

        # Inclusive scan over the requested index window
        for i in range(start_index, end_index + 1):
            c = self.candles[i]
            # If session filter is active, skip candles not in the target session
            if session is not None and c.session != session:
                continue

            if not found:
                # First matching candle seeds the aggregates
                highest_high = c.high
                lowest_low = c.low
                found = True
            else:
                # Update running extrema
                if c.high > highest_high:
                    highest_high = c.high
                if c.low < lowest_low:
                    lowest_low = c.low

        if not found:
            # Either no candles in range or none matched session filter
            raise ValueError("No matching candles found in given range.")

        return highest_high, lowest_low

    def get_high_low(self, start_index, end_index):
        """
        Finds the highest high and lowest low between start_index and end_index (inclusive).

        Returns:
            (float, float)
        """
        return self._calc_high_low_in_range(start_index, end_index)

    def get_session_high_low(self, session, start_index=0, end_index=None):
        """
        Finds the highest high and lowest low for a given session
        between start_index and end_index (inclusive).

        Args:
            session (str): One of the session labels used by Candle.session_of.
            start_index (int): Start index for the scan.
            end_index (int|None): End index (inclusive). If None, defaults to last index.
        """
        if end_index is None:
            end_index = len(self.candles) - 1
        return self._calc_high_low_in_range(start_index, end_index, session=session)

    def reset_daily_levels(self):
        """
        Initialize/reset per-day state (call at trading-day rollover).

        Data structures created:
            - _sessions_order: tracking order of major sessions (Off is ignored here).
            - _rolling_levels: dict of session -> running {'high','low'} for the *current* day/session.
            - _final_levels:   dict of session -> frozen {'high','low'} finalized when a session ends.
            - _active_session: name of session currently being updated.
            - _current_day_anchor: string label 'YYYY-MM-DD' identifying the trading day.
            - _level_hit_flags: list of dict events indicating touches of prior sessions' levels.

        This method intentionally clears all of the above to start a new trading day.
        """
        self._sessions_order = ["Asia", "London", "NY AM", "NY PM"]  # track these; ignore "Off" if you wish
        self._rolling_levels = {s: {"high": None, "low": None} for s in self._sessions_order}
        self._final_levels = {}   # e.g., {"Asia": {"high": 123, "low": 100}, ...} once session ends
        self._active_session = None
        self._current_day_anchor = None  # e.g., "YYYY-MM-DD" label for the day
        self._level_hit_flags = []       # list of dict events (see below)

    def _parse_date_hour(self, ts_str):
        """
        Parse a timestamp string expected in "YYYY-MM-DD HH:MM:SS" into date and hour.

        Returns:
            (date_str, hour_int)

        Assumptions:
            - ts_str is a string with a single space between date and time.
        """
        # expects "YYYY-MM-DD HH:MM:SS"
        date_part, time_part = ts_str.split()
        hour = int(time_part.split(':')[0])
        return date_part, hour

    def is_new_trading_day(self, prev_ts, curr_ts, rollover_hour=18):
        """
        Detect trading-day rollover at `rollover_hour` local (NY) clock WITHOUT external libraries.

        The logic flags a new day in two scenarios:
          1) Same calendar date, but the hour crosses the rollover boundary (prev_hour < R <= curr_hour)
          2) Date changes AND the current hour is at/after rollover (conservative fallback)

        Args:
            prev_ts (str|None): previous timestamp string, or None if no previous candle
            curr_ts (str): current timestamp string
            rollover_hour (int): hour (0-23) defining the trading-day boundary (default 18 == 6pm)

        Returns:
            bool: True if rollover detected, else False.

        Caveat:
            - This naive approach assumes timestamps are contiguous and monotonically increasing.
        """
        if prev_ts is None:
            return True
        prev_date, prev_hour = self._parse_date_hour(prev_ts)
        curr_date, curr_hour = self._parse_date_hour(curr_ts)
        if prev_date == curr_date and prev_hour < rollover_hour <= curr_hour:
            return True
        if prev_date != curr_date and curr_hour >= rollover_hour:
            return True
        return False

    def _finalize_session_if_needed(self, new_session):
        """
        When the session changes, finalize the previous session’s rolling levels.

        Behavior:
            - If we were actively tracking 'prev' and it differs from new_session, freeze its
              current rolling high/low (if present) into _final_levels (only once).
            - Then set _active_session = new_session.

        Notes:
            - This relies on Candle.session to indicate which bucket the new candle belongs to.
        """
        prev = self._active_session
        if prev is not None and prev != new_session:
            rl = self._rolling_levels.get(prev)
            if rl and rl["high"] is not None and rl["low"] is not None:
                # Freeze only once
                if prev not in self._final_levels:
                    self._final_levels[prev] = {"high": rl["high"], "low": rl["low"]}
        # Switch active
        self._active_session = new_session

    def _update_rolling_levels(self, session, high, low):
        """
        In-place update of the rolling high/low for the given session using the current candle.
        """
        cell = self._rolling_levels.get(session)
        if cell is None:
            return
        if cell["high"] is None or high > cell["high"]:
            cell["high"] = high
        if cell["low"] is None or low < cell["low"]:
            cell["low"] = low

    def _check_hits_vs_final_levels(self, day_label, curr_session, c_index, c_timestamp, c_high, c_low):
        """
        If the current candle's [low, high] interval touches any *prior-session* finalized high/low,
        append a flag event dict to self._level_hit_flags.

        Matching condition:
            - A level L is considered touched if c_low <= L <= c_high.

        Appended flag format:
            {
                "day": day_label,
                "touched_session": <prior session name>,
                "which": "high"|"low",
                "level": float,
                "by_session": curr_session,
                "at_index": int,
                "timestamp": <candle timestamp>
            }
        """
        for sess, lv in self._final_levels.items():
            if sess == curr_session:
                continue  # only flag touches of *prior* sessions
            fh = lv["high"]
            fl = lv["low"]

            # Check low-level touch
            if fl is not None and c_low <= fl <= c_high:
                self._level_hit_flags.append({
                    "day": day_label,
                    "touched_session": sess,
                    "which": "low",
                    "level": fl,
                    "by_session": curr_session,
                    "at_index": c_index,
                    "timestamp": c_timestamp
                })
            # Check high-level touch
            if fh is not None and c_low <= fh <= c_high:
                self._level_hit_flags.append({
                    "day": day_label,
                    "touched_session": sess,
                    "which": "high",
                    "level": fh,
                    "by_session": curr_session,
                    "at_index": c_index,
                    "timestamp": c_timestamp
                })

    def update_key_levels_for_candle(self, c, c_index, rollover_hour=18):
        """
        Stream this per new candle (call once per incoming candle in chronological order):

          1) Rollover? -> reset daily state structures for a new trading day.
          2) Session change? -> finalize the prior session’s rolling levels (freeze into _final_levels).
          3) Update rolling levels for the *current* session with this candle’s high/low.
          4) Check if this candle touches any *prior finalized* session levels and record flags.

        Args:
            c (Candle): The current candle.
            c_index (int): Index of 'c' within self.candles.
            rollover_hour (int): Trading-day boundary hour (default 18).

        Returns:
            dict: Snapshot like:
                {
                    "rolling": <dict of current rolling levels>,
                    "final":   <dict of finalized prior-session levels>,
                    "flags_count": <int number of accumulated touch flags>
                }

        Expected usage:
            - Call for each new candle during live processing/backtest to keep
              levels updated and to collect "touch" events.

        Complexity:
            - O(S) per call for touch checks, where S = number of finalized sessions today (<= 3).
        """
        # Lazy init (if consumer didn't call reset_daily_levels() manually)
        if not hasattr(self, "_rolling_levels"):
            self.reset_daily_levels()

        # Rollover detection (new trading day?)
        if self.is_new_trading_day(self._current_day_anchor, c.timestamp, rollover_hour):
            # Start new day state
            self.reset_daily_levels()
            date_part, hour = self._parse_date_hour(c.timestamp)
            # Anchor label for the current day. (Current code uses the same date_part either way.)
            # NOTE: The conditional here is redundant because both branches assign date_part.
            self._current_day_anchor = date_part if hour >= rollover_hour else date_part  # simple label

        # If session changed from the last candle, freeze prior session levels
        self._finalize_session_if_needed(c.session)

        # Update rolling levels with current candle’s extrema for the active session
        self._update_rolling_levels(c.session, c.high, c.low)

        # Check touches of any *finalized* prior-session levels
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

    def consume_level_hit_flags(self):
        """
        Return and clear accumulated 'level hit' flags.

        Each flag is a dict of the form:
          {
            'day': 'YYYY-MM-DD',
            'touched_session': 'Asia'|'London'|'NY AM'|'NY PM',
            'which': 'high'|'low',
            'level': float,
            'by_session': current session string,
            'at_index': int (index in self.candles),
            'timestamp': candle.timestamp
          }

        Usage:
            events = series.consume_level_hit_flags()
            # ... handle events ...
        """
        out = self._level_hit_flags
        self._level_hit_flags = []
        return out

    def get_today_final_levels(self):
        """
        Return a shallow copy of frozen levels for sessions already completed today.

        Returns:
            dict: e.g. {"Asia": {"high": 123.0, "low": 100.0}, ...}
        """
        return dict(self._final_levels)

    # --- BATCH (OPTIONAL): compute per trading day over whole series ----------
    def compute_daily_session_levels(self, rollover_hour=18):
        """
        Iterate the entire series, resetting at each trading-day rollover,
        and return a list of per-day session levels. Also stores it on
        self.daily_levels_history.

        Returns:
            list of dicts like:
              [
                {
                  "day": "YYYY-MM-DD",
                  "levels": {
                      "Asia": {"high": <float|None>, "low": <float|None>},
                      "London": {...}, "NY AM": {...}, "NY PM": {...}
                  }
                },
                ...
              ]

        IMPORTANT NOTE:
            This function references self._daily_levels, but no method in this class
            initializes _daily_levels. As written, calling this will raise an
            AttributeError unless _daily_levels is created elsewhere. See suggestions
            below for a fix.
        """
        self.reset_daily_levels()
        self.daily_levels_history = []
        self._current_day_anchor = None

        prev_ts = None
        for c in self.candles:
            # Detect potential rollover
            if self.is_new_trading_day(prev_ts, c.timestamp, rollover_hour):
                # If we already had a day in progress, flush its levels to history
                if self._current_day_anchor is not None:
                    self.daily_levels_history.append({
                        "day": self._current_day_anchor,
                        "levels": {
                            s: {"high": v["high"], "low": v["low"]}
                            for s, v in self._daily_levels.items()
                        }
                    })
                # Reset for new day
                self.reset_daily_levels()
                date_part, hour = self._parse_date_hour(c.timestamp)
                # Only sets anchor if hour >= rollover; otherwise keeps prior anchor
                self._current_day_anchor = date_part if hour >= rollover_hour else self._current_day_anchor

            # Update daily levels for the current candle's session
            # NOTE: _daily_levels is NOT initialized anywhere in current code.
            s = c.session
            cell = self._daily_levels.get(s)
            if cell is not None:
                if cell["high"] is None or c.high > cell["high"]:
                    cell["high"] = c.high
                if cell["low"] is None or c.low < cell["low"]:
                    cell["low"] = c.low

            prev_ts = c.timestamp

        # Flush the final day's levels if we have an anchor
        if self._current_day_anchor is not None:
            self.daily_levels_history.append({
                "day": self._current_day_anchor,
                "levels": {
                    s: {"high": v["high"], "low": v["low"]}
                    for s, v in self._daily_levels.items()
                }
            })

        return self.daily_levels_history

    def __repr__(self):
        """
        Returns a human-readable string representation of the Candles object,
        showing the number of candles and the date range covered.

        Returns:
            str: Summary of the Candles object.
        """
        # Handle empty series
        if not self.candles:
            return "Candles(0 candles)"
        first_ts = self.candles[0].timestamp
        last_ts = self.candles[-1].timestamp
        return f"Candles({len(self.candles)} candles from {first_ts} to {last_ts})"
