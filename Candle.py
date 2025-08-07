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
        """
        self.timestamp = timestamp
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def body(self):
        """
        Calculate the size of the candle's body (difference between open and close).

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
