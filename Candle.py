class Candle:
    def __init__(self,close,open_,high,low,volume,timestamp):
        self.timestamp = timestamp
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def body(self):
        return abs(self.close - self.open)

    def is_bullish(self):
        return self.close > self.open

    def is_bearish(self):
        return self.close < self.open

    def wick_top(self):
        return self.high - max(self.open,self.close)

    def wick_bottom(self):
        return min(self.open,self.close) - self.low

    def __repr__(self):
        return (f"Candle({self.timestamp}, Open:{self.open},Close:{self.close},"
                f"High:{self.high},Low:{self.low},Volume:{self.volume})")
