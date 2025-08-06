from backtesting import Strategy
from price_action import signal

class strat(Strategy):
    def init(self):
        super().init()
        self.signal1 = self.I(signal)

    def next(self):
        super().next(self)
        if self.signal1 == 2:
            sl1 = self.data.Close[-1] - 750e-4 #We are going to import better TP/SL logic later
            tp1 = self.data.Close[-1] + 600e-4
            self.buy(sl = sl1, tp = tp1)

        elif self.signal1 == 1:
            sl1 = self.data.Close[-1] - 750e-4  # We are going to import better TP/SL logic later
            tp1 = self.data.Close[-1] + 600e-4
            self.sell(sl = sl1, tp = tp1)