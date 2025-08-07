from backtesting import Strategy
from price_action import signal

from random_signals import gen_signals

class strat(Strategy):
    def init(self):
        super().init()

        self.signal1 = self.I(gen_signals, self.data.Close)

    def next(self):
        super().next()
        if self.signal1 == 2:
            sl1 = self.data.Close[-1] - 750e-4 #We are going to import better TP/SL logic later
            tp1 = self.data.Close[-1] + 600e-4
            self.buy(sl = sl1, tp = tp1)

        elif self.signal1 == 1:
            sl1 = self.data.Close[-1] + 750e-4  # We are going to import better TP/SL logic later
            tp1 = self.data.Close[-1] - 600e-4
            self.sell(sl = sl1, tp = tp1)

