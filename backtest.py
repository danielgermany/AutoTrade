from backtesting import Backtest
from db import dataF
from Strat import strat

bt = Backtest(dataF,strat, cash= 50_000)
stat = bt.run()

print(stat)