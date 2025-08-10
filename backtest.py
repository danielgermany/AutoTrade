from backtesting import Backtest
from db import df
from Strat import strat

bt = Backtest(df,strat, cash= 50_000)
stat = bt.run()

print(stat)