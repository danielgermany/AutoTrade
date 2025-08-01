from price_action import *
ratio = 1.1 #TP/SL Ratio


def target(bars_ahead,df):
    length = len(df1)
    high = df['High'].to_list()
    low = df['Low'].to_list()
    close_ = df['Close'].to_list()
    open_ = df['Open'].to_list()
    trendcat = [0] * length
    amount = [0] * length

    #back_candles is to check how far we want to look for success/failure of trade
    #This logic needs to be changed later
    for line in range(back_candles, length-back_candles - n2):

        if signal[line] == 1: #Shorts
            SL = max(high[line-1:line+1]) #These can change to whatever
            TP = close_[line]-ratio*(SL-close_[line])

            for i in range(1,bars_ahead + 1):
                if low[line + i] <= TP and high[line + i] >= SL:
                    trendcat[line] = 3
                    break

                elif low[line + i] <= TP: #win trend 1 in signal 1
                    trendcat[line] = 1
                    amount[line] = close_[line] - low[line+1]
                    break

                elif high[line + i] >= SL: #loss trend 2 in signal 1
                    trendcat[line] = 2
                    amount[line] = close_[line] - high[line+1]
                    break

        if signal[line] == 2: #Longs
            SL = min(low[line - 1:line + 1]) #These can change to whatever
            TP = close_[line] + ratio * (close_[line] - SL)

            for i in range(1, bars_ahead + 1):
                if high[line + 1] >= TP and low[line + i] <= SL:
                    trendcat[line] = 3
                    break

                elif high[line + 1] >= TP: #win trend 2 in signal 2
                    trendcat[line] = 2
                    amount[line] = high[line + 1] - close_[line]
                    break

                elif low[line + i] <= SL: #loss trend 1 in signal 2
                    trendcat[line] = 1
                    amount[line] = low[line + i] - close_[line]
                    break

    return amount

trend = target(16,dataF)