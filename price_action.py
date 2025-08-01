from platform import android_ver

from db import *

#Support and Resistance

#df1 is the dataframe used
#candle is the index of the row we are looking at

#Checks the high and lows of the candles from the dataframe

def support(df1, candle, candles_before, candles_after):
    if candle - candles_before < 0 or candle + candles_after >= len(df1):
        return 0

    for i in range(candle - candles_before + 1, candle + 1):
        # Sanity check here
            #print(f"[SUPPORT BEFORE] i={i}, i-1={i-1}, Low[i]={df1['Low'].iloc[i]}, Low[i-1]={df1['Low'].iloc[i - 1]}")
        if df1['Low'].iloc[i] > df1['Low'].iloc[i - 1]:
            return 0

    for i in range(candle + 1, candle + candles_after + 1):
        # Sanity check here
            #print(f"[SUPPORT AFTER] i={i}, i-1={i-1}, Low[i]={df1['Low'].iloc[i]}, Low[i-1]={df1['Low'].iloc[i - 1]}")
        if df1['Low'].iloc[i] < df1['Low'].iloc[i - 1]:
            return 0

    return 1


def resistance(df1, candle, candles_before, candles_after):
    if candle - candles_before < 0 or candle + candles_after >= len(df1):
        return 0

    for i in range(candle - candles_before + 1, candle + 1):
            #print(f"[RESISTANCE BEFORE] i={i}, i-1={i-1}, High[i]={df1['High'].iloc[i]}, High[i-1]={df1['High'].iloc[i - 1]}")
        if df1['High'].iloc[i] < df1['High'].iloc[i - 1]:
            return 0

    for i in range(candle + 1, candle + candles_after + 1):
            #print(f"[RESISTANCE AFTER] i={i}, i-1={i-1}, High[i]={df1['High'].iloc[i]}, High[i-1]={df1['High'].iloc[i - 1]}")
        if df1['High'].iloc[i] > df1['High'].iloc[i - 1]:
            return 0

    return 1

#determine variable for functions
    #find a cleaner way to do this later

length = len(dataF)
high = dataF['High'].to_list()
low = dataF['Low'].to_list()
close_ = dataF['Close'].to_list()
open_ = dataF['Open'].to_list()

bodydiff, highdiff, lowdiff, ratio1, ratio2 = ([0] * length for _ in range(5))

bodydiffmin = 0.002

def engulfing(candle):
    bodydiff[candle] = abs(open_[candle] - close_[candle])
    if bodydiff[candle]<0.000001:
        bodydiff[candle] = 0.000001

    if (bodydiff[candle] < bodydiffmin < bodydiff[candle - 1] and
            open_[candle - 1] < close_[candle - 1] and
            open_[candle] > close_[candle] and
            (open_[candle] - close_[candle - 1] >= -0e-5 and
            close_[candle] < open_[candle - 1])):
        return 1

    elif (bodydiff[candle] > bodydiffmin and bodydiff[candle - 1] > bodydiffmin and
            open_[candle - 1] > close_[candle - 1] and
            open_[candle] < close_[candle] and
            (open_[candle] - close_[candle - 1]) <= +0e-5 and
            close_[candle] > open_[candle - 1]):
        return 2
    else:
        return 0

def star(candle):
    highdiff[candle] = high[candle] - max(open_[candle], close_[candle])
    lowdiff[candle] = min(open_[row], close_[candle]) - low[candle]
    bodydiff[candle] = abs(open_[candle] - close_[candle])
    if bodydiff[candle] < 0.000001:
        bodydiff[candle] = 0.000001

    ratio1[candle] = highdiff[candle] / bodydiff[candle]
    ratio2[candle] = lowdiff[candle] / bodydiff[candle]

    if ratio1[candle] <= 1 or lowdiff[candle] >= 0.2 * highdiff[candle] or bodydiff[candle] <= bodydiffmin:
        return 1
    elif ratio2[candle] > 1 and highdiff[candle] < 0.2 * lowdiff[candle] and bodydiff[candle] > bodydiffmin:
        return 2
    else: return 0

def close_resistance(candle, levels, lim): #lim = how close to touching a level
    if len(levels) == 0:
        return 0
    c1 = abs(dataF.high[candle] - min(levels,
                                      key=lambda x:abs(x-dataF.high[candle])))<=lim
    c2 = abs(max(dataF.open_[candle],
                 dataF.close_[candle]) - min(levels,
                                             key=lambda x:abs(x-dataF.low[candle])))<=lim
    c3 = max(dataF.open_[candle],
             dataF.close_[candle])>min(levels,
                                       key=lambda x:abs(x-dataF.low[candle]))
    c4 = dataF.low[candle] < min(levels,
                                 key=lambda x:abs(x-dataF.high[candle]))

    if(c1 or c2) and c3 and c4:
        return 1
    else:
        return 0

def close_support(candle, levels, lim):
    if len(levels) == 0:
        return 0

    c1 = abs(dataF.low[candle] - min(levels,
                                     key=lambda x:abs(x-dataF.low[candle])))<=lim
    c2 = abs(min(dataF.open_[candle],
                 dataF.close_[candle]) - min(levels,
                                             key=lambda x:abs(x-dataF.low[candle])))<=lim
    c3 = max(dataF.open_[candle],
             dataF.close_[candle])>min(levels,
                                       key=lambda x:abs(x-dataF.low[candle]))
    c4 = dataF.high[candle] > min(levels,
                                  key=lambda x:abs(x-dataF.low[candle]))

    if (c1 or c2) and c3 and c4:
        return 1
    else:
        return 0

#Support and resistance levels


back_candles = 45
signal = [0] * length

n1=3 #Change these to see how many candles you wanna go back
n2=2 #Or forward

#Test Support Levels
#Less candles = less sensitive

for row in range(back_candles, len(dataF) - n2):
    support_levels = []
    resistance_levels = []

    for subrow in range(row-back_candles + n1, row + 1):
        if support(dataF,subrow,n1,n2):
            support_levels.append(dataF.Low[subrow])

        if resistance(dataF,subrow,n1,n2):
            resistance_levels.append(dataF.High[subrow])

    if engulfing(row) == 1 or star(row) and close_resistance(row, resistance_levels, 150e-5):
        signal[row] = 1
    elif engulfing(row) or star(row) and close_support(row, support_levels, 150e-5):
        signal[row] = 2
    else:
        signal[row] = 0
