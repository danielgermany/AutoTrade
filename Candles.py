from Candle import Candle

class Candles:
    def __init__(self,df):
        self.df = df.reset_index()
        self.candles = [self.row_to_candle(row) for _, row in self.df.iterrows()]

    def row_to_candle(self,row):
        return Candle(
            timestamp = row['Datetime'] if 'Datetime' in row else row.name,
            open_ = row['Open'],
            high = row['High'],
            low = row['Low'],
            close = row['Close'],
            volume = row['Volume'],
        )

    def __getitem__(self,idx):
        return self.candles[idx]

    def __len__(self):
        return len(self.candles)

    def slice(self,start,end):
        return self.candles[start:end]

    def __repr__(self):
        return (f"Candles({len(self.candles)} candles from "
                f"{self.candles[0]} to {self.candles[-1].timestamp})")
