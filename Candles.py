# Import the Candle class from the Candle module.
from Candle import Candle

# Define a class to manage and operate on a list of Candle objects
class Candles:
    def __init__(self, df):
        """
        Initializes the Candles object with a DataFrame of OHLCV data.

        Args:
            df (pandas.DataFrame): A DataFrame containing at least the columns:
                ['Open', 'High', 'Low', 'Close', 'Volume'] and optionally 'Datetime'.
        """
        # Reset the index so that 'Datetime' becomes a column (if it was the index)
        self.df = df.reset_index()

        # Convert each row in the DataFrame into a Candle object and store in a list
        self.candles = [self.row_to_candle(row) for _, row in self.df.iterrows()]

    def row_to_candle(self, row):
        """
        Converts a single row of the DataFrame into a Candle object.

        Args:
            row (pandas.Series): A single row of OHLCV data.

        Returns:
            Candle: A Candle object created from the row data.
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
        """
        return self.candles[idx]

    def __len__(self):
        """
        Returns the number of Candle objects stored.

        Returns:
            int: The total number of candles.
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
        """
        return self.candles[start:end]

    def __repr__(self):
        """
        Returns a human-readable string representation of the Candles object,
        showing the number of candles and the date range covered.

        Returns:
            str: Summary of the Candles object.
        """
        return (f"Candles({len(self.candles)} candles from "
                f"{self.candles[0]} to {self.candles[-1].timestamp})")
