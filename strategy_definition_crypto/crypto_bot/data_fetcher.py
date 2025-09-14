import pandas as pd
import ccxt
import time
import os


class DataFetcher:
    """
    A class to fetch and preprocess OHLCV stock data either from CSV files or live
    via Alpha Vantage API.

    Attributes
    ----------
    symbol : str
        Stock ticker symbol.
    api_key : str
        API key for Alpha Vantage.
    files_paths : dict
        Dictionary mapping timeframes to local CSV file paths.
    market_timeframe : str
        Market session filter: 'regular', 'pre', 'post', 'full', or other custom modes.
    """

    def __init__(
        self,
        symbol: str,
        api_key: str,
        files_paths: dict = None,
        start_datetime: str = None,
        end_date: str = None,
    ) -> None:
        """
        Initialize the DataFetcher.

        Parameters
        ----------
        symbol : str
            The stock symbol to fetch data for.
        api_key : str
            Your Alpha Vantage API key.
        files_paths : dict
            Dictionary of file paths for backtesting data by timeframe.
        market_timeframe : str, optional
            Defines whether to use 'regular' (9:30 AM to 4:00 PM ET),
            'pre_regular' (4:00 AM to 4:00 PM ET), 'full' (4:00 AM to 8:00 PM ET).
            Default is 'regular'.
        """
        self.symbol = symbol
        self.api_key = api_key
        self.files_paths = files_paths
        self.start_datetime = start_datetime
        self.end_date = end_date

    def build_1min_dataset(
        self,
    ):

        binance = ccxt.binance()
        symbol = f"{self.symbol}/USDT"  # Change to your preferred trading pair
        since = binance.parse8601(self.start_datetime)  # Start date
        end_time = binance.parse8601(self.end_date)  # End date
        # since = binance.parse8601("2025-09-07T06:00:00Z")  # Start date
        # end_time = binance.parse8601("2025-09-07T12:00:00Z")  # End date
        all_ohlcv = []
        while since < end_time:
            try:
                candles = binance.fetch_ohlcv(
                    symbol, timeframe="1m", since=since, limit=1000
                )
            except Exception as e:
                print(f"Error fetching candles: {e}")
                break
            if not candles:
                break
            all_ohlcv.extend(candles)
            since = candles[-1][0] + 60_000  # Move forward by 1 minute
            time.sleep(1)  # Respect rate limits

        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["datetime", "open", "high", "low", "close", "volume"]]
        df.set_index("datetime", inplace=True)

        all_data = df.sort_index().copy()

        return all_data

    def fetch_all_timeframes(self) -> dict:
        """
        Fetch OHLCV data for all required timeframes, either from local CSV files
        or by building the dataset from Alpha Vantage.
        If local files are provided, it reads them directly. Otherwise, it fetches
        the data from Alpha Vantage and aggregates it into the required timeframes.
        Parameters
        ----------
        target_days : int
            The number of unique trading days to include in the dataset.
        Returns
        -------
        dict
            Dictionary containing OHLCV DataFrames for various timeframes, including:
            - "1min", "3min", "5min", "1h", "4h"
            - "latest": most recent 1min candle as dictionary
        """
        path_1min = self.files_paths.get("1min")

        df_1min = self.build_1min_dataset()

        end_date = self.end_date.replace("T", " ").replace("Z", "")
        end_date = pd.to_datetime(end_date)
        df_1min = df_1min[df_1min.index <= end_date]

        try:
            df_3min = self.aggregate_ohlcv(df_1min, "3min")
            df_5min = self.aggregate_ohlcv(df_1min, "5min")
            df_1h = self.aggregate_ohlcv(df_1min, "1h")
            df_4h = self.aggregate_ohlcv(df_1min, "4h")
        except Exception as e:
            print(f"Error during aggregation: {e}")
            raise

        if not os.path.exists(path_1min):
            file_path = self.files_paths.get("1min")
            if file_path:
                df_1min.to_csv(file_path)

        """
        print("Here are the lenghts of the dataframes:")
        print(
            f"1min: {len(df_1min)}, 3min: {len(df_3min)}, "
            f"5min: {len(df_5min)}, 1h: {len(df_1h)}, 4h: {len(df_4h)}"
        )
        """
        print("fetch_all_timeframes")
        return {
            "1min": df_1min,
            "3min": df_3min,
            "5min": df_5min,
            "1h": df_1h,
            "4h": df_4h,
            "latest": df_1min.iloc[-1].to_dict(),
        }

    def aggregate_ohlcv(self, base_df: pd.DataFrame, new_interval: str) -> pd.DataFrame:
        """
        Aggregate raw OHLCV data to a new time interval.

        Parameters
        ----------
        base_df : pd.DataFrame
            Base dataframe (e.g., 1min candles).
        new_interval : str
            Target aggregation interval (e.g., '3min', '1h').

        Returns
        -------
        pd.DataFrame
            Aggregated OHLCV dataframe.
        """
        df = base_df.copy()
        df.index = pd.to_datetime(df.index)
        resampled = (
            df.resample(new_interval)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        candle_count = df.resample(new_interval).size()
        resampled["candle_count"] = candle_count.reindex(resampled.index)

        return resampled
