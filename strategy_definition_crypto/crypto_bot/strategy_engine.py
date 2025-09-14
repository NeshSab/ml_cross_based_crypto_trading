"""
Flow Summary:
1. Fetch data (1m, 3m, 5m, 1h, 4h)
2. Generate short / medium / long term signals
3. Check for valid bullish/bearish signal:
   - Volume
   - RSI
   - ADX
   - DI alignment
4. If confirmations are valied, place order (buy/sell) if risk conditions satisfied
5. Monitor for exit signal or reversal
6. Log result and prepare for next signal
"""

from datetime import datetime, time as dtime
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD
import pandas as pd
import numpy as np
from scipy.stats import linregress


from strategy_definition_crypto.crypto_bot.strategy_params import (
    StrategyParams,
)


class StrategyEngine:
    """
    Executes a trading strategy using short-term and medium-term EMA crossover signals,
    confirmed by multi-indicator analysis and long-term trend alignment.

    Attributes
    ----------
    data_fetcher : object
        Responsible for retrieving market data.
    signal_generator : object
        Evaluates EMA crossovers and confirmations.
    risk_manager : object
        Assesses whether a trade should be placed based on risk rules.
    broker : object
        Places orders through the brokerage API.
    """

    def __init__(
        self,
        data_fetcher: object,
        signal_generator: object,
        risk_manager: object,
        broker: object,
        params: StrategyParams,
        trade_logger: object = None,
    ) -> None:
        self.data_fetcher = data_fetcher
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.broker = broker
        self.params = params
        self.trade_logger = trade_logger

        self.term_trade_type = params.term_trade
        self.fast_window = params.fast_slow_windows[0]
        self.slow_window = params.fast_slow_windows[1]
        self.atr_multiplier = params.atr_multiplier
        self.atr_window = params.atr_window
        self.end_of_day_min = 55
        self.stop_placing_orders_min = 40
        self.start_placing_orders_min = 45
        self.close_end_of_day = True

    @staticmethod
    def add_indicator_columns(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        df = df.sort_index()

        df["ema_fast"] = EMAIndicator(
            df["close"], window=params.fast_slow_windows[0]
        ).ema_indicator()
        df["ema_slow"] = EMAIndicator(
            df["close"], window=params.fast_slow_windows[1]
        ).ema_indicator()
        df["rsi"] = RSIIndicator(
            df["close"], window=params.confirmation_indicator_window
        ).rsi()
        adx = ADXIndicator(
            df["high"],
            df["low"],
            df["close"],
            window=params.confirmation_indicator_window,
        )
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()
        df["atr"] = AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=params.atr_window,
        ).average_true_range()

        macd = MACD(
            df["close"],
            window_slow=params.macd_slow,
            window_fast=params.macd_fast,
            window_sign=params.macd_signal,
        )
        df["macd_line"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        bb = BollingerBands(
            df["close"], window=params.bb_window, window_dev=params.bb_dev
        )
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()

        return df

    def calculate_atr_stop_loss(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
    ) -> float:
        """
        Calculate an ATR-based stop-loss level.

        Parameters
        ----------
        df : pd.DataFrame
            The price data to calculate ATR from.
        direction : str
            Trade direction, 'bullish' or 'bearish'.
        entry_price : float
            The price at which the trade was entered.
        multiplier : float, optional
            Multiplier for ATR to adjust the stop-loss distance, by default 1.5.

        Returns
        -------
        float
            Calculated stop-loss level.
        """
        multiplier = self.atr_multiplier
        atr = df["atr"].iloc[-1]
        if direction == "bullish":
            stop = entry_price - (multiplier * atr)
        elif direction == "bearish":
            stop = entry_price + (multiplier * atr)
        else:
            raise ValueError("Invalid direction for stop-loss calculation")
        return stop, atr

    def backtest_strategy_modular(
        self,
        start_time: str,
        trigger_price_multiplier: float = 1.005,
        total_portfolio: float = 10000,
        play_size: float = 3000,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Modular backtest strategy.
        """
        trades = []
        start_time = start_time.replace("T", " ").replace("Z", "")
        start_time = pd.to_datetime(start_time) + pd.DateOffset(days=1)

        historical_data = self.prepare_historical_data()
        df = historical_data["1min"]
        filtered_index = df.index[df.index >= start_time]

        if filtered_index.empty:
            print(f"No available data at or after start_time: {start_time}.")
            return pd.DataFrame(), {}

        start_index = df.index.get_indexer_for(filtered_index)[0]

        for i in range(start_index, len(df)):
            window = df.iloc[: i + 1]
            signal_info = self.generate_signal_for_index(i, window, historical_data)
            if not signal_info:
                continue

            (
                signal_type,
                active_signal,
                fast_df,
                confirmation_df,
                future_window_df,
                fast_df_count,
                confirmation_df_count,
            ) = signal_info
            confirmation = self.confirm_signal(
                signal_type, confirmation_df, active_signal
            )
            # print(f"Confirmation for {signal_type}: {confirmation}")
            if not confirmation.get("valid", False):
                trades.append(
                    self.log_not_confirmed_trade(
                        window,
                        signal_type,
                        active_signal,
                        fast_df_count,
                        confirmation_df_count,
                        confirmation,
                    )
                )
                continue

            signal_detection_time, signal_detection_price = (
                window.index[-1],
                window["open"].iloc[-1],
            )

            trigger_price = signal_detection_price * trigger_price_multiplier
            shares_count = play_size / trigger_price

            stop_loss = self.calculate_initial_stop_loss(
                trigger_price,
                total_portfolio,
                shares_count,
                confirmation_df,
                active_signal,
            )

            trade = self.simulate_trade(
                df,
                i,
                signal_detection_price,
                signal_detection_time,
                trigger_price,
                stop_loss,
                active_signal,
                signal_type,
                future_window_df,
                fast_df_count,
                confirmation_df_count,
                confirmation,
            )
            if trade:
                trades.append(trade)

        trades_df = pd.DataFrame(trades)
        return trades_df

    def prepare_historical_data(self):
        historical_data = self.data_fetcher.fetch_all_timeframes()
        if self.term_trade_type == "long":
            # need to leave only 1h and 4h data
            historical_data = {
                "1min": historical_data["1min"],
                "1h": historical_data["1h"],
                "4h": historical_data["4h"],
            }
        elif self.term_trade_type == "medium":
            # need to leave only 3min, 5min and 1h data
            historical_data = {
                "1min": historical_data["1min"],
                "3min": historical_data["3min"],
                "5min": historical_data["5min"],
            }
        elif self.term_trade_type == "short":
            # need to leave only 1min, 3min and 5min data
            historical_data = {
                "1min": historical_data["1min"],
                "3min": historical_data["3min"],
            }
        for tf, df in historical_data.items():
            if isinstance(df, pd.DataFrame):
                historical_data[tf] = self.add_indicator_columns(df, self.params)
        return historical_data

    def generate_signal_for_index(self, i, window, historical_data):
        signal_time = window.index[-1]
        signal_hour_min = signal_time.time()
        signal_type = None
        active_signal = None
        fast_df = None
        confirmation_df = None
        future_window_df = None
        fast_df_count = []
        confirmation_df_count = []

        if not (
            dtime(9, self.start_placing_orders_min)
            < signal_hour_min
            < dtime(15, self.stop_placing_orders_min)
        ):
            return None

        signal_minute = signal_time.minute
        signal_hour = signal_time.hour

        if (
            signal_minute == 0
            and signal_hour % 4 == 0
            and self.term_trade_type == "long"
        ):
            fast_df = historical_data["1h"][historical_data["1h"].index < signal_time]
            confirmation_df = historical_data["4h"][
                historical_data["4h"].index < signal_time
            ]
            if len(fast_df) >= (self.slow_window + 1) and len(confirmation_df) >= (
                self.fast_window + 1
            ):
                long_term_signal = self.signal_generator.evaluate_ema_crossover(
                    fast_df, confirmation_df
                )
                if long_term_signal:
                    signal_type = "long"
                    active_signal = long_term_signal
                    future_window_df = historical_data["4h"]
                    fast_df_count = fast_df["candle_count"][-4:].tolist()
                    confirmation_df_count = confirmation_df["candle_count"][
                        -4:
                    ].tolist()

        elif signal_minute % 5 == 0 and self.term_trade_type == "medium":
            confirmation_df = historical_data["5min"][
                historical_data["5min"].index < signal_time
            ]
            fast_df = historical_data["3min"][
                historical_data["3min"].index < signal_time
            ]
            if signal_minute % 15 != 0:
                fast_df = fast_df[:-1]  # Exclude the last candle for 3min data

            medium_term_signal = self.signal_generator.evaluate_ema_crossover(
                fast_df, confirmation_df
            )
            if medium_term_signal:
                signal_type = "medium"
                active_signal = medium_term_signal
                future_window_df = historical_data["5min"]
                fast_df_count = fast_df["candle_count"][-4:].tolist()
                confirmation_df_count = confirmation_df["candle_count"][-4:].tolist()

        elif signal_minute % 3 == 0 and self.term_trade_type == "short":
            fast_df = historical_data["1min"][
                historical_data["1min"].index < signal_time
            ]
            confirmation_df = historical_data["3min"][
                historical_data["3min"].index < signal_time
            ]
            short_term_signal = self.signal_generator.evaluate_ema_crossover(
                fast_df, confirmation_df
            )
            if short_term_signal:
                signal_type = "short"
                active_signal = short_term_signal
                future_window_df = historical_data["3min"]
                fast_df_count = [1, 1, 1, 1]
                confirmation_df_count = confirmation_df["candle_count"][-4:].tolist()

        if signal_type:
            return (
                signal_type,
                active_signal,
                fast_df,
                confirmation_df,
                future_window_df,
                fast_df_count,
                confirmation_df_count,
            )
        return None

    def confirm_signal(self, signal_type, confirmation_df, active_signal):
        confirmation = self.signal_generator.check_confirmations(
            df=confirmation_df, trade_signal=active_signal
        )
        # if signal_type in ["long"]:
        #    confirmation["valid"] = True

        return confirmation

    def log_not_confirmed_trade(
        self,
        window,
        signal_type,
        active_signal,
        fast_df_count,
        confirmation_df_count,
        confirmation,
        mod_count=0,
    ):
        entry_time = window.index[-1]
        entry_price = window["open"].iloc[-1]
        return {
            "signal_detection_time": entry_time,
            "entry_time": entry_time,
            "exit_time": entry_time,
            "direction": active_signal,
            "term_trade_type": signal_type,
            "confirmation": "not_confirmed",
            "signal_detected_price": entry_price,
            "entry_price": -999,
            "exit_price": -999,
            "mod_count": mod_count,
            "fast_df_count": fast_df_count,
            "confirmation_df_count": confirmation_df_count,
            "conf_indicators": confirmation["indicator_dictionary"],
        }

    def calculate_initial_stop_loss(
        self,
        trigger_price,
        total_portfolio,
        shares_count,
        confirmation_df,
        active_signal,
    ):

        stop_loss_atr, _ = self.calculate_atr_stop_loss(
            confirmation_df,
            active_signal,
            trigger_price,
        )
        if active_signal == "bullish":
            stop_loss_pct = trigger_price - (
                ((total_portfolio * 0.01) - 2) / shares_count
            )
        else:
            stop_loss_pct = trigger_price + (
                ((total_portfolio * 0.01) - 2) / shares_count
            )
        if self.term_trade_type == "long":
            print(
                f"\nInitial stop loss for long trade: {stop_loss_pct}, "
                f"ATR-based stop loss: {stop_loss_atr}"
            )
        return (
            max(stop_loss_pct, stop_loss_atr)
            if active_signal == "bullish"
            else min(stop_loss_pct, stop_loss_atr)
        )

    def adjust_stop_loss(
        self,
        future_window_df,
        entry_time,
        datetime_j,
        active_signal,
        stop_loss,
        mod_count,
    ):
        """
        Adjusts the stop loss using ATR trailing logic,
        including future_window calculation.
        """
        # Determine offset based on trade type
        if self.term_trade_type == "short":
            offset = pd.Timedelta(minutes=3)
        elif self.term_trade_type == "medium":
            offset = pd.Timedelta(minutes=5)
        elif self.term_trade_type == "long":
            offset = pd.Timedelta(hours=4)
            # entry_time = entry_time.replace(minute=0, second=0, microsecond=0)
            # entry_time = entry_time - offset
        else:
            offset = pd.Timedelta(0)

        future_window = future_window_df[future_window_df.index <= datetime_j - offset]
        relevant_window = future_window[future_window.index >= entry_time]

        if relevant_window.empty:
            return stop_loss, mod_count
        else:
            if active_signal == "bullish":
                anchor_price = relevant_window["high"].max()
            else:
                anchor_price = relevant_window["low"].min()

        trailing_stop, _ = self.calculate_atr_stop_loss(
            future_window, active_signal, anchor_price
        )

        if (active_signal == "bullish" and trailing_stop > stop_loss) or (
            active_signal == "bearish" and trailing_stop < stop_loss
        ):
            return trailing_stop, mod_count + 1
        return stop_loss, mod_count

    def simulate_trade(
        self,
        df,
        i,
        signal_detection_price,
        signal_detection_time,
        trigger_price,
        stop_loss,
        active_signal,
        signal_type,
        future_window_df,
        fast_df_count,
        confirmation_df_count,
        confirmation,
    ):
        flag = "not_triggered"
        mod_count = 0
        position = 0
        entry_time = None
        for j in range(i, len(df)):
            end_of_day = (
                (
                    df.index[j].time().hour == 15
                    and df.index[j].time().minute >= self.end_of_day_min
                )
                if self.close_end_of_day
                else df.index[j] == df.index[-1]
            )

            stop_placing_orders = (
                (
                    df.index[j].time().hour == 15
                    and df.index[j].time().minute >= self.stop_placing_orders_min
                )
                if self.stop_placing_orders_min
                else False
            )

            if position == 0:
                if not stop_placing_orders:
                    highest_price = df.iloc[j]["high"]
                    if trigger_price < highest_price:
                        position = 1
                        flag = "confirmed"
                        entry_time = df.index[j]
                else:
                    try:
                        conf_indicators = confirmation.get(
                            "indicator_dictionary", "empty"
                        )
                    except Exception as e:
                        print(f"Error accessing confirmation indicators: {e}")

                    return {
                        "signal_detection_time": signal_detection_time,
                        "entry_time": entry_time,
                        "exit_time": None,
                        "direction": active_signal,
                        "term_trade_type": signal_type,
                        "confirmation": flag,  # "not_triggered",
                        "signal_detected_price": signal_detection_price,
                        "entry_price": None,
                        "exit_price": None,
                        "mod_count": mod_count,
                        "fast_df_count": fast_df_count,
                        "confirmation_df_count": confirmation_df_count,
                        "conf_indicators": conf_indicators,
                    }
            else:
                current_price = df.iloc[j]["low"]
                datetime_j = df.index[j]
                """
                if signal_type == "short":
                    min_candle_count = 4
                elif signal_type == "medium":
                    min_candle_count = 12
                else:  # signal_type == "long"
                    min_candle_count = 180

                if j - i == min_candle_count:
                    # Check if price is rising gradually over the last 5 candles
                    # before checking stop loss
                    is_price_rising = (
                        self.is_price_rising_gradually_over_last_5_candles(
                            active_signal, df.iloc[i : j + 1], trigger_price
                        )
                    )
                    if not is_price_rising:
                        exit_price = df.iloc[j]["open"]
                        try:
                            conf_indicators = confirmation.get(
                                "indicator_dictionary", "empty"
                            )
                        except Exception as e:
                            print(f"Error accessing confirmation indicators: {e}")
                        
                        return {
                            "signal_detection_time": signal_detection_time,
                            "entry_time": entry_time,
                            "exit_time": df.index[j],
                            "direction": active_signal,
                            "term_trade_type": signal_type,
                            "confirmation": flag,  # "confirmed",
                            "signal_detected_price": signal_detection_price,
                            "entry_price": trigger_price,
                            "exit_price": exit_price,
                            "mod_count": mod_count,
                            "fast_df_count": fast_df_count,
                            "confirmation_df_count": confirmation_df_count,
                            "conf_indicators": conf_indicators,
                            "prev_low": df.iloc[j - 1]["low"],
                            "now_low_current_price": current_price,
                            "next_low": "exited_early",
                        }
                """
                # print(f"Current price: {current_price}, Stop loss: {stop_loss}")
                stop_hit = (
                    active_signal == "bullish" and current_price < stop_loss
                ) or (active_signal == "bearish" and current_price > stop_loss)
                if stop_hit or end_of_day:

                    stop_loss = (
                        df.iloc[j]["high"]
                        if stop_loss > df.iloc[j]["high"]
                        else stop_loss
                    )
                    exit_price = stop_loss if stop_hit else df.iloc[j]["open"]
                    try:
                        conf_indicators = confirmation.get(
                            "indicator_dictionary", "empty"
                        )
                    except Exception as e:
                        print(f"Error accessing confirmation indicators: {e}")
                    return {
                        "signal_detection_time": signal_detection_time,
                        "entry_time": entry_time,
                        "exit_time": df.index[j],
                        "direction": active_signal,
                        "term_trade_type": signal_type,
                        "confirmation": flag,  # "confirmed",
                        "signal_detected_price": signal_detection_price,
                        "entry_price": trigger_price,
                        "exit_price": exit_price,
                        "mod_count": mod_count,
                        "fast_df_count": fast_df_count,
                        "confirmation_df_count": confirmation_df_count,
                        "conf_indicators": conf_indicators,
                        "prev_low": df.iloc[j - 1]["low"],
                        "now_low_current_price": current_price,
                        "next_low": df.iloc[j + 1]["low"] if j + 1 < len(df) else None,
                    }

                stop_loss, mod_count = self.adjust_stop_loss(
                    future_window_df,
                    entry_time,
                    datetime_j,
                    active_signal,
                    stop_loss,
                    mod_count,
                )

        return None

    def is_price_rising_gradually_over_last_5_candles(
        self,
        active_signal: str,
        df: pd.DataFrame,
        trigger_price: float,
    ) -> bool:
        """
        Check if the price is rising by a certain threshold.
        Also check if slope of the last 5 candles is positive for bullish trades.

        """
        price = df["close"].iloc[-1]
        slope = self.calculate_slope(df["close"])
        print(
            f"Price: {price}, Trigger Price: {trigger_price}, Slope: {slope}, "
            f"Active Signal: {active_signal}"
        )
        if active_signal == "bullish":
            if price > trigger_price or slope > 0:
                return True
        elif active_signal == "bearish":
            if price < trigger_price or slope < 0:
                return True
        return False

    @staticmethod
    def calculate_slope(
        series: pd.Series,
    ) -> float:
        """
        Calculate the normalized slope of a pandas Series over a specified window.

        Parameters
        ----------
        series : pd.Series
            Time series data such as RSI, EMA, or price.
        window : int, optional
            Number of most recent points to include in slope calculation. Default is 5.
        normalize : {'last', 'mean', None}, optional
            Method used to normalize the slope value:
            - 'last': Normalize by the last value in the window.
            - 'mean': Normalize by the mean value in the window.
            - None: Return the raw slope value.
            Default is 'last'.

        Returns
        -------
        float
            The normalized slope of the series. Returns np.nan if insufficient data.

        Notes
        -----
        This method uses linear regression to calculate the slope over a rolling window.
        Normalizing the slope allows for comparison across different price levels
        and indicators.
        """

        y = series.values
        x = np.arange(len(series))

        slope, _, _, _, _ = linregress(x, y)

        return slope / y[-1]
