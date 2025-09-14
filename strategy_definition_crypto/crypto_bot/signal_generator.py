import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Literal
from strategy_definition_crypto.crypto_bot.strategy_params import (
    StrategyParams,
)


class SignalGenerator:
    """
    Generates trading signals and confirms them using technical indicators.

    Methods
    -------
    evaluate_ema_crossover(fast_df, confirm_df, fast, slow)
        Checks for bullish or bearish EMA crossover signals,
        confirmed on a higher timeframe.

    evaluate_long_term_trend(df_1h, df_4h)
        Evaluates long-term trend direction based on EMA 50/200 crossovers
        on 1h and 4h charts.

    check_confirmations(df, trend_bias, trade_signal)
        Validates a trade signal using multiple momentum and volume indicators.

    calculate_slope(series, window, normalize)
        Computes a normalized slope for a time series using linear regression over
        a lookback window.
    """

    def __init__(
        self,
        params: StrategyParams,
        indicators: dict | None = None,
    ):
        """
        Initialize SignalGenerator with customizable indicator toggles.

        Parameters
        ----------
        indicators : dict, optional
            Dictionary specifying which indicators to include in confirmation, e.g.,
            {'rsi': True, 'stoch': True, 'adx': True, 'di': True,
            'volume': True, 'bias': True}
        """
        default_indicators = {
            "rsi": True,
            "rsi_trend_reverse": True,
            "adx": True,
            "di": True,
            "volume": True,
            "macd": False,
            "bollinger": False,
            "atr": True,
            "allowed_rejections": 0,
        }
        self.indicators = indicators if indicators else default_indicators
        self.term_trade_signal = params.term_trade
        self.close_slope_threshold = params.close_slope_threshold
        self.confirmation_slope_threshold = params.confirmation_slope_threshold
        self.close_slope_window = params.close_slope_window
        self.confirmation_slope_window = params.confirmation_slope_window
        self.confirmation_indicator_window = params.confirmation_indicator_window
        self.rsi_value = params.rsi_value
        self.rsi_X = params.rsi_X
        self.rsi_slope_threshold = params.rsi_slope_threshold
        self.rsi_overbought = params.rsi_overbought
        self.rsi_oversold = params.rsi_oversold
        self.adx_threshold = params.adx_threshold
        self.atr_multiplier = params.atr_multiplier
        self.atr_window = params.atr_window

    def evaluate_ema_crossover(
        self,
        fast_df: pd.DataFrame,  # 1m data with ['close','ema_fast','ema_slow','atr'] (atr optional)
        confirm_df: pd.DataFrame,  # 3m data with ['ema_fast'] aligned to completed bars
        lookback_bars: int = 3,  # search window on 1m for a recent cross
        persistence_bars: int = 0,  # require fast>slow for N bars after the cross
        eps: float = 0.0,  # tolerance around equality at the cross
        min_delta_k_atr: float = 0.0,  # require fast - slow >= k * ATR after cross (0 = disabled)
        use_ema_slope_only: bool = True,
        slope_window_fast: int = 3,  # slope window on 1m
        slope_norm: str = "mean",  # 'range'|'mean'|'none'
        slope_threshold: float = 0.0,  # lower than your current threshold to admit more signals
        confirm_slope_window: int = 2,
        confirm_slope_threshold: float = 0.0005,
    ) -> str | None:
        """
        This method is meant just for bullish plays.
        More permissive EMA cross:
        - detects a cross within the last `lookback_bars` 1m candles
        - optional persistence requirement
        - optional ATR-proportional delta filter
        - single (or softer) slope confirmations
        """

        def slope(series: pd.Series, window: int, norm: str) -> float:
            s = series.dropna().iloc[-window:]
            if len(s) < window:
                return 0.0
            m = (s.iloc[-1] - s.iloc[0]) / (window - 1)
            if norm == "range":
                rng = s.max() - s.min()
                return m / (rng if rng != 0 else 1.0)
            elif norm == "mean":
                mu = abs(s.mean())
                return m / (mu if mu != 0 else 1.0)
            return m

        if self.term_trade_signal in ["short", "medium"]:
            lookback_bars = 5
        else:
            lookback_bars = 3

        if self.term_trade_signal in ["short", "medium"]:
            persistence_bars = 1

        last_row_atr = fast_df.iloc[-1]["atr"]
        eps = 0.01 * last_row_atr if last_row_atr > 0 else 0.0

        fe = fast_df["ema_fast"]
        se = fast_df["ema_slow"]

        # 1) Find a recent bullish cross within last N bars
        cross_mask = (fe.shift(1) <= se.shift(1) + eps) & (fe > se - eps)
        recent = cross_mask.tail(lookback_bars)
        if not recent.any():
            return None

        # Pick the most recent cross index
        cross_idx = recent[recent].index[-1]

        if persistence_bars > 0:
            post = fast_df.loc[cross_idx:].head(persistence_bars + 1)
            # Only apply persistence check if we have enough data
            if len(post) >= persistence_bars + 1:
                if not (post["ema_fast"] > post["ema_slow"]).all():
                    return None

        # 3) Optional ATR-proportional delta after cross (more forgiving than instant acceleration)
        if min_delta_k_atr > 0 and "atr" in fast_df.columns:
            last_row = fast_df.iloc[-1]
            delta = last_row["ema_fast"] - last_row["ema_slow"]
            if last_row["atr"] > 0 and delta < min_delta_k_atr * last_row["atr"]:
                return None

        # 4) Slope confirmation (prefer EMA slope; drop double-gating)
        if use_ema_slope_only:
            s_fast = slope(fe, slope_window_fast, slope_norm)
            if s_fast <= slope_threshold:
                return None
        else:
            s_close = slope(fast_df["close"], slope_window_fast, slope_norm)
            if s_close <= slope_threshold:
                return None

        # 5) 3m confirmation slope (soften threshold)
        ce = confirm_df["ema_fast"].dropna()
        if len(ce) >= confirm_slope_window:
            conf_s = slope(
                ce,
                confirm_slope_window,
                "mean" if slope_norm == "none" else slope_norm,
            )
            if conf_s <= confirm_slope_threshold:
                return None

        return "bullish"

    def check_confirmations(
        self,
        df: pd.DataFrame,
        trade_signal: str | None,
    ) -> dict:
        """
        Validate a signal using only the enabled confirmation indicators.

        Parameters
        ----------
        df : pd.DataFrame
            Data used to compute confirmation indicators.
        trade_signal : str or None
            Short/medium-term trade signal to confirm.

        Returns
        -------
        dict
            Dictionary with 'valid': bool and 'direction': str.
        """
        indicator_dictionary = {}
        validation = True
        false_indicators = 0

        adx_value = df["adx"].iloc[-1]
        adx_slope = self.calculate_slope(
            df["adx"], window=self.confirmation_slope_window, normalize="mean"
        )
        if self.indicators.get("adx"):
            if adx_value <= self.adx_threshold:
                validation = False
                false_indicators += 1

        indicator_dictionary["adx"] = (adx_slope, round(adx_value, 1))

        di_plus = df["adx_pos"].iloc[-1]
        di_minus = df["adx_neg"].iloc[-1]
        di_plus_slope = self.calculate_slope(
            df["adx_pos"], window=self.confirmation_slope_window, normalize="mean"
        )
        di_minus_slope = self.calculate_slope(
            df["adx_neg"], window=self.confirmation_slope_window, normalize="mean"
        )
        if self.indicators.get("di"):
            di_confirms = (di_plus > di_minus and trade_signal == "bullish") or (
                di_minus > di_plus and trade_signal == "bearish"
            )
            if not di_confirms:
                validation = False
                false_indicators += 1

        indicator_dictionary["di"] = (
            (round(di_plus_slope, 3), round(di_minus_slope, 3)),
            round(di_plus, 1),
            round(di_minus, 1),
        )

        volume = df["volume"].iloc[-1]
        avg_volume = (
            df["volume"]
            .rolling(window=self.confirmation_indicator_window)
            .mean()
            .iloc[-1]
        )
        volume_slope = self.calculate_slope(
            df["volume"], window=self.confirmation_slope_window, normalize="mean"
        )
        vol_multiplier = 1
        if self.term_trade_signal == "short":
            vol_multiplier = 1.3
        elif self.term_trade_signal == "medium":
            vol_multiplier = 1.2
        elif self.term_trade_signal == "long":
            vol_multiplier = 1.1
        if self.indicators.get("volume"):
            if not (volume > avg_volume * vol_multiplier):
                validation = False
                false_indicators += 1

        indicator_dictionary["volume"] = (
            round(volume_slope, 3),
            (
                round(volume),
                round(avg_volume),
            ),
        )

        rsi_series = df["rsi"]
        rsi = rsi_series.iloc[-1]
        rsi_last = rsi_series.iloc[-self.rsi_X :]
        if self.indicators.get("rsi"):
            rsi_confirms = (rsi > self.rsi_value and trade_signal == "bullish") or (
                rsi < self.rsi_value and trade_signal == "bearish"
            )
            if not rsi_confirms:
                validation = False
                false_indicators += 1
        
        rsi_last_mean = rsi_last.mean()
        rsi_ratio = rsi / rsi_last_mean
        indicator_dictionary["rsi"] = (round(rsi_ratio, 2), round(rsi, 1))
        
        rsi_slope = self.calculate_slope(
            rsi_series, window=self.rsi_X, normalize="mean"
        )
        rsi_long_slope = self.calculate_slope(
            rsi_series, window=self.confirmation_indicator_window, normalize="mean"
        )
        if self.indicators.get("rsi_trend_reverse"):
            reversing = (
                (
                    all(r >= self.rsi_overbought for r in rsi_last)
                    and rsi_slope > self.rsi_slope_threshold
                )
                if trade_signal == "bullish"
                else (
                    all(r <= self.rsi_oversold for r in rsi_last)
                    and abs(rsi_slope) > self.rsi_slope_threshold
                )
            )
            if reversing:
                validation = False
                false_indicators += 1

        indicator_dictionary["rsi_trend_reverse_slope"] = (
            round(rsi_long_slope, 2),
            round(rsi_slope, 2),
        )
        
        indicator_dictionary["rsi_trend_reverse_over"] = (
            round(rsi_last_mean, 1),
            [round(float(x), 1) for x in rsi_last.values],
        )

        """
        macd_line = df["macd_line"].iloc[-1]
        macd_signal = df["macd_signal"].iloc[-1]
        macd_line_prev = df["macd_line"].iloc[-2]
        macd_signal_prev = df["macd_signal"].iloc[-2]
        bullish_macd = (
            trade_signal == "bullish"
            and macd_line > macd_signal
            and macd_line_prev <= macd_signal_prev
        )
        bearish_macd = (
            trade_signal == "bearish"
            and macd_line < macd_signal
            and macd_line_prev >= macd_signal_prev
        )
        if self.indicators.get("macd"):
            if not (bullish_macd if trade_signal == "bullish" else bearish_macd):
                validation = False
                false_indicators += 1

        indicator_dictionary["macd"] = (
            (
                round(macd_line, 3),
                round(macd_signal, 3),
                round(macd_line_prev, 3),
                round(macd_signal_prev, 3),
            ),
            "bullish" if bullish_macd else "bearish",
        )

        price = df["close"].iloc[-1]
        middle = df["bb_middle"].iloc[-1]
        upper = df["bb_upper"].iloc[-1]
        lower = df["bb_lower"].iloc[-1]
        bullish_bb = trade_signal == "bullish" and price > middle and price < lower
        bearish_bb = trade_signal == "bearish" and price < middle and price > upper
        if self.indicators.get("bollinger"):
            if not (bullish_bb if trade_signal == "bullish" else bearish_bb):
                validation = False
                false_indicators += 1

        indicator_dictionary["bollinger"] = (
            (round(price, 2), round(middle, 2), round(lower, 2), round(upper, 2)),
            "bullish" if bullish_bb else "bearish",
        )
        """
        price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1]
        atr_avg = df["atr"].rolling(window=self.atr_window).mean().iloc[-1]
        atr_slope = self.calculate_slope(
            df["atr"], window=self.atr_window, normalize="mean"
        )
        stop_loss_price = (
            (price - self.atr_multiplier * atr)
            if trade_signal == "bullish"
            else (price + self.atr_multiplier * atr)
        )
        stop_loss_to_price_ratio = abs(((stop_loss_price / price) - 1) * 100)
        if self.indicators.get("atr"):
            if not 54 < rsi < 80:
                validation = False
                false_indicators += 1

            if not stop_loss_to_price_ratio > 0.4:
                validation = False
                false_indicators += 1
        indicator_dictionary["atr"] = (
            round(atr_avg, 4),
            (
                round(atr, 4),
                round(atr_slope, 3),
                round(stop_loss_to_price_ratio, 3),
            ),
        )
        validation = False
        if false_indicators <= self.indicators.get("allowed_rejections", 0):
            validation = True

        validation = True
        return {"valid": validation, "indicator_dictionary": indicator_dictionary}

    def check_confirmations_faster(
        self,
        df: pd.DataFrame,
        trade_signal: str | None,
    ) -> dict:
        """
        Validate a signal using only the enabled confirmation indicators.

        Parameters
        ----------
        df : pd.DataFrame
            Data used to compute confirmation indicators.
        trade_signal : str or None
            Short/medium-term trade signal to confirm.

        Returns
        -------
        dict
            Dictionary with 'valid': bool and 'direction': str.
        """
        indicator_dictionary = {}
        if self.indicators.get("adx") or self.indicators.get("di"):
            adx_value = df["adx"].iloc[-1]

        if self.indicators.get("adx"):
            if adx_value <= self.adx_threshold:
                return {"valid": False, "indicator_dictionary": "adx"}
            indicator_dictionary["adx"] = (self.adx_threshold, round(adx_value, 1))

        if self.indicators.get("di"):
            di_plus = df["adx_pos"].iloc[-1]
            di_minus = df["adx_neg"].iloc[-1]
            di_confirms = (di_plus > di_minus and trade_signal == "bullish") or (
                di_minus > di_plus and trade_signal == "bearish"
            )
            if not di_confirms:
                return {"valid": False, "indicator_dictionary": "di"}
            indicator_dictionary["di"] = (
                ("di+ > d-", round(di_plus, 1), round(di_minus, 1))
                if trade_signal == "bullish"
                else ("di- > d+", round(di_minus, 1), round(di_plus, 1))
            )
        if self.indicators.get("volume"):
            volume = df["volume"].iloc[-1]
            avg_volume = (
                df["volume"]
                .rolling(window=self.confirmation_indicator_window)
                .mean()
                .iloc[-1]
            )
            vol_multiplier = 1
            if self.term_trade_signal == "short":
                vol_multiplier = 1.3
            elif self.term_trade_signal == "medium":
                vol_multiplier = 1.2
            elif self.term_trade_signal == "long":
                vol_multiplier = 1.1

            if not (volume > avg_volume * vol_multiplier):
                return {"valid": False, "indicator_dictionary": "volume"}
            indicator_dictionary["volume"] = (
                round(avg_volume * vol_multiplier, 1),
                round(volume, 1),
            )

        if self.indicators.get("rsi") or self.indicators.get("rsi_trend_reverse"):
            rsi_series = df["rsi"]
            rsi = rsi_series.iloc[-1]

        if self.indicators.get("rsi"):
            rsi_confirms = (rsi > self.rsi_value and trade_signal == "bullish") or (
                rsi < self.rsi_value and trade_signal == "bearish"
            )
            if not rsi_confirms:
                return {"valid": False, "indicator_dictionary": "rsi"}
            indicator_dictionary["rsi"] = (self.rsi_value, round(rsi, 1))

        if self.indicators.get("rsi_trend_reverse"):
            rsi_last = rsi_series.iloc[-self.rsi_X :]
            rsi_slope = self.calculate_slope(
                rsi_series, window=self.rsi_X, normalize="mean"
            )
            reversing = (
                (
                    all(r >= self.rsi_overbought for r in rsi_last)
                    and rsi_slope > self.rsi_slope_threshold
                )
                if trade_signal == "bullish"
                else (
                    all(r <= self.rsi_oversold for r in rsi_last)
                    and abs(rsi_slope) > self.rsi_slope_threshold
                )
            )
            if reversing:
                return {"valid": False, "indicator_dictionary": "rsi_trend_reverse"}
            indicator_dictionary["rsi_trend_reverse_slope"] = (
                self.rsi_slope_threshold,
                round(rsi_slope, 2),
            )
            indicator_dictionary["rsi_trend_reverse_over"] = (
                self.rsi_overbought if trade_signal == "bullish" else self.rsi_oversold,
                [round(x, 1) for x in rsi_last.values],
            )

        return {"valid": True, "indicator_dictionary": indicator_dictionary}

    @staticmethod
    def calculate_slope(
        series: pd.Series,
        window: int = 3,
        normalize: Literal["first", "mean", None] = "first",
    ) -> float:
        """
        Calculate the normalized slope of a pandas Series over a specified window.

        Parameters
        ----------
        series : pd.Series
            Time series data such as RSI, EMA, or price.
        window : int, optional
            Number of most recent points to include in slope calculation. Default is 5.
        normalize : {'first', 'mean', None}, optional
            Method used to normalize the slope value:
            - 'first': Normalize by the first value in the window.
            - 'mean': Normalize by the mean value in the window.
            - None: Return the raw slope value.
            Default is 'first'.

        Returns
        -------
        float
            The normalized slope of the series. Returns np.nan if insufficient data.

        """
        if len(series) < window:
            return np.nan

        y = series.iloc[-window:].values
        x = np.arange(window)

        slope, _, _, _, _ = linregress(x, y)

        if normalize == "first":
            return slope / y[0]
        elif normalize == "mean":
            return slope / np.mean(y)
        else:
            return slope

    def evaluate_ema_crossover_bc(
        self,
        fast_df: pd.DataFrame,
        confirm_df: pd.DataFrame,
        aggregation_span: int = 4,
    ) -> str | None:
        """
        Check if a bullish or bearish EMA crossover occurred within the last completed
        confirmation timeframe candle, and if confirmation candle confirms the trend
        direction, as well as the slope

        Parameters:
            df_1m (pd.DataFrame): 1-minute OHLCV data with datetime index
            df_3m (pd.DataFrame): 3-minute OHLCV data with datetime index
            fast_window (int): EMA window for fast EMA
            slow_window (int): EMA window for slow EMA
            aggregation_span (int): Number of candles needed to aggregate
            lower into higher timeframe

        Returns:
            str: "bullish", "bearish", or None
        """

        fast_ema_fast_df = fast_df["ema_fast"]
        slow_ema_fast_df = fast_df["ema_slow"]
        fast_ema_confirm = confirm_df["ema_fast"]

        if self.term_trade_signal in ["short"]:
            aggregation_span = 5
        elif self.term_trade_signal in ["medium"]:
            aggregation_span = 4
        else:
            aggregation_span = 3

        candle_count = aggregation_span
        in_range_fast = fast_ema_fast_df.dropna().iloc[-candle_count:]
        in_range_slow = slow_ema_fast_df.dropna().iloc[-candle_count:]

        if (
            len(in_range_fast) < 3
            or in_range_fast.isna().any()
            or in_range_slow.isna().any()
        ):
            return None

        bullish_cross = False
        bearish_cross = False

        for i in range(1, len(in_range_fast) - 1):
            next_next_fast = None
            next_next_slow = None
            x = 0
            prev_fast = in_range_fast.iloc[i - 1]
            prev_slow = in_range_slow.iloc[i - 1]
            curr_fast = in_range_fast.iloc[i]
            curr_slow = in_range_slow.iloc[i]
            next_fast = in_range_fast.iloc[i + 1]
            next_slow = in_range_slow.iloc[i + 1]
            x = len(in_range_fast) - 1 - i
            if x > 1:
                next_next_fast = in_range_fast.iloc[
                    i + x
                ]  # better name would be last_fast
                next_next_slow = in_range_slow.iloc[i + x]

            if (
                prev_fast < prev_slow
                and curr_fast >= curr_slow
                and next_fast > next_slow
                and next_fast > curr_fast
            ):
                bullish_cross = True
                if next_next_fast is not None and next_next_slow is not None:
                    if next_next_fast > next_next_slow and next_next_fast > next_fast:
                        bullish_cross = True
                    else:
                        bullish_cross = False
                break

            elif (
                prev_fast > prev_slow
                and curr_fast <= curr_slow
                and next_fast < next_slow
                and next_fast < curr_fast
            ):
                bearish_cross = True
                if next_next_fast is not None and next_next_slow is not None:
                    if next_next_fast < next_next_slow and next_next_fast < next_fast:
                        bearish_cross = True
                    else:
                        bearish_cross = False
                break

        slope_confirms = False
        if bullish_cross or bearish_cross:
            close_slope = self.calculate_slope(
                fast_df["close"],
                window=aggregation_span,  # self.close_slope_window,
                normalize="mean",
            )
            if bullish_cross:
                slope_confirms = close_slope > self.close_slope_threshold
            elif bearish_cross:
                slope_confirms = abs(close_slope) > self.close_slope_threshold

        longer_candle_slope = self.calculate_slope(
            fast_ema_confirm,
            window=2,
            normalize="first",  # window=self.confirmation_slope_window
        )
        longer_candle_confirms = False
        if (
            bullish_cross
            and slope_confirms
            and longer_candle_slope > self.confirmation_slope_threshold
        ):
            longer_candle_confirms = True
        elif (
            bearish_cross
            and slope_confirms
            and abs(longer_candle_slope) > self.confirmation_slope_threshold
        ):
            longer_candle_confirms = True

        # slope_confirms = True
        # longer_candle_confirms = True
        if bullish_cross and slope_confirms and longer_candle_confirms:
            return "bullish"
        elif bearish_cross and slope_confirms and longer_candle_confirms:
            return None  # "bearish" - faster to evalute only bullish signals
        else:
            return None

    def vwap(self, df, trade_signal):
        # VWAP: approximate using typical price and cumulative volume intraday grouping
        # Requires that df has intraday timestamps and volume
        df["tpv"] = (df["high"] + df["low"] + df["close"]) / 3 * df["volume"]
        df["cum_tpv"] = df.groupby(df.index.date)["tpv"].cumsum()
        df["cum_vol"] = df.groupby(df.index.date)["volume"].cumsum()
        df["vwap"] = df["cum_tpv"] / df["cum_vol"]

        indicator_dictionary = {}

        if self.indicators.get("vwap"):
            price = df["close"].iloc[-1]
            vwap = df["vwap"].iloc[-1]
            vol = df["volume"].iloc[-1]
            avg_vol = (
                df["volume"]
                .rolling(window=self.confirmation_indicator_window)
                .mean()
                .iloc[-1]
            )

            bullish_vwap = trade_signal == "bullish" and price > vwap and vol > avg_vol
            bearish_vwap = trade_signal == "bearish" and price < vwap and vol > avg_vol
            if not (bullish_vwap if trade_signal == "bullish" else bearish_vwap):
                validation = False
            indicator_dictionary["vwap"] = (
                (round(price, 2), round(vwap, 2), round(vol, 1), round(avg_vol, 1)),
                "bullish" if bullish_vwap else "bearish",
            )
        return validation, indicator_dictionary
