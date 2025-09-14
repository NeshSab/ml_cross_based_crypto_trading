from dataclasses import dataclass
from typing import Literal


@dataclass
class StrategyParams:
    """
    Container for all strategy parameters used in signal generation and execution.
    """

    company_type: Literal["emerging_volatile", "large_cap_tech", "moderate_midcap"]
    term_trade: Literal["short", "medium", "long"]
    atr_multiplier: float
    atr_window: int
    fast_slow_windows: tuple
    close_slope_threshold: float
    confirmation_slope_threshold: float
    close_slope_window: int
    confirmation_slope_window: int
    confirmation_indicator_window: int
    rsi_value: int
    rsi_X: int
    rsi_slope_threshold: float
    rsi_overbought: int
    rsi_oversold: int
    adx_threshold: float
    macd_slow: int = 26
    macd_fast: int = 12
    macd_signal: int = 9
    bb_window: int = 20
    bb_dev: float = 2.0
