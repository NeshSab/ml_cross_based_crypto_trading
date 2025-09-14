import pandas as pd
import json

test_identifier = "ml_day_before"

STRATEGY_METADATA_PATH = (
    "strategy_definition/trading_bot_development/tickers_tests_metadata.json"
)

from utils.helper_functions import (
    delete_symbol_csvs,
    unnest_conf_indicators,
    save_df_with_metadata,
)
import utils.eda_utils as eda

from strategy_definition_crypto.crypto_bot.strategy_engine import (
    StrategyEngine,
)
from strategy_definition_crypto.crypto_bot.data_fetcher import (
    DataFetcher,
)
from strategy_definition_crypto.crypto_bot.signal_generator import (
    SignalGenerator,
)
from strategy_definition_crypto.crypto_bot.strategy_params import (
    StrategyParams,
)
from strategy_definition_crypto.crypto_bot.trade_logger import (
    TradeLogger,
)
from strategy_definition_crypto.crypto_bot.parameter_manager import (
    ParameterManager,
)


class DummyRiskManager:
    def assess_trade(self, direction, latest_price):
        return {"approve": True, "side": "buy", "symbol": "AAPL", "quantity": 10}


class DummyBroker:
    def place_order(self, side, symbol, quantity):
        print(f"[Broker] Placing {side.upper()} order for {symbol}, qty={quantity}")


def load_strategy_metadata():
    """Load strategy metadata from config file"""
    strategy_metadata_path = STRATEGY_METADATA_PATH
    with open(strategy_metadata_path, "r") as f:
        strategy_metadata = json.load(f)

    return {
        "test_date": strategy_metadata.get("test_date", "unknown_test_date"),
        "signal_discovery_strategy": strategy_metadata.get(
            "signal_discovery_strategy", "unknown_signal_discovery_strategy"
        ),
        "default_parameters": strategy_metadata.get(
            "default_parameters", "unknown_default_parameters"
        ),
        "timeframes": strategy_metadata.get("timeframes", "unknown_timeframes"),
        "confirmation_strategy": strategy_metadata.get(
            "confirmation_strategy", "unknown_confirmation_strategy"
        ),
    }


def run_trading(
    start_datetime, end_datetime, metadata_path, ticker_path, default_path, api_key
):
    """
    Run trading strategy across multiple symbols and timeframes.

    Parameters:
    -----------
    start_datetime : str
        Start datetime for backtesting (e.g., "2025-05-09 09:24:00")
    end_datetime : str
        End datetime for backtesting (e.g., "2025-05-09 16:00:00")
    metadata_path : str
        Path to ticker metadata JSON file
    ticker_path : str
        Path to ticker parameters JSON file
    api_key : str, optional
        API key for data fetching

    Returns:
    --------
    tuple
        (all_trades_results, failed, metadata_dict)
    """

    risk_manager = DummyRiskManager()
    broker = DummyBroker()
    trade_logger = TradeLogger("trades_backtest.db")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    symbols_all = [ticker for ticker, data in metadata.items()]
    start_date = start_datetime.split(" ")[0]
    end_date = end_datetime.split(" ")[0]
    ticker_set_name = metadata_path.split("/")[-1]

    strategy_meta = load_strategy_metadata()

    all_trades_list = []
    failed = []
    empty = []

    for symbol in symbols_all:
        company_type = metadata[symbol]["group"]
        for term_trade in ["short", "medium", "long"]:
            files_paths = {
                "1min": f"data_all/{symbol}_1min_{start_date}.csv",
            }
            try:
                print(f"\nProcessing {symbol} for {term_trade} trade")
                params = ParameterManager.load_params(
                    symbol=symbol,
                    term_trade=term_trade,
                    company_type=company_type,
                    ticker_path=ticker_path,
                    default_path=strategy_meta["default_parameters"],
                )

                signal_generator_obj = SignalGenerator(params)
                data_fetcher_obj = DataFetcher(
                    symbol, api_key, files_paths, end_datetime
                )
                engine_obj = StrategyEngine(
                    data_fetcher_obj,
                    signal_generator_obj,
                    risk_manager,
                    broker,
                    params,
                )

                trigger_price_multiplier = 1.001
                trades_df = engine_obj.backtest_strategy_modular(
                    start_datetime, trigger_price_multiplier=trigger_price_multiplier
                )

                if not trades_df.empty and not trades_df.isna().all().all():
                    clean_trades_df = trades_df.dropna(how="all")
                    if not clean_trades_df.empty:
                        clean_trades_df["symbol"] = symbol
                        clean_trades_df["company_type"] = company_type

                        all_trades_list.append(clean_trades_df)
                    else:
                        empty.append((symbol, term_trade))
                else:
                    empty.append((symbol, term_trade))

            except Exception as e:
                failed.append((symbol, term_trade, str(e)))
                print(f"Failed to process {symbol} for {term_trade} trade: {e}")

        delete_symbol_csvs(files_paths)

    if trade_logger:
        trade_logger.close()

    metadata_dict = {
        "start_date": start_date,
        "end_date": end_date,
        "ticker_set_name": ticker_set_name,
        **strategy_meta,
    }
    if all_trades_list:
        all_trades_results = pd.concat(all_trades_list, ignore_index=True)
    else:
        all_trades_results = pd.DataFrame()

    return all_trades_results, failed, metadata_dict


def analyze_and_save_trades_performance(
    all_trades_results,
    failed,
    metadata_dict,
    initial_portfolio=10000,
    play_size=3000,
    save_results=True,
):
    """
    Analyze trading performance and optionally save results.

    Parameters:
    -----------
    all_trades_results : pd.DataFrame
        DataFrame containing all trade results
    failed : list
        List of failed trades
    metadata_dict : dict
        Dictionary containing run metadata
    initial_portfolio : float, optional
        Initial portfolio value (default: 10000)
    play_size : float, optional
        Amount invested per trade (default: 3000)
    save_results : bool, optional
        Whether to save results to files (default: True)

    Returns:
    --------
    dict
        Dictionary containing analysis results
    """

    if failed:
        print(f"\nFailed trades: {len(failed)}\n")
        for f in failed:
            print(f)

    if all_trades_results.empty:
        print("No trades to analyze")
        return {"status": "no_trades", "failed_count": len(failed)}

    all_trades_results = all_trades_results.sort_values(by="entry_time").reset_index(
        drop=True
    )
    bullish_trades_df = all_trades_results[
        all_trades_results["direction"] == "bullish"
    ].copy()
    bullish_trades_df = bullish_trades_df.reset_index(drop=True)

    bullish_trades_df.drop(columns=["direction"], inplace=True)
    bullish_trades_df["shares"] = play_size / bullish_trades_df["entry_price"]
    # max(per_unit_rate ($0.0051) × quantity, min_fee ($1))
    bullish_trades_df["commission_fee"] = (bullish_trades_df["shares"] * 0.0051).clip(
        lower=1, upper=play_size * 0.01
    ) * 2
    bullish_trades_df["pnl_usd"] = (
        bullish_trades_df["exit_price"] - bullish_trades_df["entry_price"]
    ) * bullish_trades_df["shares"] - bullish_trades_df["commission_fee"]

    bullish_trades_df["trade_duration_min"] = (
        bullish_trades_df["exit_time"] - bullish_trades_df["entry_time"]
    ).dt.total_seconds() / 60

    bullish_trades_df["pnl_pct_per_trade"] = round(
        (bullish_trades_df["pnl_usd"] / play_size * 100), 3
    )

    confirmed_trades = bullish_trades_df[
        bullish_trades_df["confirmation"] == "confirmed"
    ]

    if len(confirmed_trades) > 0:

        confirmed_bullish_trades_df = confirmed_trades.copy()
        confirmed_bullish_trades_df.sort_values(
            by="signal_detection_time", inplace=True
        )

        conf_df = confirmed_bullish_trades_df[
            [
                "conf_indicators",
                "signal_detection_time",
                "symbol",
                "pnl_usd",
                "commission_fee",
                "company_type",
                "term_trade_type",
                "trade_duration_min",
            ]
        ].copy()
        unnested_df = unnest_conf_indicators(conf_df)
        # unnested_df["ratio"] = unnested_df.apply(safe_ratio_calc, axis=1)

        performance_summary = eda.get_strategy_performance_summary(
            bullish_trades_df, play_size, initial_portfolio
        )

        if save_results:
            test_date = metadata_dict.get("test_date", "unknown")

            save_df_with_metadata(
                unnested_df,
                f"{test_identifier}_conf_indicators_summary_{test_date}.csv",
            )

            try:
                # need metadata_dict to have metadata keys ant it's order
                start_date = metadata_dict.get("start_date", "unknown_start_date")
                end_date = metadata_dict.get("end_date", "unknown_end_date")
                signal_discovery_strategy = metadata_dict.get(
                    "signal_discovery_strategy", "unknown_signal_discovery_strategy"
                )
                default_parameters = metadata_dict.get(
                    "default_parameters", "unknown_default_parameters"
                )
                timeframes = metadata_dict.get("timeframes", "unknown_timeframes")
                confirmation_strategy = metadata_dict.get(
                    "confirmation_strategy", "unknown_confirmation_strategy"
                )
                ticker_set_name = metadata_dict.get(
                    "ticker_set_name", "unknown_ticker_set_name"
                )
                metadata = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "strategy_version": signal_discovery_strategy,
                    "default_parameters": default_parameters,
                    "timeframes": timeframes,
                    "confirmation_strategy": confirmation_strategy,
                    "ticker_set_name": ticker_set_name,
                }
                save_df_with_metadata(
                    performance_summary,
                    f"{test_identifier}_performance_summary_detailed_{test_date}.csv",
                    metadata_dict=metadata,
                    append=True,
                )
            except ValueError as e:
                print(f"Column mismatch error: {e}")
                new_filename = f"{test_identifier}_performance_summary_detailed_mismatch_{test_date}.csv"
                save_df_with_metadata(
                    performance_summary,
                    new_filename,
                    metadata_dict=metadata_dict,
                    append=False,
                )
                print(f"Saved to new file instead: {new_filename}")

        return {
            "status": "success",
            "bullish_trades_df": bullish_trades_df,
            "confirmed_trades_df": confirmed_bullish_trades_df,
            "unnested_indicators": unnested_df,
            "performance_summary": performance_summary,
            "failed_count": len(failed),
        }

    else:
        print("No confirmed trades found")
        return {
            "status": "no_confirmed_trades",
            "bullish_trades_df": bullish_trades_df,
            "failed_count": len(failed),
        }


def run_strategy_analysis(config_params):
    """
    Main function to run complete strategy analysis.

    Parameters:
    -----------
    config_params : dict
        Dictionary containing configuration parameters:
        - start_datetime : str
        - end_datetime : str
        - metadata_path : str
        - ticker_path : str
        - api_key : str (optional)
        - initial_portfolio : float (optional)
        - play_size : float (optional)
        - save_results : bool (optional)

    Returns:
    --------
    dict
        Analysis results dictionary
    """
    start_datetime = config_params["start_datetime"]
    end_datetime = config_params["end_datetime"]
    metadata_path = config_params["metadata_path"]
    ticker_path = config_params["ticker_path"]
    default_path = config_params.get("default_path", "default_params.json")

    api_key = config_params.get("api_key", "SGUFIZB1UFKXLVDE")
    initial_portfolio = config_params.get("initial_portfolio", 10000)
    play_size = config_params.get("play_size", 3000)
    save_results = config_params.get("save_results", True)

    print(f"Running strategy analysis from {start_datetime} to {end_datetime}")
    print(f"Using metadata: {metadata_path}")
    print(f"Using ticker params: {ticker_path}")

    all_trades_results, failed, metadata_dict = run_trading(
        start_datetime, end_datetime, metadata_path, ticker_path, default_path, api_key
    )

    results = analyze_and_save_trades_performance(
        all_trades_results,
        failed,
        metadata_dict,
        initial_portfolio,
        play_size,
        save_results,
    )

    return results


if __name__ == "__main__":
    from datetime import datetime

    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Script started at {time_now}")
    config_1 = {
        "start_datetime": "2025-07-08 09:24:00",
        "end_datetime": "2025-07-08 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_704.json",
        "ticker_path": "ticker_params.json",
    }
    config_2 = {
        "start_datetime": "2025-07-09 09:24:00",
        "end_datetime": "2025-07-09 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_708.json",
        "ticker_path": "ticker_params.json",
    }
    config_3 = {
        "start_datetime": "2025-07-10 09:24:00",
        "end_datetime": "2025-07-10 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_709.json",
        "ticker_path": "ticker_params.json",
    }
    config_4 = {
        "start_datetime": "2025-07-11 09:24:00",
        "end_datetime": "2025-07-11 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_710.json",
        "ticker_path": "ticker_params.json",
    }
    config_5 = {
        "start_datetime": "2025-07-14 09:24:00",
        "end_datetime": "2025-07-14 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_711.json",
        "ticker_path": "ticker_params.json",
    }
    config_6 = {
        "start_datetime": "2025-07-15 09:24:00",
        "end_datetime": "2025-07-15 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_714.json",
        "ticker_path": "ticker_params.json",
    }
    config_7 = {
        "start_datetime": "2025-07-16 09:24:00",
        "end_datetime": "2025-07-16 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_715.json",
        "ticker_path": "ticker_params.json",
    }
    config_8 = {
        "start_datetime": "2025-07-17 09:24:00",
        "end_datetime": "2025-07-17 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_716.json",
        "ticker_path": "ticker_params.json",
    }
    config_9 = {
        "start_datetime": "2025-07-18 09:24:00",
        "end_datetime": "2025-07-18 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_717.json",
        "ticker_path": "ticker_params.json",
    }
    config_10 = {
        "start_datetime": "2025-07-21 09:24:00",
        "end_datetime": "2025-07-21 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_718.json",
        "ticker_path": "ticker_params.json",
    }
    config_11 = {
        "start_datetime": "2025-07-22 09:24:00",
        "end_datetime": "2025-07-22 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_721.json",
        "ticker_path": "ticker_params.json",
    }
    config_12 = {
        "start_datetime": "2025-07-23 09:24:00",
        "end_datetime": "2025-07-23 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_722.json",
        "ticker_path": "ticker_params.json",
    }
    config_13 = {
        "start_datetime": "2025-07-24 09:24:00",
        "end_datetime": "2025-07-24 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_723.json",
        "ticker_path": "ticker_params.json",
    }
    config_14 = {
        "start_datetime": "2025-07-25 09:24:00",
        "end_datetime": "2025-07-25 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_724.json",
        "ticker_path": "ticker_params.json",
    }
    config_15 = {
        "start_datetime": "2025-07-28 09:24:00",
        "end_datetime": "2025-07-28 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_725.json",
        "ticker_path": "ticker_params.json",
    }
    config_16 = {
        "start_datetime": "2025-07-29 09:24:00",
        "end_datetime": "2025-07-29 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_728.json",
        "ticker_path": "ticker_params.json",
    }
    config_17 = {
        "start_datetime": "2025-07-30 09:24:00",
        "end_datetime": "2025-07-30 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_729.json",
        "ticker_path": "ticker_params.json",
    }
    config_18 = {
        "start_datetime": "2025-07-31 09:24:00",
        "end_datetime": "2025-07-31 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_730.json",
        "ticker_path": "ticker_params.json",
    }
    config_19 = {
        "start_datetime": "2025-08-01 09:24:00",
        "end_datetime": "2025-08-01 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_731.json",
        "ticker_path": "ticker_params.json",
    }
    config_20 = {
        "start_datetime": "2025-08-04 09:24:00",
        "end_datetime": "2025-08-04 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_801.json",
        "ticker_path": "ticker_params.json",
    }
    config_21 = {
        "start_datetime": "2025-08-05 09:24:00",
        "end_datetime": "2025-08-05 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_804.json",
        "ticker_path": "ticker_params.json",
    }
    config_22 = {
        "start_datetime": "2025-08-06 09:24:00",
        "end_datetime": "2025-08-06 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_805.json",
        "ticker_path": "ticker_params.json",
    }
    config_23 = {
        "start_datetime": "2025-08-07 09:24:00",
        "end_datetime": "2025-08-07 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_806.json",
        "ticker_path": "ticker_params.json",
    }
    config_24 = {
        "start_datetime": "2025-08-08 09:24:00",
        "end_datetime": "2025-08-08 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_807.json",
        "ticker_path": "ticker_params.json",
    }
    config_25 = {
        "start_datetime": "2025-08-11 09:24:00",
        "end_datetime": "2025-08-11 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_808.json",
        "ticker_path": "ticker_params.json",
    }
    config_26 = {
        "start_datetime": "2025-08-12 09:24:00",
        "end_datetime": "2025-08-12 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_811.json",
        "ticker_path": "ticker_params.json",
    }
    config_27 = {
        "start_datetime": "2025-08-13 09:24:00",
        "end_datetime": "2025-08-13 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_812.json",
        "ticker_path": "ticker_params.json",
    }
    config_28 = {
        "start_datetime": "2025-08-14 09:24:00",
        "end_datetime": "2025-08-14 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_813.json",
        "ticker_path": "ticker_params.json",
    }
    config_29 = {
        "start_datetime": "2025-08-15 09:24:00",
        "end_datetime": "2025-08-15 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/month/ticker_metadata_uw_814.json",
        "ticker_path": "ticker_params.json",
    }
    config_30 = {
        "start_datetime": "2025-08-18 09:24:00",
        "end_datetime": "2025-08-18 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/validation/ticker_metadata_uw_2025-08-18.json",
        "ticker_path": "ticker_params.json",
    }
    config_31 = {
        "start_datetime": "2025-08-19 09:24:00",
        "end_datetime": "2025-08-19 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/validation/ticker_metadata_uw_2025-08-19.json",
        "ticker_path": "ticker_params.json",
    }
    config_32 = {
        "start_datetime": "2025-08-20 09:24:00",
        "end_datetime": "2025-08-20 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/validation/ticker_metadata_uw_2025-08-20.json",
        "ticker_path": "ticker_params.json",
    }
    config_33 = {
        "start_datetime": "2025-08-21 09:24:00",
        "end_datetime": "2025-08-21 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/validation/ticker_metadata_uw_2025-08-21.json",
        "ticker_path": "ticker_params.json",
    }
    config_34 = {
        "start_datetime": "2025-08-22 09:24:00",
        "end_datetime": "2025-08-22 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/validation/ticker_metadata_uw_2025-08-22.json",
        "ticker_path": "ticker_params.json",
    }
    config_35 = {
        "start_datetime": "2025-08-25 09:24:00",
        "end_datetime": "2025-08-25 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/validation/ticker_metadata_uw_2025-08-25.json",
        "ticker_path": "ticker_params.json",
    }
    config_36 = {
        "start_datetime": "2025-08-26 09:24:00",
        "end_datetime": "2025-08-26 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/validation/ticker_metadata_uw_2025-08-26.json",
        "ticker_path": "ticker_params.json",
    }
    config_37 = {
        "start_datetime": "2025-08-27 09:24:00",
        "end_datetime": "2025-08-27 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/validation/ticker_metadata_uw_2025-08-27.json",
        "ticker_path": "ticker_params.json",
    }
    config_38 = {
        "start_datetime": "2025-08-28 09:24:00",
        "end_datetime": "2025-08-28 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/validation/ticker_metadata_uw_2025-08-28.json",
        "ticker_path": "ticker_params.json",
    }
    config_39 = {
        "start_datetime": "2025-08-29 09:24:00",
        "end_datetime": "2025-08-29 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/validation/ticker_metadata_uw_2025-08-29.json",
        "ticker_path": "ticker_params.json",
    }
    config40 = {
        "start_datetime": "2025-09-02 09:24:00",
        "end_datetime": "2025-09-02 16:00:00",
        "metadata_path": "strategy_definition/trading_bot_development/bot_modules_calib/config/config_uw/validation/ticker_metadata_uw_2025-09-02.json",
        "ticker_path": "ticker_params.json",
    }
    
    configs = [
        config_1,
        config_2,
        config_3,
        config_4,
        config_5,
        config_6,
        config_7,
        config_8,
        config_9,
        config_10,
        config_11,
        config_12,
        config_13,
        config_14,
        config_15,
        config_16,
        config_17,
        config_18,
        config_19,
        config_20,
        config_21,
        config_22,
        config_23,
        config_24,
        config_25,
        config_26,
        config_27,
        config_28,
        config_29,
        config_30,
        config_31,
        config_32,
        config_33,
        config_34,
        config_35,
        config_36,
        config_37,
        config_38,
        config_39,
    ]

    for config in configs:
        results = run_strategy_analysis(config)

    print(f"\nAnalysis completed with status: {results['status']}")
    print(f"Failed trades: {results['failed_count']}")
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Script ended at {time_now}")
