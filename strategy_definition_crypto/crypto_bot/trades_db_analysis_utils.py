"""
Provides utility functions for analyzing and post-processing trade logs from
the trading bot. Includes routines to update incomplete trades, calculate
performance metrics, and generate detailed statistics on trade outcomes,
slippage, PnL, win rates, and portfolio impact.

Main features:
- Updates 'cancel_sl_eod' trades with data from matching closed trades for accurate PnL
- Computes and prints summary statistics for real bot performance
- Unnests and extracts confirmation indicator details from trade parameters
- Supports grouped analysis by trade type and highlights best/worst trades

Designed for use in performance review and research of live results.
"""

import pandas as pd


def update_cancel_eod_with_closed_data(not_completed_trades):
    """
    Update cancel_sl_eod rows with exit data from matching closed trades by symbol.
    Returns updated dataframe with recalculated PnL.
    """
    updated_df = not_completed_trades.copy()

    cancel_eod_trades = updated_df[updated_df["order_status"] == "cancel_sl_eod"]
    closed_trades = updated_df[updated_df["order_status"] == "closed"]

    updates_made = []

    for cancel_id, cancel_row in cancel_eod_trades.iterrows():
        symbol = cancel_row["symbol"]

        matching_closed = closed_trades[closed_trades["symbol"] == symbol]

        if not matching_closed.empty:

            closed_row = matching_closed.iloc[0]
            closed_id = closed_row.name

            updated_df.loc[cancel_id, "exit_commission"] = closed_row[
                "entry_commission"
            ]
            updated_df.loc[cancel_id, "exit_fill_price"] = closed_row["exit_fill_price"]
            updated_df.loc[cancel_id, "exit_fill_time"] = closed_row["exit_fill_time"]

            updated_df.loc[cancel_id, "real_time_pnl"] = updated_df.loc[
                cancel_id, "entry_fill_qty"
            ] * (
                updated_df.loc[cancel_id, "exit_fill_price"]
                - updated_df.loc[cancel_id, "entry_fill_price"]
            )

            updates_made.append(
                {"cancel_id": cancel_id, "closed_id": closed_id, "symbol": symbol}
            )
        else:
            print(
                f"No matching closed trade found for "
                f"cancel_sl_eod trade {cancel_id} ({symbol})"
            )

    return updated_df[updated_df["order_status"] == "cancel_sl_eod"], updates_made


def analyze_real_bot_performance(df: pd.DataFrame, initial_portfolio: float):
    df = df.copy()
    total_signals = len(df)

    df = df[df["exit_order_id"].notnull()].copy()
    all_completed_trades = df[
        (df["term_trade_type"] != "instant") & (df["order_status"] != "cancel_sl_eod")
    ]
    not_completed_trades = df[
        (df["term_trade_type"] == "instant") | (df["order_status"] == "cancel_sl_eod")
    ]

    cancel_eod_updated, _ = update_cancel_eod_with_closed_data(not_completed_trades)
    print(len(cancel_eod_updated))
    df = pd.concat([all_completed_trades, cancel_eod_updated], ignore_index=False)

    if df.empty:
        print("No closed trades to analyze.")
        return

    df["entry_fill_time"] = pd.to_datetime(df["entry_fill_time"], errors="coerce")
    df["exit_fill_time"] = pd.to_datetime(df["exit_fill_time"], errors="coerce")

    df["entry_slippage"] = (
        (df["entry_fill_price"] - df["entry_trigger_price"])
        / df["entry_trigger_price"]
        * 100
    )
    df["exit_slippage"] = (
        (df["exit_fill_price"] - df["amended_stop_loss"])
        / df["amended_stop_loss"]
        * 100
    )

    df["gross_pnl_usd"] = (df["exit_fill_price"] - df["entry_fill_price"]) * df[
        "entry_fill_qty"
    ]

    df["total_commission"] = df["entry_commission"].fillna(0) + df[
        "exit_commission"
    ].fillna(0)
    df["net_pnl_usd"] = df["gross_pnl_usd"] - df["total_commission"]

    df["trade_value"] = df["entry_fill_price"] * df["entry_fill_qty"]
    df["pnl_pct"] = df["net_pnl_usd"] / df["trade_value"] * 100
    df["portfolio_impact"] = df["net_pnl_usd"] / initial_portfolio * 100

    avg_entry_slippage = df["entry_slippage"].mean()
    avg_exit_slippage = df["exit_slippage"].mean()

    win_rate = (df["net_pnl_usd"] > 0).sum() / len(df) * 100

    df["trade_duration"] = df["exit_fill_time"] - df["entry_fill_time"]

    print("\n==== Real Bot Trade Analysis ====\n")
    print(f"Initial Portfolio Value:      ${initial_portfolio:.2f}")
    print(
        f"Trades Executed:              {len(df)} out of {total_signals} total signals "
        f"({len(df) / total_signals * 100:.2f}%)"
    )
    print(f"Win Rate:                     {win_rate:.2f}%")

    print(
        f"\nTotal Net PnL:                ${df['net_pnl_usd'].sum():.2f} | Gross PnL: "
        f"${df['gross_pnl_usd'].sum():.2f}"
    )
    print(f"Portfolio Impact:             {df['portfolio_impact'].sum():.2f}%")

    print(f"\nAverage Net PnL per Trade:    ${df['net_pnl_usd'].mean():.2f}")
    print(f"Average PnL % per Trade:      {df['pnl_pct'].mean():.2f}%")

    print(f"\nAverage Entry Slippage:       {avg_entry_slippage:.3f}%")
    print(f"Average Exit Slippage:        {avg_exit_slippage:.3f}%")

    if df["trade_duration"].notna().any():
        print("\n==== Trade Durations ====")
        print(f"Average Duration:             {df['trade_duration'].mean()}")
        print(f"Shortest:                     {df['trade_duration'].min()}")
        print(f"Longest:                      {df['trade_duration'].max()}")

    best_trade = df.loc[df["net_pnl_usd"].idxmax()]
    worst_trade = df.loc[df["net_pnl_usd"].idxmin()]
    print("\n==== Best Trade ====")
    print(
        f"Symbol: {best_trade['symbol']}, Net PnL: ${best_trade['net_pnl_usd']:.2f}, "
        f"Pct: {best_trade['pnl_pct']:.2f}%"
    )
    print(f"From {best_trade['entry_fill_time']} → {best_trade['exit_fill_time']}")

    print("\n==== Worst Trade ====")
    print(
        f"Symbol: {worst_trade['symbol']}, Net PnL: ${worst_trade['net_pnl_usd']:.2f}, "
        f"Pct: {worst_trade['pnl_pct']:.2f}%"
    )
    print(f"From {worst_trade['entry_fill_time']} → {worst_trade['exit_fill_time']}")

    print("\n==== Performance by Trade Type ====")
    grouped = df.groupby("term_trade_type").agg(
        trade_count=("net_pnl_usd", "count"),
        avg_net_pnl_usd=("net_pnl_usd", "mean"),
        avg_pnl_pct=("pnl_pct", "mean"),
        total_net_pnl_usd=("net_pnl_usd", "sum"),
        win_rate=("net_pnl_usd", lambda x: (x > 0).sum() / len(x) * 100),
    )
    from IPython.display import display

    display(grouped.round(2))

    df = df.sort_values("entry_fill_time")
    timeline = pd.concat(
        [
            pd.DataFrame({"time": df["entry_fill_time"], "delta": df["trade_value"]}),
            pd.DataFrame({"time": df["exit_fill_time"], "delta": -df["trade_value"]}),
        ]
    )
    timeline = timeline.sort_values("time")
    timeline["cash_in_use"] = timeline["delta"].cumsum()
    max_cash = timeline["cash_in_use"].max()
    print(f"\nMax Cash in Use at Once:      ${max_cash:.2f}")
    if max_cash > initial_portfolio:
        print("⚠️ Warning: Max cash in use exceeded initial portfolio value!")

    return df


def unnest_conf_indicators(df):
    """
    Unnest the conf_indicators column into separate columns.
    """
    df_copy = df.copy()

    df_copy["adx_slope"] = None
    df_copy["adx_value"] = None
    df_copy["di_slopes"] = None
    df_copy["di_plus"] = None
    df_copy["di_minus"] = None

    df_copy["volume_slope"] = None
    df_copy["volume_value"] = None
    df_copy["volume_avg"] = None
    df_copy["rsi_ratio"] = None
    df_copy["rsi_value"] = None
    df_copy["rsi_trend_reverse_slope_long"] = None
    df_copy["rsi_trend_reverse_slope_value"] = None
    df_copy["rsi_mean"] = None
    df_copy["rsi_trend_reverse"] = None
    """
    df_copy["macd_line"] = None
    df_copy["macd_signal"] = None
    df_copy["macd_line_prev"] = None
    df_copy["macd_signal_prev"] = None
    df_copy["macd_direction"] = None
    df_copy["bollinger_price"] = None
    df_copy["bollinger_middle"] = None
    df_copy["bollinger_lower"] = None
    df_copy["bollinger_upper"] = None
    df_copy["bollinger_direction"] = None
    """
    df_copy["atr_avg"] = None
    df_copy["atr_value"] = None
    df_copy["atr_slope"] = None
    df_copy["atr_stop_loss_to_price_ratio"] = None

    for idx, row in df_copy.iterrows():
        conf_indicators_str = row["params"].split("}")[0] + "}"
        conf_indicators = (
            eval(conf_indicators_str)
            if isinstance(conf_indicators_str, str)
            else conf_indicators_str
        )

        if conf_indicators == "no_confirmation_needed":
            df_copy.at[idx, "adx_slope"] = "N/A"
            df_copy.at[idx, "adx_value"] = "N/A"
            df_copy.at[idx, "di_slopes"] = "N/A"
            df_copy.at[idx, "di_plus"] = "N/A"
            df_copy.at[idx, "di_minus"] = "N/A"
            df_copy.at[idx, "volume_slope"] = "N/A"
            df_copy.at[idx, "volume_value"] = "N/A"
            df_copy.at[idx, "volume_avg"] = "N/A"
            df_copy.at[idx, "rsi_ratio"] = "N/A"
            df_copy.at[idx, "rsi_value"] = "N/A"
            df_copy.at[idx, "rsi_trend_reverse_slope_long"] = "N/A"
            df_copy.at[idx, "rsi_trend_reverse_slope_value"] = "N/A"
            df_copy.at[idx, "rsi_mean"] = "N/A"
            df_copy.at[idx, "rsi_trend_reverse"] = "N/A"
            """
            df_copy.at[idx, "macd_line"] = "N/A"
            df_copy.at[idx, "macd_signal"] = "N/A"
            df_copy.at[idx, "macd_line_prev"] = "N/A"
            df_copy.at[idx, "macd_signal_prev"] = "N/A"
            df_copy.at[idx, "macd_direction"] = "N/A"
            df_copy.at[idx, "bollinger_price"] = "N/A"
            df_copy.at[idx, "bollinger_middle"] = "N/A"
            df_copy.at[idx, "bollinger_lower"] = "N/A"
            df_copy.at[idx, "bollinger_upper"] = "N/A"
            df_copy.at[idx, "bollinger_direction"] = "N/A"
            """
            df_copy.at[idx, "atr_avg"] = "N/A"
            df_copy.at[idx, "atr_value"] = "N/A"
            df_copy.at[idx, "atr_slope"] = "N/A"
            df_copy.at[idx, "atr_stop_loss_to_price_ratio"] = "N/A"

        elif isinstance(conf_indicators, dict):
            adx_data = conf_indicators["adx"]
            if isinstance(adx_data, tuple) and len(adx_data) == 2:
                df_copy.at[idx, "adx_slope"] = adx_data[0]
                df_copy.at[idx, "adx_value"] = adx_data[1]

            di_data = conf_indicators["di"]
            if isinstance(di_data, tuple) and len(di_data) >= 3:
                df_copy.at[idx, "di_slopes"] = di_data[0]
                df_copy.at[idx, "di_plus"] = di_data[1]
                df_copy.at[idx, "di_minus"] = di_data[2]

            volume_data = conf_indicators["volume"]
            if isinstance(volume_data, tuple) and len(volume_data) == 2:
                df_copy.at[idx, "volume_slope"] = volume_data[0]
                df_copy.at[idx, "volume_value"] = volume_data[1][0]
                df_copy.at[idx, "volume_avg"] = volume_data[1][1]
            rsi_data = conf_indicators["rsi"]
            if isinstance(rsi_data, tuple) and len(rsi_data) == 2:
                df_copy.at[idx, "rsi_ratio"] = rsi_data[0]
                df_copy.at[idx, "rsi_value"] = rsi_data[1]
            rsi_trend_reverse_slope_data = conf_indicators.get(
                "rsi_trend_reverse_slope", (None, None)
            )
            if (
                isinstance(rsi_trend_reverse_slope_data, tuple)
                and len(rsi_trend_reverse_slope_data) == 2
            ):
                df_copy.at[idx, "rsi_trend_reverse_slope_long"] = (
                    rsi_trend_reverse_slope_data[0]
                )
                df_copy.at[idx, "rsi_trend_reverse_slope_value"] = (
                    rsi_trend_reverse_slope_data[1]
                )
            rsi_trend_reverse_over_data = conf_indicators.get(
                "rsi_trend_reverse_over", (None, None)
            )
            if (
                isinstance(rsi_trend_reverse_over_data, tuple)
                and len(rsi_trend_reverse_over_data) == 2
            ):
                df_copy.at[idx, "rsi_mean"] = rsi_trend_reverse_over_data[0]
                df_copy.at[idx, "rsi_trend_reverse"] = rsi_trend_reverse_over_data[1]

            """
            macd_data = conf_indicators["macd"]
            if isinstance(macd_data, tuple) and len(macd_data) == 2:
                df_copy.at[idx, "macd_line"] = macd_data[0][0]
                df_copy.at[idx, "macd_signal"] = macd_data[0][1]
                df_copy.at[idx, "macd_line_prev"] = macd_data[0][2]
                df_copy.at[idx, "macd_signal_prev"] = macd_data[0][3]
                df_copy.at[idx, "macd_direction"] = macd_data[1]

            bollinger_data = conf_indicators["bollinger"]
            if isinstance(bollinger_data, tuple) and len(bollinger_data) == 2:
                df_copy.at[idx, "bollinger_price"] = bollinger_data[0][0]
                df_copy.at[idx, "bollinger_middle"] = bollinger_data[0][1]
                df_copy.at[idx, "bollinger_lower"] = bollinger_data[0][2]
                df_copy.at[idx, "bollinger_upper"] = bollinger_data[0][3]
                df_copy.at[idx, "bollinger_direction"] = bollinger_data[1]
            """

            atr_data = conf_indicators["atr"]
            if isinstance(atr_data, tuple) and len(atr_data) == 2:
                df_copy.at[idx, "atr_avg"] = atr_data[0]
                df_copy.at[idx, "atr_value"] = atr_data[1][0]
                df_copy.at[idx, "atr_slope"] = atr_data[1][1]
                df_copy.at[idx, "atr_stop_loss_to_price_ratio"] = atr_data[1][2]

    df_copy = df_copy.drop(columns=["params"])
    return df_copy


def rejecting_conf_indicators(df):
    failed = []
    for idx, row in df.iterrows():
        try:
            conf_indicators_str = row["params"].split("}")[1].split(": ")[1]
            print(f"{idx}: {conf_indicators_str}")
        except Exception:
            failed.append(idx)
    return failed
