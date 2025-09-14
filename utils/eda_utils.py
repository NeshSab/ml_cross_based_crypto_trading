"""
This module provides utility functions for performing exploratory data analysis (EDA)
on pandas DataFrames. The functions help retrieve group-specific values, identify and
fix inconsistencies, check for missing values, an so on.

These utilities streamline the EDA process by automating common data quality checks
and facilitating data preparation tasks.
"""

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.io as pio
from datetime import timedelta
from typing import Literal
from IPython.display import display

PASTEL_COLORS_RGB = {
    "coral orange": "rgb(248, 156, 116)",
    "aqua blue": "rgb(102, 197, 204)",
    "neutral grey": "rgb(179, 179, 179)",
    "leafy green": "rgb(135, 197, 95)",
    "sunny yellow": "rgb(246, 207, 113)",
    "soft purple": "rgb(220, 176, 242)",
    "lavender blue": "rgb(158, 185, 243)",
    "bubblegum pink": "rgb(254, 136, 177)",
    "lime zest": "rgb(201, 219, 116)",
    "mint green": "rgb(139, 224, 164)",
    "purple blue": "rgb(180, 151, 231)",
}

PASTEL_COLORS_RGBA_60 = {
    "coral orange": "rgba(248, 156, 116, 0.6)",
    "aqua blue": "rgba(102, 197, 204, 0.6)",
    "neutral grey": "rgba(179, 179, 179, 0.6)",
    "leafy green": "rgba(135, 197, 95, 0.6)",
    "sunny yellow": "rgba(246, 207, 113, 0.6)",
    "soft purple": "rgba(220, 176, 242, 0.6)",
    "lavender blue": "rgba(158, 185, 243, 0.6)",
    "bubblegum pink": "rgba(254, 136, 177, 0.6)",
    "lime zest": "rgba(201, 219, 116, 0.6)",
    "mint green": "rgba(139, 224, 164, 0.6)",
    "purple blue": "rgba(180, 151, 231, 0.6)",
}

pastel_colors_list = list(PASTEL_COLORS_RGB.values())

pio.templates["plotly_dark"].layout.colorway = pastel_colors_list
pio.templates.default = "plotly_dark"

aqua_blue_grayish_colorscale = [
    [0, "rgb(224, 224, 224)"],
    [0.25, "rgb(102, 170, 170)"],
    [0.5, "rgb(76, 140, 140)"],
    [0.75, "rgb(51, 110, 110)"],
    [1.0, "rgb(25, 80, 80)"],
]


def get_strategy_performance_summary(bullish_trades_df, play_size, initial_portfolio):
    """
    Calculate strategy performance metrics and return as DataFrame.

    Parameters:
    -----------
    bullish_trades_df : pd.DataFrame
        DataFrame containing trade results
    play_size : float
        Amount invested per trade
    initial_portfolio : float
        Initial portfolio value

    Returns:
    --------
    pd.DataFrame
        Performance summary with overall and term-type breakdowns
    """

    # Overall performance metrics
    confirmed_trades = bullish_trades_df[
        bullish_trades_df["confirmation"] == "confirmed"
    ]
    total_signals = len(bullish_trades_df)
    confirmed_signals = len(confirmed_trades)
    trades_executed = confirmed_signals  # Assuming all confirmed trades are executed
    trades_skipped = 0  # Assuming no capital constraints for now
    not_triggered = total_signals - confirmed_signals

    total_pnl_usd = confirmed_trades["pnl_usd"].sum()
    total_pnl_pct = (total_pnl_usd / initial_portfolio) * 100
    win_rate = (confirmed_trades["pnl_usd"] > 0).mean() * 100
    avg_pnl_usd = confirmed_trades["pnl_usd"].mean()
    avg_pnl_pct = (avg_pnl_usd / play_size) * 100

    # Overall summary
    overall_summary = pd.DataFrame(
        [
            {
                "metric_type": "overall",
                "term_trade_type": "all",
                "total_pnl_usd": total_pnl_usd,
                "total_pnl_pct": total_pnl_pct,
                "trades_executed": trades_executed,
                "trades_skipped": trades_skipped,
                "confirmed_signals": confirmed_signals,
                "total_signals": total_signals,
                "not_triggered_signals": not_triggered,
                "win_rate": win_rate,
                "avg_pnl_usd": avg_pnl_usd,
                "avg_pnl_pct": avg_pnl_pct,
                "trade_count": trades_executed,
            }
        ]
    )
    if not confirmed_trades.empty:
        term_performance = (
            confirmed_trades.groupby("term_trade_type")
            .agg(
                {
                    "pnl_usd": [
                        "sum",
                        "mean",
                        "count",
                        lambda x: (x > 0).mean() * 100,
                    ]  # Fixed: all in one list
                }
            )
            .round(2)
        )

        term_performance.columns = [
            "total_pnl_usd",
            "avg_pnl_usd",
            "trade_count",
            "win_rate",
        ]
        term_performance = term_performance.reset_index()

        # Calculate percentage metrics for term types
        term_performance["total_pnl_pct"] = (
            term_performance["total_pnl_usd"] / initial_portfolio
        ) * 100
        term_performance["avg_pnl_pct"] = (
            term_performance["avg_pnl_usd"] / play_size
        ) * 100

        # Add metric type and fill other columns
        term_performance["metric_type"] = "by_term_type"
        term_performance["trades_executed"] = term_performance["trade_count"]
        term_performance["trades_skipped"] = 0
        term_performance["confirmed_signals"] = term_performance["trade_count"]
        term_performance["total_signals"] = term_performance[
            "trade_count"
        ]  # Simplified
        term_performance["not_triggered_signals"] = 0

        # Combine results
        result_df = pd.concat([overall_summary, term_performance], ignore_index=True)
    else:
        # If no confirmed trades, just return overall summary
        result_df = overall_summary

    # Reorder columns for better readability
    column_order = [
        "metric_type",
        "term_trade_type",
        "total_pnl_usd",
        "total_pnl_pct",
        "trades_executed",
        "trades_skipped",
        "confirmed_signals",
        "total_signals",
        "not_triggered_signals",
        "win_rate",
        "avg_pnl_usd",
        "avg_pnl_pct",
        "trade_count",
    ]

    result_df = result_df[column_order].round(2)

    return result_df


def simulate_capital_allocation(confirmed_trades, play_size, initial_portfolio):
    """
    Simulate realistic capital allocation and return executable vs capital-constrained
    trades.
    """
    confirmed_trades = confirmed_trades.copy()
    confirmed_trades["entry_time"] = pd.to_datetime(confirmed_trades["entry_time"])
    confirmed_trades["exit_time"] = pd.to_datetime(confirmed_trades["exit_time"])

    # Sort by entry time
    confirmed_trades = confirmed_trades.sort_values("entry_time").reset_index(drop=True)

    available_cash = initial_portfolio
    open_positions = []
    executable_trades = []
    capital_constrained_trades = []

    # Create timeline of all entry and exit events
    events = []
    for idx, trade in confirmed_trades.iterrows():
        events.append(
            {
                "time": trade["entry_time"],
                "type": "entry",
                "trade_idx": idx,
                "trade": trade,
            }
        )
        if pd.notna(trade["exit_time"]):
            events.append(
                {
                    "time": trade["exit_time"],
                    "type": "exit",
                    "trade_idx": idx,
                    "trade": trade,
                }
            )

    # Sort events by time
    events.sort(key=lambda x: x["time"])

    # Process events chronologically
    for event in events:
        if event["type"] == "entry":
            # Check if we have enough cash
            if available_cash >= play_size:
                # Execute the trade
                available_cash -= play_size
                open_positions.append(
                    {"trade_idx": event["trade_idx"], "trade": event["trade"]}
                )
                executable_trades.append(event["trade"])
            else:
                # Not enough capital
                capital_constrained_trades.append(event["trade"])

        elif event["type"] == "exit":
            # Find the corresponding open position
            for i, pos in enumerate(open_positions):
                if pos["trade_idx"] == event["trade_idx"]:
                    # Close the position and free up capital
                    pnl = pos["trade"]["pnl_usd"]
                    available_cash += play_size + pnl
                    open_positions.pop(i)
                    break

    return pd.DataFrame(executable_trades), pd.DataFrame(capital_constrained_trades)


def analyze_strategy_performance_multiple_tickers_updated(
    trades_df: pd.DataFrame, play_size: int, initial_portfolio: float
):
    trades_df = trades_df.copy()
    total_signals = len(trades_df)
    total_days = trades_df["signal_detection_time"].dt.date.nunique()
    not_triggered_signals = len(trades_df[trades_df["confirmation"] == "not_triggered"])

    # First, filter for confirmed trades and sort by entry time
    confirmed_trades = trades_df[trades_df["confirmation"] == "confirmed"].copy()
    confirmed_trades = confirmed_trades.sort_values("entry_time").reset_index(drop=True)

    if not confirmed_trades.empty:
        # Simulate realistic capital allocation
        executable_trades, capital_constrained_trades = simulate_capital_allocation(
            confirmed_trades, play_size, initial_portfolio
        )

        # Update the trades dataframe with execution status
        trades_df = update_trades_execution_status(
            trades_df, executable_trades, capital_constrained_trades
        )

        # Use only executable trades for analysis
        analysis_trades = executable_trades.copy()

        # Calculate metrics
        analysis_trades["pnl_pct"] = analysis_trades["pnl_usd"] / play_size * 100
        analysis_trades["pnl_portfolio_pct"] = (
            analysis_trades["pnl_usd"] / initial_portfolio * 100
        )

        print("\n==== Overall Performance (Capital-Constrained) ====")
        total_pnl = analysis_trades["pnl_usd"].sum()
        final_portfolio = initial_portfolio + total_pnl
        total_pnl_pct = (total_pnl / initial_portfolio) * 100
        average_pnl = analysis_trades["pnl_usd"].mean()
        average_pnl_pct = analysis_trades["pnl_pct"].mean()
        average_pnl_portfolio = analysis_trades["pnl_portfolio_pct"].mean()

        win_rate = (
            len(analysis_trades[analysis_trades["pnl_usd"] > 0])
            / len(analysis_trades)
            * 100
        )

        # Calculate max concurrent trades from executable trades
        max_concurrent = calculate_max_concurrent_trades(analysis_trades)
        max_cash_used = max_concurrent * play_size

        print(f"Initial Portfolio Value:       ${initial_portfolio:.2f}")
        print(f"Cash per Play:                 ${play_size}")
        print(
            f"Max Cash in Play at Once:      ${max_cash_used} ({max_concurrent} concurrent trades)\n"
        )
        print(f"Final Portfolio Value:         ${final_portfolio:.2f}")
        print(f"Total PnL:                     ${total_pnl:.2f} ({total_pnl_pct:.2f}%)")
        print(f"\nTrades Actually Executed:      {len(analysis_trades)}")
        print(f"Trades Skipped (No Capital):   {len(capital_constrained_trades)}")
        print(f"Total Confirmed Signals:       {len(confirmed_trades)}")
        print(f"Total Signals:                 {total_signals}")
        print(f"Not Triggered Signals:         {not_triggered_signals}")
        print(f"Win rate:                      {win_rate:.2f}%\n")

        print(
            f"Average PnL per Trade:          ${average_pnl:.2f} ({average_pnl_pct:.2f}%)"
        )
        print(f"Average Portfolio Impact:       {average_pnl_portfolio:.2f}%\n")

        if not analysis_trades.empty:
            best_trade = analysis_trades.loc[analysis_trades["pnl_usd"].idxmax()]
            worst_trade = analysis_trades.loc[analysis_trades["pnl_usd"].idxmin()]
            print("\n== Best Trade ==")
            print(f"Ticker: {best_trade['symbol']}")
            print(f"PnL:  ${best_trade['pnl_usd']:.2f} ({best_trade['pnl_pct']:.2f}%)")
            print(f"Time: {best_trade['entry_time']} → {best_trade['exit_time']}")
            print("\n== Worst Trade ==")
            print(f"Ticker: {worst_trade['symbol']}")
            print(f"PnL: ${worst_trade['pnl_usd']:.2f} ({worst_trade['pnl_pct']:.2f}%)")
            print(f"Time: {worst_trade['entry_time']} → {worst_trade['exit_time']}")

            print("\n\n==== Performance by Term Type ====")
            grouped = analysis_trades.groupby("term_trade_type").agg(
                trade_count=("pnl_usd", "count"),
                avg_pnl_usd=("pnl_usd", "mean"),
                avg_pnl_pct=("pnl_pct", "mean"),
                avg_portfolio_impact=("pnl_portfolio_pct", "mean"),
                total_pnl_usd=("pnl_usd", "sum"),
                win_rate=("pnl_usd", lambda x: (x > 0).sum() / len(x) * 100),
            )

            display(grouped.round(2))

            # Trade duration stats
            analysis_trades["trade_duration"] = pd.to_datetime(
                analysis_trades["exit_time"]
            ) - pd.to_datetime(analysis_trades["entry_time"])

            valid_durations = analysis_trades["trade_duration"].dropna()

            if not valid_durations.empty:
                avg_duration = valid_durations.mean()
                min_duration = valid_durations.min()
                max_duration = valid_durations.max()

                print("\n\n==== Trade Duration Stats ====")
                print(f"Average Trade Duration: {avg_duration}\n")
                print(f"Shortest Trade:   {min_duration}")
                print(f"Longest Trade:    {max_duration}\n")
                effect_trade_days = analysis_trades["entry_time"].dt.date.nunique()
                print(f"Effective Trading Days: {effect_trade_days} of {total_days}")
                print(
                    f"Average Trades per Day: {len(analysis_trades) / effect_trade_days:.1f}"
                )

        # Show capital constraint impact
        if not capital_constrained_trades.empty:
            print("\n\n==== Capital Constraint Impact ====")
            missed_pnl = capital_constrained_trades["pnl_usd"].sum()
            print(f"Missed PnL due to capital constraints: ${missed_pnl:.2f}")
            print(
                f"Potential total PnL (if unlimited capital): ${total_pnl + missed_pnl:.2f}"
            )
            print(
                f"Capital efficiency: {len(analysis_trades) / len(confirmed_trades) * 100:.1f}%"
            )

            # Show which symbols/terms were most affected
            missed_by_symbol = (
                capital_constrained_trades.groupby("symbol")["pnl_usd"]
                .sum()
                .sort_values(ascending=False)
            )
            print(f"\nTop symbols missed due to capital constraints:")
            for symbol, missed in missed_by_symbol.head(5).items():
                print(f"  {symbol}: ${missed:.2f}")
    else:
        print("No confirmed trades found.")


def calculate_max_concurrent_trades(trades_df):
    """
    Calculate the maximum number of concurrent trades.
    """
    if trades_df.empty:
        return 0

    trades_df = trades_df.copy()
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

    # Handle missing exit times
    reference_exit = (
        trades_df["exit_time"].max()
        if not trades_df["exit_time"].isna().all()
        else pd.Timestamp.now()
    )
    trades_df["adjusted_exit"] = trades_df["exit_time"].fillna(reference_exit)

    # Create timeline
    timeline = pd.concat(
        [
            pd.DataFrame({"time": trades_df["entry_time"], "delta": 1}),
            pd.DataFrame({"time": trades_df["adjusted_exit"], "delta": -1}),
        ]
    )
    timeline = timeline.sort_values("time")
    timeline["active_trades"] = timeline["delta"].cumsum()

    return timeline["active_trades"].max()


def update_trades_execution_status(
    trades_df, executable_trades, capital_constrained_trades
):
    """
    Update the original trades dataframe with execution status.
    """
    trades_df = trades_df.copy()
    trades_df["execution_status"] = "confirmed"  # Default for confirmed trades

    # Mark capital-constrained trades
    if not capital_constrained_trades.empty:
        for _, constrained_trade in capital_constrained_trades.iterrows():
            mask = (
                (trades_df["entry_time"] == constrained_trade["entry_time"])
                & (trades_df["symbol"] == constrained_trade["symbol"])
                & (trades_df["term_trade_type"] == constrained_trade["term_trade_type"])
            )
            trades_df.loc[mask, "execution_status"] = "capital_constrained"

    return trades_df


def analyze_strategy_performance_multiple_tickers(
    trades_df: pd.DataFrame, play_size: int, initial_portfolio: float
):
    total_signals = len(trades_df)
    not_triggered_signals = len(trades_df[trades_df["confirmation"] == "not_triggered"])
    trades_df["pnl_pct"] = trades_df["pnl_usd"] / play_size * 100
    trades_df["pnl_portfolio_pct"] = trades_df["pnl_usd"] / initial_portfolio * 100
    confirmed_trades = trades_df[trades_df["confirmation"] == "confirmed"].copy()
    if not confirmed_trades.empty:

        print("\n==== Overall Performance ====")
        total_pnl = trades_df["pnl_usd"].sum()
        final_portfolio = initial_portfolio + total_pnl
        total_pnl_pct = (total_pnl / initial_portfolio) * 100
        average_pnl = trades_df[
            "pnl_usd"
        ].mean()  # if not trades_df["pnl_usd"].isna().all() else 0

        average_pnl_pct = trades_df[
            "pnl_pct"
        ].mean()  # if not trades_df["pnl_pct"].isna().all() else 0
        average_pnl_portfolio = trades_df["pnl_portfolio_pct"].mean()

        win_rate = (
            len(confirmed_trades[confirmed_trades["pnl_usd"] > 0])
            / len(confirmed_trades)
            * 100
        )
        df = trades_df.copy()

        df = df[df["confirmation"] == "confirmed"]
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"])

        reference_exit = (
            pd.Timestamp(df["exit_time"].max())
            if not df["exit_time"].isna().all()
            else pd.Timestamp.now()
        )
        df["adjusted_exit"] = df["exit_time"].fillna(reference_exit)
        timeline = pd.concat(
            [
                pd.DataFrame({"time": df["entry_time"], "delta": 1}),
                pd.DataFrame({"time": df["adjusted_exit"], "delta": -1}),
            ]
        )
        timeline = timeline.sort_values("time")
        timeline["active_trades"] = timeline["delta"].cumsum()
        max_active_trades = timeline["active_trades"].max()
        max_active_trades_count_true = timeline["active_trades"] == max_active_trades
        max_active_trades_count = max_active_trades_count_true.sum()

        max_cash_used = max_active_trades * play_size
        if max_cash_used > initial_portfolio:
            warning = (
                "Warning: Max cash used exceeds initial portfolio value! "
                + "Please, increase your initial cash amount to get all stats correct."
            )
        else:
            warning = ""
        print(f"Initial Portfolio Value:       ${initial_portfolio:.2f}")
        print(f"Cash per Play:                 ${play_size}")
        print(
            f"Max Cash in Play at Once:      ${max_cash_used} ({max_active_trades_count} times) {warning}\n"
        )
        print(f"Final Portfolio Value:         ${final_portfolio:.2f}")
        print(f"Total PnL:                     ${total_pnl:.2f} ({total_pnl_pct:.2f}%)")
        print(f"\nTrades Executed:               {len(confirmed_trades)}")
        print(f"Total Signals:                 {total_signals}")
        print(f"Not Triggered Signals:         {not_triggered_signals}")
        print(f"Win rate:                      {win_rate:.2f}%\n")

        print(
            f"Average PnL per Trade:          ${average_pnl:.2f} ({average_pnl_pct:.2f}%)"
        )
        print(f"Average Portfolio Impact:       {average_pnl_portfolio:.2f}%\n")

        best_trade = confirmed_trades.loc[confirmed_trades["pnl_usd"].idxmax()]
        worst_trade = confirmed_trades.loc[confirmed_trades["pnl_usd"].idxmin()]
        print("\n== Best Trade ==")
        print(f"Ticker: {best_trade['symbol']}")
        print(f"PnL:  ${best_trade['pnl_usd']:.2f} ({best_trade['pnl_pct']:.2f}%)")
        print(f"Time: {best_trade['entry_time']} → {best_trade['exit_time']}")
        print("\n== Worst Trade ==")
        print(f"Ticker: {worst_trade['symbol']}")
        print(f"PnL: ${worst_trade['pnl_usd']:.2f} ({worst_trade['pnl_pct']:.2f}%)")
        print(f"Time: {worst_trade['entry_time']} → {worst_trade['exit_time']}")

        print("\n\n==== Performance by Term Type ====")
        grouped = confirmed_trades.groupby("term_trade_type").agg(
            trade_count=("pnl_usd", "count"),
            avg_pnl_usd=("pnl_usd", "mean"),
            avg_pnl_pct=("pnl_pct", "mean"),
            avg_portfolio_impact=("pnl_portfolio_pct", "mean"),
            total_pnl_usd=("pnl_usd", "sum"),
            win_rate=("pnl_usd", lambda x: (x > 0).sum() / len(x) * 100),
        )

        display(grouped.round(2))

        confirmed_trades["trade_duration"] = pd.to_datetime(
            confirmed_trades["exit_time"]
        ) - pd.to_datetime(confirmed_trades["entry_time"])

        valid_durations = confirmed_trades["trade_duration"].dropna()

        avg_duration = valid_durations.mean()
        min_duration = valid_durations.min()
        max_duration = valid_durations.max()

        print("\n\n==== Trade Duration Stats ====")
        print(f"Average Trade Duration: {avg_duration}\n")
        print(f"Shortest Trade:   {min_duration}")
        print(f"Longest Trade:    {max_duration}\n")
        effect_trade_days = confirmed_trades["entry_time"].dt.date.nunique()
        print(f"Effective Trading Days: {effect_trade_days}")
        print(
            f"Average Trades per Day: {len(confirmed_trades) / effect_trade_days:.1f}"
        )
        # Trade duration stats for profitable and losing trades
        """
        profitable_trades = confirmed_trades[confirmed_trades["pnl_usd"] > 50]
        losing_trades = confirmed_trades[confirmed_trades["pnl_usd"] < 0]
        if not profitable_trades.empty:
            avg_duration_profit = profitable_trades["trade_duration"].median()
            min_duration_profit = profitable_trades["trade_duration"].min()
            max_duration_profit = profitable_trades["trade_duration"].max()
            max_duration_profit_trade = profitable_trades.loc[
                profitable_trades["trade_duration"].idxmax()
            ]
            print("\n\n==== Profitable Trades Duration Stats ====")
            print(f"Average Duration: {avg_duration_profit}")
            print(f"Shortest Trade:   {min_duration_profit}")
            print(f"Longest Trade:    {max_duration_profit}")
            print(max_duration_profit_trade)

        else:
            print("\n\n==== No Profitable Trades Found ====")

        if not losing_trades.empty:
            avg_duration_loss = losing_trades["trade_duration"].median()
            min_duration_loss = losing_trades["trade_duration"].min()
            max_duration_loss = losing_trades["trade_duration"].max()
            max_duration_loss_trade = losing_trades.loc[
                losing_trades["trade_duration"].idxmax()
            ]
            print("\n\n==== Losing Trades Duration Stats ====")
            print(f"Average Duration: {avg_duration_loss}")
            print(f"Shortest Trade:   {min_duration_loss}")
            print(f"Longest Trade:    {max_duration_loss}")
            print(max_duration_loss_trade)
        else:
            print("\n\n==== No Losing Trades Found ====")
            """
    else:
        print("No confirmed trades found.")


def analyze_strategy_performance(
    trades_df: pd.DataFrame, ochlv_df: pd.DataFrame, play_size: int
):

    confirmed_trades = trades_df[trades_df["confirmation"] == "confirmed"].copy()
    if not confirmed_trades.empty:
        first_trade_time = confirmed_trades["entry_time"].min().date()
        last_trade_time = confirmed_trades["exit_time"].max().date()

        backtect_star_time = ochlv_df.index.min().date()
        backtest_end_time = ochlv_df.index.max().date()

        print("\n==== Overall Performance ====")
        initial_portfolio = ochlv_df["total_portfolio"].iloc[0]
        final_portfolio = ochlv_df["total_portfolio"].iloc[-1]
        total_pnl = final_portfolio - initial_portfolio
        total_pnl_pct = (total_pnl / initial_portfolio) * 100
        average_pnl = trades_df[
            "pnl_usd"
        ].mean()  # if not trades_df["pnl_usd"].isna().all() else 0
        average_pnl_pct = trades_df[
            "pnl_pct"
        ].mean()  # if not trades_df["pnl_pct"].isna().all() else 0
        average_pnl_portfolio = trades_df["pnl_portfolio_pct"].mean()
        average_pnl_per_profit = trades_df[trades_df["pnl_usd"] > 0]["pnl_usd"].mean()
        average_pnl_per_profit_pct = average_pnl_per_profit / play_size * 100
        average_pnl_per_loss = trades_df[trades_df["pnl_usd"] < 0]["pnl_usd"].mean()
        average_pnl_per_loss_pct = average_pnl_per_loss / play_size * 100
        max_portfolio = ochlv_df["total_portfolio"].max()
        min_portfolio = ochlv_df["total_portfolio"].min()
        drawdown_pct = ((max_portfolio - min_portfolio) / max_portfolio) * 100
        win_rate = (
            len(confirmed_trades[confirmed_trades["pnl_usd"] > 0])
            / len(confirmed_trades)
            * 100
        )
        min_portfolio_below_initial = ((min_portfolio / initial_portfolio) - 1) * 100
        max_cash_used = int(ochlv_df["open_position_count"].max()) * play_size
        if max_cash_used > initial_portfolio:
            warning = (
                "Warning: Max cash used exceeds initial portfolio value! "
                + "Please, increase your initial cash amount to get all stats correct."
            )
        else:
            warning = ""
        print(f"Initial Portfolio Value:       ${initial_portfolio:.2f}")
        print(f"Cash per Play:                 ${play_size}")
        print(f"Max Cash in Play at Once:      ${max_cash_used} {warning}\n")
        print(f"Final Portfolio Value:         ${final_portfolio:.2f}")
        print(f"Total PnL:                     ${total_pnl:.2f}")
        print(f"Total PnL per Porfolio:         {total_pnl_pct:.2f}%")
        print(f"Total PnL per Max Cash in Play: {total_pnl / max_cash_used * 100:.2f}%")
        print(f"Total Trades:                   {len(confirmed_trades)}")
        print(f"Win rate:                       {win_rate:.2f}%\n")
        print(
            f"Average PnL per Trade:          ${average_pnl:.2f} ({average_pnl_pct:.2f}%)"
        )
        print(f"Average Portfolio Impact:       {average_pnl_portfolio:.2f}%\n")
        print(f"\nMax Drawdown:             {drawdown_pct:.2f}%")
        print(f"Max Portfolio Value:      ${max_portfolio:.2f}")
        print(
            f"Min Portfolio Value:      ${min_portfolio:.2f} "
            f"({min_portfolio_below_initial:.2f}% from initial value)"
        )

        best_trade = confirmed_trades.loc[confirmed_trades["pnl_usd"].idxmax()]
        worst_trade = confirmed_trades.loc[confirmed_trades["pnl_usd"].idxmin()]
        print("\n== Best Trade ==")
        print(f"PnL:  ${best_trade['pnl_usd']:.2f} ({best_trade['pnl_pct']:.2f}%)")
        print(f"Time: {best_trade['entry_time']} → {best_trade['exit_time']}")
        print("== Worst Trade ==")
        print(f"PnL: ${worst_trade['pnl_usd']:.2f} ({worst_trade['pnl_pct']:.2f}%)")
        print(f"Time: {worst_trade['entry_time']} → {worst_trade['exit_time']}")

        print(
            f"\nTrades occurred between {first_trade_time} and {last_trade_time}, "
            f"while backtest was run from {backtect_star_time} to {backtest_end_time}."
        )
        backtest_duration = pd.to_datetime(backtest_end_time) - pd.to_datetime(
            backtect_star_time
        )
        backtest_duration = backtest_duration.days
        print(f"Backtest Duration:      {backtest_duration} days (including weekends)")
        effect_trade_days = confirmed_trades["entry_time"].dt.date.nunique()
        print(f"Effective Trading Days: {effect_trade_days}")
        print(
            f"Average Trades per Day: {len(confirmed_trades) / effect_trade_days:.1f}"
        )

        print("\n\n==== Performance by Term Type ====")
        grouped = confirmed_trades.groupby("term_trade_type").agg(
            trade_count=("pnl_usd", "count"),
            avg_pnl_usd=("pnl_usd", "mean"),
            avg_pnl_pct=("pnl_pct", "mean"),
            avg_portfolio_impact=("pnl_portfolio_pct", "mean"),
            total_pnl_usd=("pnl_usd", "sum"),
            win_rate=("pnl_usd", lambda x: (x > 0).sum() / len(x) * 100),
        )

        display(grouped.round(2))

        confirmed_trades["trade_duration"] = pd.to_datetime(
            confirmed_trades["exit_time"]
        ) - pd.to_datetime(confirmed_trades["entry_time"])

        valid_durations = confirmed_trades["trade_duration"].dropna()

        avg_duration = valid_durations.mean()
        min_duration = valid_durations.min()
        max_duration = valid_durations.max()

        print("\n\n\n==== Trade Duration Stats ====")
        print(f"Average Trade Duration: {avg_duration}\n")
        print(f"Shortest Trade:   {min_duration}")
        print(f"Longest Trade:    {max_duration}")
    else:
        print("No confirmed trades found.")


def analyze_strategy_performance_bulk(
    trades_df: pd.DataFrame, ochlv_df: pd.DataFrame, play_size: int
):

    confirmed_trades = trades_df[trades_df["confirmation"] == "confirmed"].copy()
    if not confirmed_trades.empty:
        print("\n==== Overall Performance ====")
        initial_portfolio = ochlv_df["total_portfolio"].iloc[0]
        final_portfolio = ochlv_df["total_portfolio"].iloc[-1]
        total_pnl = final_portfolio - initial_portfolio
        total_pnl_pct = (total_pnl / initial_portfolio) * 100
        average_pnl = (
            trades_df["pnl_usd"].mean() if not trades_df["pnl_usd"].isna().all() else 0
        )
        average_pnl_pct = (
            trades_df["pnl_pct"].mean() if not trades_df["pnl_pct"].isna().all() else 0
        )
        average_pnl_portfolio = (
            trades_df["pnl_portfolio_pct"].mean()
            if not trades_df["pnl_portfolio_pct"].isna().all()
            else 0
        )

        max_portfolio = ochlv_df["total_portfolio"].max()
        min_portfolio = ochlv_df["total_portfolio"].min()
        drawdown_pct = ((max_portfolio - min_portfolio) / max_portfolio) * 100
        max_cash_used = int(ochlv_df["open_position_count"].max()) * play_size
        if max_cash_used > initial_portfolio:
            warning = (
                "Warning: Max cash used exceeds initial portfolio value! "
                + "Please, increase your initial cash amount to get all stats correct."
            )
        else:
            warning = ""
        print(f"Initial Portfolio Value:  ${initial_portfolio:.2f}")
        print(f"Max Cash Used at Once:    ${max_cash_used} {warning}")
        print(f"Final Portfolio Value:    ${final_portfolio:.2f}")
        print(f"Total PnL:                ${total_pnl:.2f} ({total_pnl_pct:.2f}%)")
        print(f"Average PnL per Trade:    ${average_pnl:.2f} ({average_pnl_pct:.2f}%)")
        print(f"Average Portfolio Impact: {average_pnl_portfolio:.2f}%")
        print(f"Max Drawdown:             {drawdown_pct:.2f}%")
        print(f"Max Portfolio Value:      ${max_portfolio:.2f}")
        print(f"Min Portfolio Value:      ${min_portfolio:.2f}")
        print(f"Total Trades:             {len(confirmed_trades)}")


def compute_dynamic_portfolio_with_breakdown(
    ochlv_df: pd.DataFrame, trades_df: pd.DataFrame, initial_cash, capital_per_trade
) -> pd.DataFrame:

    ochlv_df = ochlv_df.copy()
    ochlv_df.index = pd.to_datetime(ochlv_df.index)
    trades_df = trades_df.copy()
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

    confirmed = trades_df[
        (trades_df["confirmation"] == "confirmed") & trades_df["entry_time"].notna()
    ].copy()

    open_positions = []
    portfolio_values = []
    cash_balances = []
    stocks_balances = []
    open_position_counts = []

    cash = initial_cash

    for current_time in ochlv_df.index:
        current_price = ochlv_df.loc[current_time, "close"]
        new_trades = confirmed[confirmed["entry_time"] == current_time]
        for _, trade in new_trades.iterrows():
            shares = capital_per_trade / trade["entry_price"]
            cash -= capital_per_trade
            open_positions.append(
                {
                    "shares": shares,
                    "direction": trade["term_trade_type"],
                    "entry_price": trade["entry_price"],
                    "exit_time": trade["exit_time"],
                    "exit_time_key": trade["exit_time"],
                    "entry_time": trade["entry_time"],
                }
            )

        for pos in list(open_positions):
            if pd.notna(pos["exit_time"]) and pos["exit_time"] <= current_time:
                trade_pnl_row = confirmed[
                    (confirmed["exit_time"] == pos["exit_time"])
                    & (confirmed["entry_time"] == pos["entry_time"])
                ]
                if not trade_pnl_row.empty:
                    pnl_usd = trade_pnl_row["pnl_usd"].iloc[0]
                else:
                    pnl_usd = pos["shares"] * current_price - capital_per_trade
                cash += capital_per_trade + pnl_usd
                open_positions.remove(pos)

        open_value = 0
        for pos in open_positions:
            open_value += pos["shares"] * current_price

        total_value = cash + open_value

        portfolio_values.append(total_value)
        cash_balances.append(cash)
        stocks_balances.append(open_value)
        open_position_counts.append(len(open_positions))

    ochlv_df["total_portfolio"] = portfolio_values
    ochlv_df["cash_balance"] = cash_balances
    ochlv_df["stocks_balance"] = stocks_balances
    ochlv_df["open_position_count"] = open_position_counts

    return ochlv_df


def compute_dynamic_portfolio_with_breakdown_og(
    ochlv_df: pd.DataFrame, trades_df: pd.DataFrame, initial_cash, capital_per_trade
) -> pd.DataFrame:

    ochlv_df = ochlv_df.copy()
    ochlv_df.index = pd.to_datetime(ochlv_df.index)
    trades_df = trades_df.copy()
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

    confirmed = trades_df[
        (trades_df["confirmation"] == "confirmed") & trades_df["entry_time"].notna()
    ].copy()

    open_positions = []
    portfolio_values = []
    cash_balances = []
    stocks_balances = []
    open_position_counts = []

    cash = initial_cash

    for current_time in ochlv_df.index:
        current_price = ochlv_df.loc[current_time, "close"]
        new_trades = confirmed[confirmed["entry_time"] == current_time]
        for _, trade in new_trades.iterrows():
            shares = capital_per_trade / trade["entry_price"]
            cash -= capital_per_trade
            open_positions.append(
                {
                    "shares": shares,
                    "direction": trade["term_trade_type"],
                    "entry_price": trade["entry_price"],
                    "exit_time": trade["exit_time"],
                }
            )
        for pos in list(open_positions):
            if pd.notna(pos["exit_time"]) and pos["exit_time"] <= current_time:
                cash += pos["shares"] * current_price
                open_positions.remove(pos)

        open_value = 0
        for pos in open_positions:
            open_value += pos["shares"] * current_price

        total_value = cash + open_value

        portfolio_values.append(total_value)
        cash_balances.append(cash)
        stocks_balances.append(open_value)
        open_position_counts.append(len(open_positions))

    ochlv_df["total_portfolio"] = portfolio_values
    ochlv_df["cash_balance"] = cash_balances
    ochlv_df["stocks_balance"] = stocks_balances
    ochlv_df["open_position_count"] = open_position_counts

    return ochlv_df


def plot_portfolio_trade_signals(
    trades_df: pd.DataFrame,
    ochlv_df: pd.DataFrame,
    trend_direction: Literal["bullish", "bearish"],
    symbol: str,
    width_px: int = 1100,
    height_px: int = 700,
) -> None:
    trades_df = trades_df.copy()
    ochlv_df = ochlv_df.copy()

    confirmed_trades = trades_df[trades_df["confirmation"] == "confirmed"]
    not_confirmed_trades = trades_df[trades_df["confirmation"] == "not_confirmed"]

    fig = go.Figure()

    size_map = {"short": 10, "medium": 11, "long": 12}
    color_map_entry = {
        "short": PASTEL_COLORS_RGB["aqua blue"],
        "medium": PASTEL_COLORS_RGB["lavender blue"],
        "long": PASTEL_COLORS_RGB["mint green"],
    }
    color_map_exit = {
        "short": PASTEL_COLORS_RGBA_60["aqua blue"],
        "medium": PASTEL_COLORS_RGBA_60["lavender blue"],
        "long": PASTEL_COLORS_RGBA_60["mint green"],
    }

    for term_type in confirmed_trades["term_trade_type"].unique():
        sub_df = confirmed_trades[confirmed_trades["term_trade_type"] == term_type]
        confirmed_customdata = sub_df[["pnl_pct"]].copy()
        confirmed_customdata.insert(0, "index", sub_df.index)
        fig.add_trace(
            go.Scatter(
                x=sub_df["entry_time"],
                y=sub_df["entry_price"],
                mode="markers",
                marker=dict(
                    color=color_map_entry[term_type],
                    size=size_map[term_type],
                    sizemode="diameter",
                ),
                name=f"Entry ({term_type})",
                customdata=confirmed_customdata.values,
                hovertemplate=(
                    "ID: %{customdata[0]}<br>"
                    "Entry: %{x}<br>"
                    "Price: %{y}<br>"
                    "PnL per play: %{customdata[1]:.2f}%<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=ochlv_df.index,
            y=ochlv_df["close"],
            mode="lines",
            name="Close Price (1min)",
            line=dict(color=PASTEL_COLORS_RGB["neutral grey"], width=1),
        )
    )

    for term_type in confirmed_trades["term_trade_type"].unique():
        sub_df = confirmed_trades[confirmed_trades["term_trade_type"] == term_type]
        confirmed_customdata = sub_df[["pnl_pct"]].copy()
        confirmed_customdata.insert(0, "index", sub_df.index)

        fig.add_trace(
            go.Scatter(
                x=sub_df["exit_time"],
                y=sub_df["exit_price"],
                mode="markers",
                marker=dict(
                    color=color_map_exit[term_type],
                    size=size_map[term_type],
                    sizemode="diameter",
                ),
                name=f"Exit ({term_type})",
                customdata=confirmed_customdata.values,
                hovertemplate=(
                    "ID: %{customdata[0]}<br>"
                    "Exit: %{x}<br>"
                    "Price: %{y}<br>"
                    "PnL per play: %{customdata[1]:.2f}%<extra></extra>"
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=ochlv_df.index,
            y=ochlv_df["total_portfolio"],
            mode="lines",
            name="Total Portfolio",
            yaxis="y2",
            line=dict(color=PASTEL_COLORS_RGBA_60["leafy green"], width=1),
        )
    )

    for term_type in not_confirmed_trades["term_trade_type"].unique():
        sub_df = not_confirmed_trades[
            not_confirmed_trades["term_trade_type"] == term_type
        ]
        fig.add_trace(
            go.Scatter(
                x=sub_df["entry_time"],
                y=sub_df["entry_price"],
                mode="markers",
                marker=dict(
                    color=PASTEL_COLORS_RGBA_60["neutral grey"],
                    size=size_map[term_type],
                    sizemode="diameter",
                ),
                name=f"Rejected ({term_type})",
                customdata=not_confirmed_trades[["term_trade_type"]].values,
                hovertemplate=(
                    "Entry: %{x}<br>"
                    "Price: %{y}<br>"
                    "Term trade: %{customdata}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        width=width_px,
        height=height_px,
        title=dict(
            text=f"{symbol} Trade Signals: {trend_direction.capitalize()} play",
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="bottom",
            y=0.95,
            x=0.02,
            font=dict(size=24),
        ),
        title_subtitle=dict(
            text="Short, medium and long term trades with ATR based exits",
            font=dict(size=16),
        ),
        margin=dict(l=30, r=100, t=180, b=60),
        legend=dict(
            title="Legend: ",
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            itemsizing="constant",
            font=dict(size=14),
        ),
        yaxis=dict(
            title="Stock Price, USD",
            ticks="outside",
            tickcolor="black",
            ticklen=10,
        ),
        yaxis2=dict(
            title="Total Portfolio, USD",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        xaxis=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[16, 9.5], pattern="hour"),
            ]
        ),
    )

    fig.show()


def plot_trade_signals(
    trades_df: pd.DataFrame,
    ochlv_df: pd.DataFrame,
    trend_direction: Literal["bullish", "bearish"],
    symbol: str,
    width_px: int = 1200,
    height_px: int = 500,
) -> None:
    trades_df = trades_df.copy()
    trades_df = trades_df[trades_df["direction"] == trend_direction]
    ochlv_df = ochlv_df.copy()
    # here add function?
    min_entry_time = trades_df["entry_time"].min()
    min_entry_plus_one_day = min_entry_time - timedelta(days=1)

    ochlv_df = ochlv_df[ochlv_df.index >= min_entry_plus_one_day]
    confirmed_trades = trades_df[trades_df["confirmation"] == "confirmed"]
    not_confirmed_trades = trades_df[trades_df["confirmation"] == "not_confirmed"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ochlv_df.index,
            y=ochlv_df["close"],
            mode="lines",
            name="1min Close Price",
            line=dict(color=PASTEL_COLORS_RGB["neutral grey"], width=1),
        )
    )

    customdata = confirmed_trades[["term_trade_type"]].copy()
    customdata.insert(0, "index", confirmed_trades.index)
    fig.add_trace(
        go.Scatter(
            x=confirmed_trades["entry_time"],
            y=confirmed_trades["entry_price"],
            mode="markers",
            marker=dict(color=PASTEL_COLORS_RGB["leafy green"], size=12),
            name="Confirmed Entry",
            customdata=customdata.values,
            hovertemplate=(
                "Entry: %{x}<br>"
                "Price: %{y}<br>"
                "ID: %{customdata[0]}<br>"
                "Signal: %{customdata[1]}<br>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=confirmed_trades["exit_time"],
            y=confirmed_trades["exit_price"],
            mode="markers",
            marker=dict(color=PASTEL_COLORS_RGB["lavender blue"], size=12),
            name="Exit",
            customdata=customdata.values,
            hovertemplate=(
                "Exit: %{x}<br>"
                "Price: %{y}<br>"
                "ID: %{customdata[0]}<br>"
                "Signal: %{customdata[1]}<br>"
            ),
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=not_confirmed_trades["entry_time"],
            y=not_confirmed_trades["entry_price"],
            mode="markers",
            marker=dict(color=PASTEL_COLORS_RGB["coral orange"], size=12),
            name="Not Confirmed Entry",
            customdata=not_confirmed_trades[["term_trade_type"]],
            hovertemplate=(
                "Entry: %{x}<br>Price: %{y}<br>Term type: %{customdata}<extra></extra>"
            ),
        ),
    )

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[16, 9.5], pattern="hour"),
        ]
    )

    fig.update_layout(
        width=width_px,
        height=height_px,
        title=dict(
            text=f"{symbol} Trade Signals",
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="bottom",
            y=0.92,
            x=0.02,
            font=dict(size=20),
        ),
        title_subtitle=dict(
            text=f"{trend_direction.capitalize()} play", font=dict(size=14)
        ),
        margin=dict(l=10, r=30, t=80, b=60),
    )
    fig.update_yaxes(
        ticks="outside",
        tickcolor="black",
        ticklen=10,
    )

    fig.show()


def check_missing_values(df: pd.DataFrame, missing_value: int = None) -> None:
    """
    Checks for missing values in the given DataFrame and returns a summary
    DataFrame with counts and percentages. Optionally considers a specific
    value as a missing indicator, in addition to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be checked for missing values.
    missing_value : int, optional
        A value to consider as missing, in addition to NaN. Default is None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the count and percentage of missing values
        per column.
    """
    missing_counts = df.isna().sum()

    if missing_value is not None:
        missing_counts += (df == missing_value).sum()

    total_rows = len(df)
    missing_percent = (missing_counts / total_rows) * 100

    missing_summary = pd.DataFrame(
        {
            "missing_count": missing_counts,
            "missing_percentage": round(missing_percent, 2),
        }
    )
    missing_summary = missing_summary[missing_summary["missing_count"] > 0]

    if missing_summary.empty:
        print("The dataset does not contain any missing values.")
    else:
        print("Missing value summary:")

    return missing_summary


def check_duplicates(df: pd.DataFrame, column_names: list = None) -> pd.DataFrame:
    """
    Checks for duplicate rows in the DataFrame based on the specified column names.
    If no column names are provided, checks for duplicates across the entire DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be checked for duplicates.
    column_names : list, optional
        List of column names to check for duplicates. Defaults to None, meaning full
        rows will be checked.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the duplicate rows. If no duplicates are found,
        an empty DataFrame is returned.
    """
    if column_names is None:
        column_names = df.columns.tolist()

    duplicate_rows = df[df.duplicated(subset=column_names, keep=False)]

    if not duplicate_rows.empty:
        print(
            f"There are {len(duplicate_rows)} duplicate rows based on the columns: "
            f"{column_names}"
        )
        return duplicate_rows
    else:
        print(f"No duplicate rows found based on the columns: {column_names}")


def find_outlier_rows_by_iqr(
    df: pd.DataFrame, columns: list = None, multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Identify and return rows in a DataFrame that contain outliers based on the
    Interquartile Range (IQR) method.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to analyze for outliers.
    columns : list, optional
        A list of column names to check for outliers. If None (default), the function
        will analyze all numerical columns (int64 and float64).
    multiplier : float, optional
        The multiplier for the IQR to define the outlier bounds. Default is 1.5.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing only the rows from the original DataFrame that are
        identified as having outliers in any of the specified columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    outlier_mask = pd.Series(False, index=df.index)

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        col_outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_mask |= col_outlier_mask

    return df[outlier_mask]


def find_unseen_categories(
    train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_columns: list[str]
) -> list[str]:
    """
    Compares categorical columns in train and test DataFrames.
    Returns a list of columns where the test set contains unseen categories.
    """
    columns_with_unseen_categories = []

    for col in categorical_columns:
        train_unique = set(train_df[col].dropna().unique())
        test_unique = set(test_df[col].dropna().unique())
        unseen = test_unique - train_unique
        if unseen:
            columns_with_unseen_categories.append(col)

    return columns_with_unseen_categories


def select_true_numeric(df: pd.DataFrame, threshold: int = 10) -> list[str]:
    """
    Selects columns that are truly numeric based on data type and number
    of unique values.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    threshold : int, optional
        Minimum number of unique values required to consider a column truly numeric.
        Default is 10.

    Returns
    -------
    List[str]
        A list of column names that are numeric and have more than the specified number
        of unique values.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    true_numeric = [col for col in numeric_cols if df[col].nunique() > threshold]
    return true_numeric


def find_highly_correlated_pairs(
    df: pd.DataFrame, method="pearson", threshold=0.8
) -> pd.DataFrame:
    """
    Finds feature pairs in a DataFrame that are highly correlated.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    method : str, optional
        Correlation method: 'pearson', 'spearman', or 'kendall' (default is 'pearson').
    threshold : float, optional
        Absolute correlation threshold to flag as "high" (default is 0.8).

    Returns
    -------
    pd.DataFrame
        DataFrame with feature pairs, correlation, and count of how often each feature
        appears.
    """
    corr_matrix = df.corr(method=method)
    upper = corr_matrix.where(
        ~pd.DataFrame(
            np.tri(corr_matrix.shape[0], dtype=bool),
            index=corr_matrix.index,
            columns=corr_matrix.columns,
        )
    )
    high_corr = (
        upper.stack()
        .reset_index()
        .rename(
            columns={"level_0": "feature_1", "level_1": "feature_2", 0: "corr_coef"}
        )
    )

    high_corr = high_corr[high_corr["corr_coef"].abs() > threshold]
    feature_counts = pd.concat(
        [high_corr["feature_1"], high_corr["feature_2"]]
    ).value_counts()

    high_corr["feature_1_count"] = high_corr["feature_1"].map(feature_counts)
    high_corr["feature_2_count"] = high_corr["feature_2"].map(feature_counts)

    return high_corr.sort_values(by="corr_coef", ascending=False).reset_index(drop=True)


def find_highly_correlated_binary_pairs(
    df: pd.DataFrame, threshold=0.8
) -> pd.DataFrame:
    """
    Finds highly correlated binary feature pairs using the Phi coefficient.
    Phi coefficient is found usinf Pearson method.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    threshold : float, optional
        Absolute Phi coefficient threshold to flag (default is 0.8).

    Returns
    -------
    pd.DataFrame
        DataFrame with binary feature pairs, Phi correlation, and
        feature pair frequencies.
    """
    corr_matrix = df.corr(method="pearson")
    upper = corr_matrix.where(
        ~pd.DataFrame(
            np.tri(corr_matrix.shape[0], dtype=bool),
            index=corr_matrix.index,
            columns=corr_matrix.columns,
        )
    )

    high_phi = (
        upper.stack()
        .reset_index()
        .rename(
            columns={
                "level_0": "feature_1",
                "level_1": "feature_2",
                0: "phi_corr_coef",
            }
        )
    )

    high_phi = high_phi[high_phi["phi_corr_coef"].abs() > threshold]

    feature_counts = pd.concat(
        [high_phi["feature_1"], high_phi["feature_2"]]
    ).value_counts()

    high_phi["feature_1_count"] = high_phi["feature_1"].map(feature_counts)
    high_phi["feature_2_count"] = high_phi["feature_2"].map(feature_counts)

    return high_phi.sort_values(by="phi_corr_coef", ascending=False).reset_index(
        drop=True
    )
