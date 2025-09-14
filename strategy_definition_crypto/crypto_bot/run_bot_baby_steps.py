import os

from datetime import datetime, time as dtime
from crypto_bot.parameter_manager import ParameterManager
from crypto_bot.data_fetcher import DataFetcher
from crypto_bot.signal_generator import SignalGenerator
from crypto_bot.risk_manager import RiskManager
from crypto_bot.broker import Broker
from crypto_bot.strategy_engine import StrategyEngine
from crypto_bot.trade_logger import TradeLogger

from ib_insync import IB
import asyncio
import json

SYMBOLS = ["AAPL", "LUV", "SMR"]
TRADE_TYPES = ["short", "medium", "long"]
# run calibration module as a separate, scheduled process that updates
# parameter files before the next trading week.


def check_kill_switch(kill_file="KILL_SWITCH"):
    # for no just a directory
    """
    upload a file (e.g., named KILL_SWITCH) to a specific bucket in Google Cloud Storage.
    bot checks for the existence or content of this file every minute (or at your chosen interval).
    If the file exists (or contains a certain flag), the bot shuts down or stops trading.
    The file can be empty (existence = stop).
    How does a non-coder use it?
    Go to Google Cloud Console → Storage → Your Bucket.
    Click "Upload file" and select a file named KILL_SWITCH (can be created in Notepad/TextEdit).
    To turn the bot back on: Delete the file from the bucket.
    """
    return os.path.exists(kill_file)


def is_market_open():
    now = datetime.now().time()
    return dtime(9, 45) <= now <= dtime(15, 55)  # adjust end time as needed


def delete_data_files(file_paths):
    for path in file_paths.values():
        if os.path.exists(path):
            os.remove(path)


def build_file_paths(symbol):
    return {
        "1min": f"data_live/{symbol}_1min.csv",
        "3min": f"data_live/{symbol}_3min.csv",
        "5min": f"data_live/{symbol}_5min.csv",
        "1h": f"data_live/{symbol}_1h.csv",
        "4h": f"data_live/{symbol}_4h.csv",
    }


ticker_path = "path.json"

with open(
    "trading_bot_development/bot_modules_calib/config/ticker_metadata.json", "r"
) as f:
    metadata = json.load(f)


async def connect_ib(ib):
    while not ib.isConnected():
        try:
            ib.connect("127.0.0.1", 4002, clientId=1)
            print("✅ Connected to IB Gateway")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            await asyncio.sleep(5)


async def run_realtime_bot():
    ib = IB()
    await connect_ib(ib)

    bots = []
    trade_logger = TradeLogger("trades.db")

    portfolio_morning_value = None
    last_archive_date = None
    last_portfolio_update_date = None

    for symbol in SYMBOLS:
        for trade_type in TRADE_TYPES:
            file_paths = build_file_paths(symbol)
            company_type = metadata[symbol]["group"]
            params = ParameterManager.load_params(
                symbol=symbol,
                term_trade=trade_type,
                company_type=company_type,
                ticker_path=ticker_path,
            )

            data_fetcher = DataFetcher(symbol, API_KEY, file_paths)
            signal_generator = SignalGenerator(
                indicators=params.get_indicators(), **params.to_dict()
            )
            risk_manager = RiskManager()
            broker = Broker(ib)

            engine = StrategyEngine(
                data_fetcher,
                signal_generator,
                risk_manager,
                broker,
                params,
                trade_logger,
            )

            bots.append((symbol, trade_type, engine))

    try:
        while True:
            if not ib.isConnected():
                print("🔄 Reconnecting...")
                await connect_ib()

            if check_kill_switch():
                print("Kill switch activated! Closing all positions and shutting down.")
                for _, _, engine in bots:
                    engine.broker.close_all_positions()
                break

            now = datetime.now()
            today = now.date()
            if not is_market_open():
                print(f"[{now}] Market is closed. Sleeping...")

                if now.time() > dtime(15, 55) and last_archive_date != today:
                    trade_logger.archive_todays_trades()
                    last_archive_date = today
                    print("📦 Archived today's trades.")

                if now.time() > dtime(9, 30) and last_portfolio_update_date != today:
                    portfolio_morning_value = Broker.get_portfolio_value()
                    last_portfolio_update_date = today
                    print("📈 Morning portfolio value recorded.")

                await asyncio.sleep(60)
                continue

            # Run bots in parallel
            tasks = [
                engine.run_live_strategy(portfolio_morning_value)
                for _, _, engine in bots
            ]

            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                print(f"⚠️ Error running bots: {e}")

            # 🧹 Clean up symbol data IF NEEDED, I MIGHT NOT EVEN SAVE TO CSV?
            for symbol, _, _ in bots:
                delete_data_files(build_file_paths(symbol))

            await asyncio.sleep(60)

    except KeyboardInterrupt:
        print("Bot stopped by user.")
    finally:
        trade_logger.close()
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    asyncio.run(run_realtime_bot())
