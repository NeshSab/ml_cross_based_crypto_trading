import sqlite3
import json
from datetime import datetime


class TradeLogger:
    def __init__(self, db_path="trades.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            for table in ["todays_trades", "all_trades"]:
                self.conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id INTEGER PRIMARY KEY,
                        symbol TEXT,
                        direction TEXT,
                        trade_type TEXT,
                        entry_order_id TEXT,
                        entry_order_placement_time TEXT,
                        entry_trigger_price REAL,
                        entry_fill_time TEXT,
                        entry_fill_price REAL,
                        entry_fill_qty REAL,
                        exit_order_id TEXT,
                        exit_order_placement_time TEXT,
                        exit_trigger_price REAL,
                        exit_fill_time TEXT,
                        exit_fill_price REAL,
                        exit_fill_qty REAL,
                        stop_loss REAL,
                        amended_stop_loss REAL,
                        quantity REAL,
                        order_status TEXT,
                        pnl REAL,
                        params TEXT
                    )
                """
                )

    def log_entry(self, trade, table="todays_trades"):
        with self.conn:
            self.conn.execute(
                f"""
                INSERT INTO {table} (
                    symbol, direction, trade_type, entry_order_id, entry_order_placement_time,
                    entry_trigger_price, stop_loss, amended_stop_loss, quantity, order_status, params
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade["symbol"],
                    trade["direction"],
                    trade["trade_type"],
                    trade.get("entry_order_id"),
                    trade.get("entry_order_placement_time"),
                    trade.get("entry_trigger_price"),
                    trade.get("stop_loss"),
                    trade.get("amended_stop_loss"),
                    trade["quantity"],
                    "pending",  # status is pending until filled
                    json.dumps(trade.get("params", {})),
                ),
            )

    def update_entry_fill(
        self, entry_order_id, fill_time, fill_price, fill_qty, table="todays_trades"
    ):
        with self.conn:
            self.conn.execute(
                f"""
                UPDATE {table}
                SET entry_fill_time=?, entry_fill_price=?, entry_fill_qty=?, order_status='open'
                WHERE entry_order_id=? AND order_status='pending'
                """,
                (
                    fill_time,
                    fill_price,
                    fill_qty,
                    entry_order_id,
                ),
            )

    def log_exit_order(
        self,
        entry_order_id,
        exit_order_id,
        exit_order_placement_time,
        exit_trigger_price,
        table="todays_trades",
    ):
        with self.conn:
            self.conn.execute(
                f"""
                UPDATE {table}
                SET exit_order_id=?, exit_order_placement_time=?, exit_trigger_price=?
                WHERE entry_order_id=? AND order_status='open'
                """,
                (
                    exit_order_id,
                    exit_order_placement_time,
                    exit_trigger_price,
                    entry_order_id,
                ),
            )

    def update_exit_fill(
        self,
        entry_order_id,
        exit_fill_time,
        exit_fill_price,
        exit_fill_qty,
        table="todays_trades",
    ):
        with self.conn:
            # Fetch entry fill price, qty, direction for PnL calculation
            cur = self.conn.execute(
                f"SELECT entry_fill_price, entry_fill_qty, direction FROM {table} WHERE entry_order_id=? AND order_status='open'",
                (entry_order_id,),
            )
            row = cur.fetchone()
            if not row:
                print(f"No open trade found with entry_order_id={entry_order_id}")
                return

            entry_fill_price, entry_fill_qty, direction = row
            pnl = (
                (exit_fill_price - entry_fill_price) * entry_fill_qty
                if direction == "bullish"
                else (entry_fill_price - exit_fill_price) * entry_fill_qty
            )

            self.conn.execute(
                f"""
                UPDATE {table}
                SET exit_fill_time=?, exit_fill_price=?, exit_fill_qty=?, order_status='closed', pnl=?
                WHERE entry_order_id=? AND order_status='open'
                """,
                (
                    exit_fill_time,
                    exit_fill_price,
                    exit_fill_qty,
                    pnl,
                    entry_order_id,
                ),
            )

    def update_stop_loss(
        self, entry_order_id, amended_stop_loss, table="todays_trades"
    ):
        with self.conn:
            self.conn.execute(
                f"""
                UPDATE {table}
                SET amended_stop_loss=?
                WHERE entry_order_id=? AND order_status IN ('pending', 'open')
                """,
                (
                    amended_stop_loss,
                    entry_order_id,
                ),
            )

    def get_open_trades(self, symbol=None, table="todays_trades"):
        cur = self.conn.cursor()
        if symbol:
            cur.execute(
                f"SELECT * FROM {table} WHERE order_status IN ('pending', 'open') AND symbol=?",
                (symbol,),
            )
        else:
            cur.execute(
                f"SELECT * FROM {table} WHERE order_status IN ('pending', 'open')"
            )
        return cur.fetchall()

    def get_closed_trades(self, symbol=None, table="todays_trades"):
        cur = self.conn.cursor()
        if symbol:
            cur.execute(
                f"SELECT * FROM {table} WHERE order_status='closed' AND symbol=?",
                (symbol,),
            )
        else:
            cur.execute(f"SELECT * FROM {table} WHERE order_status='closed'")
        return cur.fetchall()

    def get_all_trades(self, table="todays_trades"):
        cur = self.conn.cursor()
        cur.execute(f"SELECT * FROM {table}")
        return cur.fetchall()

    def archive_todays_trades(self):
        """
        Move all trades from todays_trades to all_trades and clear todays_trades.
        """
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO all_trades
                SELECT * FROM todays_trades
                """
            )
            self.conn.execute("DELETE FROM todays_trades")

    def close(self):
        self.conn.close()
