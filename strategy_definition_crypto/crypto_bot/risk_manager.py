
class RiskManager:
    def __init__(
        self,
        max_daily_loss_pct=0.04,
        max_ticker_loss_pct=0.04,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_ticker_loss_pct = max_ticker_loss_pct
        self.daily_loss = 0.0
        self.ticker_losses = {} 
        self.morning_portfolio_value = None

    def assess_trade(self, direction, latest_price):
        """
        Assess if a trade can be placed based on risk rules.
        Returns a dict with 'approve', 'side', 'symbol', 'quantity', etc.
        """
        # Here you would check 
        # should ticker be still traded (haven;t lost 4% today)
        # position sizing, include max allowance per trade and per ticker 
        #
        # For now, always approve
        return {
            "approve": True,
            "side": "buy" if direction == "bullish" else "sell",
            "symbol": None,  # Fill in at call site
            "quantity": 1,  # Fill in at call site
        }

    def calculate_trailing_stop(self, position):
        """
        Calculate a new trailing stop based on ATR or other logic.
        """
        # Example: Use ATR from position info if available
        atr = position.get("atr")
        if (
            atr is None
            or "current_price" not in position
            or "direction" not in position
        ):
            return None
        if position["direction"] == "bullish":
            return position["current_price"] - self.trailing_stop_atr_mult * atr
        elif position["direction"] == "bearish":
            return position["current_price"] + self.trailing_stop_atr_mult * atr
        return None

    def check_max_daily_loss_per_ticker(self, position):
        """
        Check if the loss for this ticker exceeds the max allowed for the day.
        """
        symbol = position.get("symbol")
        pnl = position.get("pnl", 0)
        if symbol not in self.ticker_losses:
            self.ticker_losses[symbol] = 0
        self.ticker_losses[symbol] += pnl
        # Assume you have access to morning portfolio value
        if self.morning_portfolio_value:
            if (
                abs(self.ticker_losses[symbol])
                > self.max_ticker_loss_pct * self.morning_portfolio_value
            ):
                return True
        return False


    def check_max_daily_loss(self, current_portfolio_value):
        """
        Check if the portfolio loss exceeds the max allowed for the day.
        """
        if self.morning_portfolio_value is None:
            self.morning_portfolio_value = current_portfolio_value
        loss = self.morning_portfolio_value - current_portfolio_value
        if loss > self.max_daily_loss_pct * self.morning_portfolio_value:
            return True
        return False
