import json
from pathlib import Path
from strategy_definition_crypto.crypto_bot.strategy_params import (
    StrategyParams,
)


class ParameterManager:

    @staticmethod
    def load_params(
        symbol: str,
        term_trade: str,
        company_type: str,
        ticker_path: str = "ticker_params.json",
        default_path: str = "default_params.json",
    ) -> StrategyParams:
        script_dir = Path(__file__).parent
        tuned_path = script_dir / "config" / ticker_path
        default_path = script_dir / "config" / default_path

        def convert_lists_to_tuples(params: dict) -> dict:
            if "fast_slow_windows" in params and isinstance(
                params["fast_slow_windows"], list
            ):
                params["fast_slow_windows"] = tuple(params["fast_slow_windows"])
            return params

        if tuned_path.exists():
            with open(tuned_path) as f:
                tuned = json.load(f)
            if symbol in tuned and term_trade in tuned[symbol]:
                params = tuned[symbol][term_trade][company_type]
                params = convert_lists_to_tuples(params)

                return StrategyParams(company_type, term_trade, **params)
            else:
                if default_path.exists():
                    with open(default_path) as f:
                        defaults = json.load(f)
                    params = defaults[term_trade][company_type]
                    params = convert_lists_to_tuples(params)
                    return StrategyParams(company_type, term_trade, **params)
                else:
                    raise ValueError(f"No parameters found for {symbol} / {term_trade}")

    def save_params(symbol: str, trade_type: str, params: dict) -> None:
        print("Write updated calibration results to ticker_params.json")
