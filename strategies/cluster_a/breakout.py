from __future__ import annotations

from dataclasses import dataclass

from strategies.base import Bar, Signal


@dataclass
class DonchianBreakout:
    """A3 — Donchian channel breakout."""

    id: str
    channel: int = 20
    cluster: str = "A"
    timeframe: str = "1D"
    product: str = "CNC"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        if state.get("warmup"):
            return Signal(action="HOLD", symbol=bar.symbol)
        upper = state.get("donchian_upper")
        lower = state.get("donchian_lower")
        if upper is None or lower is None:
            return Signal(action="HOLD", symbol=bar.symbol)
        try:
            upper_f = float(upper)
            lower_f = float(lower)
        except (TypeError, ValueError):
            return Signal(action="HOLD", symbol=bar.symbol)
        if upper_f != upper_f or lower_f != lower_f:  # NaN
            return Signal(action="HOLD", symbol=bar.symbol)
        if bar.close > upper_f:
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        if bar.close < lower_f:
            return Signal(action="SELL", symbol=bar.symbol, intended_exposure=0.0)
        return Signal(action="HOLD", symbol=bar.symbol)
