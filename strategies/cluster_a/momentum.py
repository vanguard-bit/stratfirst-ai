from __future__ import annotations

from dataclasses import dataclass

from strategies.base import Bar, Signal


@dataclass
class TimeSeriesMomentum:
    """A1 — time-series momentum on daily bars."""

    id: str
    lookback_days: int = 252
    exit_ma: int = 100
    cluster: str = "A"
    timeframe: str = "1D"
    product: str = "CNC"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        if state.get("warmup") or state.get("returns_lookback_ready") is False:
            return Signal(action="HOLD", symbol=bar.symbol)
        ret_key = f"returns_{self.lookback_days}d"
        ret = float(state.get(ret_key, state.get("returns_252d", 0.0)))
        ma_key = f"close_vs_ma{self.exit_ma}"
        vs_ma = float(state.get(ma_key, state.get("close_vs_ma100", 1.0)))

        if ret > 0 and vs_ma > 1.0:
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        if ret <= 0 or vs_ma < 1.0:
            return Signal(action="SELL", symbol=bar.symbol, intended_exposure=0.0)
        return Signal(action="HOLD", symbol=bar.symbol)


@dataclass
class DualMaCrossover:
    """A2 — dual MA cross (long-only) with slow-MA stop."""

    id: str
    fast: int = 12
    slow: int = 26
    cluster: str = "A"
    timeframe: str = "1H"
    product: str = "MIS"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        if state.get("warmup"):
            return Signal(action="HOLD", symbol=bar.symbol)
        cross = state.get("ma_cross", "none")
        raw_qty = state.get("position_qty")
        if raw_qty is not None:
            try:
                qty = int(raw_qty)
            except (TypeError, ValueError):
                qty = 1 if state.get("in_position") else 0
        else:
            qty = 1 if state.get("in_position") else 0
        try:
            ma_slow = float(state.get("ma_slow"))
        except (TypeError, ValueError):
            ma_slow = float("nan")
        close = float(bar.close)

        if qty > 0:
            if cross == "bearish":
                return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
            if ma_slow == ma_slow and close < ma_slow:
                return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
            return Signal(action="HOLD", symbol=bar.symbol)

        if cross == "bullish":
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        return Signal(action="HOLD", symbol=bar.symbol)
