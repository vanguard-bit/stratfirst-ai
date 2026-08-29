from __future__ import annotations

from dataclasses import dataclass

from strategies.base import Bar, Signal


@dataclass
class VolTargetOverlay:
    """D1 — scale exposure to hit target vol."""

    id: str
    target_vol: float = 0.12
    lookback: int = 20
    cluster: str = "D"
    timeframe: str = "1D"
    product: str = "CNC"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        realized = float(state.get("realized_vol", self.target_vol))
        scale = min(self.target_vol / max(realized, 1e-6), 1.0)
        if scale < 0.5:
            return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=scale)
        return Signal(action="HOLD", symbol=bar.symbol, intended_exposure=scale)


@dataclass
class VixRegimeFilter:
    """D2 — reduce risk when VIX above median."""

    id: str
    index: str = "INDIAVIX"
    median_window: int = 20
    cluster: str = "D"
    timeframe: str = "1D"
    product: str = "CNC"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        if state.get("vix_above_median", False):
            return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.3)
        return Signal(action="HOLD", symbol=bar.symbol, intended_exposure=1.0)


@dataclass
class AtrExpansionBreakout:
    """D3 — ATR-expansion breakout with channel stop / 1.5R target; flat not reverse."""

    id: str
    atr_window: int = 14
    expansion: float = 1.5
    target_r: float = 1.5
    cluster: str = "D"
    timeframe: str = "1H"
    product: str = "MIS"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        atr_ratio = float(state.get("atr_ratio", 1.0))
        direction = state.get("breakout_dir", "none")
        raw_qty = state.get("position_qty")
        if raw_qty is not None:
            try:
                qty = int(raw_qty)
            except (TypeError, ValueError):
                qty = 1 if state.get("in_position") else 0
        else:
            qty = 1 if state.get("in_position") else 0

        try:
            upper = float(state.get("donchian_upper"))
            lower = float(state.get("donchian_lower"))
        except (TypeError, ValueError):
            upper = lower = float("nan")
        rng = upper - lower if upper == upper and lower == lower else float("nan")
        close = float(bar.close)

        if qty > 0:
            if direction == "down":
                return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
            if lower == lower and close <= lower:
                return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
            if rng == rng and rng > 0 and upper == upper and close >= upper + self.target_r * rng:
                return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
            if upper == upper and lower == lower and lower <= close <= upper:
                return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
            return Signal(action="HOLD", symbol=bar.symbol)

        if qty < 0:
            if direction == "up":
                return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
            if upper == upper and close >= upper:
                return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
            if rng == rng and rng > 0 and lower == lower and close <= lower - self.target_r * rng:
                return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
            if upper == upper and lower == lower and lower <= close <= upper:
                return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
            return Signal(action="HOLD", symbol=bar.symbol)

        if atr_ratio >= self.expansion and direction == "up":
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        if atr_ratio >= self.expansion and direction == "down":
            return Signal(action="SELL", symbol=bar.symbol, intended_exposure=1.0)
        return Signal(action="HOLD", symbol=bar.symbol)
