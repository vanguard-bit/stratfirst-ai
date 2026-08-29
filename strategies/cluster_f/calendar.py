from __future__ import annotations

from dataclasses import dataclass

from strategies.base import Bar, Signal


@dataclass
class TurnOfMonth:
    """F1 — seasonality around month-end."""

    id: str
    days_before: int = 2
    days_after: int = 3
    cluster: str = "F"
    timeframe: str = "1D"
    product: str = "CNC"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        day = int(state.get("day_of_month", 0)) or 0
        if day <= 0:
            # Fallback boolean from build_state when day_of_month absent
            if state.get("in_turn_of_month", False):
                return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
            return Signal(action="HOLD", symbol=bar.symbol)
        # Use calendar params: last `days_before` days of month + first `days_after`
        # Approximate month length 31 for boundary (good enough for paper/replay).
        if day <= self.days_after or day > 31 - self.days_before:
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        return Signal(action="HOLD", symbol=bar.symbol)


@dataclass
class DayOfWeek:
    """F2 — weekday seasonality."""

    id: str
    long_days: list[int] | None = None
    reduce_days: list[int] | None = None
    cluster: str = "F"
    timeframe: str = "1D"
    product: str = "CNC"

    def __post_init__(self) -> None:
        self.long_days = self.long_days if self.long_days is not None else [1, 2]
        self.reduce_days = self.reduce_days if self.reduce_days is not None else [0, 4]

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        dow = int(state.get("day_of_week", 2))
        if dow in self.long_days:
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        if dow in self.reduce_days:
            # Half size — not FLAT (paper-live FLAT zeroes the book).
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=0.5)
        return Signal(action="HOLD", symbol=bar.symbol)


@dataclass
class ExpiryWeekEffect:
    """F3 — reduce size in expiry week."""

    id: str
    reduce_factor: float = 0.5
    cluster: str = "F"
    timeframe: str = "1D"
    product: str = "CNC"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        if state.get("expiry_week", False):
            return Signal(action="HOLD", symbol=bar.symbol, intended_exposure=self.reduce_factor)
        return Signal(action="HOLD", symbol=bar.symbol, intended_exposure=1.0)
