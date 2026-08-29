from __future__ import annotations

from dataclasses import dataclass

from strategies.base import Bar, Signal


@dataclass
class MomentumRank:
    """C1 — long top momentum (CNC long-only; bottom ranks flat, not short)."""

    id: str
    top_n: int = 5
    bottom_n: int = 5
    cluster: str = "C"
    timeframe: str = "1W"
    product: str = "CNC"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        rank = int(state.get("momentum_rank", 999))
        if rank <= self.top_n:
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)


@dataclass
class LowVolAnomaly:
    """C2 — prefer low-volatility quintile."""

    id: str
    lookback: int = 20
    quintile: int = 1
    cluster: str = "C"
    timeframe: str = "1W"
    product: str = "CNC"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        q = int(state.get("vol_quintile", 3))
        if q <= self.quintile:
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        return Signal(action="HOLD", symbol=bar.symbol)


@dataclass
class MeanReversionRank:
    """C3 — long recent losers."""

    id: str
    lookback: int = 5
    bottom_n: int = 5
    cluster: str = "C"
    timeframe: str = "1W"
    product: str = "CNC"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        rank = int(state.get("reversion_rank", 999))
        if rank <= self.bottom_n:
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        return Signal(action="HOLD", symbol=bar.symbol)
