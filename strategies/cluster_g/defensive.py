from __future__ import annotations

from dataclasses import dataclass

from strategies.base import Bar, Signal


@dataclass
class TrendAbsentCash:
    """G1 — reduce exposure when long-term trend is absent."""

    id: str
    ma: int = 200
    cluster: str = "G"
    timeframe: str = "1D"
    product: str = "CNC"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        vs_ma = float(state.get(f"close_vs_ma{self.ma}", state.get("close_vs_ma200", 1.0)))
        adx = float(state.get("adx", 25.0))
        if vs_ma < 1.0 or adx < 20:
            return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
        return Signal(action="HOLD", symbol=bar.symbol, intended_exposure=1.0)


@dataclass
class DrawdownCircuitBreaker:
    """G2 — portfolio-level drawdown halt."""

    id: str
    max_dd: float = 0.10
    cluster: str = "G"
    timeframe: str = "1D"
    product: str = "CNC"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        dd = float(state.get("portfolio_drawdown", 0.0))
        if dd >= self.max_dd:
            return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
        return Signal(action="HOLD", symbol=bar.symbol)


@dataclass
class LowBetaBasket:
    """G3 — defensive low-beta basket."""

    id: str
    beta_window: int = 60
    bottom_n: int = 10
    cluster: str = "G"
    timeframe: str = "1W"
    product: str = "CNC"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        beta_rank = int(state.get("beta_rank", 999))
        if beta_rank <= self.bottom_n:
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        return Signal(action="HOLD", symbol=bar.symbol)
