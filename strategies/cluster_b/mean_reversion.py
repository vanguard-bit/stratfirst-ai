from __future__ import annotations

from dataclasses import dataclass

from strategies.base import Bar, Signal


@dataclass
class BollingerZscore:
    """B1 — fade extremes via Bollinger z-score."""

    id: str
    window: int = 20
    z_entry: float = -2.0
    z_exit: float = 0.0
    cluster: str = "B"
    timeframe: str = "15m"
    product: str = "MIS"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        z = float(state.get("zscore", 0.0))
        if z <= self.z_entry:
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        if z >= self.z_exit and state.get("in_position"):
            return Signal(action="SELL", symbol=bar.symbol, intended_exposure=0.0)
        return Signal(action="HOLD", symbol=bar.symbol)


@dataclass
class Rsi2Revert:
    """B2 — Connors-style RSI(2): long dips above 200MA; exit RSI≥65."""

    id: str
    period: int = 2
    buy: float = 10.0
    sell: float = 65.0
    cluster: str = "B"
    timeframe: str = "5m"
    product: str = "MIS"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        rsi = float(state.get("rsi", 50.0))
        vs_ma = float(state.get("close_vs_ma200", 1.0))
        raw_qty = state.get("position_qty")
        if raw_qty is not None:
            try:
                qty = int(raw_qty)
            except (TypeError, ValueError):
                qty = 1 if state.get("in_position") else 0
        else:
            qty = 1 if state.get("in_position") else 0

        if qty > 0:
            if rsi >= self.sell:
                return Signal(action="SELL", symbol=bar.symbol, intended_exposure=0.0)
            return Signal(action="HOLD", symbol=bar.symbol)

        # Long-only Connors: require uptrend filter for new entries.
        if vs_ma > 1.0 and rsi <= self.buy:
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        return Signal(action="HOLD", symbol=bar.symbol)


@dataclass
class GapFade:
    """B3 — morning gap fade; VWAP cover, early time stop, hard stop if fade fails."""

    id: str
    gap_pct: float = 0.015
    entry_cutoff: str = "09:45"
    exit_time: str = "10:30"
    vwap_band: float = 0.001
    stop_extra: float = 0.005
    cluster: str = "B"
    timeframe: str = "15m"
    product: str = "MIS"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        gap = float(state.get("session_gap_pct", state.get("gap_pct", 0.0)))
        dev = float(state.get("vwap_dev", 0.0))
        hhmm = str(state.get("session_hhmm") or "00:00")[:5]
        raw_qty = state.get("position_qty")
        if raw_qty is not None:
            try:
                qty = int(raw_qty)
            except (TypeError, ValueError):
                qty = 1 if state.get("in_position") else 0
        else:
            qty = 1 if state.get("in_position") else 0

        if qty != 0:
            if hhmm >= self.exit_time:
                return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
            # Fade filled toward VWAP
            if qty > 0 and dev >= -self.vwap_band:
                return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
            if qty < 0 and dev <= self.vwap_band:
                return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
            # Hard stop: gap continues against the fade
            if qty < 0 and gap > 0 and dev >= gap + self.stop_extra:
                return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
            if qty > 0 and gap < 0 and dev <= gap - self.stop_extra:
                return Signal(action="FLAT", symbol=bar.symbol, intended_exposure=0.0)
            return Signal(action="HOLD", symbol=bar.symbol)

        if hhmm >= self.entry_cutoff:
            return Signal(action="HOLD", symbol=bar.symbol)
        if gap >= self.gap_pct:
            return Signal(action="SELL", symbol=bar.symbol, intended_exposure=1.0)
        if gap <= -self.gap_pct:
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        return Signal(action="HOLD", symbol=bar.symbol)
