from __future__ import annotations

from dataclasses import dataclass

from strategies.base import Bar, Signal


def _flat(symbol: str) -> Signal:
    return Signal(action="FLAT", symbol=symbol, intended_exposure=0.0)


def _pos_qty(state: dict) -> int:
    raw = state.get("position_qty")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return 1 if state.get("in_position") else 0


def _hhmm(state: dict, bar: Bar) -> str:
    raw = state.get("session_hhmm")
    if isinstance(raw, str) and len(raw) >= 4:
        return raw[:5]
    ts = str(bar.ts)
    # "...T14:15..." or "14:15:00"
    if "T" in ts:
        part = ts.split("T", 1)[1]
        return part[:5]
    if len(ts) >= 5 and ts[2] == ":":
        return ts[:5]
    return "00:00"


@dataclass
class OpeningRangeBreakout:
    """E1 — ORB with stop / 1.5R target / fail-back; one trade per symbol per day."""

    id: str
    orb_minutes: int = 15
    target_r: float = 1.5
    cluster: str = "E"
    timeframe: str = "5m"
    product: str = "MIS"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        if not state.get("orb_complete", False):
            return Signal(action="HOLD", symbol=bar.symbol)
        oh = float(state.get("orb_high", bar.high))
        ol = float(state.get("orb_low", bar.low))
        rng = oh - ol
        if rng <= 0:
            return Signal(action="HOLD", symbol=bar.symbol)

        qty = _pos_qty(state)
        close = float(bar.close)
        traded = bool(state.get("e1_traded_today", False))

        if qty > 0:
            if close <= ol:
                return _flat(bar.symbol)
            if close >= oh + self.target_r * rng:
                return _flat(bar.symbol)
            if ol <= close <= oh:
                return _flat(bar.symbol)
            return Signal(action="HOLD", symbol=bar.symbol)

        if qty < 0:
            if close >= oh:
                return _flat(bar.symbol)
            if close <= ol - self.target_r * rng:
                return _flat(bar.symbol)
            if ol <= close <= oh:
                return _flat(bar.symbol)
            return Signal(action="HOLD", symbol=bar.symbol)

        # Flat — one-shot: no re-entry after a trade today.
        if traded:
            return Signal(action="HOLD", symbol=bar.symbol)
        if close > oh:
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        if close < ol:
            return Signal(action="SELL", symbol=bar.symbol, intended_exposure=1.0)
        return Signal(action="HOLD", symbol=bar.symbol)


@dataclass
class VwapReversion:
    """E2 — fade VWAP stretch; exit at VWAP / stop / opposite stretch (flat, not reverse)."""

    id: str
    dev_pct: float = 0.005
    target_band: float = 0.001
    stop_extra: float = 0.005
    cluster: str = "E"
    timeframe: str = "15m"
    product: str = "MIS"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        dev = float(state.get("vwap_dev", 0.0))
        qty = _pos_qty(state)
        stop_long = -(self.dev_pct + self.stop_extra)
        stop_short = self.dev_pct + self.stop_extra

        if qty > 0:
            if dev >= -self.target_band:
                return _flat(bar.symbol)
            if dev <= stop_long:
                return _flat(bar.symbol)
            if dev >= self.dev_pct:
                return _flat(bar.symbol)
            return Signal(action="HOLD", symbol=bar.symbol)

        if qty < 0:
            if dev <= self.target_band:
                return _flat(bar.symbol)
            if dev >= stop_short:
                return _flat(bar.symbol)
            if dev <= -self.dev_pct:
                return _flat(bar.symbol)
            return Signal(action="HOLD", symbol=bar.symbol)

        if dev <= -self.dev_pct:
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        if dev >= self.dev_pct:
            return Signal(action="SELL", symbol=bar.symbol, intended_exposure=1.0)
        return Signal(action="HOLD", symbol=bar.symbol)


@dataclass
class PowerHourMomentum:
    """E3 — power-hour momentum; flat on mom flip; no new entries after 15:05 IST."""

    id: str
    start_time: str = "14:15"
    entry_cutoff: str = "15:05"
    cluster: str = "E"
    timeframe: str = "5m"
    product: str = "MIS"

    def on_bar(self, bar: Bar, state: dict) -> Signal:
        mom = float(state.get("intraday_mom", 0.0))
        qty = _pos_qty(state)

        if qty > 0:
            if mom <= 0:
                return _flat(bar.symbol)
            return Signal(action="HOLD", symbol=bar.symbol)
        if qty < 0:
            if mom >= 0:
                return _flat(bar.symbol)
            return Signal(action="HOLD", symbol=bar.symbol)

        if not state.get("in_power_hour", False):
            return Signal(action="HOLD", symbol=bar.symbol)
        if _hhmm(state, bar) >= self.entry_cutoff:
            return Signal(action="HOLD", symbol=bar.symbol)
        if mom > 0:
            return Signal(action="BUY", symbol=bar.symbol, intended_exposure=1.0)
        if mom < 0:
            return Signal(action="SELL", symbol=bar.symbol, intended_exposure=1.0)
        return Signal(action="HOLD", symbol=bar.symbol)
