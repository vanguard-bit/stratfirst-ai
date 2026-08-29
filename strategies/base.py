from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Signal:
    action: str  # BUY | SELL | HOLD | FLAT
    symbol: str | None = None
    quantity: int = 0
    confidence: float = 1.0
    intended_exposure: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Bar:
    ts: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str


class Strategy(Protocol):
    id: str
    cluster: str
    timeframe: str
    product: str

    def on_bar(self, bar: Bar, state: dict[str, Any]) -> Signal: ...
