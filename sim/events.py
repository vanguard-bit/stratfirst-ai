from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class FillEvent:
    strategy_id: str
    symbol: str
    side: str
    quantity: int
    fill_price: float
    fees: float
    ts: datetime | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.fill_price <= 0:
            raise ValueError("fill_price must be positive")
        if self.fees < 0:
            raise ValueError("fees cannot be negative")

    @property
    def timestamp(self) -> datetime:
        return self.ts or datetime.now(timezone.utc)

    @property
    def notional(self) -> float:
        return self.fill_price * self.quantity
