from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class Product(str, Enum):
    CNC = "CNC"
    MIS = "MIS"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class OrderIntent:
    strategy_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    product: Product
    limit_price: float | None = None


@dataclass
class Order:
    order_id: str
    intent: OrderIntent
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fill_price: float | None = None
    fees: float | None = None
    reject_reason: str | None = None

    @classmethod
    def from_intent(cls, intent: OrderIntent) -> Order:
        return cls(order_id=str(uuid.uuid4()), intent=intent)

    def mark_filled(self, fill_price: float, fees: float) -> None:
        self.status = OrderStatus.FILLED
        self.fill_price = fill_price
        self.fees = fees

    def mark_rejected(self, reason: str) -> None:
        self.status = OrderStatus.REJECTED
        self.reject_reason = reason

    def mark_cancelled(self, reason: str | None = None) -> None:
        self.status = OrderStatus.CANCELLED
        self.reject_reason = reason
