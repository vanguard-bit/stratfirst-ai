from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sim.fees.calculator import FeeCalculator, TradeCharges
from sim.friction.measured import MeasuredFriction, Quote
from sim.fees.sources.official import FeeRegistry


@dataclass
class SimFill:
    symbol: str
    side: str
    quantity: int
    signal_price: float
    fill_price: float
    turnover: float
    charges: TradeCharges
    friction_detail: dict[str, float]


class SimulationEngine:
    def __init__(
        self,
        fees_path: Path,
        friction: MeasuredFriction | None = None,
    ):
        self.fees = FeeCalculator.from_file(fees_path)
        self.friction = friction or MeasuredFriction()

    def execute(
        self,
        *,
        segment: str,
        side: str,
        symbol: str,
        quantity: int,
        quote: Quote,
    ) -> SimFill:
        fill_price, friction_detail = self.friction.fill_price(quote, side)
        turnover = fill_price * quantity
        charges = self.fees.compute(segment, side.lower(), turnover)
        return SimFill(
            symbol=symbol,
            side=side,
            quantity=quantity,
            signal_price=quote.ltp,
            fill_price=fill_price,
            turnover=turnover,
            charges=charges,
            friction_detail=friction_detail,
        )

    def ensure_fees_fresh(self, max_age_hours: float) -> None:
        age = self.fees.registry_age_hours()
        if age > max_age_hours:
            raise RuntimeError(
                f"Fee registry is {age:.1f}h old (max {max_age_hours}h). "
                "Run: python -m sim.fees.refresh"
            )
