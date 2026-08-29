from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sim.broker.square_off import SquareOffAction, SquareOffEngine
from sim.engine import SimFill, SimulationEngine
from sim.exchange.rules import evaluate_order
from sim.friction.measured import MeasuredFriction, Quote
from sim.orders import Order, OrderIntent, Product


@dataclass
class PipelineResult:
    status: str
    order: Order
    fill: SimFill | None = None
    reject_reason: str | None = None


class SimPipeline:
    """Unified sim path: exchange rules → friction fill → fee registry."""

    def __init__(
        self,
        registry_path: Path,
        friction: MeasuredFriction | None = None,
        square_off_time: str = "15:20",
    ):
        self.engine = SimulationEngine(registry_path, friction=friction)
        self.square_off = SquareOffEngine(square_off_time=square_off_time)

    def process(
        self,
        intent: OrderIntent,
        *,
        quote: Quote,
        uc: float,
        lc: float,
    ) -> PipelineResult:
        order = Order.from_intent(intent)

        exchange = evaluate_order(
            side=intent.side.value,
            order_type=intent.order_type.value,
            ltp=quote.ltp,
            uc=uc,
            lc=lc,
            bid=quote.bid,
            ask=quote.ask,
            limit_price=intent.limit_price,
        )
        if not exchange.allowed:
            order.mark_rejected(exchange.reason)
            return PipelineResult(
                status="REJECTED",
                order=order,
                fill=None,
                reject_reason=exchange.reason,
            )

        segment = (
            "equity_intraday" if intent.product == Product.MIS else "equity_delivery"
        )
        fill = self.engine.execute(
            segment=segment,
            side=intent.side.value,
            symbol=intent.symbol,
            quantity=intent.quantity,
            quote=quote,
        )
        order.mark_filled(fill.fill_price, fill.charges.total)
        return PipelineResult(status="FILLED", order=order, fill=fill)

    def end_of_day(self, ts: datetime, state: dict) -> list[SquareOffAction]:
        positions = state.get("positions", {})
        return self.square_off.actions_at(ts, positions)
