"""Phase 1 contract — order model and lifecycle."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.phase1


def test_order_intent_fields():
    from sim.orders import OrderIntent, OrderSide, OrderType, Product

    intent = OrderIntent(
        strategy_id="A1",
        symbol="RELIANCE",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        product=Product.CNC,
    )
    assert intent.strategy_id == "A1"
    assert intent.quantity == 10


def test_order_pending_to_filled():
    from sim.orders import Order, OrderStatus

    order = Order.from_intent(
        intent=__import__("sim.orders", fromlist=["OrderIntent"]).OrderIntent(
            strategy_id="B1",
            symbol="TCS",
            side=__import__("sim.orders", fromlist=["OrderSide"]).OrderSide.BUY,
            quantity=5,
            order_type=__import__("sim.orders", fromlist=["OrderType"]).OrderType.MARKET,
            product=__import__("sim.orders", fromlist=["Product"]).Product.MIS,
        )
    )
    assert order.status == OrderStatus.PENDING
    order.mark_filled(fill_price=4000.0, fees=25.0)
    assert order.status == OrderStatus.FILLED
    assert order.fill_price == 4000.0


def test_ledger_applies_fill_event():
    from sim.events import FillEvent
    from sim.ledger import VirtualBook

    book = VirtualBook(strategy_id="A1", cash=1_000_000)
    book.apply(
        FillEvent(
            strategy_id="A1",
            symbol="RELIANCE",
            side="BUY",
            quantity=10,
            fill_price=2500.0,
            fees=55.0,
        )
    )
    assert book.positions.get("RELIANCE") == 10
    assert book.cash < 1_000_000
