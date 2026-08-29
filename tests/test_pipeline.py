"""Phase 4 contract — unified pipeline orchestration."""

from __future__ import annotations

import pytest

from sim.friction.measured import Quote

pytestmark = pytest.mark.phase4


def test_pipeline_rejects_circuit_before_fill(registry_path, circuit_locked_uc):
    from sim.orders import OrderIntent, OrderSide, OrderType, Product
    from sim.pipeline import SimPipeline

    pipeline = SimPipeline(registry_path=registry_path)
    intent = OrderIntent(
        strategy_id="E1",
        symbol=circuit_locked_uc["symbol"],
        side=OrderSide.BUY,
        quantity=1,
        order_type=OrderType.MARKET,
        product=Product.MIS,
    )
    quote = Quote(
        circuit_locked_uc["symbol"],
        ltp=circuit_locked_uc["ltp"],
        bid=circuit_locked_uc["bid"],
        ask=circuit_locked_uc["ask"],
    )
    result = pipeline.process(intent, quote=quote, uc=circuit_locked_uc["uc"], lc=circuit_locked_uc["lc"])
    assert result.status == "REJECTED"
    assert result.fill is None


def test_pipeline_fill_records_friction_and_fees(registry_path, sample_quote):
    from sim.orders import OrderIntent, OrderSide, OrderType, Product
    from sim.pipeline import SimPipeline

    pipeline = SimPipeline(registry_path=registry_path)
    intent = OrderIntent(
        strategy_id="A1",
        symbol="RELIANCE",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        product=Product.CNC,
    )
    result = pipeline.process(intent, quote=sample_quote, uc=3000, lc=2000)
    assert result.status == "FILLED"
    assert result.fill.fill_price == sample_quote.ask
    assert result.fill.charges.total > 0


def test_pipeline_square_off_overrides_strategy_intent(registry_path, mis_square_off_time):
    from sim.pipeline import SimPipeline

    pipeline = SimPipeline(registry_path=registry_path)
    state = {"positions": {"INFY": {"qty": 5, "product": "MIS"}}}
    events = pipeline.end_of_day(mis_square_off_time, state)
    assert any(e.reason == "mis_eod_square_off" for e in events)
