from __future__ import annotations

import pytest

from sim.friction.measured import MeasuredFriction, Quote

pytestmark = pytest.mark.phase0


class TestMeasuredFriction:
    def test_buy_crosses_at_ask(self, sample_quote):
        price, detail = MeasuredFriction().fill_price(sample_quote, "BUY")
        assert price == sample_quote.ask
        assert detail["half_spread"] == pytest.approx(0.5)

    def test_sell_crosses_at_bid(self, sample_quote):
        price, _ = MeasuredFriction().fill_price(sample_quote, "SELL")
        assert price == sample_quote.bid

    def test_no_guessed_slippage_without_bid_ask(self):
        q = Quote("RELIANCE", ltp=2500)
        with pytest.raises(ValueError, match="No measured bid/ask"):
            MeasuredFriction().fill_price(q, "BUY")

    def test_invalid_spread_rejected(self):
        q = Quote("X", ltp=100, bid=101, ask=100)
        with pytest.raises(ValueError, match="Crossed bid/ask"):
            MeasuredFriction().fill_price(q, "BUY")
