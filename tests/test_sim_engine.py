from __future__ import annotations

from pathlib import Path

import pytest

from sim.engine import SimulationEngine
from sim.friction.measured import Quote

pytestmark = pytest.mark.phase0


class TestSimulationEngine:
    def test_fill_price_not_signal_price(self, registry_path: Path):
        engine = SimulationEngine(registry_path)
        q = Quote("RELIANCE", ltp=2500, bid=2499.5, ask=2500.5)
        fill = engine.execute(
            segment="equity_intraday",
            side="BUY",
            symbol="RELIANCE",
            quantity=10,
            quote=q,
        )
        assert fill.signal_price == 2500
        assert fill.fill_price == 2500.5
        assert fill.charges.total > 0

    def test_turnover_uses_fill_not_ltp(self, registry_path: Path):
        engine = SimulationEngine(registry_path)
        q = Quote("RELIANCE", ltp=2500, bid=2499.5, ask=2500.5)
        fill = engine.execute(
            segment="equity_intraday",
            side="BUY",
            symbol="RELIANCE",
            quantity=10,
            quote=q,
        )
        assert fill.turnover == pytest.approx(10 * 2500.5)
