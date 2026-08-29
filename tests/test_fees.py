from __future__ import annotations

from pathlib import Path

import pytest

from sim.fees.calculator import FeeCalculator
from sim.fees.refresh import refresh_registry
from sim.fees.sources.seed import load_seed_registry
from nse_trader.config import ROOT

pytestmark = pytest.mark.phase0


class TestFeeRegistry:
    def test_official_seed_has_required_segments(self):
        reg = load_seed_registry()
        assert {"equity_delivery", "equity_intraday", "broker_admin"}.issubset(reg.segments)

    def test_refresh_offline_writes_registry(self):
        reg = refresh_registry(offline=True)
        path = ROOT / "data" / "fees" / "registry.json"
        assert path.exists()
        assert len(reg.segments["equity_delivery"].components) >= 6

    def test_seed_components_have_source_labels(self):
        reg = load_seed_registry()
        for seg in reg.segments.values():
            for c in seg.components:
                assert c.source_label
                assert c.source_url or c.source_label


class TestFeeCalculator:
    def test_intraday_buy_includes_brokerage_and_stamp(self, registry_path: Path):
        calc = FeeCalculator.from_file(registry_path)
        r = calc.compute("equity_intraday", "buy", 100_000)
        assert r.brokerage == 20
        assert r.stamp_duty == pytest.approx(3.0, rel=0.02)
        assert r.total > r.brokerage

    def test_delivery_zero_brokerage(self, registry_path: Path):
        calc = FeeCalculator.from_file(registry_path)
        r = calc.compute("equity_delivery", "buy", 100_000)
        assert r.brokerage == 0

    def test_intraday_sell_has_stt_not_stamp(self, registry_path: Path):
        calc = FeeCalculator.from_file(registry_path)
        r = calc.compute("equity_intraday", "sell", 100_000)
        assert r.stt > 0
        assert r.stamp_duty == 0

    def test_breakdown_sums_to_total(self, registry_path: Path):
        calc = FeeCalculator.from_file(registry_path)
        r = calc.compute("equity_intraday", "buy", 50_000)
        assert r.breakdown is not None
        assert sum(r.breakdown.values()) == pytest.approx(r.total, rel=1e-6)
