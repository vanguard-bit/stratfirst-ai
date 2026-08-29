"""Plan fidelity — configs match agreed design."""

from __future__ import annotations

from pathlib import Path

import pytest

from nse_trader.config import ROOT

pytestmark = pytest.mark.phase0

EXPECTED_STRATEGIES = {
    "A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3",
    "D1", "D2", "D3", "E1", "E2", "E3", "F1", "F2", "F3", "G1", "G2", "G3",
}
EXPECTED_CLUSTERS = {"A", "B", "C", "D", "E", "F", "G"}


class TestPortfolioPlan:
    def test_total_capital_20l(self, portfolio_config):
        assert portfolio_config["portfolio"]["total_capital"] == 2_000_000

    def test_virtual_book_10l(self, portfolio_config):
        assert portfolio_config["virtual_books"]["per_strategy_notional"] == 1_000_000

    def test_universe_nifty50(self, portfolio_config):
        assert portfolio_config["portfolio"]["universe"] == "NIFTY50"

    def test_friction_mode_measured(self, portfolio_config):
        assert portfolio_config["simulation"]["friction_mode"] == "measured"

    def test_broker_zerodha(self, portfolio_config):
        assert portfolio_config["simulation"]["broker_profile"] == "zerodha"

    def test_meta_max_strategy_weight(self, portfolio_config):
        assert portfolio_config["meta_allocator"]["constraints"]["max_strategy_weight"] == 0.25

    def test_mis_flat_time(self, portfolio_config):
        assert portfolio_config["risk"]["intraday_flat_time"] == "15:20"
        assert portfolio_config["risk"]["mis_strat_flat_time"] == "15:15"
        assert float(portfolio_config["risk"]["circuit_fallback_pct"]) == 0.10


class TestStrategiesPlan:
    def test_21_strategies(self, strategies_config):
        assert set(strategies_config["strategies"]) == EXPECTED_STRATEGIES

    def test_7_clusters(self, strategies_config):
        assert set(strategies_config["clusters"]) == EXPECTED_CLUSTERS

    def test_intraday_strategies_use_mis(self, strategies_config):
        mis_ids = {"A2", "B1", "B2", "B3", "D3", "E1", "E2", "E3"}
        for sid in mis_ids:
            assert strategies_config["strategies"][sid]["product"] == "MIS", sid

    def test_daily_swing_use_cnc(self, strategies_config):
        cnc_ids = {"A1", "A3", "C1", "C2", "C3", "D1", "D2", "F1", "F2", "F3", "G1", "G2", "G3"}
        for sid in cnc_ids:
            assert strategies_config["strategies"][sid]["product"] == "CNC", sid


class TestOpsPlan:
    def test_timezone_ist(self, ops_config):
        assert ops_config["timezone"] == "Asia/Kolkata"

    def test_persistence_dirs_defined(self, ops_config):
        for key in ("state_dir", "store_dir", "logs_dir"):
            assert key in ops_config["persistence"]

    def test_retention_manual_only(self, ops_config):
        ret = ops_config["retention"]
        assert ret["rotate_after_days"] == 180
        assert ret["auto_rotate"] is False
        assert ops_config["retention"]["reminder"]["enabled"] is True

    def test_jobs_not_24_7_monolith(self, ops_config):
        jobs = ops_config["jobs"]
        assert "ingest_live" in jobs
        assert "eod_pipeline" in jobs
        assert jobs["eod_pipeline"]["type"] in {"cron", "systemd_timer"}


class TestRepoLayout:
    @pytest.mark.parametrize(
        "path",
        [
            "config/portfolio.yaml",
            "config/strategies.yaml",
            "config/costs_nse.yaml",
            "config/ops.yaml",
            "config/fees_official_seed.json",
            ".env.example",
            "deploy/systemd/nse-trader-ingest.service",
            "deploy/systemd/nse-trader-eod.timer",
        ],
    )
    def test_required_files_exist(self, path: str):
        assert (ROOT / path).exists(), f"missing {path}"
