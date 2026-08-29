from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from experiments.allocation_log import log_allocation_snapshot
from ops.monitor.allocator_audit import audit_allocations
from ops.monitor.config import MonitorConfig, MonitorThresholds
from ops.monitor.log_scan import scan_logs
from ops.monitor.models import Severity
from ops.monitor.retention import RetentionPolicy, check_retention_reminder
from ops.monitor.runner import run_health_checks
from ops.monitor.strategy_integrity import check_strategy_config, check_trade_logs

pytestmark = pytest.mark.runtime

IST = ZoneInfo("Asia/Kolkata")


class TestLogScan:
    def test_detects_traceback_in_logs(self, tmp_path):
        (tmp_path / "ingest.log").write_text(
            "INFO started\nTraceback (most recent call last):\n  File main.py\nValueError: boom\n",
            encoding="utf-8",
        )
        report = scan_logs(tmp_path, max_errors=0)
        assert any(f.check == "log_errors" and f.severity == Severity.ERROR for f in report.findings)

    def test_clean_logs_pass(self, tmp_path):
        (tmp_path / "ingest.log").write_text("INFO ingest ok\nINFO 0 errors\n", encoding="utf-8")
        report = scan_logs(tmp_path)
        assert any(f.check == "log_errors" and f.severity == Severity.OK for f in report.findings)

    def test_ignores_fyers_ws_noise(self, tmp_path):
        (tmp_path / "ingest.log").write_text(
            "INFO tick\n"
            "fyers ws error: Connection timed out\n"
            "WARNING Temporary failure in name resolution\n"
            "close_connection still blocked after 2.0s\n",
            encoding="utf-8",
        )
        report = scan_logs(tmp_path, max_errors=0)
        assert any(f.check == "log_errors" and f.severity == Severity.OK for f in report.findings)

    def test_missing_log_dir_warns(self, tmp_path):
        report = scan_logs(tmp_path / "nope")
        assert any(f.severity == Severity.WARN for f in report.findings)


class TestStrategyIntegrity:
    def test_config_has_21_strategies_7_clusters(self):
        report = check_strategy_config()
        assert report.ok

    def test_unknown_strategy_in_trades_fails(self, tmp_path, monkeypatch):
        from nse_trader import config as cfg_mod

        store = tmp_path / "data" / "store" / "experiments" / "run1"
        store.mkdir(parents=True)
        df = pd.DataFrame(
            [{"trade_id": "t1", "strategy_id": "ZZ99", "cluster": "Z", "symbol": "RELIANCE"}]
        )
        path = store / "trades.parquet"
        df.to_parquet(path, index=False)
        glob = str(path.relative_to(tmp_path))
        monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
        report = check_trade_logs(glob)
        assert any(f.check == "trade_strategy_ids" and f.severity == Severity.ERROR for f in report.findings)

    def test_cluster_mismatch_in_trades_fails(self, tmp_path, monkeypatch):
        from nse_trader import config as cfg_mod

        store = tmp_path / "data" / "store" / "experiments" / "run1"
        store.mkdir(parents=True)
        df = pd.DataFrame(
            [{"trade_id": "t1", "strategy_id": "A1", "cluster": "B", "symbol": "RELIANCE"}]
        )
        path = store / "trades.parquet"
        df.to_parquet(path, index=False)
        glob = str(path.relative_to(tmp_path))
        monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
        report = check_trade_logs(glob)
        assert any(f.check == "trade_cluster_match" and f.severity == Severity.ERROR for f in report.findings)


class TestAllocatorDiversity:
    def _stuck_history(self, path: Path, strategy: str = "A1", days: int = 15) -> None:
        base = datetime(2026, 8, 1, tzinfo=IST)
        rows = []
        for d in range(days):
            ts = (base + timedelta(days=d)).isoformat()
            rows.append({"ts": ts, "strategy_id": strategy, "weight": 1.0})
        pd.DataFrame(rows).to_parquet(path, index=False)

    def _healthy_history(self, path: Path, strategies: list[str], days: int = 15) -> None:
        base = datetime(2026, 8, 1, tzinfo=IST)
        rows = []
        for d in range(days):
            ts = (base + timedelta(days=d)).isoformat()
            for sid in strategies:
                rows.append({"ts": ts, "strategy_id": sid, "weight": 1.0 / len(strategies)})
        pd.DataFrame(rows).to_parquet(path, index=False)

    def test_flags_stuck_single_strategy_picker(self, tmp_path):
        path = tmp_path / "alloc.parquet"
        self._stuck_history(path, strategy="A1", days=15)
        thresholds = MonitorThresholds(min_unique_strategies=5, lookback_days=30)
        report = audit_allocations(path, thresholds)
        assert any(f.check == "allocator_diversity" and f.severity == Severity.ERROR for f in report.findings)

    def test_healthy_diverse_allocations_pass(self, tmp_path):
        path = tmp_path / "alloc.parquet"
        self._healthy_history(path, ["A1", "B1", "C1", "D1", "G1"], days=10)
        thresholds = MonitorThresholds(min_unique_strategies=5, lookback_days=30)
        report = audit_allocations(path, thresholds)
        assert any(f.check == "allocator_diversity" and f.severity == Severity.OK for f in report.findings)

    def test_minute_snapshots_do_not_inflate_streak(self, tmp_path):
        """Hundreds of same-day minute ticks must count as one daily rebalance."""
        path = tmp_path / "alloc.parquet"
        base = datetime(2026, 8, 20, 9, 15, tzinfo=IST)
        rows = []
        for minute in range(120):
            ts = (base + timedelta(minutes=minute)).isoformat()
            for sid, w in [("A1", 0.4), ("B1", 0.3), ("C1", 0.2), ("D1", 0.05), ("G1", 0.05)]:
                rows.append({"ts": ts, "strategy_id": sid, "weight": w})
        # Second day with a different top — streak stays 1.
        day2 = datetime(2026, 8, 21, 9, 15, tzinfo=IST)
        for sid, w in [("B1", 0.45), ("A1", 0.25), ("C1", 0.15), ("D1", 0.1), ("G1", 0.05)]:
            rows.append({"ts": day2.isoformat(), "strategy_id": sid, "weight": w})
        pd.DataFrame(rows).to_parquet(path, index=False)
        thresholds = MonitorThresholds(
            min_unique_strategies=5,
            max_consecutive_same_top=10,
            max_top_strategy_share=0.50,
            lookback_days=30,
        )
        report = audit_allocations(path, thresholds)
        assert not any(f.check == "allocator_streak" for f in report.findings)
        assert not any(f.check == "allocator_concentration" for f in report.findings)
        assert any(f.check == "allocator_diversity" and f.severity == Severity.OK for f in report.findings)

    def test_allocation_log_helper_appends(self, tmp_path):
        path = tmp_path / "meta_allocations.parquet"
        cluster_of = {"A1": "A", "B1": "B"}
        log_allocation_snapshot({"A1": 0.5, "B1": 0.5}, cluster_of=cluster_of, path=path)
        log_allocation_snapshot({"A1": 0.4, "B1": 0.6}, cluster_of=cluster_of, path=path)
        df = pd.read_parquet(path)
        assert len(df) == 4


class TestHealthRunner:
    def test_runner_merges_checks(self, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        (logs / "paper.log").write_text("INFO all good\n", encoding="utf-8")
        from ops.monitor.config import PersistencePaths

        cfg = MonitorConfig(
            logs_dir=logs,
            allocation_history=tmp_path / "missing.parquet",
            trade_glob="data/store/experiments/*/trades.parquet",
            thresholds=MonitorThresholds(),
            persistence=PersistencePaths(
                state_dir=tmp_path / "state",
                store_dir=tmp_path / "store",
                logs_dir=logs,
            ),
        )
        report = run_health_checks(cfg)
        checks = {f.check for f in report.findings}
        assert "strategy_cluster_map" in checks
        assert "log_errors" in checks
        assert "allocation_history" in checks
        assert "retention_age" in checks


class TestRetentionReminder:
    def test_young_data_no_reminder(self, tmp_path):
        store = tmp_path / "store"
        store.mkdir(parents=True)
        (store / "market.duckdb").write_text("x", encoding="utf-8")
        policy = RetentionPolicy(
            rotate_after_days=180,
            auto_rotate=False,
            reminder_state_file=tmp_path / "state" / "retention_reminder.json",
            reminder_banner_file=tmp_path / "state" / "ROTATE_WHEN_READY.txt",
        )
        report = check_retention_reminder(
            store, tmp_path / "logs", tmp_path / "state", policy, today=date(2026, 9, 1)
        )
        assert any(f.check == "retention_age" and f.severity == Severity.OK for f in report.findings)
        assert not policy.reminder_banner_file.exists()

    def test_old_data_warns_but_does_not_delete(self, tmp_path):
        store = tmp_path / "store"
        store.mkdir(parents=True)
        old_file = store / "market.duckdb"
        old_file.write_text("x", encoding="utf-8")
        # mtime ~200 days ago
        import os
        import time

        old_ts = time.time() - (200 * 86400)
        os.utime(old_file, (old_ts, old_ts))

        policy = RetentionPolicy(
            rotate_after_days=180,
            auto_rotate=False,
            reminder_state_file=tmp_path / "state" / "retention_reminder.json",
            reminder_banner_file=tmp_path / "state" / "ROTATE_WHEN_READY.txt",
        )
        report = check_retention_reminder(
            store, tmp_path / "logs", tmp_path / "state", policy, today=date(2027, 2, 1)
        )
        assert any(f.check == "retention_reminder" and f.severity == Severity.WARN for f in report.findings)
        assert old_file.exists()
        assert policy.reminder_banner_file.exists()
        assert "Auto-rotate:    OFF" in policy.reminder_banner_file.read_text(encoding="utf-8")

    def test_auto_rotate_true_is_config_error(self, tmp_path):
        policy = RetentionPolicy(rotate_after_days=180, auto_rotate=True)
        report = check_retention_reminder(
            tmp_path / "store", tmp_path / "logs", tmp_path / "state", policy
        )
        assert any(f.check == "retention_policy" and f.severity == Severity.ERROR for f in report.findings)
