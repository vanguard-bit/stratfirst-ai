from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nse_trader.config import ROOT, load_yaml


@dataclass
class MonitorThresholds:
    min_unique_strategies: int = 5
    max_top_strategy_share: float = 0.50
    max_consecutive_same_top: int = 10
    min_enabled_strategy_coverage: float = 0.30
    max_log_errors: int = 0
    lookback_days: int = 20


@dataclass
class PersistencePaths:
    state_dir: Path
    store_dir: Path
    logs_dir: Path


@dataclass
class MonitorConfig:
    logs_dir: Path
    allocation_history: Path
    trade_glob: str
    thresholds: MonitorThresholds
    persistence: PersistencePaths

    @classmethod
    def load(cls) -> MonitorConfig:
        raw = load_yaml("ops.yaml")
        mon = raw.get("monitoring", {})
        persistence = raw.get("persistence", {})
        logs_dir = ROOT / persistence.get("logs_dir", "data/logs")
        state_dir = ROOT / persistence.get("state_dir", "data/state")
        store_dir = ROOT / persistence.get("store_dir", "data/store")

        t = mon.get("thresholds", {})
        thresholds = MonitorThresholds(
            min_unique_strategies=int(t.get("min_unique_strategies", 5)),
            max_top_strategy_share=float(t.get("max_top_strategy_share", 0.50)),
            max_consecutive_same_top=int(t.get("max_consecutive_same_top", 10)),
            min_enabled_strategy_coverage=float(t.get("min_enabled_strategy_coverage", 0.30)),
            max_log_errors=int(t.get("max_log_errors", 0)),
            lookback_days=int(t.get("lookback_days", 20)),
        )

        alloc = mon.get("allocation_history")
        if alloc:
            allocation_history = ROOT / alloc
        else:
            allocation_history = state_dir / "meta_allocations.parquet"

        trade_glob = mon.get(
            "trade_glob",
            f"{store_dir.relative_to(ROOT)}/experiments/**/trades.parquet",
        )

        return cls(
            logs_dir=logs_dir,
            allocation_history=allocation_history,
            trade_glob=str(trade_glob),
            thresholds=thresholds,
            persistence=PersistencePaths(
                state_dir=state_dir,
                store_dir=store_dir,
                logs_dir=logs_dir,
            ),
        )
