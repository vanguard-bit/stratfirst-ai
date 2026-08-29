from __future__ import annotations

from pathlib import Path

import pandas as pd

from nse_trader.config import load_yaml
from ops.monitor.models import Finding, HealthReport, Severity


def load_strategy_map() -> tuple[dict[str, str], set[str]]:
    """Return cluster_of map and enabled strategy ids from config."""
    cfg = load_yaml("strategies.yaml")
    strategies = cfg.get("strategies", {})
    cluster_of: dict[str, str] = {}
    enabled: set[str] = set()
    for sid, meta in strategies.items():
        cluster_of[sid] = meta["cluster"]
        if meta.get("enabled", True):
            enabled.add(sid)
    return cluster_of, enabled


def check_strategy_config() -> HealthReport:
    report = HealthReport()
    cfg = load_yaml("strategies.yaml")
    clusters = set(cfg.get("clusters", {}))
    strategies = cfg.get("strategies", {})

    if len(strategies) != 21:
        report.add(
            Finding(
                check="strategy_count",
                severity=Severity.ERROR,
                message=f"Expected 21 strategies, found {len(strategies)}",
            )
        )

    if len(clusters) != 7:
        report.add(
            Finding(
                check="cluster_count",
                severity=Severity.ERROR,
                message=f"Expected 7 clusters, found {len(clusters)}",
            )
        )

    bad_cluster: list[str] = []
    for sid, meta in strategies.items():
        cl = meta.get("cluster")
        if cl not in clusters:
            bad_cluster.append(sid)

    if bad_cluster:
        report.add(
            Finding(
                check="strategy_cluster_map",
                severity=Severity.ERROR,
                message=f"Strategies reference unknown clusters: {bad_cluster}",
            )
        )
    else:
        report.add(
            Finding(
                check="strategy_cluster_map",
                severity=Severity.OK,
                message=f"All {len(strategies)} strategies map to valid clusters",
            )
        )

    return report


def check_trade_logs(trade_glob: str) -> HealthReport:
    report = HealthReport()
    cluster_of, enabled = load_strategy_map()
    from nse_trader.config import ROOT

    paths = sorted(ROOT.glob(trade_glob))
    if not paths:
        report.add(
            Finding(
                check="trade_logs",
                severity=Severity.WARN,
                message="No trade parquet files yet (paper/backtest not run)",
            )
        )
        return report

    unknown: set[str] = set()
    cluster_mismatch: list[dict] = []
    disabled_used: set[str] = set()
    total_rows = 0

    for path in paths:
        df = pd.read_parquet(path)
        total_rows += len(df)
        if "strategy_id" not in df.columns:
            report.add(
                Finding(
                    check="trade_schema",
                    severity=Severity.ERROR,
                    message=f"{path.name} missing strategy_id column",
                )
            )
            continue

        for _, row in df.iterrows():
            sid = str(row["strategy_id"])
            if sid not in cluster_of:
                unknown.add(sid)
                continue
            if sid not in enabled:
                disabled_used.add(sid)
            if "cluster" in df.columns:
                logged_cl = str(row["cluster"])
                if logged_cl != cluster_of[sid]:
                    cluster_mismatch.append(
                        {"strategy_id": sid, "logged": logged_cl, "expected": cluster_of[sid]}
                    )

    if unknown:
        report.add(
            Finding(
                check="trade_strategy_ids",
                severity=Severity.ERROR,
                message=f"Trades reference unknown strategies: {sorted(unknown)}",
            )
        )
    if cluster_mismatch:
        report.add(
            Finding(
                check="trade_cluster_match",
                severity=Severity.ERROR,
                message=f"{len(cluster_mismatch)} trade(s) have cluster mismatch vs config",
                detail={"samples": cluster_mismatch[:5]},
            )
        )
    if disabled_used:
        report.add(
            Finding(
                check="trade_disabled_strategies",
                severity=Severity.WARN,
                message=f"Trades from disabled strategies: {sorted(disabled_used)}",
            )
        )
    if not unknown and not cluster_mismatch:
        report.add(
            Finding(
                check="trade_logs",
                severity=Severity.OK,
                message=f"Validated {total_rows} trade row(s) across {len(paths)} file(s)",
            )
        )

    return report
