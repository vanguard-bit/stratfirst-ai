from __future__ import annotations

from pathlib import Path

import pandas as pd

from ops.monitor.config import MonitorThresholds
from ops.monitor.models import Finding, HealthReport, Severity
from ops.monitor.strategy_integrity import load_strategy_map


def _max_consecutive_same(values: list[str]) -> tuple[int, str | None]:
    if not values:
        return 0, None
    best = 1
    best_sid = values[0]
    cur = 1
    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            cur += 1
            if cur > best:
                best = cur
                best_sid = values[i]
        else:
            cur = 1
    return best, best_sid


def _prepare_rebalance_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse minute snapshots to one valid rebalance per IST calendar day.

    Paper-live logs allocations every minute. Counting those as rebalances makes
    concentration/streak checks fire on noise. Keep only snapshots whose weights
    sum to ~1.0, then take the last snapshot of each IST day.
    """
    work = df.copy()
    if "ts" not in work.columns or "strategy_id" not in work.columns or "weight" not in work.columns:
        return work.iloc[0:0]

    work["weight"] = pd.to_numeric(work["weight"], errors="coerce").fillna(0.0)
    work = work.loc[work["weight"] > 0]
    if work.empty:
        return work

    ts = pd.to_datetime(work["ts"], format="ISO8601", utc=True).dt.tz_convert("Asia/Kolkata")
    work = work.assign(_ts=ts, _day=ts.dt.date)

    sums = work.groupby("_ts", sort=False)["weight"].sum()
    valid_mask = (sums - 1.0).abs() <= 0.05
    valid_ts = set(sums.index[valid_mask].tolist())
    work = work.loc[work["_ts"].isin(valid_ts)]
    if work.empty:
        return work

    last_ts = work.groupby("_day", sort=True)["_ts"].transform("max")
    work = work.loc[work["_ts"] == last_ts]
    return work.drop(columns=["_ts", "_day"], errors="ignore")


def audit_allocations(path: Path, thresholds: MonitorThresholds) -> HealthReport:
    report = HealthReport()
    _, enabled = load_strategy_map()
    enabled_n = len(enabled)

    if not path.exists():
        report.add(
            Finding(
                check="allocation_history",
                severity=Severity.WARN,
                message=f"No allocation history at {path} (meta allocator not logging yet)",
            )
        )
        return report

    df = pd.read_parquet(path)
    if df.empty:
        report.add(
            Finding(
                check="allocation_history",
                severity=Severity.WARN,
                message="Allocation history file is empty",
            )
        )
        return report

    if not {"strategy_id", "weight"}.issubset(df.columns):
        report.add(
            Finding(
                check="allocation_schema",
                severity=Severity.ERROR,
                message="Allocation parquet must include strategy_id and weight columns",
            )
        )
        return report

    if "ts" in df.columns and thresholds.lookback_days > 0:
        cutoff = pd.Timestamp.now(tz="Asia/Kolkata") - pd.Timedelta(days=thresholds.lookback_days)
        ts = pd.to_datetime(df["ts"], format="ISO8601", utc=True).dt.tz_convert("Asia/Kolkata")
        df = df.loc[ts >= cutoff]

    df = _prepare_rebalance_frames(df)
    if df.empty:
        report.add(
            Finding(
                check="allocation_history",
                severity=Severity.WARN,
                message=(
                    "No valid daily rebalance snapshots in lookback "
                    "(need weight sum ≈ 1.0 per timestamp)"
                ),
            )
        )
        return report

    strategies_seen = set(df["strategy_id"].astype(str))
    # Mean weight across daily rebalances (missing days count as 0 for that strategy).
    day_key = pd.to_datetime(df["ts"], format="ISO8601", utc=True).dt.tz_convert("Asia/Kolkata").dt.date
    n_days = int(day_key.nunique())
    day_weights = (
        df.assign(_day=day_key)
        .groupby(["_day", "strategy_id"], as_index=False)["weight"]
        .sum()
    )
    avg_weights = day_weights.groupby("strategy_id")["weight"].sum() / max(n_days, 1)

    unique_count = len(strategies_seen)
    if unique_count < thresholds.min_unique_strategies:
        report.add(
            Finding(
                check="allocator_diversity",
                severity=Severity.ERROR,
                message=(
                    f"Meta allocator used only {unique_count} strateg{'y' if unique_count == 1 else 'ies'} "
                    f"in lookback (min {thresholds.min_unique_strategies}) — possible stuck picker"
                ),
                detail={"strategies_with_weight": sorted(strategies_seen)},
            )
        )
    else:
        report.add(
            Finding(
                check="allocator_diversity",
                severity=Severity.OK,
                message=(
                    f"{unique_count} strategies received weight across "
                    f"{n_days} daily rebalance(s)"
                ),
            )
        )

    coverage = unique_count / enabled_n if enabled_n else 0.0
    if coverage < thresholds.min_enabled_strategy_coverage:
        report.add(
            Finding(
                check="allocator_coverage",
                severity=Severity.WARN,
                message=(
                    f"Only {coverage:.0%} of enabled strategies ever got weight "
                    f"(min {thresholds.min_enabled_strategy_coverage:.0%})"
                ),
                detail={"enabled": enabled_n, "used": unique_count},
            )
        )

    if not avg_weights.empty:
        top_sid = str(avg_weights.idxmax())
        top_share = float(avg_weights.max())
        if top_share > thresholds.max_top_strategy_share:
            report.add(
                Finding(
                    check="allocator_concentration",
                    severity=Severity.WARN,
                    message=(
                        f"Strategy {top_sid} averages {top_share:.1%} weight across daily rebalances "
                        f"(max {thresholds.max_top_strategy_share:.0%}) — allocator may be stuck"
                    ),
                    detail={"top_strategy": top_sid, "avg_weight": top_share, "days": n_days},
                )
            )

    # Consecutive IST days with the same top-weighted strategy
    tops: list[str] = []
    for _, chunk in day_weights.groupby("_day", sort=True):
        w = chunk.set_index("strategy_id")["weight"]
        tops.append(str(w.idxmax()))
    streak, streak_sid = _max_consecutive_same(tops)
    if streak > thresholds.max_consecutive_same_top:
        report.add(
            Finding(
                check="allocator_streak",
                severity=Severity.WARN,
                message=(
                    f"Strategy {streak_sid} was top-weighted {streak} consecutive day(s) "
                    f"(max {thresholds.max_consecutive_same_top})"
                ),
                detail={"strategy_id": streak_sid, "streak": streak},
            )
        )

    never_used = enabled - strategies_seen
    if never_used and n_days >= 1 and unique_count >= thresholds.min_unique_strategies:
        report.add(
            Finding(
                check="allocator_never_used",
                severity=Severity.WARN,
                message=f"{len(never_used)} enabled strategies never received weight in window",
                detail={"strategies": sorted(never_used)[:10], "truncated": len(never_used) > 10},
            )
        )

    return report
