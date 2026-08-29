"""Budget gate for MCP enrichment calls."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from features.mcp_budget import BudgetConfig, can_spend, record_spend, snapshot

IST = ZoneInfo("Asia/Kolkata")


def test_can_spend_and_record(tmp_path: Path):
    cfg = BudgetConfig(max_calls_per_day=2, max_calls_per_month=40)
    state = tmp_path / "mcp_budget.json"
    log = tmp_path / "mcp_budget.jsonl"
    now = datetime(2026, 8, 29, 8, 40, tzinfo=IST)

    assert can_spend(2, cfg, state_path=state, now=now)
    assert can_spend(3, cfg, state_path=state, now=now) is False

    record_spend(
        provider="finstack",
        tool="get_morning_brief",
        cfg=cfg,
        state_path=state,
        log_path=log,
        now=now,
    )
    record_spend(
        provider="tapetide",
        tool="get_market_news",
        cfg=cfg,
        state_path=state,
        log_path=log,
        now=now,
    )
    snap = snapshot(cfg, state_path=state, now=now)
    assert snap.day_count == 2
    assert snap.remaining_today == 0
    assert can_spend(1, cfg, state_path=state, now=now) is False
    assert log.exists()
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_month_cap(tmp_path: Path):
    cfg = BudgetConfig(max_calls_per_day=10, max_calls_per_month=2)
    state = tmp_path / "mcp_budget.json"
    log = tmp_path / "mcp_budget.jsonl"
    now = datetime(2026, 8, 29, 8, 40, tzinfo=IST)
    record_spend(
        provider="finstack",
        tool="a",
        cfg=cfg,
        state_path=state,
        log_path=log,
        now=now,
    )
    record_spend(
        provider="tapetide",
        tool="b",
        cfg=cfg,
        state_path=state,
        log_path=log,
        now=now,
    )
    assert can_spend(1, cfg, state_path=state, now=now) is False


def test_day_rollover_resets_day_count(tmp_path: Path):
    cfg = BudgetConfig(max_calls_per_day=1, max_calls_per_month=40)
    state = tmp_path / "mcp_budget.json"
    log = tmp_path / "mcp_budget.jsonl"
    d1 = datetime(2026, 8, 28, 8, 40, tzinfo=IST)
    d2 = datetime(2026, 8, 29, 8, 40, tzinfo=IST)
    record_spend(
        provider="finstack",
        tool="a",
        cfg=cfg,
        state_path=state,
        log_path=log,
        now=d1,
    )
    assert can_spend(1, cfg, state_path=state, now=d2)
