"""Daily / monthly MCP call budget for llm enrichment."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from nse_trader.config import ROOT

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger(__name__)

DEFAULT_STATE = ROOT / "data" / "state" / "mcp_budget.json"
DEFAULT_LOG = ROOT / "data" / "logs" / "mcp_budget.jsonl"


@dataclass
class BudgetConfig:
    max_calls_per_day: int = 2
    max_calls_per_month: int = 40


@dataclass
class BudgetSnapshot:
    day: str
    month: str
    day_count: int
    month_count: int
    max_day: int
    max_month: int

    @property
    def remaining_today(self) -> int:
        return max(0, self.max_day - self.day_count)

    @property
    def remaining_month(self) -> int:
        return max(0, self.max_month - self.month_count)


def _today_parts(now: datetime | None = None) -> tuple[str, str]:
    now = now or datetime.now(tz=IST)
    d = now.date()
    return d.isoformat(), f"{d.year:04d}-{d.month:02d}"


def load_budget_state(path: Path | None = None) -> dict[str, Any]:
    p = Path(path or DEFAULT_STATE)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def snapshot(
    cfg: BudgetConfig,
    *,
    state_path: Path | None = None,
    now: datetime | None = None,
) -> BudgetSnapshot:
    day, month = _today_parts(now)
    raw = load_budget_state(state_path)
    day_count = int(raw.get("day_count", 0) or 0) if raw.get("day") == day else 0
    month_count = int(raw.get("month_count", 0) or 0) if raw.get("month") == month else 0
    return BudgetSnapshot(
        day=day,
        month=month,
        day_count=day_count,
        month_count=month_count,
        max_day=int(cfg.max_calls_per_day),
        max_month=int(cfg.max_calls_per_month),
    )


def can_spend(
    n: int,
    cfg: BudgetConfig,
    *,
    state_path: Path | None = None,
    now: datetime | None = None,
) -> bool:
    if n <= 0:
        return True
    snap = snapshot(cfg, state_path=state_path, now=now)
    return snap.day_count + n <= snap.max_day and snap.month_count + n <= snap.max_month


def record_spend(
    *,
    provider: str,
    tool: str,
    cfg: BudgetConfig,
    state_path: Path | None = None,
    log_path: Path | None = None,
    now: datetime | None = None,
    ok: bool = True,
    detail: str = "",
) -> BudgetSnapshot:
    """Increment counters by 1 (attempt counts even on failure once issued)."""
    now = now or datetime.now(tz=IST)
    day, month = _today_parts(now)
    state_p = Path(state_path or DEFAULT_STATE)
    log_p = Path(log_path or DEFAULT_LOG)
    raw = load_budget_state(state_p)
    day_count = int(raw.get("day_count", 0) or 0) if raw.get("day") == day else 0
    month_count = int(raw.get("month_count", 0) or 0) if raw.get("month") == month else 0
    day_count += 1
    month_count += 1
    payload = {
        "day": day,
        "month": month,
        "day_count": day_count,
        "month_count": month_count,
        "updated_at": now.isoformat(),
    }
    state_p.parent.mkdir(parents=True, exist_ok=True)
    state_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_p.parent.mkdir(parents=True, exist_ok=True)
    with log_p.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "ts": now.isoformat(),
                    "day": day,
                    "provider": provider,
                    "tool": tool,
                    "ok": ok,
                    "detail": detail[:500],
                    "day_count": day_count,
                    "month_count": month_count,
                },
                default=str,
            )
            + "\n"
        )
    return snapshot(cfg, state_path=state_p, now=now)
