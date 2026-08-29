"""IST market-session helpers for short-lived systemd ticks."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from nse_trader.config import load_yaml

IST = ZoneInfo("Asia/Kolkata")


def session_bounds() -> tuple[time, time]:
    market = load_yaml("ops.yaml").get("market", {})
    open_s = market.get("pre_open", "09:00")
    close_s = market.get("close", "15:30")
    # ingest window uses ops jobs active start; default slightly before open
    start = time.fromisoformat(str(open_s))
    end = time.fromisoformat(str(close_s))
    return start, end


def ingest_bounds() -> tuple[time, time]:
    """Mon–Fri ingest window from ops jobs.active (09:10–15:35)."""
    return time(9, 10), time(15, 35)


def now_ist() -> datetime:
    return datetime.now(tz=IST)


def is_weekday(ts: datetime | None = None) -> bool:
    stamp = ts or now_ist()
    return stamp.weekday() < 5


def in_ingest_window(ts: datetime | None = None) -> bool:
    stamp = ts or now_ist()
    if not is_weekday(stamp):
        return False
    start, end = ingest_bounds()
    return start <= stamp.time() <= end
