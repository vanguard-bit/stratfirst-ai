"""IST market-session helpers for short-lived systemd ticks."""

from __future__ import annotations

from datetime import date, datetime, time
from functools import lru_cache
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


@lru_cache(maxsize=1)
def _holiday_calendar() -> dict[int, dict[date, str]]:
    raw = load_yaml("nse_holidays.yaml") or {}
    return {
        int(year): {date.fromisoformat(str(d)): str(name) for d, name in (days or {}).items()}
        for year, days in raw.items()
    }


def holiday_calendar_covers(year: int) -> bool:
    return bool(_holiday_calendar().get(year))


def nse_holiday(ts: datetime | date | None = None) -> str | None:
    """Holiday name if the IST date is a weekday NSE trading holiday."""
    stamp = ts or now_ist()
    day = stamp.astimezone(IST).date() if isinstance(stamp, datetime) else stamp
    return _holiday_calendar().get(day.year, {}).get(day)


def is_trading_day(ts: datetime | date | None = None) -> bool:
    stamp = ts or now_ist()
    return stamp.weekday() < 5 and nse_holiday(stamp) is None


def in_ingest_window(ts: datetime | None = None) -> bool:
    stamp = ts or now_ist()
    if not is_trading_day(stamp):
        return False
    start, end = ingest_bounds()
    return start <= stamp.time() <= end
