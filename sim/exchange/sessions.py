from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# NSE cash equity regular session (IST).
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)


def is_trading_session(ts: str | datetime) -> bool:
    """True during Mon–Fri regular NSE cash session."""
    if isinstance(ts, str):
        dt = datetime.fromisoformat(ts)
    else:
        dt = ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    else:
        dt = dt.astimezone(IST)

    if dt.weekday() >= 5:
        return False

    t = dt.time()
    return SESSION_OPEN <= t <= SESSION_CLOSE
