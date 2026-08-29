"""Fyers historical candles (1m / other resolutions) for meta backfill."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from data.ingest.fyers_auth import ensure_valid_access_token, is_fyers_auth_error, rest_access_token
from data.ingest.symbols import to_fyers_symbol
from nse_trader.env import require_env

IST = ZoneInfo("Asia/Kolkata")

# Fyers: up to 100 days per request for minute resolutions
_CHUNK_DAYS = {"1": 100, "5": 100, "15": 100, "60": 100, "D": 366, "1D": 366}


def _date_chunks(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    out: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=chunk_days - 1), end)
        out.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return out


def _candles_to_df(candles: list[list[Any]], symbol: str) -> pd.DataFrame:
    """Fyers candle: [epoch, open, high, low, close, volume]."""
    if not candles:
        return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume"])
    rows = []
    for c in candles:
        if not c or len(c) < 6:
            continue
        ts = datetime.fromtimestamp(int(c[0]), tz=IST).replace(tzinfo=None)
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            }
        )
    return pd.DataFrame(rows)


def fetch_fyers_history_chunk(
    symbol: str,
    start: date,
    end: date,
    *,
    resolution: str = "1",
) -> pd.DataFrame:
    """One Fyers history request (must respect resolution day limits)."""
    from fyers_apiv3 import fyersModel

    ensure_valid_access_token(skew_seconds=300)
    app_id = require_env("FYERS_APP_ID")
    jwt = rest_access_token()
    if not jwt:
        raise RuntimeError("No Fyers access token")

    client = fyersModel.FyersModel(client_id=app_id, token=jwt, is_async=False, log_path="")
    payload = {
        "symbol": to_fyers_symbol(symbol),
        "resolution": resolution,
        "date_format": "1",
        "range_from": start.isoformat(),
        "range_to": end.isoformat(),
        "cont_flag": "1",
    }
    resp = client.history(data=payload)
    if is_fyers_auth_error(resp):
        ensure_valid_access_token(force=True)
        jwt = rest_access_token()
        client = fyersModel.FyersModel(client_id=app_id, token=jwt, is_async=False, log_path="")
        resp = client.history(data=payload)
    if not isinstance(resp, dict) or resp.get("s") != "ok":
        raise RuntimeError(f"Fyers history failed for {symbol} {start}→{end}: {resp}")
    return _candles_to_df(list(resp.get("candles") or []), symbol)


def fetch_fyers_1m(
    symbol: str,
    start: date,
    end: date,
    *,
    sleep_s: float = 0.35,
    progress: Any | None = None,
) -> pd.DataFrame:
    """Fetch multi-year 1m bars in ≤100-day chunks."""
    chunks = _date_chunks(start, end, _CHUNK_DAYS["1"])
    parts: list[pd.DataFrame] = []
    for i, (a, b) in enumerate(chunks, start=1):
        if progress:
            progress(f"    1m chunk {i}/{len(chunks)} {a}→{b}")
        df = fetch_fyers_history_chunk(symbol, a, b, resolution="1")
        if not df.empty:
            parts.append(df)
        if sleep_s > 0 and i < len(chunks):
            time.sleep(sleep_s)
    if not parts:
        return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume"])
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["ts", "symbol"]).sort_values("ts").reset_index(drop=True)
    return out
