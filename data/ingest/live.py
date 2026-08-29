"""Record live bid/ask snapshots into DuckDB (no orders)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from data.ingest.store import DataStore
from sim.friction.measured import MeasuredFriction, Quote

IST = ZoneInfo("Asia/Kolkata")
logger = __import__("logging").getLogger(__name__)


def record_spread_snapshot(
    quotes: list[Quote],
    *,
    db_path: Path | None = None,
    ts: datetime | None = None,
) -> int:
    """Persist measured spreads for friction audit / backtest replay."""
    stamp = ts or datetime.now(tz=IST)
    rows = []
    for q in quotes:
        if q.bid is None or q.ask is None or q.ask < q.bid or q.bid <= 0:
            continue
        half_bps = MeasuredFriction.half_spread_bps(q)
        rows.append(
            {
                "ts": stamp,
                "symbol": q.symbol,
                "bid": q.bid,
                "ask": q.ask,
                "ltp": q.ltp,
                "half_spread_bps": half_bps,
                "upper_ckt": q.upper_ckt,
                "lower_ckt": q.lower_ckt,
                "prev_close": q.prev_close,
            }
        )
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    with DataStore(db_path=db_path) as store:
        store.init_schema()
        return store.write_friction_spreads(df)


def run_ingest_once(symbols: list[str] | None = None, duration_sec: int = 55) -> dict:
    """Market-hours ingest — Fyers websocket when creds set, else placeholder."""
    from data.ingest.symbols import load_nifty50_symbols

    symbols = symbols or load_nifty50_symbols()[:10]
    try:
        from data.ingest.fyers_ws import run_fyers_websocket_ingest

        return run_fyers_websocket_ingest(symbols, duration_sec=duration_sec)
    except RuntimeError as exc:
        logger.warning("fyers unavailable (%s) — placeholder ingest", exc)
    except ImportError as exc:
        logger.warning("fyers-apiv3 not installed (%s) — placeholder ingest", exc)

    quotes = [
        Quote(
            s,
            ltp=2500.0,
            bid=2499.5,
            ask=2500.5,
            timestamp=datetime.now(tz=IST).isoformat(),
        )
        for s in symbols
    ]
    rows = record_spread_snapshot(quotes)
    return {"mode": "placeholder", "spread_rows": rows, "symbols": len(symbols)}


def run_live_ingest(symbols: list[str] | None = None, duration_sec: int = 55) -> dict:
    """Public API for systemd / paper --mode ingest."""
    from data.ingest.symbols import load_nifty50_symbols

    syms = symbols or load_nifty50_symbols()
    return run_ingest_once(syms, duration_sec=duration_sec)
