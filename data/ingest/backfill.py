from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from data.ingest.nse_client import get_historical_eod
from data.ingest.store import DataStore


def fetch_historical_eod(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Download EOD OHLC for an equity (yfinance) or index (nsedata)."""
    return get_historical_eod(symbol, start, end)


def backfill_eod(
    symbols: list[str],
    years: int = 5,
    *,
    db_path: Path | None = None,
) -> int:
    """
    Bulk EOD backfill via nsedata (pip install nse-archives).
    Returns number of rows written.
    """
    end = date.today()
    start = end - timedelta(days=365 * years)
    written = 0

    with DataStore(db_path=db_path) as store:
        store.init_schema()
        for symbol in symbols:
            raw = fetch_historical_eod(symbol, start, end)
            if raw is None or len(raw) == 0:
                continue
            normalized = store.normalize_bars_1d(raw, symbol)
            written += store.write_bars_1d(normalized)

    return written
