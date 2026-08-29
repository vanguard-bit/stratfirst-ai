"""Multi-year history backfill for meta bootstrap (foreground / long-running)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from data.ingest.backfill import fetch_historical_eod
from data.ingest.fyers_history import fetch_fyers_1m
from data.ingest.store import DataStore
from data.ingest.symbols import load_nifty50_symbols
from nse_trader.config import ROOT

ProgressFn = Callable[[str], None]

LOG_PATH = ROOT / "data" / "logs" / "backfill.log"
IST = ZoneInfo("Asia/Kolkata")


def _log(msg: str, progress: ProgressFn | None = None) -> None:
    line = msg.rstrip()
    if progress:
        progress(line)
    else:
        print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _resample_1d_to_1w(df_1d: pd.DataFrame) -> pd.DataFrame:
    if df_1d.empty:
        return df_1d
    df = df_1d.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    out = (
        df.groupby("symbol")
        .resample("W-FRI")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
        .reset_index()
    )
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out[["date", "symbol", "open", "high", "low", "close", "volume"]]


def backfill_history(
    symbols: list[str] | None = None,
    years: int = 3,
    *,
    db_path: Path | None = None,
    tfs: tuple[str, ...] = ("1d", "1W", "1m"),
    progress: ProgressFn | None = None,
    skip_eod: bool = False,
) -> dict[str, int]:
    """
    Backfill history into DuckDB.
    - 1d via yfinance/nsedata; 1W derived from 1d
    - 1m via Fyers history API (100-day chunks) for intraday strategy TFs
    """
    symbols = symbols or load_nifty50_symbols()
    end = date.today()
    start = end - timedelta(days=365 * int(years))
    stats: dict[str, int] = {tf: 0 for tf in tfs}

    with DataStore(db_path=db_path) as store:
        store.init_schema()
        store.ensure_bars_1w()

        if not skip_eod and ("1d" in tfs or "1W" in tfs):
            for i, symbol in enumerate(symbols, start=1):
                _log(f"[{i}/{len(symbols)}] EOD {symbol} {start}→{end}", progress)
                try:
                    raw = fetch_historical_eod(symbol, start, end)
                except Exception as exc:  # noqa: BLE001
                    _log(f"  ERROR {symbol}: {exc}", progress)
                    continue
                if raw is None or len(raw) == 0:
                    _log(f"  skip {symbol}: empty", progress)
                    continue
                try:
                    normalized = store.normalize_bars_1d(raw, symbol)
                    if "1d" in tfs:
                        n = store.write_bars_1d(normalized)
                        stats["1d"] = stats.get("1d", 0) + n
                        _log(f"  wrote 1d rows={n}", progress)
                    if "1W" in tfs:
                        weekly = _resample_1d_to_1w(normalized)
                        n_w = store.write_bars_1w(weekly)
                        stats["1W"] = stats.get("1W", 0) + n_w
                        _log(f"  wrote 1W rows={n_w}", progress)
                except Exception as exc:  # noqa: BLE001
                    _log(f"  ERROR write {symbol}: {exc}", progress)
                    continue

        if "1m" in tfs:
            for i, symbol in enumerate(symbols, start=1):
                _log(f"[{i}/{len(symbols)}] 1m Fyers {symbol} {start}→{end}", progress)
                try:
                    raw = fetch_fyers_1m(
                        symbol,
                        start,
                        end,
                        progress=lambda m: _log(m, progress),
                    )
                except Exception as exc:  # noqa: BLE001
                    _log(f"  ERROR 1m {symbol}: {exc}", progress)
                    continue
                if raw is None or raw.empty:
                    _log(f"  skip 1m {symbol}: empty", progress)
                    continue
                try:
                    # Replace prior history for this symbol (keep today's live ticks if any)
                    today_start = datetime.now(tz=IST).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    ).replace(tzinfo=None)
                    store.con.execute(
                        "DELETE FROM bars_1m WHERE symbol = ? AND ts < ?",
                        [symbol, today_start],
                    )
                    # Also drop overlapping rows from fetch that fall on today
                    hist = raw[pd.to_datetime(raw["ts"]) < today_start]
                    if hist.empty:
                        _log(f"  skip 1m {symbol}: no pre-today rows", progress)
                        continue
                    n = store.write_bars_1m(hist)
                    stats["1m"] = stats.get("1m", 0) + n
                    _log(f"  wrote 1m rows={n}", progress)
                except Exception as exc:  # noqa: BLE001
                    _log(f"  ERROR write 1m {symbol}: {exc}", progress)
                    continue

        for tf in tfs:
            if tf in {"1d", "1W", "1m"}:
                continue
            _log(
                f"TF {tf}: derive on read from bars_1m (no separate store table)",
                progress,
            )
            stats.setdefault(tf, 0)

    _log(f"backfill_history done: {stats}", progress)
    return stats
