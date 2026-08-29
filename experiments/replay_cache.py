"""Export DuckDB OHLCV to per-symbol parquet for lock-free parallel replay."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from data.ingest.store import DataStore
from data.ingest.symbols import load_nifty50_symbols
from data.resample.bars import resample_bars
from nse_trader.config import ROOT

ProgressFn = Callable[[str], None]

DEFAULT_CACHE = ROOT / "data" / "store" / "cache" / "replay_bars"

INTRADAY_TF_RULES = {
    "5m": "5min",
    "15m": "15min",
    "1H": "1h",
}


def cache_symbol_dir(cache_dir: Path, symbol: str) -> Path:
    # Sanitize for filesystem (e.g. M&M)
    safe = symbol.replace("/", "_").replace("\\", "_")
    return Path(cache_dir) / safe


def _write_resampled_tfs(m1: pd.DataFrame | None, out: Path) -> None:
    """Write 5m / 15m / 1H parquet beside 1m for faster workers."""
    empty_cols = ["ts", "symbol", "open", "high", "low", "close", "volume"]
    if m1 is None or m1.empty:
        for name in INTRADAY_TF_RULES:
            pd.DataFrame(columns=empty_cols).to_parquet(out / f"{name}.parquet", index=False)
        return
    m1 = m1.copy()
    m1["ts"] = pd.to_datetime(m1["ts"])
    for name, rule in INTRADAY_TF_RULES.items():
        try:
            tf_df = resample_bars(m1, rule)
        except Exception:  # noqa: BLE001
            tf_df = pd.DataFrame(columns=empty_cols)
        if tf_df is None or tf_df.empty:
            pd.DataFrame(columns=empty_cols).to_parquet(out / f"{name}.parquet", index=False)
        else:
            tf_df.to_parquet(out / f"{name}.parquet", index=False)


def export_replay_bars(
    symbols: list[str] | None = None,
    *,
    cache_dir: Path | None = None,
    db_path: Path | None = None,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> Path:
    """
    Write 1d / 1w / 1m / 5m / 15m / 1H parquet per symbol under cache_dir/{symbol}/.
    Returns cache_dir.
    """
    cache_dir = Path(cache_dir or DEFAULT_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    symbols = symbols or load_nifty50_symbols()
    log = progress or (lambda m: print(m, flush=True))

    with DataStore(db_path=db_path) as store:
        store.init_schema()
        for i, symbol in enumerate(symbols, start=1):
            out = cache_symbol_dir(cache_dir, symbol)
            out.mkdir(parents=True, exist_ok=True)
            paths = {
                "1d": out / "1d.parquet",
                "1w": out / "1w.parquet",
                "1m": out / "1m.parquet",
                "5m": out / "5m.parquet",
                "15m": out / "15m.parquet",
                "1H": out / "1H.parquet",
            }
            if not force and all(p.exists() and p.stat().st_size > 0 for p in paths.values()):
                log(f"[{i}/{len(symbols)}] cache hit {symbol}")
                continue

            log(f"[{i}/{len(symbols)}] export {symbol}")
            d1 = store.read_bars_1d(symbol)
            if not d1.empty:
                d1 = d1.copy()
                d1["ts"] = pd.to_datetime(d1["date"])
                d1.to_parquet(paths["1d"], index=False)
            else:
                pd.DataFrame(
                    columns=["date", "symbol", "open", "high", "low", "close", "volume", "ts"]
                ).to_parquet(paths["1d"], index=False)

            w1 = store.read_bars_1w(symbol)
            if not w1.empty:
                w1 = w1.copy()
                w1["ts"] = pd.to_datetime(w1["date"])
                w1.to_parquet(paths["1w"], index=False)
            else:
                pd.DataFrame(
                    columns=["date", "symbol", "open", "high", "low", "close", "volume", "ts"]
                ).to_parquet(paths["1w"], index=False)

            m1 = store.read_bars_1m(symbol)
            if not m1.empty:
                m1 = m1.copy()
                m1["ts"] = pd.to_datetime(m1["ts"])
                m1.to_parquet(paths["1m"], index=False)
            else:
                m1 = pd.DataFrame(
                    columns=["ts", "symbol", "open", "high", "low", "close", "volume"]
                )
                m1.to_parquet(paths["1m"], index=False)

            _write_resampled_tfs(m1 if not m1.empty else None, out)

    log(f"export_replay_bars done → {cache_dir}")
    return cache_dir
