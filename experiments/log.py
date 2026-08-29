from __future__ import annotations

from pathlib import Path

import pandas as pd

from nse_trader.config import PortfolioConfig

TRADE_LOG_COLUMNS = [
    "trade_id",
    "ts",
    "strategy_id",
    "cluster",
    "symbol",
    "side",
    "qty",
    "signal_price",
    "fill_price",
    "total_cost",
    "regime_adx",
    "regime_vix",
    "expiry_flag",
]


def log_trades(
    trades: list[dict],
    run_id: str,
    base_dir: Path | None = None,
) -> Path:
    if base_dir is not None:
        out_dir = base_dir / run_id
    else:
        cfg = PortfolioConfig.load()
        out_dir = cfg.store_path / "experiments" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "trades.parquet"
    df = pd.DataFrame(trades)
    if path.exists():
        prev = pd.read_parquet(path)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(path, index=False)
    return path
