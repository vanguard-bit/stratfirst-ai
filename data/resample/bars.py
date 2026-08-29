from __future__ import annotations

import pandas as pd


def resample_bars(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1m OHLCV to 5m, 15m, 1h, 1D locally."""
    df = df_1m.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    out = (
        df.groupby("symbol")
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    return out
