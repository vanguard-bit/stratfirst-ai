"""Daily strategy returns → top-K binary labels for meta training."""

from __future__ import annotations

import pandas as pd


def top_k_labels(returns: pd.DataFrame, *, k: int = 5) -> pd.DataFrame:
    """
    Rank strategies within each date by `ret` (desc).
    y=1 if rank <= k. Rows with NaN ret are dropped (excluded from that day's rank).
    """
    need = {"date", "strategy_id", "ret"}
    missing = need - set(returns.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    clean = returns.dropna(subset=["ret"]).copy()
    if clean.empty:
        return pd.DataFrame(columns=["date", "strategy_id", "ret", "rank", "y"])

    rows: list[pd.DataFrame] = []
    for _, g in clean.groupby("date", sort=True):
        g = g.copy()
        g["rank"] = g["ret"].rank(ascending=False, method="first").astype(int)
        g["y"] = (g["rank"] <= int(k)).astype(int)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)
