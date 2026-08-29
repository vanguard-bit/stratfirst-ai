"""Cross-sectional ranks for C1/C3/G3 — lookback return / beta, not last close."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MOM_LOOKBACK = 20
REV_LOOKBACK = 5
BETA_LOOKBACK = 60
BETA_MIN_OBS = 20


def _ts_col(df: pd.DataFrame) -> str:
    return "ts" if "ts" in df.columns else "date"


def cross_sectional_ranks(univ: pd.DataFrame, symbol: str) -> dict[str, Any]:
    """
    Rank `symbol` vs universe bars (multi-symbol OHLC).

    - momentum_rank: MOM_LOOKBACK-bar return, descending (1 = strongest)
    - reversion_rank: REV_LOOKBACK-bar return, ascending (1 = weakest / loser)
    - beta_rank: BETA_LOOKBACK-bar beta vs equal-weight universe, ascending (1 = lowest)
    - low_vol_rank: short realized vol ascending (1 = quietest)
    - vol_quintile: 1..5 from low_vol_rank
    """
    defaults: dict[str, Any] = {
        "momentum_rank": 999,
        "reversion_rank": 999,
        "beta_rank": 999,
        "low_vol_rank": 999,
        "vol_quintile": 3,
        "universe_size": 50,
    }
    if univ is None or univ.empty or "symbol" not in univ.columns or "close" not in univ.columns:
        return defaults

    ts_col = _ts_col(univ)
    df = univ.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    symbols = sorted(df["symbol"].astype(str).unique())
    if len(symbols) < 2:
        return defaults

    closes: dict[str, pd.Series] = {}
    for sym in symbols:
        g = df[df["symbol"].astype(str) == sym].sort_values(ts_col)
        closes[sym] = g["close"].astype(float).reset_index(drop=True)

    mom: dict[str, float] = {}
    rev: dict[str, float] = {}
    vol: dict[str, float] = {}
    for sym, c in closes.items():
        n = len(c)
        if n >= MOM_LOOKBACK + 1:
            mom[sym] = float(c.iloc[-1] / c.iloc[-(MOM_LOOKBACK + 1)] - 1.0)
        elif n >= 2:
            mom[sym] = float(c.iloc[-1] / c.iloc[0] - 1.0)
        else:
            mom[sym] = 0.0
        if n >= REV_LOOKBACK + 1:
            rev[sym] = float(c.iloc[-1] / c.iloc[-(REV_LOOKBACK + 1)] - 1.0)
        elif n >= 2:
            rev[sym] = float(c.iloc[-1] / c.iloc[0] - 1.0)
        else:
            rev[sym] = 0.0
        if n >= 6:
            vol[sym] = float(c.pct_change().tail(5).std() or 0.0)
        else:
            vol[sym] = 0.0

    # Align last BETA_LOOKBACK+1 closes for beta vs equal-weight market
    betas: dict[str, float] = {}
    min_len = min(len(closes[s]) for s in symbols)
    use = min(min_len, BETA_LOOKBACK + 1)
    if use >= BETA_MIN_OBS + 1:
        mat = np.column_stack([closes[s].iloc[-use:].to_numpy(dtype=float) for s in symbols])
        rets = np.diff(mat, axis=0) / np.maximum(mat[:-1], 1e-12)
        mkt = rets.mean(axis=1)
        var_m = float(np.var(mkt))
        for i, sym in enumerate(symbols):
            y = rets[:, i]
            if var_m < 1e-18:
                betas[sym] = 1.0
            else:
                betas[sym] = float(np.cov(y, mkt, ddof=0)[0, 1] / var_m)
    else:
        for sym in symbols:
            betas[sym] = 1.0

    mom_s = pd.Series(mom)
    rev_s = pd.Series(rev)
    beta_s = pd.Series(betas)
    vol_s = pd.Series(vol)

    mom_rank = mom_s.rank(ascending=False, method="first").astype(int)
    rev_rank = rev_s.rank(ascending=True, method="first").astype(int)
    beta_rank = beta_s.rank(ascending=True, method="first").astype(int)
    low_vol_rank = vol_s.rank(ascending=True, method="first").astype(int)

    sym = str(symbol)
    n_u = len(symbols)
    lv = int(low_vol_rank.get(sym, 999))
    quint = 3
    if lv != 999 and n_u > 0:
        quint = int(min(5, max(1, np.ceil(lv / n_u * 5))))

    return {
        "momentum_rank": int(mom_rank.get(sym, 999)),
        "reversion_rank": int(rev_rank.get(sym, 999)),
        "beta_rank": int(beta_rank.get(sym, 999)),
        "low_vol_rank": lv,
        "vol_quintile": quint,
        "universe_size": n_u,
    }
