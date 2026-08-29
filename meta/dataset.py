"""Build feature/label panel for LightGBM meta training (no lookahead)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from meta.labels import top_k_labels

# Frozen for train + shadow parity. Do not include ret_fwd / y / date / strategy_id.
FEATURE_COLS: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "cluster_code",
    "tf_code",
    "regime_adx",
    "regime_vix",
    "regime_vix_high",
    "regime_expiry_week",
    "regime_nifty_ret_20d",
    "llm_sentiment",
    "llm_high_mat",
]

_TF_CODE = {"5m": 0, "15m": 1, "1H": 2, "1h": 2, "1D": 3, "1d": 3, "1W": 4, "1w": 4}
_CLUSTER_CODE = {c: i for i, c in enumerate("ABCDEFG")}


def _default_strategy_meta() -> dict[str, dict[str, Any]]:
    try:
        from strategies.registry import load_enabled_strategies

        out: dict[str, dict[str, Any]] = {}
        for sid, strat in load_enabled_strategies().items():
            cluster = str(getattr(strat, "cluster", "A"))
            tf = str(getattr(strat, "timeframe", "1D"))
            out[sid] = {
                "cluster": cluster,
                "cluster_code": int(_CLUSTER_CODE.get(cluster, 0)),
                "tf_code": int(_TF_CODE.get(tf, 3)),
            }
        return out
    except Exception:  # noqa: BLE001
        return {}


def _trailing_by_strategy(returns: pd.DataFrame) -> pd.DataFrame:
    """Per strategy-day trailing returns/vol using info available at EOD of `date` (incl. that day's ret)."""
    parts: list[pd.DataFrame] = []
    for sid, g in returns.sort_values("date").groupby("strategy_id", sort=False):
        g = g.copy()
        r = g["ret"].astype(float)
        g["ret_1d"] = r
        g["ret_5d"] = r.rolling(5, min_periods=1).sum()
        g["ret_20d"] = r.rolling(20, min_periods=1).sum()
        g["vol_20d"] = r.rolling(20, min_periods=2).std().fillna(0.0)
        parts.append(g)
    if not parts:
        return returns.copy()
    return pd.concat(parts, ignore_index=True)


def build_meta_panel(
    returns: pd.DataFrame,
    *,
    k: int = 5,
    regime_by_date: dict[str, dict[str, float]] | None = None,
    strategy_meta: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Features/regime for date t; y and ret_fwd from ranks of ret on t+1.

    Input columns: date, strategy_id, ret.
    """
    need = {"date", "strategy_id", "ret"}
    missing = need - set(returns.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    base = returns.dropna(subset=["ret"]).copy()
    base["date"] = base["date"].astype(str)
    if base.empty:
        cols = ["date", "strategy_id", "y", "ret_fwd", *FEATURE_COLS]
        return pd.DataFrame(columns=cols)

    feat = _trailing_by_strategy(base)

    # Labels from next calendar day in the panel's date set (per strategy join on date)
    dates = sorted(feat["date"].unique())
    next_map = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    labels = top_k_labels(base, k=k)
    labels["date"] = labels["date"].astype(str)
    fwd = labels.rename(columns={"date": "fwd_date", "ret": "ret_fwd"})[
        ["fwd_date", "strategy_id", "ret_fwd", "y"]
    ]

    feat["fwd_date"] = feat["date"].map(next_map)
    panel = feat.merge(fwd, on=["fwd_date", "strategy_id"], how="inner")
    panel = panel.drop(columns=["fwd_date", "ret"], errors="ignore")

    meta = strategy_meta if strategy_meta is not None else _default_strategy_meta()
    cluster_codes: list[int] = []
    tf_codes: list[int] = []
    for sid in panel["strategy_id"]:
        m = meta.get(str(sid), {})
        if "cluster_code" in m:
            cluster_codes.append(int(m["cluster_code"]))
        else:
            cluster_codes.append(int(_CLUSTER_CODE.get(str(m.get("cluster", "A")), 0)))
        if "tf_code" in m:
            tf_codes.append(int(m["tf_code"]))
        else:
            tf_codes.append(int(_TF_CODE.get(str(m.get("timeframe", "1D")), 3)))
    panel["cluster_code"] = cluster_codes
    panel["tf_code"] = tf_codes

    regime_by_date = regime_by_date or {}
    defaults = {
        "regime_adx": 0.0,
        "regime_vix": 0.0,
        "regime_vix_high": 0.0,
        "regime_expiry_week": 0.0,
        "regime_nifty_ret_20d": 0.0,
        "llm_sentiment": 0.0,
        "llm_high_mat": 0.0,
    }
    for col, default in defaults.items():
        panel[col] = [
            float(regime_by_date.get(d, {}).get(col, default)) for d in panel["date"]
        ]

    # Aliases accepted in regime dict
    for i, d in enumerate(panel["date"]):
        rd = regime_by_date.get(d, {})
        if "llm_sentiment_mean" in rd:
            panel.iat[i, panel.columns.get_loc("llm_sentiment")] = float(rd["llm_sentiment_mean"])
        if "regime_llm_sentiment" in rd:
            panel.iat[i, panel.columns.get_loc("llm_sentiment")] = float(rd["regime_llm_sentiment"])
        if "regime_llm_high_mat" in rd:
            panel.iat[i, panel.columns.get_loc("llm_high_mat")] = float(rd["regime_llm_high_mat"])

    for c in FEATURE_COLS:
        if c not in panel.columns:
            panel[c] = 0.0
        panel[c] = panel[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    keep = ["date", "strategy_id", "y", "ret_fwd", *FEATURE_COLS]
    return panel[keep].reset_index(drop=True)
