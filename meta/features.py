from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RegimeFeatures:
    adx: float = 0.0
    vix: float = 0.0
    vix_above_median: bool = False
    expiry_week: bool = False
    nifty_return_20d: float = 0.0
    # LLM side-channel (from latest features/llm_*.parquet; 0 = neutral / missing)
    llm_sentiment_mean: float = 0.0
    llm_high_materiality: int = 0
    llm_as_of: str = ""


def strategy_features(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    """Flatten per-strategy metrics for meta model."""
    out: dict[str, float] = {}
    for sid, m in metrics.items():
        for k, v in m.items():
            out[f"{sid}_{k}"] = float(v)
    return out


def build_feature_vector(
    regime: RegimeFeatures,
    metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    vec = {
        "regime_adx": regime.adx,
        "regime_vix": regime.vix,
        "regime_vix_high": float(regime.vix_above_median),
        "regime_expiry_week": float(regime.expiry_week),
        "regime_nifty_ret_20d": regime.nifty_return_20d,
        "regime_llm_sentiment": float(regime.llm_sentiment_mean),
        "regime_llm_high_mat": float(regime.llm_high_materiality),
    }
    vec.update(strategy_features(metrics))
    return vec
