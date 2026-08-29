from __future__ import annotations

from typing import Any

from meta.features import RegimeFeatures

# Defaults match pre-config hardcodes (portfolio.yaml may raise bull/bear bumps).
DEFAULT_LLM_TILT: dict[str, float] = {
    "bull_threshold": 0.25,
    "bear_threshold": -0.25,
    "bull_bump_A": 0.08,
    "bear_bump_G": 0.08,
    "high_materiality_min": 5,
    "high_bump_G": 0.05,
    "high_scale_E": 0.9,
}


def _llm_tilt_cfg(llm_tilt: dict[str, Any] | None) -> dict[str, float]:
    cfg = dict(DEFAULT_LLM_TILT)
    if llm_tilt:
        for k, v in llm_tilt.items():
            if k in cfg and v is not None:
                cfg[k] = float(v)
    return cfg


def rules_v0_weights(
    strategy_ids: list[str],
    cluster_of: dict[str, str],
    regime: RegimeFeatures,
    llm_tilt: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Day-1 meta allocator — interpretable regime routing."""
    n = len(strategy_ids)
    weights = {sid: 1.0 / n for sid in strategy_ids}
    tilt = _llm_tilt_cfg(llm_tilt)

    def bump(cluster: str, delta: float):
        members = [s for s in strategy_ids if cluster_of.get(s) == cluster]
        if not members:
            return
        each = delta / len(members)
        for m in members:
            weights[m] += each

    def scale(cluster: str, factor: float):
        for sid in strategy_ids:
            if cluster_of.get(sid) == cluster:
                weights[sid] *= factor

    if regime.adx > 25:
        bump("A", 0.15)
        scale("B", 0.7)
    elif regime.adx < 20:
        bump("B", 0.15)
        scale("A", 0.7)

    if regime.vix_above_median:
        bump("G", 0.20)
        scale("E", 0.5)

    if regime.expiry_week:
        bump("F", 0.05)
        scale("E", 0.8)

    # Soft LLM tilt — small vs ADX/VIX so missing headlines stay near-neutral
    mean_s = float(getattr(regime, "llm_sentiment_mean", 0.0) or 0.0)
    if mean_s > tilt["bull_threshold"]:
        bump("A", tilt["bull_bump_A"])
    elif mean_s < tilt["bear_threshold"]:
        bump("G", tilt["bear_bump_G"])

    high_n = int(getattr(regime, "llm_high_materiality", 0) or 0)
    if high_n >= int(tilt["high_materiality_min"]):
        bump("G", tilt["high_bump_G"])
        scale("E", tilt["high_scale_E"])

    total = sum(max(w, 0.0) for w in weights.values())
    if total <= 0:
        return {sid: 1.0 / n for sid in strategy_ids}
    return {sid: max(weights[sid], 0.0) / total for sid in strategy_ids}
