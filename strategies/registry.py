from __future__ import annotations

from typing import Any

from nse_trader.config import load_yaml
from strategies.cluster_a.breakout import DonchianBreakout
from strategies.cluster_a.momentum import DualMaCrossover, TimeSeriesMomentum
from strategies.cluster_b.mean_reversion import BollingerZscore, GapFade, Rsi2Revert
from strategies.cluster_c.cross_sectional import LowVolAnomaly, MeanReversionRank, MomentumRank
from strategies.cluster_d.vol_regime import AtrExpansionBreakout, VixRegimeFilter, VolTargetOverlay
from strategies.cluster_e.intraday import OpeningRangeBreakout, PowerHourMomentum, VwapReversion
from strategies.cluster_f.calendar import DayOfWeek, ExpiryWeekEffect, TurnOfMonth
from strategies.cluster_g.defensive import DrawdownCircuitBreaker, LowBetaBasket, TrendAbsentCash

_BUILDERS: dict[str, type] = {
    "time_series_momentum": TimeSeriesMomentum,
    "dual_ma_crossover": DualMaCrossover,
    "donchian_breakout": DonchianBreakout,
    "bollinger_zscore": BollingerZscore,
    "rsi2_revert": Rsi2Revert,
    "gap_fade": GapFade,
    "momentum_rank": MomentumRank,
    "low_vol_anomaly": LowVolAnomaly,
    "mean_reversion_rank": MeanReversionRank,
    "vol_target_overlay": VolTargetOverlay,
    "vix_regime_filter": VixRegimeFilter,
    "atr_expansion_breakout": AtrExpansionBreakout,
    "opening_range_breakout": OpeningRangeBreakout,
    "vwap_reversion": VwapReversion,
    "power_hour_momentum": PowerHourMomentum,
    "turn_of_month": TurnOfMonth,
    "day_of_week": DayOfWeek,
    "expiry_week_effect": ExpiryWeekEffect,
    "trend_absent_cash": TrendAbsentCash,
    "drawdown_circuit_breaker": DrawdownCircuitBreaker,
    "low_beta_basket": LowBetaBasket,
}


def build_strategy(strategy_id: str, cfg: dict[str, Any] | None = None) -> Any:
    """Instantiate a strategy from config/strategies.yaml entry."""
    all_cfg = load_yaml("strategies.yaml")["strategies"]
    if strategy_id not in all_cfg:
        raise KeyError(f"unknown strategy {strategy_id!r}")
    entry = all_cfg[strategy_id]
    name = entry["name"]
    cls = _BUILDERS.get(name)
    if cls is None:
        raise NotImplementedError(f"strategy {strategy_id} ({name}) not implemented yet")
    params = dict(entry.get("params", {}))
    strat = cls(id=strategy_id, **params)
    # YAML is source of truth for routing metadata
    if "timeframe" in entry:
        strat.timeframe = entry["timeframe"]
    if "product" in entry:
        strat.product = entry["product"]
    if "cluster" in entry:
        strat.cluster = entry["cluster"]
    return strat


def load_enabled_strategies() -> dict[str, Any]:
    cfg = load_yaml("strategies.yaml")["strategies"]
    out: dict[str, Any] = {}
    for sid, entry in cfg.items():
        if not entry.get("enabled", True):
            continue
        out[sid] = build_strategy(sid, entry)
    return out


def all_strategy_ids() -> list[str]:
    return sorted(load_yaml("strategies.yaml")["strategies"].keys())
