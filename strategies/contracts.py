"""Per-strategy state-key contracts and yaml alignment for audit gates."""

from __future__ import annotations

from typing import Any

from nse_trader.config import load_yaml
from strategies.registry import all_strategy_ids, build_strategy, load_enabled_strategies

# Keys each strategy reads from `state` (must be present after warm build_state,
# or injected by paper-live: in_position, vix_above_median, portfolio_drawdown, expiry_week).
REQUIRED_KEYS: dict[str, frozenset[str]] = {
    "A1": frozenset({"returns_252d", "close_vs_ma100", "returns_lookback_ready"}),
    "A2": frozenset({"ma_cross", "warmup", "ma_slow", "in_position"}),
    "A3": frozenset({"warmup", "donchian_upper", "donchian_lower"}),
    "B1": frozenset({"zscore", "in_position"}),
    "B2": frozenset({"rsi", "close_vs_ma200", "in_position"}),
    "B3": frozenset({"session_gap_pct", "vwap_dev", "in_position", "session_hhmm"}),
    "C1": frozenset({"momentum_rank"}),
    "C2": frozenset({"vol_quintile"}),
    "C3": frozenset({"reversion_rank"}),
    "D1": frozenset({"realized_vol"}),
    "D2": frozenset({"vix_above_median"}),
    "D3": frozenset({"atr_ratio", "breakout_dir", "donchian_upper", "donchian_lower", "in_position"}),
    "E1": frozenset({"orb_complete", "orb_high", "orb_low", "in_position", "e1_traded_today"}),
    "E2": frozenset({"vwap_dev", "in_position"}),
    "E3": frozenset({"in_power_hour", "intraday_mom", "in_position"}),
    "F1": frozenset({"in_turn_of_month", "day_of_month"}),
    "F2": frozenset({"day_of_week"}),
    "F3": frozenset({"expiry_week"}),
    "G1": frozenset({"close_vs_ma200", "adx"}),
    "G2": frozenset({"portfolio_drawdown"}),
    "G3": frozenset({"beta_rank"}),
}

# Keys that are intentional stubs / live-injected (not missing, but quality P2).
STUB_OR_INJECTED_KEYS: frozenset[str] = frozenset(
    {
        "adx",  # constant stub until full ADX
        "vix_above_median",  # False in build_state; paper overlays regime
        "portfolio_drawdown",  # caller-supplied
        "in_position",  # paper books / replay from exposure
        "position_qty",  # signed qty; paper + replay inject
        "e1_traded_today",  # E1 one-shot; paper session file / replay flag
        "session_hhmm",  # IST HH:MM for E3 cutoff; paper + replay inject
        "vol_quintile",  # short vol proxy when multi-symbol frame present
        # CS ranks (momentum/reversion/beta) filled by features.cs_ranks when universe present;
        # stay 999 on single-symbol / state_series paths.
    }
)


def assert_required_keys_complete() -> None:
    ids = all_strategy_ids()
    missing = [sid for sid in ids if sid not in REQUIRED_KEYS]
    extra = [sid for sid in REQUIRED_KEYS if sid not in ids]
    if missing or extra:
        raise AssertionError(f"REQUIRED_KEYS mismatch missing={missing} extra={extra}")


def yaml_code_alignment() -> list[dict[str, Any]]:
    """Compare strategies.yaml timeframe/product/cluster to built instances."""
    cfg = load_yaml("strategies.yaml")["strategies"]
    rows: list[dict[str, Any]] = []
    for sid in all_strategy_ids():
        entry = cfg[sid]
        strat = build_strategy(sid)
        rows.append(
            {
                "id": sid,
                "name": entry["name"],
                "yaml_tf": entry.get("timeframe"),
                "code_tf": getattr(strat, "timeframe", None),
                "yaml_product": entry.get("product"),
                "code_product": getattr(strat, "product", None),
                "yaml_cluster": entry.get("cluster"),
                "code_cluster": getattr(strat, "cluster", None),
                "tf_ok": entry.get("timeframe") == getattr(strat, "timeframe", None),
                "product_ok": entry.get("product") == getattr(strat, "product", None),
                "cluster_ok": entry.get("cluster") == getattr(strat, "cluster", None),
                "required_keys": sorted(REQUIRED_KEYS[sid]),
            }
        )
    return rows


def enabled_strategy_ids() -> list[str]:
    return sorted(load_enabled_strategies().keys())
