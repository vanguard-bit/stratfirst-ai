"""Per-cluster on_bar entry/exit/warmup behaviour for all 21 strategies."""

from __future__ import annotations

from strategies.base import Bar
from strategies.registry import build_strategy


def _bar(symbol: str = "RELIANCE", close: float = 100.0, tf: str = "1D") -> Bar:
    return Bar(
        ts="2026-08-11",
        symbol=symbol,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1e6,
        timeframe=tf,
    )


# --- Cluster A ---


def test_a1_holds_without_lookback():
    a1 = build_strategy("A1")
    sig = a1.on_bar(_bar(), {"returns_lookback_ready": False, "returns_252d": 0.0, "close_vs_ma100": 1.1})
    assert sig.action == "HOLD"


def test_a1_buy_and_sell():
    a1 = build_strategy("A1")
    buy = a1.on_bar(
        _bar(),
        {"returns_lookback_ready": True, "returns_252d": 0.1, "close_vs_ma100": 1.02, "warmup": False},
    )
    sell = a1.on_bar(
        _bar(),
        {"returns_lookback_ready": True, "returns_252d": -0.1, "close_vs_ma100": 0.99, "warmup": False},
    )
    assert buy.action == "BUY"
    assert sell.action == "SELL"


def test_a2_cross_and_warmup():
    a2 = build_strategy("A2")
    assert a2.on_bar(_bar(tf="1H"), {"warmup": True, "ma_cross": "bullish"}).action == "HOLD"
    assert a2.on_bar(
        _bar(tf="1H", close=110),
        {"warmup": False, "ma_cross": "bullish", "ma_slow": 100.0, "position_qty": 0},
    ).action == "BUY"
    # Bearish while long → flat
    assert (
        a2.on_bar(
            _bar(tf="1H", close=110),
            {
                "warmup": False,
                "ma_cross": "bearish",
                "ma_slow": 100.0,
                "position_qty": 1,
                "in_position": True,
            },
        ).action
        == "FLAT"
    )
    # Slow-MA stop while long
    assert (
        a2.on_bar(
            _bar(tf="1H", close=99),
            {
                "warmup": False,
                "ma_cross": "none",
                "ma_slow": 100.0,
                "position_qty": 1,
                "in_position": True,
            },
        ).action
        == "FLAT"
    )
    assert a2.on_bar(
        _bar(tf="1H", close=110),
        {"warmup": False, "ma_cross": "none", "ma_slow": 100.0, "position_qty": 0},
    ).action == "HOLD"


def test_a3_donchian_edges():
    a3 = build_strategy("A3")
    assert a3.on_bar(_bar(close=100), {"warmup": True}).action == "HOLD"
    assert a3.on_bar(_bar(close=100), {"warmup": False, "donchian_upper": float("nan"), "donchian_lower": 90}).action == "HOLD"
    assert a3.on_bar(_bar(close=110), {"warmup": False, "donchian_upper": 105.0, "donchian_lower": 95.0}).action == "BUY"
    assert a3.on_bar(_bar(close=90), {"warmup": False, "donchian_upper": 105.0, "donchian_lower": 95.0}).action == "SELL"
    assert a3.on_bar(_bar(close=100), {"warmup": False, "donchian_upper": 105.0, "donchian_lower": 95.0}).action == "HOLD"


# --- Cluster B ---


def test_b1_zscore_entry_exit():
    b1 = build_strategy("B1")
    assert b1.on_bar(_bar(tf="15m"), {"zscore": -2.5}).action == "BUY"
    assert b1.on_bar(_bar(tf="15m"), {"zscore": 0.5, "in_position": True}).action == "SELL"
    assert b1.on_bar(_bar(tf="15m"), {"zscore": 0.5, "in_position": False}).action == "HOLD"


def test_b2_rsi_edges():
    b2 = build_strategy("B2")
    # Connors: trend filter required for entry
    assert b2.on_bar(_bar(tf="5m"), {"rsi": 5, "close_vs_ma200": 0.99, "position_qty": 0}).action == "HOLD"
    assert b2.on_bar(_bar(tf="5m"), {"rsi": 5, "close_vs_ma200": 1.01, "position_qty": 0}).action == "BUY"
    # Exit at RSI >= 65 (not 90)
    ex = b2.on_bar(
        _bar(tf="5m"),
        {"rsi": 66, "close_vs_ma200": 1.01, "position_qty": 1, "in_position": True},
    )
    assert ex.action in ("SELL", "FLAT")
    assert abs(ex.intended_exposure) == 0.0
    # RSI 95 while flat is not a short
    assert b2.on_bar(_bar(tf="5m"), {"rsi": 95, "close_vs_ma200": 1.01, "position_qty": 0}).action == "HOLD"
    assert b2.on_bar(_bar(tf="5m"), {"rsi": 50, "close_vs_ma200": 1.01, "position_qty": 0}).action == "HOLD"


def test_b3_gap_fade():
    b3 = build_strategy("B3")
    # Morning fade: gap up ≥1.5% → short
    assert (
        b3.on_bar(
            _bar(tf="15m"),
            {
                "session_gap_pct": 0.02,
                "vwap_dev": 0.015,
                "session_hhmm": "09:30",
                "position_qty": 0,
            },
        ).action
        == "SELL"
    )
    assert (
        b3.on_bar(
            _bar(tf="15m"),
            {
                "session_gap_pct": -0.02,
                "vwap_dev": -0.015,
                "session_hhmm": "09:30",
                "position_qty": 0,
            },
        ).action
        == "BUY"
    )
    # 1% gap no longer enough
    assert (
        b3.on_bar(
            _bar(tf="15m"),
            {
                "session_gap_pct": 0.01,
                "vwap_dev": 0.01,
                "session_hhmm": "09:30",
                "position_qty": 0,
            },
        ).action
        == "HOLD"
    )
    # No new entries after 09:45
    assert (
        b3.on_bar(
            _bar(tf="15m"),
            {
                "session_gap_pct": 0.02,
                "vwap_dev": 0.015,
                "session_hhmm": "09:45",
                "position_qty": 0,
            },
        ).action
        == "HOLD"
    )
    # Cover at VWAP
    assert (
        b3.on_bar(
            _bar(tf="15m"),
            {
                "session_gap_pct": 0.02,
                "vwap_dev": 0.0,
                "session_hhmm": "10:00",
                "position_qty": -1,
                "in_position": True,
            },
        ).action
        == "FLAT"
    )
    # Time stop 10:30
    assert (
        b3.on_bar(
            _bar(tf="15m"),
            {
                "session_gap_pct": 0.02,
                "vwap_dev": 0.01,
                "session_hhmm": "10:30",
                "position_qty": -1,
                "in_position": True,
            },
        ).action
        == "FLAT"
    )
    # Hard stop: fade fails (dev stretches past gap+0.5%)
    assert (
        b3.on_bar(
            _bar(tf="15m"),
            {
                "session_gap_pct": 0.02,
                "vwap_dev": 0.026,
                "session_hhmm": "10:00",
                "position_qty": -1,
                "in_position": True,
            },
        ).action
        == "FLAT"
    )


# --- Cluster C ---


def test_c1_rank_buckets():
    c1 = build_strategy("C1")
    assert c1.on_bar(_bar(tf="1W"), {"momentum_rank": 1, "universe_size": 50}).action == "BUY"
    # CNC long-only: bottom ranks flat (no fake short)
    assert c1.on_bar(_bar(tf="1W"), {"momentum_rank": 48, "universe_size": 50}).action == "FLAT"
    assert c1.on_bar(_bar(tf="1W"), {"momentum_rank": 20, "universe_size": 50}).action == "FLAT"


def test_c2_vol_quintile():
    c2 = build_strategy("C2")
    assert c2.on_bar(_bar(tf="1W"), {"vol_quintile": 1}).action == "BUY"
    assert c2.on_bar(_bar(tf="1W"), {"vol_quintile": 3}).action == "HOLD"


def test_c3_reversion_rank():
    c3 = build_strategy("C3")
    assert c3.on_bar(_bar(tf="1W"), {"reversion_rank": 2}).action == "BUY"
    assert c3.on_bar(_bar(tf="1W"), {"reversion_rank": 20}).action == "HOLD"


# --- Cluster D ---


def test_d1_vol_target_scale():
    d1 = build_strategy("D1")
    flat = d1.on_bar(_bar(), {"realized_vol": 0.40})
    hold = d1.on_bar(_bar(), {"realized_vol": 0.10})
    assert flat.action == "FLAT"
    assert hold.action == "HOLD"
    assert hold.intended_exposure > 0


def test_d2_vix_filter():
    d2 = build_strategy("D2")
    assert d2.on_bar(_bar(), {"vix_above_median": True}).action == "FLAT"
    assert d2.on_bar(_bar(), {"vix_above_median": False}).action == "HOLD"


def test_d3_atr_breakout():
    d3 = build_strategy("D3")
    base = {
        "atr_ratio": 2.0,
        "breakout_dir": "up",
        "donchian_upper": 105.0,
        "donchian_lower": 95.0,
        "position_qty": 0,
    }
    assert d3.on_bar(_bar(tf="1H", close=110), base).action == "BUY"
    assert d3.on_bar(_bar(tf="1H", close=90), {**base, "breakout_dir": "down"}).action == "SELL"
    assert d3.on_bar(_bar(tf="1H", close=110), {**base, "atr_ratio": 1.0}).action == "HOLD"
    # Long: opposite dir → flat not reverse
    sig = d3.on_bar(
        _bar(tf="1H", close=100),
        {
            **base,
            "breakout_dir": "down",
            "position_qty": 1,
            "in_position": True,
        },
    )
    assert sig.action == "FLAT"
    # Long: stop at opposite channel
    assert (
        d3.on_bar(
            _bar(tf="1H", close=94),
            {**base, "breakout_dir": "up", "position_qty": 1, "in_position": True},
        ).action
        == "FLAT"
    )
    # Long: 1.5R target from upper (105 + 15 = 120)
    assert (
        d3.on_bar(
            _bar(tf="1H", close=120),
            {**base, "breakout_dir": "up", "position_qty": 1, "in_position": True},
        ).action
        == "FLAT"
    )


# --- Cluster E ---


def test_e1_orb():
    e1 = build_strategy("E1")
    base = {"orb_complete": True, "orb_high": 105.0, "orb_low": 95.0, "e1_traded_today": False}
    assert e1.on_bar(_bar(tf="5m", close=100), {"orb_complete": False}).action == "HOLD"
    assert e1.on_bar(_bar(tf="5m", close=110), {**base, "position_qty": 0}).action == "BUY"
    assert e1.on_bar(_bar(tf="5m", close=90), {**base, "position_qty": 0}).action == "SELL"
    # Long: stop at opposite side
    assert e1.on_bar(_bar(tf="5m", close=94), {**base, "position_qty": 1, "in_position": True}).action == "FLAT"
    # Long: 1.5R target from high (105 + 1.5*10 = 120)
    assert e1.on_bar(_bar(tf="5m", close=120), {**base, "position_qty": 1, "in_position": True}).action == "FLAT"
    # Long: fail back inside
    assert e1.on_bar(_bar(tf="5m", close=100), {**base, "position_qty": 1, "in_position": True}).action == "FLAT"
    # One-shot: no re-entry after traded
    assert (
        e1.on_bar(
            _bar(tf="5m", close=110),
            {**base, "position_qty": 0, "e1_traded_today": True},
        ).action
        == "HOLD"
    )


def test_e2_vwap():
    e2 = build_strategy("E2")
    assert e2.on_bar(_bar(tf="15m"), {"vwap_dev": -0.01, "position_qty": 0}).action == "BUY"
    assert e2.on_bar(_bar(tf="15m"), {"vwap_dev": 0.01, "position_qty": 0}).action == "SELL"
    # Long: target near VWAP
    assert e2.on_bar(_bar(tf="15m"), {"vwap_dev": -0.0005, "position_qty": 1, "in_position": True}).action == "FLAT"
    # Long: stop at -1%
    assert e2.on_bar(_bar(tf="15m"), {"vwap_dev": -0.011, "position_qty": 1, "in_position": True}).action == "FLAT"
    # Long: opposite stretch → flat, not reverse
    sig = e2.on_bar(_bar(tf="15m"), {"vwap_dev": 0.01, "position_qty": 1, "in_position": True})
    assert sig.action == "FLAT"
    assert abs(sig.intended_exposure) == 0.0


def test_e3_power_hour():
    e3 = build_strategy("E3")
    assert e3.on_bar(_bar(tf="5m"), {"in_power_hour": False, "intraday_mom": 0.1, "position_qty": 0}).action == "HOLD"
    assert e3.on_bar(
        _bar(tf="5m"),
        {"in_power_hour": True, "intraday_mom": 0.1, "position_qty": 0, "session_hhmm": "14:30"},
    ).action == "BUY"
    assert e3.on_bar(
        _bar(tf="5m"),
        {"in_power_hour": True, "intraday_mom": -0.1, "position_qty": 0, "session_hhmm": "14:30"},
    ).action == "SELL"
    # Mom flip → flat
    assert e3.on_bar(
        _bar(tf="5m"),
        {"in_power_hour": True, "intraday_mom": -0.01, "position_qty": 1, "in_position": True},
    ).action == "FLAT"
    # No new entries after 15:05
    assert e3.on_bar(
        _bar(tf="5m"),
        {"in_power_hour": True, "intraday_mom": 0.1, "position_qty": 0, "session_hhmm": "15:05"},
    ).action == "HOLD"


# --- Cluster F ---


def test_f1_turn_of_month_params():
    f1 = build_strategy("F1")
    assert f1.on_bar(_bar(), {"day_of_month": 2}).action == "BUY"
    assert f1.on_bar(_bar(), {"day_of_month": 31}).action == "BUY"
    assert f1.on_bar(_bar(), {"day_of_month": 15}).action == "HOLD"


def test_f2_day_of_week():
    f2 = build_strategy("F2")
    assert f2.on_bar(_bar(), {"day_of_week": 1}).action == "BUY"  # Tue
    tue = f2.on_bar(_bar(), {"day_of_week": 1})
    assert tue.intended_exposure == 1.0
    mon = f2.on_bar(_bar(), {"day_of_week": 0})  # Mon reduce → half long
    assert mon.action == "BUY" and mon.intended_exposure == 0.5
    assert f2.on_bar(_bar(), {"day_of_week": 3}).action == "HOLD"  # Thu


def test_f3_expiry():
    f3 = build_strategy("F3")
    hi = f3.on_bar(_bar(), {"expiry_week": False})
    lo = f3.on_bar(_bar(), {"expiry_week": True})
    assert hi.action == "HOLD" and hi.intended_exposure == 1.0
    assert lo.action == "HOLD" and lo.intended_exposure == 0.5


# --- Cluster G ---


def test_g1_trend_absent():
    g1 = build_strategy("G1")
    assert g1.on_bar(_bar(), {"close_vs_ma200": 0.95, "adx": 25}).action == "FLAT"
    assert g1.on_bar(_bar(), {"close_vs_ma200": 1.05, "adx": 15}).action == "FLAT"
    assert g1.on_bar(_bar(), {"close_vs_ma200": 1.05, "adx": 25}).action == "HOLD"


def test_g2_drawdown():
    g2 = build_strategy("G2")
    assert g2.on_bar(_bar(), {"portfolio_drawdown": 0.12}).action == "FLAT"
    assert g2.on_bar(_bar(), {"portfolio_drawdown": 0.01}).action == "HOLD"


def test_g3_low_beta():
    g3 = build_strategy("G3")
    assert g3.on_bar(_bar(tf="1W"), {"beta_rank": 3}).action == "BUY"
    assert g3.on_bar(_bar(tf="1W"), {"beta_rank": 40}).action == "HOLD"
