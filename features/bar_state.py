"""Bar-state features for strategy.on_bar — shared live/backtest path."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data.resample.bars import resample_bars

IST = ZoneInfo("Asia/Kolkata")

TF_TO_RULE: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1H": "1h",
    "1D": "1D",
    "1W": "W-FRI",
}


def bar_just_closed(ts: datetime | pd.Timestamp, timeframe: str) -> bool:
    """True if `ts` is the last 1m stamp of a closed TF bucket (clock-aligned)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(IST)
    else:
        t = t.tz_convert(IST)

    if timeframe == "1m":
        return True
    # Fire once at the closing minute — not every minute after 15:29.
    if timeframe == "1D":
        return t.hour == 15 and t.minute == 29
    if timeframe == "1W":
        return t.dayofweek == 4 and t.hour == 15 and t.minute == 29

    rule = TF_TO_RULE.get(timeframe)
    if not rule:
        return False
    if rule.endswith("min"):
        minutes = int(rule.replace("min", ""))
        start = t.floor(f"{minutes}min")
        return t == start + pd.Timedelta(minutes=minutes - 1)
    if rule == "1h":
        return t.minute == 59
    return False


def closed_timeframes(ts: datetime | pd.Timestamp) -> list[str]:
    return [tf for tf in TF_TO_RULE if bar_just_closed(ts, tf)]


def _ist_minute(ts: datetime | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(IST)
    else:
        t = t.tz_convert(IST)
    return t.floor("min")


def last_bar_minute(bars_1m, fallback: datetime | pd.Timestamp) -> pd.Timestamp:
    """Latest 1m bar minute (IST), else wall-clock minute."""
    fb = _ist_minute(fallback)
    if bars_1m is None or getattr(bars_1m, "empty", True) or "ts" not in getattr(bars_1m, "columns", []):
        return fb
    last = pd.to_datetime(bars_1m["ts"]).max()
    if pd.isna(last):
        return fb
    return _ist_minute(last)


def closed_timeframes_for_tick(now: datetime | pd.Timestamp, bars_1m=None) -> list[str]:
    """
    TF closes from the last 1m bar, not wall clock after a ~45s ingest.

    1D/1W also catch up in 15:29–15:35 IST if the exact 15:29 tick was skipped
    but a bar at/after 15:29 exists that day.
    """
    wall = _ist_minute(now)
    eval_ts = last_bar_minute(bars_1m, wall)
    tfs: list[str] = []
    # Union wall + last bar: ingest wait can push wall past the close minute
    # while the 1m bar still sits on 15:29 (or tests pass `now=` without full-day bars).
    for tf in closed_timeframes(wall) + closed_timeframes(eval_ts):
        if tf not in tfs:
            tfs.append(tf)
    extra: list[str] = []
    in_close_window = bool(wall.hour == 15 and 29 <= int(wall.minute) <= 35)
    last_ready = bool(
        eval_ts.date() == wall.date()
        and ((eval_ts.hour == 15 and int(eval_ts.minute) >= 29) or eval_ts.hour > 15)
    )
    if in_close_window and last_ready:
        if "1D" not in tfs:
            extra.append("1D")
        if int(wall.dayofweek) == 4 and "1W" not in tfs:
            extra.append("1W")
    return tfs + extra


def _rsi(closes: pd.Series, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    delta = closes.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    return 50.0 if np.isnan(val) else val


def _atr_ratio(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return 1.0
    prev_c = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_c).abs(), (low - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    atr_now = float(atr.iloc[-1])
    atr_ref = float(atr.iloc[-period - 1 : -1].mean()) if len(atr.dropna()) > period else atr_now
    if atr_ref <= 1e-12 or np.isnan(atr_now) or np.isnan(atr_ref):
        return 1.0
    return float(atr_now / atr_ref)


def _donchian(high: pd.Series, low: pd.Series, channel: int = 20) -> tuple[float, float]:
    """Prior-bar Donchian (exclude current bar) so close==channel is not automatic."""
    if len(high) < channel + 1:
        return float("nan"), float("nan")
    window = slice(-(channel + 1), -1)
    return float(high.iloc[window].max()), float(low.iloc[window].min())


def _ma_cross(closes: pd.Series, fast: int = 12, slow: int = 26) -> str:
    if len(closes) < slow + 1:
        return "none"
    f = closes.rolling(fast).mean()
    s = closes.rolling(slow).mean()
    if any(np.isnan(x) for x in (f.iloc[-1], f.iloc[-2], s.iloc[-1], s.iloc[-2])):
        return "none"
    prev_diff = float(f.iloc[-2] - s.iloc[-2])
    cur_diff = float(f.iloc[-1] - s.iloc[-1])
    if prev_diff <= 0 < cur_diff:
        return "bullish"
    if prev_diff >= 0 > cur_diff:
        return "bearish"
    return "none"


def _zscore(closes: pd.Series, window: int = 20) -> float:
    if len(closes) < window:
        return 0.0
    windowed = closes.iloc[-window:]
    mu = float(windowed.mean())
    sigma = float(windowed.std(ddof=0))
    if sigma <= 1e-12:
        return 0.0
    return float((windowed.iloc[-1] - mu) / sigma)


def build_state(
    bars_tf: pd.DataFrame,
    symbol: str,
    *,
    timeframe: str,
    universe_bars: pd.DataFrame | None = None,
    now: datetime | pd.Timestamp | None = None,
    portfolio_drawdown: float = 0.0,
) -> dict[str, Any]:
    """
    Build strategy `state` dict from TF OHLCV for one symbol.
    `bars_tf` must contain rows for `symbol` (and optionally others for ranks).
    """
    now_ts = pd.Timestamp(now or datetime.now(tz=IST))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize(IST)

    sym = bars_tf[bars_tf["symbol"] == symbol].sort_values("ts" if "ts" in bars_tf.columns else "date")
    if sym.empty:
        return {"warmup": True}

    close_col = "close"
    closes = sym[close_col].astype(float)
    highs = sym["high"].astype(float) if "high" in sym.columns else closes
    lows = sym["low"].astype(float) if "low" in sym.columns else closes
    last = sym.iloc[-1]
    close = float(last[close_col])
    high = float(last.get("high", close))
    low = float(last.get("low", close))
    open_px = float(last.get("open", close))

    ma20 = float(closes.tail(20).mean()) if len(closes) >= 1 else close
    ma100 = float(closes.tail(100).mean()) if len(closes) >= 1 else close
    ma200 = float(closes.tail(200).mean()) if len(closes) >= 1 else close
    ma12 = float(closes.tail(12).mean()) if len(closes) >= 12 else close
    ma26 = float(closes.tail(26).mean()) if len(closes) >= 26 else close

    don_upper, don_lower = _donchian(highs, lows, 20)
    ret_lookback_ready = len(closes) >= 253
    ret_20 = float(closes.iloc[-1] / closes.iloc[-21] - 1.0) if len(closes) >= 21 else 0.0
    ret_252 = (
        float(closes.iloc[-1] / closes.iloc[-253] - 1.0) if ret_lookback_ready else 0.0
    )
    gap_pct = 0.0
    if len(closes) >= 2:
        prev_close = float(closes.iloc[-2])
        if prev_close > 0:
            gap_pct = (open_px - prev_close) / prev_close

    # Overnight / session gap: first open of today vs prior calendar day's close.
    session_gap_pct = 0.0
    if "ts" in sym.columns and len(sym) >= 2:
        ts_all = pd.to_datetime(sym["ts"])
        today = now_ts.date()
        day_mask = ts_all.dt.date == today
        prior_mask = ts_all.dt.date < today
        if bool(day_mask.any()) and bool(prior_mask.any()):
            day_open = float(sym.loc[day_mask, "open"].iloc[0])
            prev_day_close = float(sym.loc[prior_mask, "close"].iloc[-1])
            if prev_day_close > 0:
                session_gap_pct = (day_open - prev_day_close) / prev_day_close

    # ORB: first ~15 minutes of session — bar count depends on TF
    orb_minutes = 15
    if timeframe in {"5m", "5min"}:
        orb_n = max(1, orb_minutes // 5)
    elif timeframe in {"15m", "15min"}:
        orb_n = max(1, orb_minutes // 15)
    elif timeframe in {"1H", "1h"}:
        orb_n = 1
    else:
        orb_n = orb_minutes  # 1m (and daily fallback)
    orb_high = high
    orb_low = low
    orb_complete = len(sym) >= orb_n
    if "ts" in sym.columns:
        day = sym[pd.to_datetime(sym["ts"]).dt.date == now_ts.date()]
        if len(day) >= 1:
            head = day.head(orb_n)
            orb_high = float(head["high"].max())
            orb_low = float(head["low"].min())
            orb_complete = len(day) >= orb_n

    vwap = close
    if "volume" in sym.columns and float(sym["volume"].sum()) > 0:
        typ = (sym["high"].astype(float) + sym["low"].astype(float) + closes) / 3.0
        vwap = float((typ * sym["volume"].astype(float)).sum() / sym["volume"].astype(float).sum())

    from meta.regime import is_expiry_week

    day_of_month = int(now_ts.day)
    # Default TOM window matches F1 params days_before=2, days_after=3
    in_tom = day_of_month <= 3 or day_of_month >= 30

    # Power hour: from 14:15 through 15:19 IST (MIS flat 15:20)
    in_power = (now_ts.hour > 14 or (now_ts.hour == 14 and now_ts.minute >= 15)) and (
        now_ts.hour < 15 or (now_ts.hour == 15 and now_ts.minute < 20)
    )

    state: dict[str, Any] = {
        "warmup": bool(len(closes) < 2 or np.isnan(don_upper)),
        "rsi": _rsi(closes, 2),
        "rsi14": _rsi(closes, 14),
        "zscore": _zscore(closes, 20),
        "close_vs_ma20": close / ma20 if ma20 else 1.0,
        "close_vs_ma100": close / ma100 if ma100 else 1.0,
        "close_vs_ma200": close / ma200 if ma200 else 1.0,
        "ma_cross": _ma_cross(closes, 12, 26),
        "ma_fast": ma12,
        "ma_slow": ma26,
        "donchian_upper": don_upper,
        "donchian_lower": don_lower,
        "returns_20d": ret_20,
        "returns_252d": ret_252,
        "returns_lookback_ready": ret_lookback_ready,
        "gap_pct": gap_pct,
        "session_gap_pct": session_gap_pct,
        "adx": 22.0,  # approx stub until full ADX series
        "realized_vol": float(closes.pct_change().tail(20).std() * np.sqrt(252))
        if len(closes) > 5
        else 0.12,
        "atr_ratio": _atr_ratio(highs, lows, closes, 14),
        "breakout_dir": (
            "up"
            if not np.isnan(don_upper) and close > don_upper
            else ("down" if not np.isnan(don_lower) and close < don_lower else "none")
        ),
        "orb_high": orb_high,
        "orb_low": orb_low,
        "orb_complete": orb_complete,
        "vwap_dev": (close - vwap) / vwap if vwap else 0.0,
        "in_power_hour": in_power,
        "intraday_mom": float(closes.pct_change().tail(5).sum()) if len(closes) > 5 else 0.0,
        "day_of_month": day_of_month,
        "in_turn_of_month": in_tom,
        "day_of_week": int(now_ts.dayofweek),
        "expiry_week": bool(is_expiry_week(now_ts.date())),
        "vix_above_median": False,  # filled by live regime overlay when available
        "portfolio_drawdown": float(portfolio_drawdown),
        "in_position": False,
        "position_qty": 0,
        "e1_traded_today": False,
        "session_hhmm": f"{int(now_ts.hour):02d}:{int(now_ts.minute):02d}",
        "universe_size": 50,
        "momentum_rank": 999,
        "low_vol_rank": 999,
        "reversion_rank": 999,
        "beta_rank": 999,
        "vol_quintile": 3,
        "timeframe": timeframe,
    }

    # Cross-sectional ranks if universe provided (lookback return / beta — not last close)
    univ = universe_bars if universe_bars is not None else bars_tf
    if univ is not None and not univ.empty and "symbol" in univ.columns:
        from features.cs_ranks import cross_sectional_ranks

        state.update(cross_sectional_ranks(univ, symbol))

    return state


def resample_symbol_tf(bars_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = TF_TO_RULE.get(timeframe, "15min")
    if timeframe == "1m":
        out = bars_1m.copy()
        out["ts"] = pd.to_datetime(out["ts"])
        return out
    return resample_bars(bars_1m, rule)
