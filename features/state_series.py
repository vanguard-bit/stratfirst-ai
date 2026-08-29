"""Vectorized per-bar state frame for fast replay (O(n) vs build_state O(n²))."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from meta.regime import is_expiry_week
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def _orb_n(timeframe: str) -> int:
    orb_minutes = 15
    if timeframe in {"5m", "5min"}:
        return max(1, orb_minutes // 5)
    if timeframe in {"15m", "15min"}:
        return max(1, orb_minutes // 15)
    if timeframe in {"1H", "1h"}:
        return 1
    return orb_minutes


def _rsi_series(closes: pd.Series, period: int) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0.0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _zscore_series(closes: pd.Series, window: int = 20) -> pd.Series:
    mu = closes.rolling(window, min_periods=window).mean()
    sigma = closes.rolling(window, min_periods=window).std(ddof=0)
    z = (closes - mu) / sigma.replace(0.0, np.nan)
    out = z.fillna(0.0)
    out = out.where(sigma.notna() & (sigma > 1e-12), 0.0)
    # match build_state: insufficient window → 0
    out = out.where(closes.expanding().count() >= window, 0.0)
    return out


def _atr_ratio_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_c).abs(), (low - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    # build_state: if len(atr.dropna()) <= period → use atr_now as ref → ratio 1.0
    valid_count = atr.notna().cumsum()
    atr_ref = atr.shift(1).rolling(period, min_periods=1).mean()
    ratio = atr / atr_ref.replace(0.0, np.nan)
    out = ratio.fillna(1.0)
    n = close.expanding().count()
    out = out.where(n >= period + 1, 1.0)
    out = out.where(valid_count > period, 1.0)
    out = out.where(atr_ref.notna() & (atr_ref > 1e-12) | (valid_count <= period), 1.0)
    # when valid_count <= period, force 1.0
    out = out.mask(valid_count <= period, 1.0)
    return out


def _ma_cross_series(closes: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    f = closes.rolling(fast, min_periods=fast).mean()
    s = closes.rolling(slow, min_periods=slow).mean()
    prev_diff = f.shift(1) - s.shift(1)
    cur_diff = f - s
    out = pd.Series("none", index=closes.index, dtype=object)
    bull = (prev_diff <= 0) & (cur_diff > 0) & f.notna() & s.notna() & f.shift(1).notna() & s.shift(1).notna()
    bear = (prev_diff >= 0) & (cur_diff < 0) & f.notna() & s.notna() & f.shift(1).notna() & s.shift(1).notna()
    out = out.mask(bull, "bullish")
    out = out.mask(bear, "bearish")
    # need slow+1 bars for cross detection (same as build_state)
    out = out.where(closes.expanding().count() >= slow + 1, "none")
    return out


def _state_frame_one_symbol(
    sym: pd.DataFrame,
    *,
    timeframe: str,
    portfolio_drawdown: float,
) -> pd.DataFrame:
    sym = sym.sort_values("ts").reset_index(drop=True)
    n = len(sym)
    if n == 0:
        return pd.DataFrame()

    closes = sym["close"].astype(float)
    highs = sym["high"].astype(float) if "high" in sym.columns else closes
    lows = sym["low"].astype(float) if "low" in sym.columns else closes
    opens = sym["open"].astype(float) if "open" in sym.columns else closes
    vols = sym["volume"].astype(float) if "volume" in sym.columns else pd.Series(0.0, index=sym.index)

    ts = pd.to_datetime(sym["ts"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST)
    else:
        ts = ts.dt.tz_convert(IST)
    dates = ts.dt.date

    count = np.arange(1, n + 1)

    ma20 = closes.rolling(20, min_periods=1).mean()
    ma100 = closes.rolling(100, min_periods=1).mean()
    ma200 = closes.rolling(200, min_periods=1).mean()
    ma12 = closes.rolling(12, min_periods=1).mean()
    ma26 = closes.rolling(26, min_periods=1).mean()
    # build_state uses tail(12) only if len>=12 else close — rolling min_periods=1 differs for len<12
    ma12 = ma12.where(count >= 12, closes)
    ma26 = ma26.where(count >= 26, closes)

    don_upper = highs.shift(1).rolling(20, min_periods=20).max()
    don_lower = lows.shift(1).rolling(20, min_periods=20).min()

    ret_lookback_ready = count >= 253
    ret_20 = (closes / closes.shift(20) - 1.0).where(count >= 21, 0.0)
    ret_252 = (closes / closes.shift(252) - 1.0).where(ret_lookback_ready, 0.0)

    prev_close = closes.shift(1)
    gap_pct = ((opens - prev_close) / prev_close.replace(0.0, np.nan)).fillna(0.0)
    gap_pct = gap_pct.where(count >= 2, 0.0)

    # Overnight session gap (constant within each calendar day)
    session_gap_pct = pd.Series(0.0, index=sym.index)
    for _, idx in sym.groupby(dates).groups.items():
        idx = list(idx)
        if not idx:
            continue
        day_open = float(opens.iloc[idx[0]])
        # prior bar before this day's first index
        first_pos = idx[0]
        if first_pos > 0:
            prev_day_close = float(closes.iloc[first_pos - 1])
            # ensure prior bar is previous calendar day
            if dates.iloc[first_pos - 1] < dates.iloc[first_pos] and prev_day_close > 0:
                sg = (day_open - prev_day_close) / prev_day_close
                for pos in idx:
                    session_gap_pct.iloc[pos] = sg

    rsi2 = _rsi_series(closes, 2)
    rsi14 = _rsi_series(closes, 14)
    zscore = _zscore_series(closes, 20)
    atr_ratio = _atr_ratio_series(highs, lows, closes, 14)
    ma_cross = _ma_cross_series(closes, 12, 26)

    # Expanding VWAP over full history (matches build_state sum)
    typ = (highs + lows + closes) / 3.0
    cum_pv = (typ * vols).cumsum()
    cum_v = vols.cumsum()
    vwap = (cum_pv / cum_v.replace(0.0, np.nan)).fillna(closes)
    vwap = vwap.where(cum_v > 0, closes)
    vwap_dev = ((closes - vwap) / vwap.replace(0.0, np.nan)).fillna(0.0)

    pct = closes.pct_change()
    realized = pct.rolling(20, min_periods=2).std() * np.sqrt(252)
    realized_vol = realized.where(count > 5, 0.12).fillna(0.12)

    intraday_mom = pct.rolling(5, min_periods=1).sum().where(count > 5, 0.0).fillna(0.0)

    breakout_dir = np.where(
        don_upper.notna() & (closes > don_upper),
        "up",
        np.where(don_lower.notna() & (closes < don_lower), "down", "none"),
    )

    warmup = (count < 2) | don_upper.isna()

    # ORB per calendar day
    orb_n = _orb_n(timeframe)
    orb_high = highs.copy()
    orb_low = lows.copy()
    orb_complete = pd.Series(False, index=sym.index)
    for _, idx in sym.groupby(dates).groups.items():
        idx = list(idx)
        head = idx[:orb_n]
        oh = float(highs.iloc[head].max()) if head else float("nan")
        ol = float(lows.iloc[head].min()) if head else float("nan")
        for j, pos in enumerate(idx):
            orb_high.iloc[pos] = oh
            orb_low.iloc[pos] = ol
            orb_complete.iloc[pos] = (j + 1) >= orb_n

    hour = ts.dt.hour
    minute = ts.dt.minute
    in_power = ((hour > 14) | ((hour == 14) & (minute >= 15))) & (
        (hour < 15) | ((hour == 15) & (minute < 20))
    )

    day_of_month = ts.dt.day.astype(int)
    in_tom = (day_of_month <= 3) | (day_of_month >= 30)
    day_of_week = ts.dt.dayofweek.astype(int)
    expiry = [bool(is_expiry_week(d)) for d in dates]

    out = pd.DataFrame(
        {
            "ts": ts,
            "symbol": sym["symbol"].astype(str).values,
            "warmup": warmup.astype(bool).values,
            "rsi": rsi2.values,
            "rsi14": rsi14.values,
            "zscore": zscore.values,
            "close_vs_ma20": (closes / ma20.replace(0.0, np.nan)).fillna(1.0).values,
            "close_vs_ma100": (closes / ma100.replace(0.0, np.nan)).fillna(1.0).values,
            "close_vs_ma200": (closes / ma200.replace(0.0, np.nan)).fillna(1.0).values,
            "ma_cross": ma_cross.values,
            "ma_fast": ma12.values,
            "ma_slow": ma26.values,
            "donchian_upper": don_upper.values,
            "donchian_lower": don_lower.values,
            "returns_20d": ret_20.fillna(0.0).values,
            "returns_252d": ret_252.fillna(0.0).values,
            "returns_lookback_ready": ret_lookback_ready,
            "gap_pct": gap_pct.values,
            "session_gap_pct": session_gap_pct.values,
            "adx": 22.0,
            "realized_vol": realized_vol.values,
            "atr_ratio": atr_ratio.values,
            "breakout_dir": breakout_dir,
            "orb_high": orb_high.values,
            "orb_low": orb_low.values,
            "orb_complete": orb_complete.astype(bool).values,
            "vwap_dev": vwap_dev.values,
            "in_power_hour": in_power.astype(bool).values,
            "intraday_mom": intraday_mom.values,
            "day_of_month": day_of_month.values,
            "in_turn_of_month": in_tom.astype(bool).values,
            "day_of_week": day_of_week.values,
            "expiry_week": expiry,
            "vix_above_median": False,
            "portfolio_drawdown": float(portfolio_drawdown),
            "in_position": False,
            "universe_size": 50,
            "momentum_rank": 999,
            "low_vol_rank": 999,
            "reversion_rank": 999,
            "beta_rank": 999,
            "vol_quintile": 3,
            "timeframe": timeframe,
        }
    )
    return out


def build_state_frame(
    bars: pd.DataFrame,
    *,
    timeframe: str,
    portfolio_drawdown: float = 0.0,
) -> pd.DataFrame:
    """
    One state row per bar (sorted by symbol, ts). O(n) vectorized features for replay.
    Single-symbol path matches replay (no universe_bars). Multi-symbol: per-symbol
    features only; CS ranks stay at defaults (999) — live paper uses build_state +
    features.cs_ranks with universe_bars instead.
    """
    if bars is None or bars.empty:
        return pd.DataFrame()

    df = bars.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    if "symbol" not in df.columns:
        df["symbol"] = "UNKNOWN"

    parts = [
        _state_frame_one_symbol(g, timeframe=timeframe, portfolio_drawdown=portfolio_drawdown)
        for _, g in df.groupby("symbol", sort=False)
    ]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def state_row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a state-frame row to a strategy state dict."""
    if hasattr(row, "to_dict"):
        d = row.to_dict()
    else:
        d = dict(row)
    d.pop("ts", None)
    d.pop("symbol", None)
    d.pop("close", None)
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (np.floating,)):
            fv = float(v)
            out[k] = fv
        elif isinstance(v, float):
            out[k] = v
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        elif pd.isna(v) if not isinstance(v, (str, bytes, list, dict)) else False:
            out[k] = float("nan")
        else:
            out[k] = v
    return out
