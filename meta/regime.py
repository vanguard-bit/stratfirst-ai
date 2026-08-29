"""Regime features for meta allocator — ADX, India VIX, expiry calendar."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from meta.features import RegimeFeatures
from nse_trader.config import ROOT, load_yaml

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger(__name__)

VIX_CACHE = ROOT / "data" / "state" / "india_vix_cache.json"
WEIGHTS_CACHE = ROOT / "data" / "state" / "meta_weights_day.json"
META_LLM_DUAL_LOG = ROOT / "data" / "logs" / "meta_llm_dual.jsonl"


def weight_delta_top(
    weights: dict[str, float],
    weights_no_llm: dict[str, float],
    *,
    k: int = 5,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for sid in sorted(set(weights) | set(weights_no_llm)):
        delta = float(weights.get(sid, 0.0)) - float(weights_no_llm.get(sid, 0.0))
        rows.append({"strategy_id": sid, "delta": delta})
    rows.sort(key=lambda r: abs(float(r["delta"])), reverse=True)
    return rows[:k]


def l1_weight_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return float(sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys))


def _llm_tilt_from_portfolio(llm_tilt: dict | None = None) -> dict | None:
    if llm_tilt is not None:
        return llm_tilt
    try:
        port = load_yaml("portfolio.yaml")
        raw = (port.get("meta_allocator") or {}).get("llm_tilt")
        return dict(raw) if isinstance(raw, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _append_llm_dual_log(payload: dict) -> None:
    META_LLM_DUAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with META_LLM_DUAL_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")



def last_thursday_of_month(year: int, month: int) -> date:
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    d = nxt - timedelta(days=1)
    while d.weekday() != 3:  # Thursday
        d -= timedelta(days=1)
    return d


def is_expiry_week(today: date | None = None) -> bool:
    """
    True near NSE monthly F&O expiry (last Thursday) or on weekly expiry weekday (Tuesday).
    """
    today = today or datetime.now(tz=IST).date()
    # Weekly Nifty/BankNifty expiry is Tuesday
    if today.weekday() == 1:
        return True
    monthly = last_thursday_of_month(today.year, today.month)
    return abs((monthly - today).days) <= 4


def wilder_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Return latest ADX; 0 if insufficient history."""
    if len(close) < period * 2:
        return 0.0
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    val = float(adx.iloc[-1])
    return 0.0 if np.isnan(val) else val


def index_ohlc_from_universe(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight synthetic index OHLC from universe 1m bars."""
    df = bars_1m.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    g = df.groupby("ts", as_index=False).agg(
        open=("open", "mean"),
        high=("high", "mean"),
        low=("low", "mean"),
        close=("close", "mean"),
        volume=("volume", "sum"),
    )
    return g.sort_values("ts")


def nifty_return_20d(daily_closes: pd.Series) -> float:
    if len(daily_closes) < 21:
        return 0.0
    a = float(daily_closes.iloc[-21])
    b = float(daily_closes.iloc[-1])
    if a == 0:
        return 0.0
    return (b - a) / a


def fetch_india_vix(*, timeout: float = 8.0) -> float | None:
    """Best-effort India VIX; None on failure (caller uses cache/fallback)."""
    try:
        import requests

        # NSE all-indices JSON (may require cookies; fail soft)
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }
        r = requests.get(
            "https://www.nseindia.com/api/allIndices",
            headers=headers,
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        for row in data.get("data", []):
            name = str(row.get("index", "") or row.get("indexSymbol", "")).upper()
            if "INDIA VIX" in name or name == "INDIA VIX":
                last = row.get("last") or row.get("lastPrice")
                if last is not None:
                    return float(last)
    except Exception as exc:  # noqa: BLE001
        logger.debug("india vix fetch failed: %s", exc)
    return None


def load_vix_cache() -> tuple[float | None, float | None]:
    if not VIX_CACHE.exists():
        return None, None
    try:
        raw = json.loads(VIX_CACHE.read_text(encoding="utf-8"))
        return (
            float(raw["vix"]) if raw.get("vix") is not None else None,
            float(raw["median"]) if raw.get("median") is not None else None,
        )
    except Exception:  # noqa: BLE001
        return None, None


def save_vix_cache(vix: float, median: float) -> None:
    VIX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    VIX_CACHE.write_text(
        json.dumps(
            {
                "vix": vix,
                "median": median,
                "updated_at": datetime.now(tz=IST).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def resolve_vix() -> tuple[float, float, bool]:
    """
    Returns (vix, median, above_median).
    Falls back to cache, then 16.0 / 16.0.
    """
    cached_vix, cached_med = load_vix_cache()
    live = fetch_india_vix()
    vix = live if live is not None else (cached_vix if cached_vix is not None else 16.0)
    # Rolling median proxy: blend cache median with current
    median = cached_med if cached_med is not None else 16.0
    if live is not None:
        median = 0.9 * median + 0.1 * live if cached_med is not None else live
        save_vix_cache(vix, median)
    return float(vix), float(median), bool(vix > median)


def build_regime(
    bars_1m: pd.DataFrame | None,
    *,
    now: datetime | None = None,
) -> RegimeFeatures:
    """Build RegimeFeatures from live 1m universe bars + VIX + calendar."""
    stamp = now or datetime.now(tz=IST)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=IST)
    today = stamp.date()

    adx = 0.0
    nifty_ret = 0.0
    if bars_1m is not None and not getattr(bars_1m, "empty", True):
        idx = index_ohlc_from_universe(bars_1m)
        if len(idx) >= 30:
            # Resample to ~15m for stabler ADX
            idx = idx.set_index("ts")
            ohlc = idx.resample("15min").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna()
            if len(ohlc) >= 30:
                adx = wilder_adx(ohlc["high"], ohlc["low"], ohlc["close"], 14)
            daily = idx["close"].resample("1D").last().dropna()
            nifty_ret = nifty_return_20d(daily)

    vix, median, above = resolve_vix()
    return RegimeFeatures(
        adx=adx,
        vix=vix,
        vix_above_median=above,
        expiry_week=is_expiry_week(today),
        nifty_return_20d=nifty_ret,
    )


def load_or_compute_daily_weights(
    *,
    strategy_ids: list[str],
    cluster_of: dict[str, str],
    regime: RegimeFeatures,
    mode: str,
    constraints,
    today: date | None = None,
    llm_tilt: dict | None = None,
) -> dict[str, float]:
    """Cache meta weights once per IST day (matches daily rebalance intent).

    On cache miss also stores ``weights_no_llm`` (same rules with LLM fields
    zeroed) for dual-dump / meta-status — live fills use ``weights`` only.
    """
    from dataclasses import replace

    from meta.allocator import MetaAllocator
    from meta.drawdown_kill import apply_strategy_drawdown_zero

    today = today or datetime.now(tz=IST).date()
    if WEIGHTS_CACHE.exists():
        try:
            raw = json.loads(WEIGHTS_CACHE.read_text(encoding="utf-8"))
            if raw.get("date") == today.isoformat() and raw.get("mode") == mode:
                cached = {k: float(v) for k, v in raw.get("weights", {}).items()}
                # Re-apply kill each load so glance updates bite without waiting for day roll.
                return apply_strategy_drawdown_zero(cached)
        except Exception:  # noqa: BLE001
            pass

    tilt = _llm_tilt_from_portfolio(llm_tilt)
    alloc = MetaAllocator(
        strategy_ids=strategy_ids,
        cluster_of=cluster_of,
        constraints=constraints,
        mode=mode,
        llm_tilt=tilt,
    )
    weights = apply_strategy_drawdown_zero(alloc.allocate(regime))
    regime_no_llm = replace(
        regime,
        llm_sentiment_mean=0.0,
        llm_high_materiality=0,
    )
    weights_no_llm = apply_strategy_drawdown_zero(alloc.allocate(regime_no_llm))
    deltas = weight_delta_top(weights, weights_no_llm, k=5)
    llm_path = ""
    try:
        from features.llm_store import latest_llm_parquet

        p = latest_llm_parquet()
        llm_path = str(p) if p is not None else ""
    except Exception:  # noqa: BLE001
        llm_path = ""

    llm_block = {
        "mean": float(regime.llm_sentiment_mean or 0.0),
        "high_n": int(regime.llm_high_materiality or 0),
        "as_of": str(regime.llm_as_of or ""),
        "path": llm_path,
    }
    updated_at = datetime.now(tz=IST).isoformat()
    WEIGHTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_CACHE.write_text(
        json.dumps(
            {
                "date": today.isoformat(),
                "mode": mode,
                "regime": {
                    "adx": regime.adx,
                    "vix": regime.vix,
                    "vix_above_median": regime.vix_above_median,
                    "expiry_week": regime.expiry_week,
                    "nifty_return_20d": regime.nifty_return_20d,
                    "llm_sentiment_mean": regime.llm_sentiment_mean,
                    "llm_high_materiality": regime.llm_high_materiality,
                    "llm_as_of": regime.llm_as_of,
                },
                "llm": llm_block,
                "weights": weights,
                "weights_no_llm": weights_no_llm,
                "weight_delta_top": deltas,
                "updated_at": updated_at,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        _append_llm_dual_log(
            {
                "date": today.isoformat(),
                "mean": llm_block["mean"],
                "high_n": llm_block["high_n"],
                "as_of": llm_block["as_of"],
                "l1_distance": l1_weight_distance(weights, weights_no_llm),
                "n_strategies": len(strategy_ids),
                "updated_at": updated_at,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("meta_llm_dual log failed: %s", exc)
    return weights

