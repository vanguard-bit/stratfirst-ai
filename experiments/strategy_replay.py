"""Replay strategies on historical bars → daily returns panel for meta labels."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from data.ingest.store import DataStore
from data.ingest.symbols import load_nifty50_symbols
from experiments.replay_cache import DEFAULT_CACHE, cache_symbol_dir, export_replay_bars
from features.bar_state import build_state
from features.state_series import build_state_frame, state_row_to_dict
from nse_trader.config import load_yaml
from sim.fees.calculator import FeeCalculator
from strategies.base import Bar
from strategies.registry import build_strategy, load_enabled_strategies

ProgressFn = Callable[[str], None]


def _resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df_1m is None or df_1m.empty:
        return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume"])
    df = df_1m.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    parts: list[pd.DataFrame] = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.set_index("ts").sort_index()
        ohlc = (
            g.resample(rule, label="left", closed="left")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["close"])
            .reset_index()
        )
        ohlc["symbol"] = sym
        parts.append(ohlc)
    if not parts:
        return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume"])
    return pd.concat(parts, ignore_index=True)[
        ["ts", "symbol", "open", "high", "low", "close", "volume"]
    ]


def _bars_for_tf(store: DataStore, timeframe: str, symbol: str) -> pd.DataFrame:
    tf = str(timeframe)
    if tf in {"1D", "1d"}:
        df = store.read_bars_1d(symbol)
        if not df.empty:
            df = df.rename(columns={"date": "ts"})
            df["ts"] = pd.to_datetime(df["ts"])
        return df
    if tf in {"1W", "1w"}:
        df = store.read_bars_1w(symbol)
        if not df.empty:
            df = df.rename(columns={"date": "ts"})
            df["ts"] = pd.to_datetime(df["ts"])
        return df
    m1 = store.read_bars_1m(symbol)
    if m1.empty:
        return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume"])
    if tf in {"1m", "1M"}:
        m1["ts"] = pd.to_datetime(m1["ts"])
        return m1.sort_values("ts").reset_index(drop=True)
    rule = {"5m": "5min", "15m": "15min", "1H": "1h", "1h": "1h"}.get(tf)
    if rule is None:
        return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume"])
    return _resample_ohlcv(m1, rule)


def _bars_from_cache(cache_dir: Path, timeframe: str, symbol: str) -> pd.DataFrame:
    """Load TF bars from parquet cache (prefer pre-resampled; else resample 1m)."""
    tf = str(timeframe)
    root = cache_symbol_dir(cache_dir, symbol)
    empty = pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume"])
    if tf in {"1D", "1d"}:
        path = root / "1d.parquet"
        if not path.exists():
            return empty
        df = pd.read_parquet(path)
        if df.empty:
            return empty
        if "ts" not in df.columns and "date" in df.columns:
            df["ts"] = pd.to_datetime(df["date"])
        else:
            df["ts"] = pd.to_datetime(df["ts"])
        return df
    if tf in {"1W", "1w"}:
        path = root / "1w.parquet"
        if not path.exists():
            return empty
        df = pd.read_parquet(path)
        if df.empty:
            return empty
        if "ts" not in df.columns and "date" in df.columns:
            df["ts"] = pd.to_datetime(df["date"])
        else:
            df["ts"] = pd.to_datetime(df["ts"])
        return df

    # Prefer cached resampled TF parquet (5m / 15m / 1H)
    tf_file = {"5m": "5m.parquet", "15m": "15m.parquet", "1H": "1H.parquet", "1h": "1H.parquet"}.get(tf)
    if tf_file:
        cached = root / tf_file
        if cached.exists() and cached.stat().st_size > 0:
            df = pd.read_parquet(cached)
            if not df.empty:
                df = df.copy()
                df["ts"] = pd.to_datetime(df["ts"])
                return df.sort_values("ts").reset_index(drop=True)

    path = root / "1m.parquet"
    if not path.exists():
        return empty
    m1 = pd.read_parquet(path)
    if m1.empty:
        return empty
    m1["ts"] = pd.to_datetime(m1["ts"])
    if tf in {"1m", "1M"}:
        return m1.sort_values("ts").reset_index(drop=True)
    rule = {"5m": "5min", "15m": "15min", "1H": "1h", "1h": "1h"}.get(tf)
    if rule is None:
        return empty
    return _resample_ohlcv(m1, rule)


def load_replay_fee_calculator(registry_path: Path | None = None) -> FeeCalculator | None:
    """Fee table only (no bid/ask). None if registry missing."""
    path = registry_path
    if path is None:
        try:
            from nse_trader.config import PortfolioConfig

            path = PortfolioConfig.load().fees_registry_path
        except Exception:  # noqa: BLE001
            return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        return FeeCalculator.from_file(path)
    except Exception:  # noqa: BLE001
        return None


def _fee_drag_frac(
    calc: FeeCalculator,
    *,
    segment: str,
    side: str,
    turnover: float,
    notional: float,
) -> float:
    """Charges as a fraction of strategy book notional (never guessed spread)."""
    if notional <= 0 or turnover <= 0:
        return 0.0
    charges = calc.compute(segment, side.lower(), float(turnover))
    return float(charges.total) / float(notional)


def replay_strategy_daily_returns(
    strategy_id: str,
    bars: pd.DataFrame,
    *,
    symbol: str = "RELIANCE",
    strategy: Any | None = None,
    notional: float | None = None,
    fee_calculator: FeeCalculator | None = None,
    apply_fees: bool = True,
    registry_path: Path | None = None,
) -> pd.DataFrame:
    """
    Walk TF bars; position exposure from on_bar intended_exposure;
    daily ret = prev_exposure * close-to-close return, minus **fee-table**
    drag on exposure changes (STT/brokerage/etc). No bid/ask spread.

    MIS is flattened at each date change (no overnight gap, flatten pays fees).

    Uses vectorized `build_state_frame` (O(n)) instead of per-bar `build_state(hist)`.
    """
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["date", "strategy_id", "ret"])

    strat = strategy or build_strategy(strategy_id)
    df = bars.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    df = df[df["symbol"].astype(str) == str(symbol)].reset_index(drop=True)
    if len(df) < 2:
        return pd.DataFrame(columns=["date", "strategy_id", "ret"])

    n0 = float(notional if notional is not None else default_notional_per_symbol())
    fees = fee_calculator
    if apply_fees and fees is None:
        fees = load_replay_fee_calculator(registry_path)
    product = str(getattr(strat, "product", "CNC") or "CNC").upper()
    is_mis = product == "MIS"
    segment = "equity_intraday" if is_mis else "equity_delivery"

    tf = getattr(strat, "timeframe", "1D")
    state_frame = build_state_frame(df, timeframe=str(tf))

    exposure = 0.0
    e1_traded_today = False
    rows: list[dict] = []
    closes = df["close"].astype(float).to_numpy()
    opens = df["open"].astype(float).to_numpy() if "open" in df.columns else closes
    highs = df["high"].astype(float).to_numpy() if "high" in df.columns else closes
    lows = df["low"].astype(float).to_numpy() if "low" in df.columns else closes
    vols = df["volume"].astype(float).to_numpy() if "volume" in df.columns else np.zeros(len(df))
    tss = pd.to_datetime(df["ts"])
    dates = [pd.Timestamp(t).date().isoformat() for t in tss]
    prev_date = dates[0]
    sid = str(getattr(strat, "id", strategy_id) or strategy_id)

    for i in range(1, len(df)):
        prev_c = float(closes[i - 1])
        cur_c = float(closes[i])
        if prev_c <= 0:
            continue
        cur_date = dates[i]
        day_ret = 0.0

        if is_mis and cur_date != prev_date and abs(exposure) > 1e-12:
            if fees is not None:
                side = "sell" if exposure > 0 else "buy"
                day_ret -= _fee_drag_frac(
                    fees,
                    segment=segment,
                    side=side,
                    turnover=abs(exposure) * n0,
                    notional=n0,
                )
            exposure = 0.0
        if cur_date != prev_date:
            e1_traded_today = False
        prev_date = cur_date

        day_ret += exposure * (cur_c / prev_c - 1.0)

        state = state_row_to_dict(state_frame.iloc[i])
        state["in_position"] = abs(exposure) > 1e-12
        if exposure > 1e-12:
            state["position_qty"] = 1
        elif exposure < -1e-12:
            state["position_qty"] = -1
        else:
            state["position_qty"] = 0
        state["e1_traded_today"] = e1_traded_today
        ts_i = pd.Timestamp(tss.iloc[i])
        state["session_hhmm"] = f"{int(ts_i.hour):02d}:{int(ts_i.minute):02d}"
        bar = Bar(
            ts=str(tss.iloc[i]),
            symbol=str(symbol),
            open=float(opens[i]),
            high=float(highs[i]),
            low=float(lows[i]),
            close=cur_c,
            volume=float(vols[i]),
            timeframe=tf,
        )
        new_exp = exposure
        try:
            sig = strat.on_bar(bar, state)
            action = (sig.action or "HOLD").upper()
            exp_attr = getattr(sig, "intended_exposure", None)
            if action == "HOLD":
                new_exp = exposure
            elif action == "FLAT":
                new_exp = 0.0
            elif action == "BUY":
                if exp_attr is not None and abs(float(exp_attr)) < 1e-12:
                    new_exp = 0.0
                else:
                    mag = abs(
                        float(exp_attr)
                        if exp_attr is not None and abs(float(exp_attr)) > 1e-12
                        else float(getattr(sig, "confidence", 1.0) or 1.0)
                    )
                    new_exp = mag
            elif action == "SELL":
                if exp_attr is not None and abs(float(exp_attr)) < 1e-12:
                    new_exp = 0.0
                else:
                    mag = abs(
                        float(exp_attr)
                        if exp_attr is not None and abs(float(exp_attr)) > 1e-12
                        else float(getattr(sig, "confidence", 1.0) or 1.0)
                    )
                    new_exp = -mag if is_mis else 0.0
            if (
                sid == "E1"
                and abs(exposure) < 1e-12
                and abs(new_exp) > 1e-12
            ):
                e1_traded_today = True
        except Exception:  # noqa: BLE001
            new_exp = exposure

        delta = new_exp - exposure
        if fees is not None and abs(delta) > 1e-12:
            side = "buy" if delta > 0 else "sell"
            day_ret -= _fee_drag_frac(
                fees,
                segment=segment,
                side=side,
                turnover=abs(delta) * n0,
                notional=n0,
            )
        exposure = new_exp

        rows.append(
            {
                "date": cur_date,
                "strategy_id": strategy_id,
                "ret": float(day_ret),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["date", "strategy_id", "ret"])
    out = pd.DataFrame(rows)
    return out.groupby(["date", "strategy_id"], as_index=False)["ret"].sum()


def replay_all_enabled(
    *,
    db_path: Path | None = None,
    symbol: str = "RELIANCE",
    strategy_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Replay all enabled strategies that have bars for their TF (single symbol)."""
    cfg = load_yaml("strategies.yaml")["strategies"]
    strategies = load_enabled_strategies()
    ids = strategy_ids or list(strategies.keys())
    panels: list[pd.DataFrame] = []
    with DataStore(db_path=db_path) as store:
        store.init_schema()
        for sid in ids:
            entry = cfg.get(sid, {})
            tf = entry.get("timeframe", getattr(strategies[sid], "timeframe", "1D"))
            bars = _bars_for_tf(store, str(tf), symbol)
            if bars.empty:
                continue
            panels.append(
                replay_strategy_daily_returns(
                    sid,
                    bars,
                    symbol=symbol,
                    strategy=strategies[sid],
                )
            )
    if not panels:
        return pd.DataFrame(columns=["date", "strategy_id", "ret"])
    return pd.concat(panels, ignore_index=True)


def replay_symbol_all_strategies(
    symbol: str,
    *,
    cache_dir: Path,
    strategy_ids: list[str] | None = None,
    notional: float = 1.0,
) -> pd.DataFrame:
    """
    Replay all enabled strategies on one symbol from parquet cache.
    Returns date, strategy_id, symbol, ret, pnl (= ret * notional).
    """
    cfg = load_yaml("strategies.yaml")["strategies"]
    strategies = load_enabled_strategies()
    ids = strategy_ids or list(strategies.keys())
    panels: list[pd.DataFrame] = []
    fees = load_replay_fee_calculator()
    for sid in ids:
        if sid not in strategies:
            continue
        entry = cfg.get(sid, {})
        tf = entry.get("timeframe", getattr(strategies[sid], "timeframe", "1D"))
        bars = _bars_from_cache(cache_dir, str(tf), symbol)
        if bars.empty:
            continue
        out = replay_strategy_daily_returns(
            sid,
            bars,
            symbol=symbol,
            strategy=strategies[sid],
            notional=float(notional),
            fee_calculator=fees,
        )
        if out.empty:
            continue
        out = out.copy()
        out["symbol"] = symbol
        out["pnl"] = out["ret"].astype(float) * float(notional)
        panels.append(out)
    if not panels:
        return pd.DataFrame(columns=["date", "strategy_id", "symbol", "ret", "pnl"])
    return pd.concat(panels, ignore_index=True)


def _worker_replay_symbol(payload: dict[str, Any]) -> pd.DataFrame:
    """Picklable process-pool worker."""
    return replay_symbol_all_strategies(
        str(payload["symbol"]),
        cache_dir=Path(payload["cache_dir"]),
        strategy_ids=payload.get("strategy_ids"),
        notional=float(payload["notional"]),
    )


def aggregate_fixed_notional_book(
    per_symbol: pd.DataFrame,
    *,
    n_universe: int,
    notional_per_symbol: float,
) -> pd.DataFrame:
    """
    Book return r_{s,t} = sum(pnl) / (N0 * n_universe).
    Missing symbol-days contribute 0 PnL; capital still in denominator.
    """
    if per_symbol is None or per_symbol.empty:
        return pd.DataFrame(columns=["date", "strategy_id", "ret"])
    denom = float(notional_per_symbol) * max(int(n_universe), 1)
    if denom <= 0:
        raise ValueError("notional_per_symbol * n_universe must be > 0")
    g = per_symbol.groupby(["date", "strategy_id"], as_index=False)["pnl"].sum()
    g["ret"] = g["pnl"] / denom
    return g[["date", "strategy_id", "ret"]]


def default_notional_per_symbol(n_symbols: int | None = None) -> float:
    cfg = load_yaml("portfolio.yaml")
    port = cfg.get("portfolio") if isinstance(cfg.get("portfolio"), dict) else cfg
    capital = float(port.get("total_capital", 2_000_000))
    n = int(n_symbols or 50)
    return capital / max(n, 1)


def replay_universe_book_returns(
    symbols: list[str] | None = None,
    *,
    cache_dir: Path | None = None,
    workers: int | None = None,
    notional_per_symbol: float | None = None,
    strategy_ids: list[str] | None = None,
    export: bool = True,
    force_export: bool = False,
    db_path: Path | None = None,
    progress: ProgressFn | None = None,
) -> pd.DataFrame:
    """
    Parallel per-symbol replay → fixed-notional book daily returns for meta labels.
    Output columns: date, strategy_id, ret
    """
    log = progress or (lambda m: print(m, flush=True))
    symbols = list(symbols or load_nifty50_symbols())
    n_universe = len(symbols)
    cache_dir = Path(cache_dir or DEFAULT_CACHE)
    n0 = float(
        notional_per_symbol
        if notional_per_symbol is not None
        else default_notional_per_symbol(n_universe)
    )
    n_workers = workers if workers is not None else min(10, max(1, (os.cpu_count() or 4) - 2))

    if export:
        log(f"exporting replay cache for {n_universe} symbols → {cache_dir}")
        export_replay_bars(
            symbols,
            cache_dir=cache_dir,
            db_path=db_path,
            force=force_export,
            progress=log,
        )

    payloads = [
        {
            "symbol": sym,
            "cache_dir": str(cache_dir),
            "strategy_ids": strategy_ids,
            "notional": n0,
        }
        for sym in symbols
    ]

    parts: list[pd.DataFrame] = []
    log(f"universe replay start symbols={n_universe} workers={n_workers} notional={n0:.2f}")
    if n_workers <= 1:
        for i, payload in enumerate(payloads, start=1):
            log(f"[{i}/{n_universe}] replay {payload['symbol']}")
            parts.append(_worker_replay_symbol(payload))
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_worker_replay_symbol, p): p["symbol"] for p in payloads}
            for fut in as_completed(futs):
                sym = futs[fut]
                done += 1
                try:
                    parts.append(fut.result())
                    log(f"[{done}/{n_universe}] done {sym}")
                except Exception as exc:  # noqa: BLE001
                    log(f"[{done}/{n_universe}] ERROR {sym}: {exc}")

    if not parts:
        return pd.DataFrame(columns=["date", "strategy_id", "ret"])
    per = pd.concat(parts, ignore_index=True)
    book = aggregate_fixed_notional_book(
        per, n_universe=n_universe, notional_per_symbol=n0
    )
    log(
        f"universe replay done rows={len(book)} strategies={book['strategy_id'].nunique()} "
        f"agg=fixed_notional n_universe={n_universe}"
    )
    return book
