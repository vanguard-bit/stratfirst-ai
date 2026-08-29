"""Shared risk/return metrics for offline + forward bake-off tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nse_trader.config import ROOT

METRICS_SCHEMA = 2
SPARSE_DAYS = 20
TRADING_DAYS_YEAR = 252
HORIZON_5D = 5


def sharpe_nd(r: pd.Series, *, horizon: int = HORIZON_5D) -> float | None:
    """Annualized Sharpe of non-overlapping `horizon`-day compounded returns.

    Evaluation-only: model still predicts t+1. Needs ≥2 complete blocks.
    """
    s = pd.to_numeric(r, errors="coerce").dropna().astype(float)
    if horizon < 2 or len(s) < horizon * 2:
        return None
    k = (len(s) // horizon) * horizon
    blocks = s.iloc[:k].to_numpy().reshape(-1, horizon)
    r_n = np.prod(1.0 + blocks, axis=1) - 1.0
    if len(r_n) < 2:
        return None
    std = float(r_n.std(ddof=0))
    if std <= 1e-15:
        return None
    return float(r_n.mean() / std * np.sqrt(TRADING_DAYS_YEAR / horizon))

BAKEOFF_DIR = ROOT / "data" / "store" / "experiments" / "meta_bakeoff"
OFFLINE_METRICS_JSON = BAKEOFF_DIR / "offline_metrics.json"
OFFLINE_METRICS_CSV = BAKEOFF_DIR / "offline_metrics.csv"
FORWARD_METRICS_JSON = BAKEOFF_DIR / "forward_metrics.json"
FORWARD_METRICS_CSV = BAKEOFF_DIR / "forward_metrics.csv"
MEASURE_FILLS = BAKEOFF_DIR / "measure_fills.parquet"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def equity_from_returns(r: pd.Series) -> pd.Series:
    """Compound equity curve from daily fractional returns (start 1.0)."""
    s = pd.to_numeric(r, errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return (1.0 + s.astype(float)).cumprod()


def equity_from_level_return(level: pd.Series) -> pd.Series:
    """Level return from buy (0 at buy) → equity E = 1 + level."""
    s = pd.to_numeric(level, errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return 1.0 + s.astype(float)


def metrics_from_returns(
    r: pd.Series,
    *,
    sleeve: str,
    block: str = "policy",
    trades: float | None = None,
    turnover: float | None = None,
    turnover_kind: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    s = pd.to_numeric(r, errors="coerce").dropna().astype(float)
    n = int(len(s))
    out: dict[str, Any] = {
        "sleeve": sleeve,
        "block": block,
        "n_days": n,
        "cagr": None,
        "max_dd": None,
        "sharpe": None,
        "sharpe_5d": None,
        "trades": trades,
        "turnover": turnover,
        "turnover_kind": turnover_kind,
        "sparse": n < SPARSE_DAYS,
        "note": note,
    }
    if n == 0:
        return out
    # Guard: absolute PnL or first-day MTM artifacts must not be compounded as returns
    abs_med = float(s.abs().median())
    abs_max = float(s.abs().max())
    if abs_med > 0.15 or abs_max > 0.35:
        out["note"] = (
            (note + "; " if note else "")
            + "large daily |r|; CAGR/Sharpe skipped (additive MaxDD on cumsum only)"
        )
        cum = s.cumsum()
        peak = cum.cummax()
        dd = cum - peak
        out["max_dd"] = float(dd.min()) if len(dd) else None
        out["cagr"] = None
        out["sharpe"] = None
        out["sharpe_5d"] = None
        return out
    eq = equity_from_returns(s)
    end = float(eq.iloc[-1])
    if end > 0 and n > 0 and np.isfinite(end) and end < 1e6:
        try:
            cagr = float(end ** (TRADING_DAYS_YEAR / n) - 1.0)
            out["cagr"] = cagr if np.isfinite(cagr) and abs(cagr) < 1e6 else None
        except (OverflowError, ValueError):
            out["cagr"] = None
    peak = eq.cummax()
    dd = eq / peak - 1.0
    out["max_dd"] = float(dd.min())
    std = float(s.std(ddof=0))
    if std > 1e-15:
        out["sharpe"] = float(s.mean() / std * np.sqrt(TRADING_DAYS_YEAR))
    else:
        out["sharpe"] = None
    out["sharpe_5d"] = sharpe_nd(s, horizon=HORIZON_5D)
    return out


def metrics_from_level(
    level: pd.Series,
    *,
    sleeve: str,
    block: str = "policy",
    trades: float | None = None,
    turnover: float | None = None,
    turnover_kind: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Metrics from buy-hold level return series (0 at buy date)."""
    s = pd.to_numeric(level, errors="coerce").dropna().astype(float)
    n = int(len(s))
    out: dict[str, Any] = {
        "sleeve": sleeve,
        "block": block,
        "n_days": n,
        "cagr": None,
        "max_dd": None,
        "sharpe": None,
        "sharpe_5d": None,
        "trades": trades,
        "turnover": turnover,
        "turnover_kind": turnover_kind,
        "sparse": n < SPARSE_DAYS,
        "note": note or "level return from buy_date",
    }
    if n == 0:
        return out
    eq = equity_from_level_return(s)
    end = float(eq.iloc[-1])
    if end > 0 and n > 1:
        # n points → n-1 return intervals for annualization
        out["cagr"] = float(end ** (TRADING_DAYS_YEAR / (n - 1)) - 1.0)
    elif end > 0 and n == 1:
        out["cagr"] = 0.0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    out["max_dd"] = float(dd.min())
    # Daily r from equity
    if n >= 2:
        daily_r = eq.pct_change().dropna()
        std = float(daily_r.std(ddof=0))
        if std > 1e-15:
            out["sharpe"] = float(daily_r.mean() / std * np.sqrt(TRADING_DAYS_YEAR))
        out["sharpe_5d"] = sharpe_nd(daily_r, horizon=HORIZON_5D)
    return out


def selection_turnover(memberships: list[set[str]]) -> float | None:
    """Mean day-to-day top-k churn: 1 - |A∩B|/max(|A|,|B|,1)."""
    if len(memberships) < 2:
        return None
    changes: list[float] = []
    for a, b in zip(memberships[:-1], memberships[1:]):
        k = max(len(a), len(b), 1)
        changes.append(1.0 - (len(a & b) / k))
    return float(np.mean(changes)) if changes else None


def top5_memberships_from_oof(oof: pd.DataFrame) -> list[set[str]]:
    """Per-date predicted top-5 strategy sets from OOF scores."""
    if oof is None or oof.empty or "p" not in oof.columns:
        return []
    o = oof.copy()
    o["date"] = o["date"].astype(str).str.slice(0, 10)
    o = o.groupby(["date", "strategy_id"], as_index=False)["p"].mean()
    out: list[set[str]] = []
    for _, g in o.groupby("date", sort=True):
        g = g.sort_values("p", ascending=False)
        top = g["strategy_id"].astype(str).head(5).tolist()
        out.append(set(top))
    return out


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    cols = [
        "sleeve",
        "block",
        "n_days",
        "cagr",
        "max_dd",
        "sharpe",
        "sharpe_5d",
        "trades",
        "turnover",
        "turnover_kind",
        "sparse",
        "note",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def write_metrics_bundle(
    rows: list[dict[str, Any]],
    *,
    json_path: Path,
    csv_path: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    df = rows_to_frame(rows)
    # JSON-safe: replace NaN trades/turnover with None
    for col in ("trades", "turnover", "cagr", "max_dd", "sharpe", "sharpe_5d"):
        if col in df.columns:
            df[col] = df[col].where(pd.notna(df[col]), None)
    # Sort: policies first by sharpe desc, then strategies
    def _sort_key(r: pd.Series) -> tuple:
        block_ord = 0 if r["block"] == "policy" else 1
        sh = r["sharpe"]
        sh_v = float(sh) if sh is not None and pd.notna(sh) else -1e18
        return (block_ord, -sh_v, str(r["sleeve"]))

    if len(df):
        df = df.assign(_k=df.apply(_sort_key, axis=1)).sort_values("_k").drop(columns=["_k"])
        df = df.reset_index(drop=True)
    payload: dict[str, Any] = {
        "metrics_schema": METRICS_SCHEMA,
        "rows": df.to_dict(orient="records"),
        "n_rows": int(len(df)),
        "sparse": bool(df["sparse"].any()) if len(df) and "sparse" in df.columns else True,
    }
    if extra:
        payload.update(extra)
    _atomic_write_text(json_path, json.dumps(payload, indent=2, default=str))
    _atomic_write_csv(csv_path, df)
    payload["json_path"] = str(json_path)
    payload["csv_path"] = str(csv_path)
    return payload


def build_offline_metrics(
    daily: pd.DataFrame,
    oof: pd.DataFrame | None = None,
    *,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Policy + strategy metrics from offline daily panel and OOF."""
    out_dir = Path(out_dir or BAKEOFF_DIR)
    rows: list[dict[str, Any]] = []
    policy_cols = [
        "model_top5_eq",
        "model_score_all",
        "eq_all",
        "rand1_E",
        "rand5_eq",
        "rules_proxy",
        "oracle_best1",
    ]
    sel_to = None
    if oof is not None and not oof.empty:
        sel_to = selection_turnover(top5_memberships_from_oof(oof))

    for col in policy_cols:
        if daily is None or col not in daily.columns:
            continue
        note = None
        turnover = None
        tkind = None
        if col == "oracle_best1":
            note = "hindsight upper bound only"
        if col == "model_top5_eq":
            turnover = sel_to
            tkind = "selection"
        elif col in ("eq_all", "rand1_E"):
            turnover = 0.0
            tkind = "selection"
        rows.append(
            metrics_from_returns(
                daily[col],
                sleeve=col,
                block="policy",
                trades=None,
                turnover=turnover,
                turnover_kind=tkind,
                note=note,
            )
        )

    if daily is not None and "bh_nifty50" in daily.columns and daily["bh_nifty50"].notna().any():
        rows.append(
            metrics_from_level(
                daily["bh_nifty50"],
                sleeve="bh_nifty50",
                block="policy",
                trades=None,
                turnover=0.0,
                turnover_kind="selection",
                note="equal-weight buy-hold; no rebalance/fees",
            )
        )

    if oof is not None and not oof.empty and "ret_fwd" in oof.columns:
        o = oof.copy()
        o["date"] = o["date"].astype(str).str.slice(0, 10)
        o = (
            o.groupby(["date", "strategy_id"], as_index=False)["ret_fwd"]
            .mean()
            .sort_values(["strategy_id", "date"])
        )
        for sid, g in o.groupby("strategy_id", sort=True):
            rows.append(
                metrics_from_returns(
                    g["ret_fwd"],
                    sleeve=str(sid),
                    block="strategy",
                    trades=None,
                    turnover=None,
                    turnover_kind=None,
                    note="OOF ret_fwd (fold-deduped); trades n/a",
                )
            )

    return write_metrics_bundle(
        rows,
        json_path=out_dir / "offline_metrics.json",
        csv_path=out_dir / "offline_metrics.csv",
        extra={
            "track": "offline",
            "notes": [
                "Cat1 OOF with fee-table drag (no bid/ask); trades always n/a",
                "model_top5_eq turnover = selection (top5 membership churn)",
                "sharpe_5d = ann. Sharpe of non-overlapping 5d compounds (eval only; model still t+1)",
                f"sparse if n_days < {SPARSE_DAYS}",
            ],
        },
    )


def _notional_turnover_by_strategy(
    fills: pd.DataFrame,
    *,
    unit_notional: float = 10_000.0,
    n_days: int,
) -> dict[str, float]:
    if fills is None or fills.empty or n_days <= 0:
        return {}
    out: dict[str, float] = {}
    f = fills.copy()
    f["notional"] = f["fill_price"].astype(float) * f["qty"].astype(float).abs()
    for sid, g in f.groupby("strategy_id"):
        traded = float(g["notional"].sum())
        out[str(sid)] = traded / (unit_notional * n_days)
    return out


def build_forward_metrics(
    forward_daily: pd.DataFrame | None = None,
    strat_daily: pd.DataFrame | None = None,
    fills: pd.DataFrame | None = None,
    *,
    out_dir: Path | None = None,
    forward_path: Path | None = None,
    strat_path: Path | None = None,
    fills_path: Path | None = None,
) -> dict[str, Any]:
    """Policy + strategy metrics from forward measure track."""
    out_dir = Path(out_dir or BAKEOFF_DIR)
    forward_path = Path(forward_path or out_dir / "forward_daily.parquet")
    strat_path = Path(strat_path or out_dir / "bakeoff_strat_daily.parquet")
    fills_path = Path(fills_path or MEASURE_FILLS)

    if forward_daily is None and forward_path.exists():
        forward_daily = pd.read_parquet(forward_path)
    if strat_daily is None and strat_path.exists():
        strat_daily = pd.read_parquet(strat_path)
    if fills is None and fills_path.exists():
        fills = pd.read_parquet(fills_path)

    forward_daily = forward_daily if forward_daily is not None else pd.DataFrame()
    strat_daily = strat_daily if strat_daily is not None else pd.DataFrame()
    fills = fills if fills is not None else pd.DataFrame()

    n_fwd = int(len(forward_daily)) if forward_daily is not None else 0
    rows: list[dict[str, Any]] = []

    # Selection turnover from ml_top5 column if present
    sel_to = None
    if not forward_daily.empty and "ml_top5" in forward_daily.columns:
        mems: list[set[str]] = []
        for raw in forward_daily.sort_values("date")["ml_top5"]:
            try:
                if isinstance(raw, str):
                    mems.append(set(json.loads(raw)))
                elif isinstance(raw, (list, tuple, set)):
                    mems.append(set(str(x) for x in raw))
                else:
                    mems.append(set())
            except Exception:  # noqa: BLE001
                mems.append(set())
        sel_to = selection_turnover(mems)

    to_by_sid = _notional_turnover_by_strategy(fills, n_days=max(n_fwd, 1))
    trades_by_sid: dict[str, int] = {}
    if not fills.empty and "strategy_id" in fills.columns:
        trades_by_sid = fills.groupby("strategy_id").size().astype(int).to_dict()
        trades_by_sid = {str(k): int(v) for k, v in trades_by_sid.items()}

    policy_map = [
        ("ml_top5_eq", "selection", sel_to),
        ("eq_all", "selection", 0.0),
        ("rand1_E", "selection", 0.0),
        ("rand5_eq", "selection", None),
        ("rules_capital", "notional", None),
    ]
    for col, tkind, tval in policy_map:
        if forward_daily.empty or col not in forward_daily.columns:
            continue
        trades = None
        turnover = tval
        note = None
        if col == "ml_top5_eq" and trades_by_sid:
            # Sum fills for strats that appeared in any day's top5
            top_ids: set[str] = set()
            if "ml_top5" in forward_daily.columns:
                for raw in forward_daily["ml_top5"]:
                    try:
                        if isinstance(raw, str):
                            top_ids |= set(json.loads(raw))
                        elif isinstance(raw, (list, tuple)):
                            top_ids |= set(str(x) for x in raw)
                    except Exception:  # noqa: BLE001
                        pass
            trades = float(sum(trades_by_sid.get(s, 0) for s in top_ids)) if top_ids else None
            if to_by_sid and top_ids:
                turnover = float(np.mean([to_by_sid.get(s, 0.0) for s in top_ids]))
                tkind = "notional"
                note = "trades=measure fills for union of top5; turnover=mean notional"
        if col == "rules_capital":
            note = "from paper day pnl if present; trades often n/a"
        if col == "eq_all" and trades_by_sid:
            trades = float(sum(trades_by_sid.values()))
            if to_by_sid:
                turnover = float(np.mean(list(to_by_sid.values()))) if to_by_sid else None
                tkind = "notional"
        rows.append(
            metrics_from_returns(
                forward_daily[col],
                sleeve=col,
                block="policy",
                trades=trades,
                turnover=turnover,
                turnover_kind=tkind,
                note=note,
            )
        )

    if not strat_daily.empty and "strategy_id" in strat_daily.columns:
        ret_col = "ret" if "ret" in strat_daily.columns else None
        if ret_col:
            for sid, g in strat_daily.groupby("strategy_id", sort=True):
                g = g.sort_values("date")
                sid_s = str(sid)
                rows.append(
                    metrics_from_returns(
                        g[ret_col],
                        sleeve=sid_s,
                        block="strategy",
                        trades=float(trades_by_sid[sid_s]) if sid_s in trades_by_sid else None,
                        turnover=to_by_sid.get(sid_s),
                        turnover_kind="notional" if sid_s in to_by_sid else None,
                        note="measure ledger ret",
                    )
                )

    return write_metrics_bundle(
        rows,
        json_path=out_dir / "forward_metrics.json",
        csv_path=out_dir / "forward_metrics.csv",
        extra={
            "track": "forward",
            "notes": [
                f"sparse if n_days < {SPARSE_DAYS}",
                "strategy trades/turnover from measure_fills when present",
                "sharpe_5d = ann. Sharpe of non-overlapping 5d compounds (eval only)",
            ],
        },
    )


def format_metrics_table(payload: dict[str, Any] | None, *, title: str = "metrics") -> str:
    if not payload or not payload.get("rows"):
        return f"{title}: (empty)"
    lines = [
        f"{title} schema={payload.get('metrics_schema')} "
        f"rows={payload.get('n_rows')} sparse={payload.get('sparse')}"
    ]
    hdr = f"{'sleeve':16} {'block':8} {'n':>4} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7} {'Sh5d':>7} {'trd':>5} {'t/o':>6}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in payload["rows"]:
        def _f(x: Any, pct: bool = False) -> str:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "n/a"
            if pct:
                return f"{100.0 * float(x):+.1f}%"
            return f"{float(x):.2f}"

        lines.append(
            f"{str(r.get('sleeve'))[:16]:16} "
            f"{str(r.get('block'))[:8]:8} "
            f"{int(r.get('n_days') or 0):4d} "
            f"{_f(r.get('cagr'), True):>8} "
            f"{_f(r.get('max_dd'), True):>8} "
            f"{_f(r.get('sharpe')):>7} "
            f"{_f(r.get('sharpe_5d')):>7} "
            f"{_trades(r.get('trades')):>5} "
            f"{_f(r.get('turnover')):>6}"
        )
    return "\n".join(lines)


def _trades(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "n/a"
        return str(int(float(x)))
    except (TypeError, ValueError):
        return "n/a"


def load_metrics_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
