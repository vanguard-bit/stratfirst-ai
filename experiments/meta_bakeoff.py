"""Meta allocator bake-off: offline (Cat1) + forward marks / glance (Cat2)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nse_trader.config import ROOT

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger(__name__)

BAKEOFF_DIR = ROOT / "data" / "store" / "experiments" / "meta_bakeoff"
OOF_DEFAULT = ROOT / "data" / "store" / "experiments" / "meta_train" / "oof_preds.parquet"
GLANCE_PATH = ROOT / "data" / "state" / "meta_bakeoff_glance.json"
FORWARD_DAILY = BAKEOFF_DIR / "forward_daily.parquet"
STRAT_DAILY = BAKEOFF_DIR / "bakeoff_strat_daily.parquet"
WEIGHTS_DAY = ROOT / "data" / "state" / "meta_weights_day.json"


def _llm_dual_from_weights_day(path: Path | None = None) -> dict[str, Any] | None:
    """Load dual-dump summary from meta_weights_day.json (missing → None)."""
    p = Path(path or WEIGHTS_DAY)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if "weights_no_llm" not in raw and "llm" not in raw:
        return None
    llm = raw.get("llm") or {}
    deltas = raw.get("weight_delta_top") or []
    w = raw.get("weights") or {}
    w0 = raw.get("weights_no_llm") or {}
    l1 = 0.0
    if w and w0:
        keys = set(w) | set(w0)
        l1 = float(sum(abs(float(w.get(k, 0.0)) - float(w0.get(k, 0.0))) for k in keys))
    return {
        "date": raw.get("date"),
        "mean": llm.get("mean"),
        "high_n": llm.get("high_n"),
        "as_of": llm.get("as_of"),
        "path": llm.get("path"),
        "l1_distance": l1,
        "weight_delta_top": deltas,
    }



def _max_drawdown(cum: pd.Series) -> float:
    if cum.empty:
        return float("nan")
    peak = cum.cummax()
    dd = cum - peak
    return float(dd.min())


def nifty50_buyhold_from(
    buy_date: str,
    *,
    sell_dates: list[str] | None = None,
) -> pd.DataFrame:
    """
    Equal-weight buy-and-hold of Nifty50 names: buy at close on `buy_date`,
    mark at close on each later date y → mean_i(P_i[y]/P_i[buy] - 1).

    No daily rebalance. Missing names on a day are skipped in that day's mean.
    """
    from data.ingest.store import DataStore
    from data.ingest.symbols import load_nifty50_symbols

    syms = [str(s) for s in load_nifty50_symbols()]
    with DataStore() as store:
        store.init_schema()
        df = store.con.execute(
            "SELECT CAST(date AS VARCHAR) AS date, symbol, close FROM bars_1d ORDER BY 1, 2"
        ).fetchdf()
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "bh_nifty50", "buy_date"])
    df = df[df["symbol"].astype(str).isin(set(syms))].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "bh_nifty50", "buy_date"])

    df["date"] = df["date"].astype(str).str.slice(0, 10)
    buy = str(buy_date)[:10]
    wide = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    wide = wide.sort_index()
    if buy not in wide.index:
        prior = [d for d in wide.index if d <= buy]
        if not prior:
            return pd.DataFrame(columns=["date", "bh_nifty50", "buy_date"])
        buy = prior[-1]
    base = wide.loc[buy]
    levels = (wide.divide(base, axis=1) - 1.0).mean(axis=1, skipna=True)
    out = levels.rename("bh_nifty50").reset_index()
    out["date"] = out["date"].astype(str)
    out["buy_date"] = buy
    if sell_dates is not None:
        want = set(str(d)[:10] for d in sell_dates)
        out = out[out["date"].isin(want) | (out["date"] == buy)].copy()
    return out.reset_index(drop=True)


def attach_nifty50_buyhold(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach bh_nifty50 level series (buy = first offline date) to policy daily panel."""
    if daily is None or daily.empty or "date" not in daily.columns:
        return daily, {}
    buy = str(daily["date"].min())[:10]
    sell = str(daily["date"].max())[:10]
    try:
        bh = nifty50_buyhold_from(buy, sell_dates=daily["date"].astype(str).tolist())
    except Exception as e:  # noqa: BLE001
        logger.warning("nifty50 buyhold failed: %s", e)
        return daily, {"error": str(e)}
    if bh.empty:
        return daily, {"buy_date": buy, "sell_date": sell, "error": "no bars"}
    buy_used = str(bh["buy_date"].iloc[0])
    merged = daily.merge(bh[["date", "bh_nifty50"]], on="date", how="left")
    merged.loc[merged["date"].astype(str) == buy_used, "bh_nifty50"] = 0.0
    end_val = merged.loc[merged["date"].astype(str) == sell, "bh_nifty50"]
    meta = {
        "label": "bh_nifty50",
        "description": (
            f"Equal-weight buy all Nifty50 at close on {buy_used}, "
            f"sell all at close on {sell} (no rebalance, no fees)"
        ),
        "buy_date": buy_used,
        "sell_date": sell,
        "total_return": float(end_val.iloc[0]) if len(end_val) else None,
        "n_days": int(merged["bh_nifty50"].notna().sum()),
    }
    return merged, meta


def _policy_daily_from_oof(oof: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """One row per date with book returns for each Cat1 policy."""
    rng = np.random.default_rng(seed)
    # OOF walk-forward can emit multiple folds per date×strategy — collapse first
    oof = oof.copy()
    oof["date"] = oof["date"].astype(str).str.slice(0, 10)
    agg = {"ret_fwd": "mean"}
    if "p" in oof.columns:
        agg["p"] = "mean"
    if "w_rules" in oof.columns:
        agg["w_rules"] = "mean"
    keep = ["date", "strategy_id"] + [c for c in agg if c in oof.columns]
    oof = oof[keep].groupby(["date", "strategy_id"], as_index=False).agg(agg)

    rows: list[dict[str, Any]] = []
    for date, g in oof.groupby("date", sort=True):
        g = g.copy()
        n = len(g)
        if n == 0:
            continue
        rets = g["ret_fwd"].astype(float).to_numpy()
        scores = g["p"].astype(float).to_numpy() if "p" in g.columns else np.ones(n)

        order = np.argsort(-scores)
        top_idx = order[: min(5, n)]
        pnl_top5 = float(rets[top_idx].mean())

        w = np.clip(scores, 0.0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
        pnl_score = float((w * rets).sum())

        pnl_eq = float(rets.mean())

        if n >= 5:
            pick = rng.choice(n, size=5, replace=False)
        else:
            pick = np.arange(n)
        pnl_rand5 = float(rets[pick].mean())

        if "w_rules" in g.columns:
            wr = g["w_rules"].astype(float).to_numpy()
            wr = wr / wr.sum() if wr.sum() > 0 else np.full(n, 1.0 / n)
            pnl_rules = float((wr * rets).sum())
        else:
            pnl_rules = pnl_eq

        pnl_oracle = float(rets.max())
        mc = rets[rng.integers(0, n, size=200)]
        p_beat_rand1 = float((pnl_top5 > mc).mean())

        rows.append(
            {
                "date": str(date),
                "model_top5_eq": pnl_top5,
                "model_score_all": pnl_score,
                "eq_all": pnl_eq,
                "rand1_E": pnl_eq,
                "rand5_eq": pnl_rand5,
                "rules_proxy": pnl_rules,
                "oracle_best1": pnl_oracle,
                "p_top5_beats_rand1": p_beat_rand1,
                "n_strat": int(n),
            }
        )
    return pd.DataFrame(rows)


def _summarize_offline(daily: pd.DataFrame) -> dict[str, Any]:
    policies = [
        "model_top5_eq",
        "model_score_all",
        "eq_all",
        "rand1_E",
        "rand5_eq",
        "rules_proxy",
        "oracle_best1",
    ]
    out: dict[str, Any] = {
        "n_days": int(len(daily)),
        "date_start": str(daily["date"].min()) if len(daily) else None,
        "date_end": str(daily["date"].max()) if len(daily) else None,
        "mean_p_top5_beats_rand1": float(daily["p_top5_beats_rand1"].mean())
        if len(daily)
        else None,
        "policies": {},
        "notes": [
            "Cat1 OOF ret_fwd with fee-table drag (no bid/ask; L6 partial)",
            "rand1_E equals eq_all by linearity",
            "oracle_best1 is hindsight upper bound only",
            "bh_nifty50 = equal-weight buy-all on buy_date, hold to each date (no rebalance/fees)",
        ],
    }
    for p in policies:
        if p not in daily.columns:
            continue
        s = daily[p].astype(float)
        cum = s.cumsum()
        out["policies"][p] = {
            "mean_daily": float(s.mean()),
            "std_daily": float(s.std(ddof=0)),
            "sum": float(s.sum()),
            "max_drawdown": _max_drawdown(cum),
        }
    if "bh_nifty50" in daily.columns and daily["bh_nifty50"].notna().any():
        bh = daily["bh_nifty50"].astype(float)
        out["policies"]["bh_nifty50"] = {
            "mean_daily": None,
            "level_end": float(bh.dropna().iloc[-1]),
            "max_drawdown": _max_drawdown(bh.fillna(0.0)),
            "note": "level return from buy_date (not sum of daily)",
        }
    if len(daily):
        out["hit_rate_top5_vs_eq"] = float((daily["model_top5_eq"] > daily["eq_all"]).mean())
        out["hit_rate_top5_vs_rand1"] = float(daily["p_top5_beats_rand1"].mean())
    return out


def run_offline_bakeoff(
    *,
    oof_path: Path | None = None,
    out_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    oof_path = Path(oof_path or OOF_DEFAULT)
    out_dir = Path(out_dir or BAKEOFF_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF preds missing: {oof_path}")
    oof = pd.read_parquet(oof_path)
    need = {"date", "strategy_id", "ret_fwd"}
    missing = need - set(oof.columns)
    if missing:
        raise ValueError(f"OOF missing columns: {sorted(missing)}")
    daily = _policy_daily_from_oof(oof, seed=seed)
    daily, bh_meta = attach_nifty50_buyhold(daily)
    summary = _summarize_offline(daily)
    summary["buyhold_nifty50"] = bh_meta
    summary["written_at"] = datetime.now(tz=IST).isoformat()
    summary["oof_path"] = str(oof_path)
    daily_path = out_dir / "offline_daily.parquet"
    sum_path = out_dir / "offline_summary.json"
    daily.to_parquet(daily_path, index=False)
    try:
        from experiments.metrics_table import build_offline_metrics

        metrics = build_offline_metrics(daily, oof, out_dir=out_dir)
        summary["metrics_table"] = {
            "n_rows": metrics.get("n_rows"),
            "sparse": metrics.get("sparse"),
            "json_path": metrics.get("json_path"),
            "csv_path": metrics.get("csv_path"),
            "rows": metrics.get("rows"),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("offline metrics failed: %s", e)
        summary["metrics_table"] = {"error": str(e)}
    sum_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["daily_path"] = str(daily_path)
    summary["summary_path"] = str(sum_path)
    return summary


def mark_policies_on_strat_day(
    strat_day: pd.DataFrame,
    *,
    top5: list[str] | None,
    seed: int = 42,
) -> dict[str, float]:
    """Weight measure-ledger strat day returns into policy book returns."""
    if strat_day is None or strat_day.empty:
        return {
            "ml_top5_eq": 0.0,
            "eq_all": 0.0,
            "rand1_E": 0.0,
            "rand5_eq": 0.0,
        }
    g = strat_day.copy()
    rets = g["ret"].astype(float)
    ids = g["strategy_id"].astype(str)
    eq = float(rets.mean()) if len(rets) else 0.0
    top5 = [str(x) for x in (top5 or [])]
    if top5:
        mask = ids.isin(top5)
        ml = float(rets[mask].mean()) if mask.any() else 0.0
    else:
        ml = 0.0
    rng = np.random.default_rng(seed)
    n = len(rets)
    if n >= 5:
        pick = rng.choice(n, size=5, replace=False)
        rand5 = float(rets.iloc[pick].mean())
    else:
        rand5 = eq
    return {
        "ml_top5_eq": ml,
        "eq_all": eq,
        "rand1_E": eq,
        "rand5_eq": rand5,
    }


def _rules_capital_day_pnl(day: str) -> float | None:
    for base in (
        ROOT / "data" / "store" / "experiments" / "paper",
        ROOT / "data" / "store" / "experiments" / "live",
        ROOT / "data" / "store" / "experiments" / "eod" / "paper",
    ):
        day_file = base / f"day_{day}.json"
        if day_file.exists():
            try:
                payload = json.loads(day_file.read_text(encoding="utf-8"))
                for key in ("pnl", "day_pnl", "net_pnl"):
                    if key in payload and payload[key] is not None:
                        return float(payload[key])
            except Exception:  # noqa: BLE001
                pass
    return None


def _load_shadow_top5(day: str) -> tuple[list[str], list[str]]:
    from meta.shadow import SHADOW_HISTORY_PATH, SHADOW_PATH

    reasons: list[str] = []
    top5: list[str] = []
    if SHADOW_PATH.exists():
        try:
            payload = json.loads(SHADOW_PATH.read_text(encoding="utf-8"))
            top5 = list(payload.get("top5") or [])
            reasons = list(payload.get("top5_reasons") or [])
            if str(payload.get("as_of")) != str(day):
                # still use latest scores but note mismatch via empty if too stale
                pass
        except Exception:  # noqa: BLE001
            pass
    if SHADOW_HISTORY_PATH.exists():
        try:
            hist = pd.read_parquet(SHADOW_HISTORY_PATH)
            h = hist[hist["as_of"].astype(str) == str(day)]
            if not h.empty and "in_top5" in h.columns:
                top5 = h.loc[h["in_top5"].astype(bool), "strategy_id"].astype(str).tolist()
                if "reason" in h.columns:
                    reasons = h.loc[h["in_top5"].astype(bool), "reason"].astype(str).tolist()
        except Exception:  # noqa: BLE001
            pass
    return top5, reasons


def _cluster_of() -> dict[str, str]:
    try:
        from strategies.registry import load_enabled_strategies

        return {
            sid: str(getattr(st, "cluster", "?"))
            for sid, st in load_enabled_strategies().items()
        }
    except Exception:  # noqa: BLE001
        return {}


def build_glance(
    *,
    day: str | None = None,
    strat_daily_path: Path | None = None,
    forward_path: Path | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    day = day or datetime.now(tz=IST).date().isoformat()
    strat_daily_path = Path(strat_daily_path or STRAT_DAILY)
    forward_path = Path(forward_path or FORWARD_DAILY)
    clusters_map = _cluster_of()
    warns = list(warnings or [])

    strat_today = pd.DataFrame()
    strat_all = pd.DataFrame()
    if strat_daily_path.exists():
        try:
            strat_all = pd.read_parquet(strat_daily_path)
            strat_today = strat_all[strat_all["date"].astype(str) == str(day)]
        except Exception as e:  # noqa: BLE001
            warns.append(f"strat_daily read failed: {e}")

    top5, reasons = _load_shadow_top5(day)
    marks = mark_policies_on_strat_day(strat_today, top5=top5)

    cum_by_sid: dict[str, float] = {}
    today_by_sid: dict[str, float] = {}
    if not strat_all.empty:
        for sid, g in strat_all.groupby("strategy_id"):
            cum_by_sid[str(sid)] = float(g["ret"].astype(float).sum())
    if not strat_today.empty:
        for r in strat_today.itertuples():
            today_by_sid[str(r.strategy_id)] = float(r.ret)

    strategies = []
    for sid in sorted(set(list(cum_by_sid) + list(today_by_sid) + list(clusters_map))):
        strategies.append(
            {
                "strategy_id": sid,
                "cluster": clusters_map.get(sid, "?"),
                "today": today_by_sid.get(sid, 0.0),
                "cum": cum_by_sid.get(sid, 0.0),
                "in_ml_top5": sid in top5,
            }
        )
    strategies.sort(key=lambda r: r["cum"], reverse=True)
    for i, row in enumerate(strategies, start=1):
        row["rank_cum"] = i

    cluster_roll: dict[str, dict[str, Any]] = {}
    for row in strategies:
        cl = row["cluster"]
        bucket = cluster_roll.setdefault(
            cl, {"today": 0.0, "cum": 0.0, "n": 0, "in_ml_top5": []}
        )
        bucket["today"] += float(row["today"])
        bucket["cum"] += float(row["cum"])
        bucket["n"] += 1
        if row["in_ml_top5"]:
            bucket["in_ml_top5"].append(row["strategy_id"])
    for bucket in cluster_roll.values():
        n = max(int(bucket["n"]), 1)
        bucket["today"] = float(bucket["today"]) / n
        bucket["cum"] = float(bucket["cum"]) / n

    rules_today = _rules_capital_day_pnl(day)
    tracks = {
        "rules_capital": {"today": rules_today, "note": "from paper day json if present"},
        "ml_top5_eq": {"today": marks["ml_top5_eq"]},
        "rand1_E": {"today": marks["rand1_E"]},
        "eq_all": {"today": marks["eq_all"]},
        "rand5_eq": {"today": marks["rand5_eq"]},
    }
    if forward_path.exists():
        try:
            fwd = pd.read_parquet(forward_path)
            for key in ("ml_top5_eq", "rand1_E", "eq_all", "rand5_eq", "rules_capital"):
                if key in fwd.columns:
                    tracks.setdefault(key, {})
                    tracks[key]["cum"] = float(pd.to_numeric(fwd[key], errors="coerce").sum())
        except Exception as e:  # noqa: BLE001
            warns.append(f"forward_daily read failed: {e}")

    if not strat_daily_path.exists() or strat_today.empty:
        warns.append("measure ledger empty for today — friction marks unavailable")
    if not top5:
        warns.append("no ML shadow top5 for today")

    llm_dual = _llm_dual_from_weights_day()
    if llm_dual is None:
        warns.append("no LLM dual dump in meta_weights_day.json — delete cache to recompute")

    return {
        "as_of": day,
        "written_at": datetime.now(tz=IST).isoformat(),
        "tracks": tracks,
        "clusters": cluster_roll,
        "strategies": strategies,
        "ml_top5": top5,
        "ml_top5_reasons": reasons,
        "llm_dual": llm_dual,
        "warnings": warns,
    }


def append_forward_day(
    *,
    day: str,
    strat_day: pd.DataFrame | None = None,
    top5: list[str] | None = None,
    out_path: Path | None = None,
    placeholder_quotes: bool | None = None,
) -> dict[str, Any]:
    out_path = Path(out_path or FORWARD_DAILY)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if strat_day is None and STRAT_DAILY.exists():
        all_s = pd.read_parquet(STRAT_DAILY)
        strat_day = all_s[all_s["date"].astype(str) == str(day)]
    if top5 is None:
        top5, _ = _load_shadow_top5(day)
    marks = mark_policies_on_strat_day(
        strat_day if strat_day is not None else pd.DataFrame(), top5=top5
    )
    rules = _rules_capital_day_pnl(day)
    overlap = None
    weights_path = ROOT / "data" / "state" / "meta_weights_day.json"
    if weights_path.exists() and top5:
        try:
            wpayload = json.loads(weights_path.read_text(encoding="utf-8"))
            weights = wpayload.get("weights") or {}
            heavy = {s for s, w in weights.items() if float(w) > 0}
            overlap = len(heavy & set(top5)) / max(len(top5), 1)
        except Exception:  # noqa: BLE001
            overlap = None

    row = {
        "date": str(day),
        "ml_top5_eq": marks["ml_top5_eq"],
        "eq_all": marks["eq_all"],
        "rand1_E": marks["rand1_E"],
        "rand5_eq": marks["rand5_eq"],
        "rules_capital": rules if rules is not None else float("nan"),
        "ml_top5": json.dumps(list(top5 or [])),
        "overlap_rules_heavy": overlap,
        "placeholder_quotes": placeholder_quotes,
        "written_at": datetime.now(tz=IST).isoformat(),
    }
    new_df = pd.DataFrame([row])
    if out_path.exists():
        old = pd.read_parquet(out_path)
        old = old[old["date"].astype(str) != str(day)]
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return row


def refresh_forward_and_glance(
    *,
    day: str | None = None,
    warnings: list[str] | None = None,
    placeholder_quotes: bool | None = None,
) -> dict[str, Any]:
    day = day or datetime.now(tz=IST).date().isoformat()
    warns = list(warnings or [])
    try:
        append_forward_day(day=day, placeholder_quotes=placeholder_quotes)
    except Exception as e:  # noqa: BLE001
        logger.warning("forward bakeoff append failed: %s", e)
        warns.append(f"forward append failed: {e}")
    try:
        from experiments.metrics_table import build_forward_metrics

        build_forward_metrics(out_dir=BAKEOFF_DIR)
    except Exception as e:  # noqa: BLE001
        logger.warning("forward metrics failed: %s", e)
        warns.append(f"forward metrics failed: {e}")
    glance = build_glance(day=day, warnings=warns)
    GLANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = GLANCE_PATH.with_suffix(GLANCE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(glance, indent=2, default=str), encoding="utf-8")
    tmp.replace(GLANCE_PATH)
    try:
        from ops.dashboard import write_dashboard

        write_dashboard()
    except Exception as e:  # noqa: BLE001
        logger.warning("dashboard refresh failed: %s", e)
    return glance


def format_glance(glance: dict[str, Any]) -> str:
    lines = [
        f"meta-status as_of={glance.get('as_of')} written={glance.get('written_at')}",
        "tracks:",
    ]
    for name, t in (glance.get("tracks") or {}).items():
        lines.append(f"  {name}: today={t.get('today')} cumulative={t.get('cum')}")
    lines.append(f"ml_top5: {glance.get('ml_top5')}")
    for r in (glance.get("ml_top5_reasons") or [])[:5]:
        lines.append(f"  reason: {r}")
    dual = glance.get("llm_dual")
    if dual is None:
        dual = _llm_dual_from_weights_day()
    if dual:
        lines.append(
            "llm_dual: "
            f"date={dual.get('date')} mean={dual.get('mean')} "
            f"high_n={dual.get('high_n')} as_of={dual.get('as_of')} "
            f"l1={dual.get('l1_distance')}"
        )
        for row in (dual.get("weight_delta_top") or [])[:5]:
            lines.append(
                f"  delta: {row.get('strategy_id')} {float(row.get('delta', 0.0)):+.5f}"
            )
    else:
        lines.append("llm_dual: (none)")
    lines.append("clusters:")
    for cl, b in sorted((glance.get("clusters") or {}).items()):
        lines.append(
            f"  {cl}: today={b.get('today'):+.5f} cumulative={b.get('cum'):+.5f} ml={b.get('in_ml_top5')}"
        )
    lines.append("strategies (top by cumulative):")
    for row in (glance.get("strategies") or [])[:12]:
        flag = "*" if row.get("in_ml_top5") else " "
        lines.append(
            f"  {flag}{row['strategy_id']} [{row.get('cluster')}] "
            f"today={row.get('today'):+.5f} cumulative={row.get('cum'):+.5f}"
        )
    for w in glance.get("warnings") or []:
        lines.append(f"WARN: {w}")
    try:
        from experiments.metrics_table import (
            FORWARD_METRICS_JSON,
            OFFLINE_METRICS_JSON,
            format_metrics_table,
            load_metrics_json,
        )

        off = load_metrics_json(OFFLINE_METRICS_JSON)
        if off.get("rows"):
            slim = {
                **off,
                "rows": [r for r in off["rows"] if r.get("block") == "policy"][:10],
            }
            lines.append(format_metrics_table(slim, title="offline metrics (policies)"))
        fwd = load_metrics_json(FORWARD_METRICS_JSON)
        if fwd.get("rows"):
            slim = {
                **fwd,
                "rows": [r for r in fwd["rows"] if r.get("block") == "policy"][:10],
            }
            lines.append(format_metrics_table(slim, title="forward metrics (policies)"))
    except Exception as e:  # noqa: BLE001
        lines.append(f"WARN: metrics table load failed: {e}")
    return "\n".join(lines)


def format_offline_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"offline bakeoff days={summary.get('n_days')} "
        f"{summary.get('date_start')}→{summary.get('date_end')}",
        f"P(top5>rand1)≈{summary.get('mean_p_top5_beats_rand1')}",
        f"hit_rate top5>eq={summary.get('hit_rate_top5_vs_eq')}",
    ]
    bh = summary.get("buyhold_nifty50") or {}
    if bh.get("total_return") is not None:
        lines.append(
            f"  bh_nifty50: buy {bh.get('buy_date')} → sell {bh.get('sell_date')} "
            f"total={float(bh['total_return']):+.4f} (eq-weight, no fees)"
        )
    for name, m in (summary.get("policies") or {}).items():
        if name == "bh_nifty50":
            end = m.get("level_end")
            lines.append(
                f"  {name}: level_end={float(end):+.4f} mdd={m.get('max_drawdown'):+.4f}"
                if end is not None
                else f"  {name}: (n/a)"
            )
            continue
        mean = m.get("mean_daily")
        sm = m.get("sum")
        lines.append(
            f"  {name}: mean={mean:+.6f} sum={sm:+.4f} mdd={m.get('max_drawdown'):+.4f}"
            if mean is not None and sm is not None
            else f"  {name}: {m}"
        )
    mt = summary.get("metrics_table") or {}
    if mt.get("error"):
        lines.append(f"metrics_table error: {mt['error']}")
    elif mt.get("rows"):
        from experiments.metrics_table import format_metrics_table

        lines.append(
            format_metrics_table(
                {
                    "metrics_schema": 1,
                    "n_rows": mt.get("n_rows"),
                    "sparse": mt.get("sparse"),
                    "rows": mt["rows"],
                },
                title="metrics table",
            )
        )
    return "\n".join(lines)


def format_forward_summary(path: Path | None = None) -> str:
    path = Path(path or FORWARD_DAILY)
    if not path.exists():
        return "forward bakeoff: no data yet"
    df = pd.read_parquet(path)
    lines = [f"forward bakeoff days={len(df)} {df['date'].min()}→{df['date'].max()}"]
    for col in ("ml_top5_eq", "eq_all", "rand1_E", "rules_capital"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            lines.append(f"  {col}: mean={s.mean():+.6f} sum={s.sum():+.4f}")
    try:
        from experiments.metrics_table import (
            FORWARD_METRICS_JSON,
            build_forward_metrics,
            format_metrics_table,
            load_metrics_json,
        )

        if not FORWARD_METRICS_JSON.exists():
            build_forward_metrics(out_dir=BAKEOFF_DIR)
        payload = load_metrics_json(FORWARD_METRICS_JSON)
        if payload.get("rows"):
            lines.append(format_metrics_table(payload, title="forward metrics"))
    except Exception as e:  # noqa: BLE001
        lines.append(f"forward metrics: {e}")
    return "\n".join(lines)
