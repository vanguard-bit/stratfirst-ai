"""LightGBM meta shadow scorer — never changes live allocation weights."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from meta.dataset import FEATURE_COLS
from meta.train_lgbm import DEFAULT_MODEL_NAME
from nse_trader.config import ROOT

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger(__name__)

SHADOW_PATH = ROOT / "data" / "state" / "meta_lgbm_shadow.json"
SHADOW_HISTORY_PATH = ROOT / "data" / "state" / "meta_lgbm_shadow_history.parquet"
SHADOW_JSONL_PATH = ROOT / "data" / "logs" / "meta_shadow_daily.jsonl"
DEFAULT_MODEL = ROOT / "data" / "store" / "models" / f"{DEFAULT_MODEL_NAME}.txt"
DEFAULT_FEATURES = ROOT / "data" / "store" / "models" / f"{DEFAULT_MODEL_NAME}.features.json"

# Human / LLM-facing names for feature attributions
_FEATURE_EN: dict[str, str] = {
    "ret_1d": "1-day strategy return",
    "ret_5d": "5-day strategy return",
    "ret_20d": "20-day strategy return",
    "vol_20d": "20-day return volatility",
    "cluster_code": "strategy cluster",
    "tf_code": "native timeframe",
    "regime_adx": "ADX regime",
    "regime_vix": "India VIX level",
    "regime_vix_high": "high-VIX flag",
    "regime_expiry_week": "expiry-week flag",
    "regime_nifty_ret_20d": "Nifty 20-day return",
    "llm_sentiment": "LLM headline sentiment",
    "llm_high_mat": "LLM high-materiality flag",
    "bias": "model baseline",
}


def feature_rows_for_shadow(returns: pd.DataFrame, *, as_of: str | None = None) -> pd.DataFrame:
    """
    Build feature rows for shadow scoring (no labels needed).

    Uses trailing returns at EOD of `as_of`. If that date is missing (weekend /
    holiday / as_of after last replay day), falls back to the latest available
    date — labeled `build_meta_panel` drops the final day (no t+1), so EOD must
    not depend on it.
    """
    from meta.dataset import _default_strategy_meta, _trailing_by_strategy

    if returns is None or returns.empty:
        return pd.DataFrame()

    base = returns.dropna(subset=["ret"]).copy()
    if base.empty:
        return pd.DataFrame()
    base["date"] = base["date"].astype(str)
    feat = _trailing_by_strategy(base)
    feat["date"] = feat["date"].astype(str)

    meta = _default_strategy_meta()
    cluster_codes: list[int] = []
    tf_codes: list[int] = []
    for sid in feat["strategy_id"]:
        m = meta.get(str(sid), {})
        cluster_codes.append(int(m.get("cluster_code", 0)))
        tf_codes.append(int(m.get("tf_code", 3)))
    feat["cluster_code"] = cluster_codes
    feat["tf_code"] = tf_codes
    for c in FEATURE_COLS:
        if c not in feat.columns:
            feat[c] = 0.0

    want = str(as_of) if as_of else str(feat["date"].max())
    rows = feat[feat["date"] == want].copy()
    if rows.empty:
        want = str(feat["date"].max())
        rows = feat[feat["date"] == want].copy()
    if rows.empty:
        return pd.DataFrame()
    return rows[["strategy_id", *FEATURE_COLS]].reset_index(drop=True)


def _strategy_clusters(ids: list[str]) -> dict[str, str]:
    try:
        from meta.dataset import _default_strategy_meta

        meta = _default_strategy_meta()
        out: dict[str, str] = {}
        for sid in ids:
            m = meta.get(str(sid), {})
            cluster = m.get("cluster")
            if cluster:
                out[sid] = str(cluster)
            else:
                out[sid] = "?"
        return out
    except Exception:  # noqa: BLE001
        return {sid: "?" for sid in ids}


def _fmt_val(v: float) -> str:
    if abs(v) >= 10:
        return f"{v:.2f}"
    if abs(v) >= 0.01:
        return f"{v:.4f}"
    return f"{v:.6f}"


def english_reason(
    *,
    strategy_id: str,
    score: float,
    in_top5: bool,
    features: dict[str, float],
    top_contribs: list[dict[str, Any]],
) -> str:
    """One-line English summary for humans / LLM review of a shadow pick."""
    rank_bit = "in daily top-5" if in_top5 else "outside top-5"
    parts: list[str] = [
        f"{strategy_id} scored {score:.3f} ({rank_bit}; P≈in next-day top-5)."
    ]
    if top_contribs:
        bits = []
        for c in top_contribs[:3]:
            name = _FEATURE_EN.get(str(c["feature"]), str(c["feature"]))
            direction = "lifted" if float(c["contribution"]) >= 0 else "cut"
            fv = features.get(str(c["feature"]))
            if fv is None:
                bits.append(f"{name} {direction} score by {float(c['contribution']):+.3f}")
            else:
                bits.append(
                    f"{name}={_fmt_val(float(fv))} {direction} score by "
                    f"{float(c['contribution']):+.3f}"
                )
        parts.append("Drivers: " + "; ".join(bits) + ".")
    return " ".join(parts)


def _top_contribs_for_row(
    contrib_row: np.ndarray,
    feat_cols: list[str],
    *,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Map LightGBM pred_contrib row (feats + bias) → top-|k| feature attributions."""
    n = len(feat_cols)
    pairs: list[tuple[str, float]] = []
    for i, name in enumerate(feat_cols):
        if i < len(contrib_row):
            pairs.append((name, float(contrib_row[i])))
    if len(contrib_row) > n:
        pairs.append(("bias", float(contrib_row[n])))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": n, "contribution": c} for n, c in pairs[:k]]


def _append_history(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            # Replace same as_of if re-run same day
            if "as_of" in existing.columns and not new_df.empty:
                day = str(new_df["as_of"].iloc[0])
                existing = existing[existing["as_of"].astype(str) != day]
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception:  # noqa: BLE001
            combined = new_df
    else:
        combined = new_df
    tmp = path.with_suffix(path.suffix + ".tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(path)


def _append_jsonl(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def run_meta_shadow(
    *,
    as_of: date | str | None = None,
    feature_rows: pd.DataFrame | None = None,
    model_path: Path | None = None,
    features_path: Path | None = None,
    out_path: Path | None = None,
    history_path: Path | None = None,
    jsonl_path: Path | None = None,
) -> dict[str, Any] | None:
    """
    Score strategies with saved Booster; write latest JSON + append history/JSONL.
    Returns None if model missing or features empty. Does not touch live weights.
    """
    import lightgbm as lgb

    model_path = Path(model_path or DEFAULT_MODEL)
    features_path = Path(features_path or DEFAULT_FEATURES)
    out_path = Path(out_path or SHADOW_PATH)
    history_path = Path(history_path or SHADOW_HISTORY_PATH)
    jsonl_path = Path(jsonl_path or SHADOW_JSONL_PATH)

    if not model_path.exists():
        logger.warning("meta shadow skipped — no model at %s", model_path)
        return None

    day = as_of or datetime.now(tz=IST).date()
    if isinstance(day, str):
        day_s = day
    else:
        day_s = day.isoformat()

    if feature_rows is None or feature_rows.empty:
        logger.warning("meta shadow skipped — empty feature_rows for %s", day_s)
        return None

    feat_cols = list(FEATURE_COLS)
    if features_path.exists():
        try:
            meta = json.loads(features_path.read_text())
            feat_cols = list(meta.get("feature_cols") or FEATURE_COLS)
        except Exception as e:  # noqa: BLE001
            logger.warning("could not read features json: %s", e)

    missing = [c for c in feat_cols if c not in feature_rows.columns]
    if missing:
        logger.warning("meta shadow skipped — missing features %s", missing)
        return None

    booster = lgb.Booster(model_file=str(model_path))
    X = feature_rows[feat_cols].astype(float).fillna(0.0)
    probs = booster.predict(X)
    try:
        contribs = np.asarray(booster.predict(X, pred_contrib=True))
    except Exception as e:  # noqa: BLE001
        logger.warning("pred_contrib failed: %s", e)
        contribs = None

    ids = feature_rows["strategy_id"].astype(str).tolist()
    scores = {sid: float(p) for sid, p in zip(ids, probs, strict=False)}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top5 = [sid for sid, _ in ranked[:5]]
    written_at = datetime.now(tz=IST).isoformat()
    clusters = _strategy_clusters(ids)

    per_strat: dict[str, Any] = {}
    history_rows: list[dict[str, Any]] = []
    for i, sid in enumerate(ids):
        feat_map = {c: float(X.iloc[i][c]) for c in feat_cols}
        top_c: list[dict[str, Any]] = []
        if contribs is not None and i < len(contribs):
            top_c = _top_contribs_for_row(np.asarray(contribs[i]), feat_cols, k=3)
        in_top = sid in top5
        score = float(scores[sid])
        reason = english_reason(
            strategy_id=sid,
            score=score,
            in_top5=in_top,
            features=feat_map,
            top_contribs=top_c,
        )
        rank = next(r for r, (s, _) in enumerate(ranked, start=1) if s == sid)
        per_strat[sid] = {
            "score": score,
            "rank": rank,
            "in_top5": in_top,
            "cluster": clusters.get(sid, "?"),
            "features": feat_map,
            "top_contribs": top_c,
            "reason": reason,
        }
        history_rows.append(
            {
                "as_of": day_s,
                "written_at": written_at,
                "strategy_id": sid,
                "cluster": clusters.get(sid, "?"),
                "score": score,
                "rank": rank,
                "in_top5": in_top,
                "reason": reason,
                "top_contribs_json": json.dumps(top_c),
                **{f"f_{c}": feat_map[c] for c in feat_cols},
            }
        )

    top5_reasons = [per_strat[s]["reason"] for s in top5 if s in per_strat]

    payload: dict[str, Any] = {
        "as_of": day_s,
        "mode": "lightgbm_shadow",
        "model_path": str(model_path),
        "scores": scores,
        "top5": top5,
        "top5_reasons": top5_reasons,
        "strategies": per_strat,
        "written_at": written_at,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(out_path)

    try:
        _append_history(history_rows, history_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("shadow history append failed: %s", e)

    try:
        _append_jsonl(
            {
                "as_of": day_s,
                "written_at": written_at,
                "mode": "lightgbm_shadow",
                "top5": top5,
                "top5_reasons": top5_reasons,
                "scores": scores,
                "strategies": {
                    sid: {
                        "score": per_strat[sid]["score"],
                        "rank": per_strat[sid]["rank"],
                        "in_top5": per_strat[sid]["in_top5"],
                        "reason": per_strat[sid]["reason"],
                        "top_contribs": per_strat[sid]["top_contribs"],
                    }
                    for sid in ids
                },
            },
            jsonl_path,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("shadow jsonl append failed: %s", e)

    try:
        from experiments.allocation_log import log_allocation_snapshot

        w = {sid: (1.0 / 5.0 if sid in top5 else 0.0) for sid in scores}
        log_allocation_snapshot(w, cluster_of=clusters, mode="lightgbm_shadow")
    except Exception as e:  # noqa: BLE001
        logger.warning("shadow allocation log failed: %s", e)

    return payload
