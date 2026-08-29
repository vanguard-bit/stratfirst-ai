"""Walk-forward LightGBM meta train → artifacts (shadow-only live)."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from meta.dataset import FEATURE_COLS
from nse_trader.config import ROOT

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

DEFAULT_MODEL_NAME = "meta_lgbm_v0"


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Mann–Whitney / Wilcoxon form of AUC
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _top5_precision(df: pd.DataFrame, score_col: str = "p", k: int = 5) -> float:
    """Mean over days of (# true top-k among predicted top-k) / k."""
    if df.empty:
        return float("nan")
    precs: list[float] = []
    for _, g in df.groupby("date", sort=True):
        g = g.sort_values(score_col, ascending=False)
        pred = set(g.head(k)["strategy_id"])
        true = set(g.loc[g["y"] == 1, "strategy_id"])
        precs.append(len(pred & true) / float(k))
    return float(np.mean(precs)) if precs else float("nan")


def _book_pnl(df: pd.DataFrame, weight_col: str) -> float:
    """Mean daily sum of ret_fwd * weight (weights should sum ~1 per day)."""
    if df.empty or weight_col not in df.columns:
        return float("nan")
    daily = df.groupby("date", sort=True).apply(
        lambda g: float((g["ret_fwd"] * g[weight_col]).sum()),
        include_groups=False,
    )
    return float(daily.mean()) if len(daily) else float("nan")


def _normalize_day_weights(scores: pd.Series) -> pd.Series:
    s = scores.clip(lower=0.0)
    total = float(s.sum())
    if total <= 0:
        return pd.Series(np.full(len(s), 1.0 / max(len(s), 1)), index=s.index)
    return s / total


def make_synthetic_panel(n_days: int = 120, n_strat: int = 10, seed: int = 0) -> pd.DataFrame:
    """Tiny panel for unit tests (also useful for smoke train)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days).strftime("%Y-%m-%d").tolist()
    rows: list[dict[str, Any]] = []
    for d in dates:
        rets = rng.normal(0.0, 0.01, size=n_strat)
        order = np.argsort(-rets)
        top = set(order[:5].tolist())
        for i in range(n_strat):
            rows.append(
                {
                    "date": d,
                    "strategy_id": f"S{i}",
                    "y": 1 if i in top else 0,
                    "ret_fwd": float(rets[i]),
                    "ret_1d": float(rng.normal(0, 0.01)),
                    "ret_5d": float(rng.normal(0, 0.02)),
                    "ret_20d": float(rng.normal(0, 0.04)),
                    "vol_20d": float(abs(rng.normal(0.01, 0.005))),
                    "cluster_code": float(i % 7),
                    "tf_code": float(i % 5),
                    "regime_adx": float(rng.uniform(10, 40)),
                    "regime_vix": float(rng.uniform(10, 25)),
                    "regime_vix_high": float(rng.random() > 0.5),
                    "regime_expiry_week": float(rng.random() > 0.8),
                    "regime_nifty_ret_20d": float(rng.normal(0, 0.03)),
                    "llm_sentiment": 0.0,
                    "llm_high_mat": 0.0,
                }
            )
    return pd.DataFrame(rows)


def train_meta_lgbm(
    panel: pd.DataFrame,
    *,
    out_dir: Path | None = None,
    embargo_days: int = 3,
    test_days: int = 63,
    step_days: int = 21,
    min_train_days: int = 40,
    model_name: str = DEFAULT_MODEL_NAME,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Expanding walk-forward LightGBM binary classifier.
    Writes model + features JSON + experiment manifest under out_dir.
    """
    import lightgbm as lgb

    out_dir = Path(out_dir or (ROOT / "data" / "store"))
    models_dir = out_dir / "models" if (out_dir / "models").exists() or out_dir.name != "models" else out_dir
    # Prefer out_dir/models and out_dir/experiments when out_dir is store root;
    # for tmp_path tests, write models/ + experiments/ under out_dir.
    models_dir = out_dir / "models"
    exp_dir = out_dir / "experiments" / "meta_train"
    models_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    need = {"date", "strategy_id", "y", "ret_fwd", *FEATURE_COLS}
    missing = need - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    df = panel.copy()
    df["date"] = df["date"].astype(str)
    dates = sorted(df["date"].unique())
    if len(dates) < min_train_days + embargo_days + 5:
        raise ValueError(
            f"need >= {min_train_days + embargo_days + 5} unique dates, got {len(dates)}"
        )

    fold_rows: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []
    last_model: Any = None
    fold_i = 0

    # Expanding: train on dates[:train_end], test next test_days after embargo
    train_end_idx = min_train_days - 1
    while True:
        test_start_idx = train_end_idx + 1 + int(embargo_days)
        test_end_idx = test_start_idx + int(test_days) - 1
        if test_start_idx >= len(dates):
            break
        test_end_idx = min(test_end_idx, len(dates) - 1)
        train_dates = set(dates[: train_end_idx + 1])
        test_dates = set(dates[test_start_idx : test_end_idx + 1])
        if not test_dates:
            break

        tr = df[df["date"].isin(train_dates)]
        te = df[df["date"].isin(test_dates)]
        if tr.empty or te.empty:
            break

        y_tr = tr["y"].astype(int)
        pos = max(int(y_tr.sum()), 1)
        neg = max(int((1 - y_tr).sum()), 1)
        scale = neg / pos
        dtrain = lgb.Dataset(tr[FEATURE_COLS], label=y_tr, feature_name=list(FEATURE_COLS))
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 5,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "scale_pos_weight": scale,
            "seed": seed,
            "verbosity": -1,
        }
        model = lgb.train(params, dtrain, num_boost_round=80)
        last_model = model

        p = model.predict(te[FEATURE_COLS])
        scored = te[["date", "strategy_id", "y", "ret_fwd"]].copy()
        scored["p"] = p
        scored["fold"] = fold_i
        # Book weights
        scored["w_score"] = scored.groupby("date")["p"].transform(_normalize_day_weights)
        scored["w_eq"] = scored.groupby("date")["strategy_id"].transform(lambda s: 1.0 / len(s))
        # Rules proxy: prefer recent ret_1d if present else equal
        if "ret_1d" in te.columns:
            scored["w_rules"] = te.groupby("date")["ret_1d"].transform(
                lambda s: _normalize_day_weights(s.clip(lower=0) + 1e-6)
            )
        else:
            scored["w_rules"] = scored["w_eq"]

        auc = _auc(scored["y"].to_numpy(), scored["p"].to_numpy())
        t5 = _top5_precision(scored, "p", k=5)
        fold_rows.append(
            {
                "fold": fold_i,
                "train_end": dates[train_end_idx],
                "test_start": dates[test_start_idx],
                "test_end": dates[test_end_idx],
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "auc": auc,
                "top5_precision": t5,
                "pnl_score": _book_pnl(scored, "w_score"),
                "pnl_rules": _book_pnl(scored, "w_rules"),
                "pnl_eq": _book_pnl(scored, "w_eq"),
            }
        )
        pred_parts.append(scored)
        fold_i += 1
        # Step train end forward
        next_end = train_end_idx + int(step_days)
        if next_end <= train_end_idx:
            break
        train_end_idx = next_end
        if train_end_idx >= len(dates) - embargo_days - 2:
            break

    if last_model is None:
        raise RuntimeError("no folds produced — check panel size / params")

    # Final fit on all but last embargo+test slice for published model
    final_cut = max(len(dates) - max(test_days // 3, 5) - embargo_days, min_train_days)
    final_dates = set(dates[:final_cut])
    final = df[df["date"].isin(final_dates)]
    y_f = final["y"].astype(int)
    pos = max(int(y_f.sum()), 1)
    neg = max(int((1 - y_f).sum()), 1)
    dfinal = lgb.Dataset(final[FEATURE_COLS], label=y_f, feature_name=list(FEATURE_COLS))
    final_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "scale_pos_weight": neg / pos,
        "seed": seed,
        "verbosity": -1,
    }
    final_model = lgb.train(final_params, dfinal, num_boost_round=100)

    model_path = models_dir / f"{model_name}.txt"
    tmp_model = models_dir / f"{model_name}.txt.tmp"
    final_model.save_model(str(tmp_model))
    os.replace(tmp_model, model_path)

    feat_path = models_dir / f"{model_name}.features.json"
    feat_payload = {
        "feature_cols": list(FEATURE_COLS),
        "categorical": ["cluster_code", "tf_code"],
        "model": model_name,
        "written_at": _utc_now_iso(),
    }
    tmp_feat = models_dir / f"{model_name}.features.json.tmp"
    tmp_feat.write_text(json.dumps(feat_payload, indent=2))
    os.replace(tmp_feat, feat_path)

    folds_df = pd.DataFrame(fold_rows)
    folds_path = exp_dir / "folds.parquet"
    folds_df.to_parquet(folds_path, index=False)
    if pred_parts:
        pd.concat(pred_parts, ignore_index=True).to_parquet(exp_dir / "oof_preds.parquet", index=False)

    manifest: dict[str, Any] = {
        "model_name": model_name,
        "model_path": str(model_path),
        "features_path": str(feat_path),
        "n_folds": int(len(fold_rows)),
        "n_rows": int(len(df)),
        "n_dates": int(len(dates)),
        "embargo_days": int(embargo_days),
        "test_days": int(test_days),
        "step_days": int(step_days),
        "feature_cols": list(FEATURE_COLS),
        "fold_metrics": fold_rows,
        "mean_auc": float(np.nanmean([f["auc"] for f in fold_rows])) if fold_rows else None,
        "mean_top5_precision": (
            float(np.nanmean([f["top5_precision"] for f in fold_rows])) if fold_rows else None
        ),
        "written_at": _utc_now_iso(),
        "mis_shorts": True,
        "cnc_long_only": True,
        "cluster_e_exits": True,
        "trad_leak_fixes": True,
        "b2_connors": True,
        "d3_exits": True,
        "overlays_disabled": ["D1", "D2", "F3", "G1", "G2"],
        "c1_long_only": True,
        "b3_gap_fade": True,
        "a2_ma_stop": True,
        "f2_half_reduce": True,
        "e2_disabled": True,
        "b3_tightened": True,
        "e1_e3_disabled": True,
        "strategy_drawdown_zero": True,
        "cs_ranks_lookback": True,
    }
    man_path = exp_dir / "manifest.json"
    tmp_man = exp_dir / "manifest.json.tmp"
    tmp_man.write_text(json.dumps(manifest, indent=2, default=str))
    os.replace(tmp_man, man_path)
    logger.info("meta-train wrote %s (%d folds)", model_path, len(fold_rows))
    return manifest


def run_meta_train_cli(
    *,
    years: int = 3,
    log_path: Path | None = None,
    symbols: list[str] | None = None,
    workers: int | None = None,
    notional_per_symbol: float | None = None,
    force_export: bool = False,
) -> dict[str, Any]:
    """
    Universe fixed-notional replay → panel → walk-forward LightGBM.
    Logs to data/logs/meta-train.log.
    """
    log_path = Path(log_path or (ROOT / "data" / "logs" / "meta-train.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def progress(msg: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    from data.ingest.symbols import load_nifty50_symbols
    from experiments.strategy_replay import (
        default_notional_per_symbol,
        replay_universe_book_returns,
    )
    from meta.dataset import build_meta_panel

    syms = list(symbols) if symbols else load_nifty50_symbols()
    n0 = (
        float(notional_per_symbol)
        if notional_per_symbol is not None
        else default_notional_per_symbol(len(syms))
    )
    n_workers = workers if workers is not None else min(10, max(1, (os.cpu_count() or 4) - 2))

    progress(
        f"meta-train start years={years} symbols={len(syms)} "
        f"workers={n_workers} notional_per_symbol={n0:.2f} agg=fixed_notional"
    )

    try:
        returns = replay_universe_book_returns(
            syms,
            workers=n_workers,
            notional_per_symbol=n0,
            force_export=force_export,
            progress=progress,
        )
    except Exception as e:  # noqa: BLE001
        progress(f"replay failed: {e}; refusing synthetic for CLI")
        raise

    if returns is None or returns.empty:
        progress("empty returns panel — need backfill-history first")
        raise RuntimeError("empty strategy returns; run backfill-history")

    progress(f"returns rows={len(returns)} strategies={returns['strategy_id'].nunique()}")
    panel = build_meta_panel(returns, k=5)
    progress(f"panel rows={len(panel)} dates={panel['date'].nunique()}")
    out = ROOT / "data" / "store"
    manifest = train_meta_lgbm(panel, out_dir=out)
    manifest["symbols"] = len(syms)
    manifest["agg"] = "fixed_notional"
    manifest["notional_per_symbol"] = n0
    manifest["workers"] = n_workers
    manifest["cs_replay"] = "per_symbol_approx"
    manifest["replay_fees"] = True
    manifest["replay_fees_note"] = (
        "fee-table drag on exposure changes (STT/brokerage/etc); no bid/ask spread"
    )
    # Re-write manifest with universe metadata
    man_path = out / "experiments" / "meta_train" / "manifest.json"
    tmp = man_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str))
    os.replace(tmp, man_path)
    progress(
        f"done n_folds={manifest['n_folds']} mean_auc={manifest.get('mean_auc')} "
        f"symbols={len(syms)} agg=fixed_notional"
    )
    return manifest
