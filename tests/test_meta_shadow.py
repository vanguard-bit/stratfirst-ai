"""Shadow scorer writes JSON without mutating live weights semantics."""

from __future__ import annotations

import json

import pandas as pd

from meta.dataset import FEATURE_COLS
from meta.shadow import english_reason, run_meta_shadow
from meta.train_lgbm import make_synthetic_panel, train_meta_lgbm


def test_english_reason_mentions_drivers():
    text = english_reason(
        strategy_id="B2",
        score=0.74,
        in_top5=True,
        features={"ret_5d": 0.02, "vol_20d": 0.01},
        top_contribs=[
            {"feature": "ret_5d", "contribution": 0.12},
            {"feature": "vol_20d", "contribution": -0.05},
        ],
    )
    assert "B2" in text and "top-5" in text
    assert "5-day strategy return" in text
    assert "Drivers:" in text


def test_shadow_writes_json(tmp_path):
    panel = make_synthetic_panel(n_days=80, n_strat=8, seed=1)
    train_meta_lgbm(
        panel,
        out_dir=tmp_path,
        embargo_days=1,
        test_days=15,
        step_days=12,
        min_train_days=35,
    )
    model = tmp_path / "models" / "meta_lgbm_v0.txt"
    feats = tmp_path / "models" / "meta_lgbm_v0.features.json"
    assert model.exists()

    day = panel["date"].max()
    rows = panel[panel["date"] == day].copy()
    # Drop labels so we're closer to live feature-only rows
    feature_rows = rows[["strategy_id", *FEATURE_COLS]].copy()

    out = tmp_path / "meta_lgbm_shadow.json"
    hist = tmp_path / "meta_lgbm_shadow_history.parquet"
    jsonl = tmp_path / "meta_shadow_daily.jsonl"
    live_weights = tmp_path / "live_weights.json"
    live_weights.write_text(json.dumps({"A1": 0.5, "B1": 0.5}))

    payload = run_meta_shadow(
        as_of=day,
        feature_rows=feature_rows,
        model_path=model,
        features_path=feats,
        out_path=out,
        history_path=hist,
        jsonl_path=jsonl,
    )
    assert payload is not None
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded.get("mode") == "lightgbm_shadow"
    assert "scores" in loaded
    assert len(loaded["scores"]) == len(feature_rows)
    assert loaded.get("top5_reasons") and "Drivers:" in loaded["top5_reasons"][0]
    assert hist.exists()
    hdf = pd.read_parquet(hist)
    assert len(hdf) == len(feature_rows)
    assert "reason" in hdf.columns
    assert jsonl.exists()
    line = json.loads(jsonl.read_text().strip().splitlines()[-1])
    assert line["top5"] == loaded["top5"]
    # Live weights file untouched
    assert json.loads(live_weights.read_text()) == {"A1": 0.5, "B1": 0.5}

    # Same-day re-run replaces history rows (no duplicate as_of)
    run_meta_shadow(
        as_of=day,
        feature_rows=feature_rows,
        model_path=model,
        features_path=feats,
        out_path=out,
        history_path=hist,
        jsonl_path=jsonl,
    )
    hdf2 = pd.read_parquet(hist)
    assert len(hdf2) == len(feature_rows)


def test_feature_rows_fallback_when_as_of_past_last_labeled_day():
    """EOD as_of is often the last returns day; labeled panel drops that day."""
    from meta.shadow import feature_rows_for_shadow

    dates = pd.bdate_range("2026-01-02", periods=30).strftime("%Y-%m-%d")
    rows = []
    for d in dates:
        for sid in ("A1", "B1", "E1"):
            rows.append({"date": d, "strategy_id": sid, "ret": 0.01})
    returns = pd.DataFrame(rows)
    last = dates[-1]
    feats = feature_rows_for_shadow(returns, as_of=last)
    assert not feats.empty
    assert set(feats["strategy_id"]) == {"A1", "B1", "E1"}
    # Future/weekend as_of falls back to latest available
    feats2 = feature_rows_for_shadow(returns, as_of="2099-01-01")
    assert not feats2.empty


def test_shadow_none_without_model(tmp_path):
    rows = pd.DataFrame(
        {
            "strategy_id": ["S0"],
            **{c: [0.0] for c in FEATURE_COLS},
        }
    )
    assert (
        run_meta_shadow(
            feature_rows=rows,
            model_path=tmp_path / "missing.txt",
            out_path=tmp_path / "shadow.json",
        )
        is None
    )
