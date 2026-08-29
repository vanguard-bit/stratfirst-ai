"""Cat1 offline bake-off math + glance schema."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from experiments.meta_bakeoff import (
    build_glance,
    mark_policies_on_strat_day,
    run_offline_bakeoff,
)


def _tiny_oof() -> pd.DataFrame:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    strats = [f"S{i}" for i in range(6)]
    rows = []
    rng = np.random.default_rng(0)
    for d in dates:
        rets = rng.normal(0, 0.01, size=len(strats))
        # Make S0 always best forward ret
        rets[0] = 0.05
        scores = rng.random(len(strats))
        scores[1] = 0.99  # model prefers S1
        for i, sid in enumerate(strats):
            rows.append(
                {
                    "date": d,
                    "strategy_id": sid,
                    "ret_fwd": float(rets[i]),
                    "p": float(scores[i]),
                    "y": int(i == 0),
                    "w_rules": 1.0 / len(strats),
                    "w_eq": 1.0 / len(strats),
                    "w_score": float(scores[i]),
                }
            )
    return pd.DataFrame(rows)


def test_nifty50_buyhold_levels():
    from experiments.meta_bakeoff import nifty50_buyhold_from

    # Smoke against live store if present
    bh = nifty50_buyhold_from("2024-01-02", sell_dates=["2024-01-02", "2024-06-28", "2024-12-31"])
    if bh.empty:
        return
    assert "bh_nifty50" in bh.columns
    row0 = bh[bh["date"] == bh["buy_date"].iloc[0]]
    assert abs(float(row0["bh_nifty50"].iloc[0])) < 1e-9
    later = bh[bh["date"] > bh["buy_date"].iloc[0]]
    assert len(later) >= 1


def test_offline_bakeoff_math(tmp_path):
    oof = _tiny_oof()
    oof_path = tmp_path / "oof.parquet"
    oof.to_parquet(oof_path, index=False)
    out = tmp_path / "bakeoff"
    summary = run_offline_bakeoff(oof_path=oof_path, out_dir=out, seed=1)
    assert summary["n_days"] == 3
    assert (out / "offline_daily.parquet").exists()
    assert (out / "offline_metrics.json").exists()
    assert (out / "offline_metrics.csv").exists()
    daily = pd.read_parquet(out / "offline_daily.parquet")
    # rand1_E == eq_all
    assert np.allclose(daily["rand1_E"], daily["eq_all"])
    # oracle >= top5 mean each day
    assert (daily["oracle_best1"] >= daily["model_top5_eq"] - 1e-12).all()
    assert "model_top5_eq" in summary["policies"]
    assert summary.get("metrics_table", {}).get("n_rows", 0) >= 1
    mt = json.loads((out / "offline_metrics.json").read_text())
    assert mt.get("metrics_schema") == 2
    assert "sharpe_5d" in mt["rows"][0]
    for r in mt["rows"]:
        if r["block"] == "policy" and r["sleeve"] != "bh_nifty50":
            assert r["trades"] is None


def test_mark_policies_top5():
    strat = pd.DataFrame(
        {
            "strategy_id": ["A1", "B1", "C1", "D1", "E1", "F1"],
            "ret": [0.01, 0.02, -0.01, 0.0, 0.03, 0.005],
        }
    )
    m = mark_policies_on_strat_day(strat, top5=["B1", "E1"])
    assert abs(m["ml_top5_eq"] - (0.02 + 0.03) / 2) < 1e-12
    assert abs(m["eq_all"] - strat["ret"].mean()) < 1e-12
    assert abs(m["rand1_E"] - m["eq_all"]) < 1e-12


def test_glance_schema(tmp_path, monkeypatch):
    from experiments import meta_bakeoff as mb

    monkeypatch.setattr(mb, "STRAT_DAILY", tmp_path / "strat.parquet")
    monkeypatch.setattr(mb, "FORWARD_DAILY", tmp_path / "fwd.parquet")
    monkeypatch.setattr(mb, "_cluster_of", lambda: {"A1": "A", "A2": "A", "B1": "B"})
    monkeypatch.setattr(mb, "_load_shadow_top5", lambda day: (["A1"], ["A1 scored"]))
    weights = tmp_path / "meta_weights_day.json"
    weights.write_text(
        json.dumps(
            {
                "date": "2026-08-11",
                "llm": {"mean": 0.4, "high_n": 1, "as_of": "2026-08-11", "path": ""},
                "weights": {"A1": 0.6, "B1": 0.4},
                "weights_no_llm": {"A1": 0.5, "B1": 0.5},
                "weight_delta_top": [{"strategy_id": "A1", "delta": 0.1}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mb, "WEIGHTS_DAY", weights)
    strat = pd.DataFrame(
        {
            "date": ["2026-08-11", "2026-08-11", "2026-08-11"],
            "strategy_id": ["A1", "A2", "B1"],
            "cluster": ["A", "A", "B"],
            "ret": [0.02, 0.04, -0.02],
            "pnl": [200.0, 400.0, -200.0],
        }
    )
    strat.to_parquet(tmp_path / "strat.parquet", index=False)
    glance = build_glance(day="2026-08-11", strat_daily_path=tmp_path / "strat.parquet")
    for key in ("tracks", "clusters", "strategies", "ml_top5", "warnings", "llm_dual"):
        assert key in glance
    assert glance["ml_top5"] == ["A1"]
    assert abs(glance["clusters"]["A"]["today"] - 0.03) < 1e-12
    assert glance["clusters"]["A"]["n"] == 2
    assert glance["llm_dual"]["mean"] == 0.4
    assert glance["llm_dual"]["l1_distance"] == pytest.approx(0.2)
    text = mb.format_glance(glance)
    assert "llm_dual:" in text
    assert "delta: A1" in text

