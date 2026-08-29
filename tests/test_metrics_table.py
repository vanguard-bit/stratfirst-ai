"""Unit tests for bake-off metrics table helpers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.metrics_table import (
    build_forward_metrics,
    build_offline_metrics,
    metrics_from_level,
    metrics_from_returns,
    selection_turnover,
    sharpe_nd,
)


def test_metrics_from_returns_known():
    # Constant +1% daily for 252 days → CAGR ~ (1.01^252 - 1), MaxDD ~ 0, Sharpe high
    r = pd.Series([0.01] * 252)
    m = metrics_from_returns(r, sleeve="const", block="policy")
    assert m["n_days"] == 252
    assert m["sparse"] is False
    assert abs(m["cagr"] - ((1.01**252) - 1.0)) < 1e-9
    assert abs(m["max_dd"]) < 1e-12
    assert m["sharpe"] is None  # zero std


def test_metrics_max_dd_and_sharpe():
    r = pd.Series([0.10, -0.05, -0.05, 0.02, 0.01])
    m = metrics_from_returns(r, sleeve="x", block="policy")
    assert m["max_dd"] is not None and m["max_dd"] < 0
    assert m["sharpe"] is not None
    assert m["sparse"] is True
    assert m["sharpe_5d"] is None  # need ≥2 complete 5d blocks


def test_sharpe_5d_less_noisy_than_daily():
    # Alternating +2% / -1% — daily Sharpe exists; 5d compounds too
    r = pd.Series([0.02, -0.01] * 30)  # 60 days → 12 blocks of 5
    m = metrics_from_returns(r, sleeve="alt", block="policy")
    assert m["sharpe"] is not None
    assert m["sharpe_5d"] is not None
    assert sharpe_nd(r, horizon=5) == m["sharpe_5d"]
    assert sharpe_nd(r.iloc[:9], horizon=5) is None


def test_bh_level_matches_compound_dd():
    level = pd.Series([0.0, 0.10, 0.05, 0.08])  # equity 1, 1.1, 1.05, 1.08
    m = metrics_from_level(level, sleeve="bh_nifty50")
    # Peak 1.1 → trough 1.05 → dd = 1.05/1.1 - 1
    assert abs(m["max_dd"] - (1.05 / 1.1 - 1.0)) < 1e-9


def test_selection_turnover():
    a = [{"A", "B", "C", "D", "E"}, {"A", "B", "C", "D", "E"}]
    assert selection_turnover(a) == 0.0
    b = [{"A", "B", "C", "D", "E"}, {"A", "B", "C", "D", "X"}]
    assert abs(selection_turnover(b) - 0.2) < 1e-12


def test_offline_metrics_bundle(tmp_path):
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    daily = pd.DataFrame(
        {
            "date": dates,
            "model_top5_eq": [0.01, 0.0, -0.005],
            "eq_all": [0.005, 0.001, 0.0],
            "rand1_E": [0.005, 0.001, 0.0],
            "rules_proxy": [0.002, 0.002, 0.002],
            "oracle_best1": [0.02, 0.01, 0.01],
            "bh_nifty50": [0.0, 0.01, 0.015],
        }
    )
    oof_rows = []
    for d in dates:
        for i, sid in enumerate(["S0", "S1"]):
            oof_rows.append(
                {
                    "date": d,
                    "strategy_id": sid,
                    "ret_fwd": 0.01 * (i + 1),
                    "p": 0.9 - 0.1 * i,
                }
            )
    oof = pd.DataFrame(oof_rows)
    payload = build_offline_metrics(daily, oof, out_dir=tmp_path)
    assert (tmp_path / "offline_metrics.json").exists()
    assert (tmp_path / "offline_metrics.csv").exists()
    sleeves = {r["sleeve"] for r in payload["rows"]}
    assert "model_top5_eq" in sleeves
    assert "bh_nifty50" in sleeves
    assert "S0" in sleeves
    for r in payload["rows"]:
        assert r["trades"] is None
    top5 = next(r for r in payload["rows"] if r["sleeve"] == "model_top5_eq")
    assert top5["turnover_kind"] == "selection"
    assert top5["turnover"] is not None
    assert 0.0 <= float(top5["turnover"]) <= 1.0


def test_forward_metrics_sparse_and_fills(tmp_path):
    fwd = pd.DataFrame(
        {
            "date": ["2026-08-10", "2026-08-11"],
            "ml_top5_eq": [0.01, -0.002],
            "eq_all": [0.0, 0.001],
            "rand1_E": [0.0, 0.001],
            "rand5_eq": [0.0, 0.0],
            "rules_capital": [np.nan, 100.0],
            "ml_top5": [json.dumps(["A1", "B1"]), json.dumps(["A1", "C1"])],
        }
    )
    strat = pd.DataFrame(
        {
            "date": ["2026-08-10", "2026-08-10", "2026-08-11", "2026-08-11"],
            "strategy_id": ["A1", "B1", "A1", "B1"],
            "ret": [0.01, -0.01, 0.0, 0.02],
        }
    )
    fills = pd.DataFrame(
        {
            "date": ["2026-08-10", "2026-08-11"],
            "strategy_id": ["A1", "B1"],
            "qty": [1, 2],
            "fill_price": [100.0, 50.0],
            "side": ["BUY", "SELL"],
            "total_cost": [1.0, 1.0],
        }
    )
    fwd.to_parquet(tmp_path / "forward_daily.parquet", index=False)
    strat.to_parquet(tmp_path / "bakeoff_strat_daily.parquet", index=False)
    fills.to_parquet(tmp_path / "measure_fills.parquet", index=False)
    payload = build_forward_metrics(
        out_dir=tmp_path,
        fills_path=tmp_path / "measure_fills.parquet",
    )
    assert payload["sparse"] is True
    sleeves = {r["sleeve"] for r in payload["rows"]}
    assert "ml_top5_eq" in sleeves
    assert "A1" in sleeves
    a1 = next(r for r in payload["rows"] if r["sleeve"] == "A1")
    assert a1["trades"] == 1.0
    assert a1["turnover_kind"] == "notional"
