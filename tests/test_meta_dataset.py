"""Leakage + schema checks for meta panel builder."""

from __future__ import annotations

import pandas as pd

from meta.dataset import FEATURE_COLS, build_meta_panel


def test_panel_shifts_label_to_next_day():
    # Day1 ranks by ret would put S1 first; day2 puts S0 first → y on day1 must use day2.
    returns = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 3 + ["2024-01-03"] * 3,
            "strategy_id": ["S0", "S1", "S2"] * 2,
            "ret": [0.01, 0.05, 0.02, 0.08, -0.01, 0.00],
        }
    )
    meta = {
        "S0": {"cluster": "A", "tf_code": 4},
        "S1": {"cluster": "B", "tf_code": 2},
        "S2": {"cluster": "A", "tf_code": 4},
    }
    regime = {
        "2024-01-02": {"regime_adx": 20.0, "regime_vix": 12.0},
        "2024-01-03": {"regime_adx": 22.0, "regime_vix": 13.0},
    }
    panel = build_meta_panel(returns, k=1, regime_by_date=regime, strategy_meta=meta)
    # Only day with a next day can get a label
    day1 = panel[panel["date"] == "2024-01-02"]
    assert not day1.empty
    assert float(day1.loc[day1["strategy_id"] == "S0", "ret_fwd"].iloc[0]) == 0.08
    assert int(day1.loc[day1["strategy_id"] == "S0", "y"].iloc[0]) == 1
    assert int(day1.loc[day1["strategy_id"] == "S1", "y"].iloc[0]) == 0
    # Features must not include future return
    assert "ret_fwd" not in FEATURE_COLS
    assert "y" not in FEATURE_COLS
    for c in FEATURE_COLS:
        assert c in panel.columns


def test_trailing_features_no_lookahead():
    returns = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "strategy_id": ["A"] * 3,
            "ret": [0.10, 0.20, 0.30],
        }
    )
    panel = build_meta_panel(returns, k=1)
    row = panel[panel["date"] == "2024-01-03"].iloc[0]
    # EOD features on t include that day's ret; label is next-day ret
    assert abs(float(row["ret_1d"]) - 0.20) < 1e-9
    assert abs(float(row["ret_fwd"]) - 0.30) < 1e-9
    assert "ret_fwd" not in FEATURE_COLS
