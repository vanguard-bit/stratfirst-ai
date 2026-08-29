from __future__ import annotations

import json

from meta.drawdown_kill import apply_strategy_drawdown_zero


def test_zeros_strategies_past_drawdown_threshold():
    weights = {"A1": 0.5, "E1": 0.5}
    out = apply_strategy_drawdown_zero(
        weights,
        cum_by_sid={"A1": 0.02, "E1": -0.20},
        threshold=0.15,
    )
    assert out["E1"] == 0.0
    assert abs(out["A1"] - 1.0) < 1e-9


def test_all_zeroed_falls_back_to_equal():
    weights = {"A1": 0.5, "B1": 0.5}
    out = apply_strategy_drawdown_zero(
        weights,
        cum_by_sid={"A1": -0.5, "B1": -0.5},
        threshold=0.15,
    )
    assert abs(out["A1"] - 0.5) < 1e-9
    assert abs(out["B1"] - 0.5) < 1e-9


def test_missing_cum_leaves_weight():
    weights = {"A1": 0.4, "B1": 0.6}
    out = apply_strategy_drawdown_zero(
        weights,
        cum_by_sid={"A1": -0.5},
        threshold=0.15,
    )
    assert out["A1"] == 0.0
    assert abs(out["B1"] - 1.0) < 1e-9


def test_loads_cum_from_glance_file(tmp_path):
    glance = tmp_path / "glance.json"
    glance.write_text(
        json.dumps(
            {
                "strategies": [
                    {"strategy_id": "A1", "cum": 0.01},
                    {"strategy_id": "E1", "cum": -0.5},
                ]
            }
        ),
        encoding="utf-8",
    )
    out = apply_strategy_drawdown_zero(
        {"A1": 0.5, "E1": 0.5},
        threshold=0.15,
        glance_path=glance,
    )
    assert out["E1"] == 0.0
    assert abs(out["A1"] - 1.0) < 1e-9
