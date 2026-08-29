"""Top-K label builder tests."""

from __future__ import annotations

import pandas as pd

from meta.labels import top_k_labels


def test_top5_labels_basic():
    df = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 7,
            "strategy_id": [f"S{i}" for i in range(7)],
            "ret": [0.01, 0.05, -0.02, 0.03, 0.02, 0.00, 0.04],
        }
    )
    out = top_k_labels(df, k=5)
    assert set(out.loc[out["y"] == 1, "strategy_id"]) == {"S1", "S6", "S3", "S4", "S0"}
    assert int(out["y"].sum()) == 5


def test_nan_excluded_from_rank():
    df = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 3,
            "strategy_id": ["A", "B", "C"],
            "ret": [0.1, float("nan"), -0.1],
        }
    )
    out = top_k_labels(df, k=1)
    assert list(out["strategy_id"]) == ["A", "C"]
    assert int(out.loc[out["strategy_id"] == "A", "y"].iloc[0]) == 1
    assert int(out.loc[out["strategy_id"] == "C", "y"].iloc[0]) == 0
