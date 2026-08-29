"""Daily meta weights dual dump: live LLM vs zeroed LLM."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from meta.allocator import AllocatorConstraints
from meta.features import RegimeFeatures
from meta.regime import (
    l1_weight_distance,
    load_or_compute_daily_weights,
    weight_delta_top,
)

pytestmark = pytest.mark.phase6b


def _ids_clusters():
    ids = [f"s{i}" for i in range(6)]
    cluster = {sid: c for sid, c in zip(ids, list("AABBGG"))}
    return ids, cluster


def _loose():
    return AllocatorConstraints(
        max_strategy_weight=1.0,
        max_cluster_weight=1.0,
        min_cash=0.0,
        max_cash=1.0,
    )


def test_weight_delta_top_and_l1():
    a = {"x": 0.6, "y": 0.4}
    b = {"x": 0.5, "y": 0.5}
    top = weight_delta_top(a, b, k=2)
    assert top[0]["strategy_id"] in {"x", "y"}
    assert abs(top[0]["delta"]) == pytest.approx(0.1)
    assert l1_weight_distance(a, b) == pytest.approx(0.2)


def test_dual_dump_bull_differs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from meta import regime as regime_mod

    cache = tmp_path / "meta_weights_day.json"
    log = tmp_path / "meta_llm_dual.jsonl"
    monkeypatch.setattr(regime_mod, "WEIGHTS_CACHE", cache)
    monkeypatch.setattr(regime_mod, "META_LLM_DUAL_LOG", log)
    monkeypatch.setattr(
        "meta.drawdown_kill.apply_strategy_drawdown_zero",
        lambda w, **kwargs: w,
    )

    ids, cluster = _ids_clusters()
    regime = RegimeFeatures(
        adx=22.0,
        vix=12.0,
        vix_above_median=False,
        expiry_week=False,
        llm_sentiment_mean=0.4,
        llm_high_materiality=0,
        llm_as_of="2026-08-29",
    )
    tilt = {
        "bull_threshold": 0.25,
        "bear_threshold": -0.25,
        "bull_bump_A": 0.12,
        "bear_bump_G": 0.12,
        "high_materiality_min": 5,
        "high_bump_G": 0.05,
        "high_scale_E": 0.9,
    }
    w = load_or_compute_daily_weights(
        strategy_ids=ids,
        cluster_of=cluster,
        regime=regime,
        mode="rules",
        constraints=_loose(),
        today=date(2026, 8, 29),
        llm_tilt=tilt,
    )
    raw = json.loads(cache.read_text(encoding="utf-8"))
    assert "weights_no_llm" in raw
    assert raw["llm"]["mean"] == pytest.approx(0.4)
    assert w == {k: float(v) for k, v in raw["weights"].items()}
    assert raw["weights"] != raw["weights_no_llm"]
    assert raw["weight_delta_top"]
    assert log.exists()
    line = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert line["l1_distance"] > 0
    assert line["date"] == "2026-08-29"


def test_dual_dump_neutral_equal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from meta import regime as regime_mod

    cache = tmp_path / "meta_weights_day.json"
    log = tmp_path / "meta_llm_dual.jsonl"
    monkeypatch.setattr(regime_mod, "WEIGHTS_CACHE", cache)
    monkeypatch.setattr(regime_mod, "META_LLM_DUAL_LOG", log)
    monkeypatch.setattr(
        "meta.drawdown_kill.apply_strategy_drawdown_zero",
        lambda w, **kwargs: w,
    )

    ids, cluster = _ids_clusters()
    regime = RegimeFeatures(
        adx=22.0,
        vix=12.0,
        vix_above_median=False,
        expiry_week=False,
        llm_sentiment_mean=0.0,
        llm_high_materiality=0,
        llm_as_of="",
    )
    load_or_compute_daily_weights(
        strategy_ids=ids,
        cluster_of=cluster,
        regime=regime,
        mode="rules",
        constraints=_loose(),
        today=date(2026, 8, 29),
        llm_tilt={
            "bull_threshold": 0.25,
            "bear_threshold": -0.25,
            "bull_bump_A": 0.12,
            "bear_bump_G": 0.12,
            "high_materiality_min": 5,
            "high_bump_G": 0.05,
            "high_scale_E": 0.9,
        },
    )
    raw = json.loads(cache.read_text(encoding="utf-8"))
    assert raw["weights"] == raw["weights_no_llm"]
    assert l1_weight_distance(raw["weights"], raw["weights_no_llm"]) == 0.0
