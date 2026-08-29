"""LLM feature store + rules tilt side-channel."""

from __future__ import annotations

from pathlib import Path

import pytest

from features.llm_gemini import extract_features_offline_sample, write_features_parquet
from features.llm_store import inject_symbol_llm, load_llm_map, summarize_llm
from meta.features import RegimeFeatures
from meta.rules_v0 import rules_v0_weights

pytestmark = pytest.mark.phase6b


def test_load_llm_map_and_summary(tmp_path: Path):
    rows = extract_features_offline_sample(["RELIANCE", "TCS"], as_of="2026-08-11")
    rows[0]["sentiment"] = 0.5
    rows[0]["materiality"] = "high"
    rows[1]["sentiment"] = -0.1
    write_features_parquet(rows, tmp_path / "features" / "llm_2026-08-11.parquet")

    m = load_llm_map(tmp_path)
    assert set(m) == {"RELIANCE", "TCS"}
    assert m["RELIANCE"]["sentiment"] == 0.5
    s = summarize_llm(m)
    assert s.n_symbols == 2
    assert s.high_materiality == 1
    assert abs(s.mean_sentiment - 0.2) < 1e-9

    state: dict = {}
    inject_symbol_llm(state, "RELIANCE", m)
    assert state["llm_sentiment"] == 0.5
    assert state["llm_materiality"] == "high"


def _neutral_regime(**kwargs) -> RegimeFeatures:
    base = dict(
        adx=22.0,
        vix=12.0,
        vix_above_median=False,
        expiry_week=False,
        llm_sentiment_mean=0.0,
        llm_high_materiality=0,
    )
    base.update(kwargs)
    return RegimeFeatures(**base)


def test_rules_llm_bullish_tilt():
    ids = [f"s{i}" for i in range(6)]
    cluster = {sid: c for sid, c in zip(ids, list("AABBGG"))}
    w0 = rules_v0_weights(ids, cluster, _neutral_regime())
    w1 = rules_v0_weights(ids, cluster, _neutral_regime(llm_sentiment_mean=0.4))
    assert w1["s0"] + w1["s1"] > w0["s0"] + w0["s1"]


def test_rules_llm_bearish_tilt():
    ids = [f"s{i}" for i in range(6)]
    cluster = {sid: c for sid, c in zip(ids, list("AABBGG"))}
    w0 = rules_v0_weights(ids, cluster, _neutral_regime())
    w1 = rules_v0_weights(ids, cluster, _neutral_regime(llm_sentiment_mean=-0.4))
    assert w1["s4"] + w1["s5"] > w0["s4"] + w0["s5"]


def test_rules_llm_subthreshold_no_sentiment_bump():
    ids = [f"s{i}" for i in range(6)]
    cluster = {sid: c for sid, c in zip(ids, list("AABBGG"))}
    w0 = rules_v0_weights(ids, cluster, _neutral_regime())
    w1 = rules_v0_weights(ids, cluster, _neutral_regime(llm_sentiment_mean=0.1))
    for sid in ids:
        assert abs(w1[sid] - w0[sid]) < 1e-12


def test_rules_llm_high_materiality():
    ids = [f"s{i}" for i in range(8)]
    cluster = {sid: c for sid, c in zip(ids, list("AABBGGEE"))}
    w0 = rules_v0_weights(ids, cluster, _neutral_regime())
    w1 = rules_v0_weights(ids, cluster, _neutral_regime(llm_high_materiality=5))
    assert w1["s4"] + w1["s5"] > w0["s4"] + w0["s5"]
    assert w1["s6"] + w1["s7"] < w0["s6"] + w0["s7"]


def test_rules_llm_tilt_config_override():
    ids = [f"s{i}" for i in range(6)]
    cluster = {sid: c for sid, c in zip(ids, list("AABBGG"))}
    bull = _neutral_regime(llm_sentiment_mean=0.4)
    soft = {
        "bull_threshold": 0.25,
        "bear_threshold": -0.25,
        "bull_bump_A": 0.08,
        "bear_bump_G": 0.08,
        "high_materiality_min": 5,
        "high_bump_G": 0.05,
        "high_scale_E": 0.9,
    }
    hard = {**soft, "bull_bump_A": 0.20}
    w_soft = rules_v0_weights(ids, cluster, bull, llm_tilt=soft)
    w_hard = rules_v0_weights(ids, cluster, bull, llm_tilt=hard)
    assert w_hard["s0"] + w_hard["s1"] > w_soft["s0"] + w_soft["s1"]


def test_missing_llm_is_neutral(tmp_path: Path):
    assert load_llm_map(tmp_path) == {}
    s = summarize_llm({})
    assert s.mean_sentiment == 0.0
    assert s.high_materiality == 0
