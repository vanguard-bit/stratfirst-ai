from __future__ import annotations

import pytest

from meta.allocator import AllocatorConstraints, MetaAllocator
from meta.features import RegimeFeatures, build_feature_vector

pytestmark = pytest.mark.phase0


class TestMetaAllocator:
    def test_equal_weight_sums_to_one(self):
        ids = ["A1", "B1", "G1"]
        clusters = {"A1": "A", "B1": "B", "G1": "G"}
        loose = AllocatorConstraints(
            max_strategy_weight=1.0,
            max_cluster_weight=1.0,
            min_cash=0.0,
            max_cash=1.0,
        )
        w = MetaAllocator(ids, clusters, constraints=loose, mode="equal_weight").allocate(
            RegimeFeatures()
        )
        assert pytest.approx(sum(w.values()), rel=1e-6) == 1.0

    def test_rules_boost_trend_when_adx_high(self):
        ids = ["A1", "B1", "G1"]
        clusters = {"A1": "A", "B1": "B", "G1": "G"}
        alloc = MetaAllocator(
            ids,
            clusters,
            constraints=AllocatorConstraints(max_strategy_weight=1.0, max_cluster_weight=1.0),
            mode="rules",
        )
        w_trend = alloc.allocate(RegimeFeatures(adx=30))
        w_chop = alloc.allocate(RegimeFeatures(adx=15))
        assert w_trend["A1"] > w_chop["A1"]

    def test_max_strategy_weight_cap(self):
        ids = ["A1", "A2", "A3"]
        clusters = {i: "A" for i in ids}
        w = MetaAllocator(ids, clusters, mode="equal_weight").allocate(RegimeFeatures())
        assert all(v <= 0.25 + 1e-9 for v in w.values())


class TestFeatures:
    def test_feature_vector_includes_regime(self):
        vec = build_feature_vector(RegimeFeatures(adx=25, vix=18), {})
        assert "regime_adx" in vec
        assert vec["regime_adx"] == 25
