"""Synthetic walk-forward train writes artifacts."""

from __future__ import annotations

from meta.train_lgbm import make_synthetic_panel, train_meta_lgbm


def test_train_writes_artifacts(tmp_path):
    panel = make_synthetic_panel(n_days=120, n_strat=10, seed=0)
    manifest = train_meta_lgbm(
        panel,
        out_dir=tmp_path,
        embargo_days=2,
        test_days=20,
        step_days=15,
        min_train_days=40,
    )
    assert (tmp_path / "models" / "meta_lgbm_v0.txt").exists()
    assert (tmp_path / "models" / "meta_lgbm_v0.features.json").exists()
    assert (tmp_path / "experiments" / "meta_train" / "manifest.json").exists()
    assert manifest["n_folds"] >= 1
    assert manifest.get("mean_auc") is not None
