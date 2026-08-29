"""Phase 7/8 contract — backtest and forward paper."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.phase7


@pytest.mark.phase7
def test_backtest_produces_run_manifest(tmp_path):
    from experiments.backtest import run_backtest

    manifest = run_backtest(run_id="contract-test", out_dir=tmp_path)
    assert manifest["run_id"] == "contract-test"
    assert "config_hash" in manifest


@pytest.mark.phase7
def test_backtest_walk_forward_splits():
    from experiments.backtest import walk_forward_splits

    splits = walk_forward_splits(n_days=100, train_pct=0.7, n_folds=3)
    assert len(splits) == 3
    for train, test in splits:
        assert train.stop <= test.start


@pytest.mark.phase8
def test_paper_reconciles_state_after_restart(tmp_path):
    from experiments.paper import reconcile_state

    state_file = tmp_path / "portfolio.json"
    state_file.write_text('{"last_ts": "2026-08-10T10:00:00+05:30", "positions": {}}')
    state = reconcile_state(state_file)
    assert state["last_ts"]


@pytest.mark.phase8
def test_paper_run_one_day(tmp_path):
    from experiments.paper import run_paper_day

    result = run_paper_day(date="2026-08-10", out_dir=tmp_path)
    assert result["date"] == "2026-08-10"
    assert "trades" in result
