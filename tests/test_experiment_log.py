"""Experiment logging contract."""

from __future__ import annotations

import pytest

from experiments.log import TRADE_LOG_COLUMNS, log_trades

pytestmark = pytest.mark.phase0


def test_trade_log_required_columns():
    required = {
        "trade_id", "ts", "strategy_id", "symbol", "side", "qty",
        "signal_price", "fill_price", "total_cost", "regime_adx",
    }
    assert required.issubset(set(TRADE_LOG_COLUMNS))


def test_log_trades_writes_parquet(tmp_path):
    path = log_trades(
        [{"trade_id": "t1", "ts": "2026-08-10", "strategy_id": "A1", "symbol": "X",
          "side": "BUY", "qty": 1, "signal_price": 100, "fill_price": 100.5,
          "total_cost": 1, "regime_adx": 22}],
        run_id="test-run",
        base_dir=tmp_path,
    )
    assert path.exists()
