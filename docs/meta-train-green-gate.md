# Meta-train green gate (post strategy audit)

After the in-flight meta-train job finishes or is stopped:

1. Confirm contract suite green:
   ```bash
   cd "$(dirname "$0")/.."   # project root
# or: cd /path/to/your/clone
   .venv/bin/python -m pytest \
     tests/test_strategy_contracts.py \
     tests/test_strategy_cluster_logic.py \
     tests/test_strategy_wiring_smoke.py \
     tests/test_state_series.py \
     tests/test_bar_state.py \
     tests/test_strategies.py -q
   ```
2. Confirm [docs/strategy-audit-matrix.md](strategy-audit-matrix.md) has **no open P0**.
3. Do **not** splice returns from the old (pre-fix / pre-fast-replay) run.
4. Re-export TF cache once (adds 5m/15m/1H parquet), then full universe train:
   ```bash
   .venv/bin/python main.py meta-train --years 3 --workers 10 --force-export
   ```
   Later runs can omit `--force-export` if bars unchanged.
5. Fast replay: `build_state_frame` (vectorized) — see `features/state_series.py`. Expect minutes-scale/symbol vs prior hour-scale.

Do not promote `meta_lgbm_v0` to live allocator weights until bake-off tables exist.
