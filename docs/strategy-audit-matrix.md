# 21-Strategy Implementation Audit Matrix

**As of:** 2026-08-11  
**Scope:** Config ↔ code, `build_state` keys, warmup/exits, product, paper vs replay.  
**P0/P1 fixed in this pass;** P2 deferred (noted).

## Legend

| Severity | Meaning |
|---|---|
| OK | Passes contract for next meta-train |
| P0 | Would poison labels / false always-BUY or always-SELL — fixed |
| P1 | Wrong wiring / unused params — fixed or mitigated |
| P2 | Known limitation; do not block re-train |

## Summary table

| ID | Name | TF | Product | State contract | Notes |
|---|---|---|---|---|---|
| A1 | time_series_momentum | 1D | CNC | OK | HOLD until `returns_lookback_ready` (was always-SELL on short hist) |
| A2 | dual_ma_crossover | 1H | MIS | OK | Warmup HOLD; needs `ma_cross` |
| A3 | donchian_breakout | 1D | CNC | OK | Prior-window Donchian; HOLD on NaN/warmup |
| B1 | bollinger_zscore | 15m | MIS | OK | Exit needs `in_position` (paper sets; replay False → exit rare) |
| B2 | rsi2_revert | 5m | MIS | OK | Uses `rsi` (=RSI2) |
| B3 | gap_fade | 1D | MIS | OK | `gap_pct` from open vs prior close |
| C1 | momentum_rank | 1W | CNC | OK* | *Rank = last-close CS proxy in single-symbol replay (P2) |
| C2 | low_vol_anomaly | 1W | CNC | OK* | *`vol_quintile` needs multi-symbol frame (P2 in 1-sym replay) |
| C3 | mean_reversion_rank | 1W | CNC | OK* | *Same CS approx as C1 |
| D1 | vol_target_overlay | 1D | CNC | OK | Scales via `realized_vol` |
| D2 | vix_regime_filter | 1D | CNC | OK* | *`vix_above_median` False in pure `build_state`; paper overlays (P2 replay) |
| D3 | atr_expansion_breakout | 1H | MIS | OK | `atr_ratio` + `breakout_dir` |
| E1 | opening_range_breakout | 5m | MIS | OK | ORB bar count TF-aware (was `head(15)` on 5m = 75m) |
| E2 | vwap_reversion | 15m | MIS | OK | `vwap_dev` |
| E3 | power_hour_momentum | 5m | MIS | OK | Power hour from 14:15–15:19 IST |
| F1 | turn_of_month | 1D | CNC | OK | Uses `day_of_month` + params `days_before`/`days_after` |
| F2 | day_of_week | 1D | CNC | OK* | *FLAT `intended_exposure=0.5` still full-flatten in paper (P2) |
| F3 | expiry_week_effect | 1D | CNC | OK | `expiry_week` from calendar |
| G1 | trend_absent_cash | 1D | CNC | OK* | *`adx` stub ≈22 (P2); FLAT mainly via `close_vs_ma200` |
| G2 | drawdown_circuit_breaker | 1D | CNC | OK | `portfolio_drawdown` caller-supplied |
| G3 | low_beta_basket | 1W | CNC | OK* | *`beta_rank` proxied from momentum rank (P2) |

## Yaml ↔ code

All 21: `timeframe` / `product` / `cluster` match after registry applies YAML metadata onto instances (`strategies/registry.py`).

## Shared path

- Paper-live and replay both call `features.bar_state.build_state` + `strat.on_bar`.
- Paper injects: `in_position`, regime `adx`/`vix_above_median`/`expiry_week`.
- Positions keyed `strategy_id:symbol` — no cross-strategy exits.
- CNC: no 15:20 square-off. MIS: square-off when `hhmm >= intraday_flat_time`.

## P0/P1 fixes applied

1. Donchian prior window + A3 HOLD on warmup/NaN (prior session).
2. A1 HOLD until 252d lookback ready.
3. ORB length scales with TF (5m → 3 bars for 15 minutes).
4. Power hour starts 14:15 not 14:00.
5. F1 respects `days_before` / `days_after` via `day_of_month`.
6. Registry copies YAML tf/product/cluster onto strategy objects.
7. `returns_lookback_ready`, `day_of_month` added to `build_state`.

## Open P2 (do not block meta-train)

- True cross-sectional momentum/vol/beta ranks (multi-name, lookback) in replay.
- Real ADX series.
- F2 half-FLAT sizing in paper-live.
- Replay O(n²) `build_state` performance.
- D2 VIX in offline replay (always False unless injected).

## Green gate for next meta-train

- [x] `REQUIRED_KEYS` ⊆ warm `build_state` keys (contract tests)
- [x] Per-cluster entry/exit/warmup tests
- [x] Non-degenerate synthetic smoke per sid
- [x] No open P0 in this matrix
- [ ] Kill or ignore in-flight meta-train (old code); **full** re-replay + train with this tree
- [ ] Bar cache reuse (`--force-export` off unless schema change)
