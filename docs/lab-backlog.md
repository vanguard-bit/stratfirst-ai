# Lab backlog (paper / experiment)

Not a trading-machine roadmap. Items stay **shadow / lab** unless explicitly promoted.  
Live allocator stays `rules`. No new systemd timers for research trains.

**Now (this week):** leave t+1 shadow + forward bake-off running; come-back via `diagnose` → `meta-status` → dashboard.

---

## Next (after a handful of clean t+1 forward days)

- [ ] **5-day label meta (shadow arm)** — train “top-5 over t+1…t+5”, do **not** overwrite `meta_lgbm_v0`
  - Label: top-5 by sum/compound of strategy `ret` over the next 5 sessions
  - Walk-forward embargo **≥ 5 days** (overlapping 5d labels leak otherwise)
  - New artifact: e.g. `data/store/models/meta_lgbm_h5d_v0.*` + own train manifest
  - Bake-off: **hold-5** policy (do not daily-reshuffle a 5d model)
  - Compare to current t+1 model on the same OOF window (CAGR / Max DD / Sharpe / Sh 5d)
  - Gate: only start after Cat2 `forward_daily` has enough real (non-placeholder) days to not mix two experiments
  - Status: **queued** — 2026-08-12

- [ ] **New sleeve: overnight close→open** (buy near close, sell near next open)
  - Name TBD (e.g. `F4` calendar or new id); **CNC only** (MIS cannot hold overnight)
  - Opposite of **B3 gap_fade** (B3 fades the open; this *harvests* the overnight gap)
  - Spec: enter last minutes / closing auction proxy; exit next open / post-auction
  - Honesty: NSE pre-open auction + STT + weekend/event gap risk; replay must use true open vs close, not close-to-close
  - Lab first: replay + metrics table row vs `bh_nifty50`; do not auto-enable in live weights
  - Status: **queued** — 2026-08-12

- [ ] **1m bars / last-minute path**
  - Today: ingest ticks mostly close only `1m`; **no enabled strategy is 1m**, so paper-live often logs `signals=0` between 5m/15m/1H/1D closes (by design, not a dead book)
  - Lab options (pick explicitly, don’t do all):
    - (A) Overnight sleeve entry via **last N minutes** on 1m (ties to close→open item above)
    - (B) Optional 1m strategy id(s) for lab only — replay + metrics row; do not auto-weight live
    - (C) Diagnostics: count non-HOLD actions on higher-TF closes so “quiet 1m ticks” aren’t misread as no-signal day
  - Honesty: 1m replay is expensive (already dominates universe train); measured spreads on 1m still incomplete
  - Status: **queued** — 2026-08-12

- [x] **1D close miss (15:29)** — paper-live used wall clock after ~45s ingest, so `1D` never fired Wed/Thu. Now `closed_timeframes_for_tick` unions last 1m bar + 15:29–15:35 catch-up (TMPV/TRENT fill path).
- [x] **MIS strat flat + real UC/LC** — 2026-08-15: `MIS_STRAT_FLAT` @15:15 under strat id; BROKER @15:20 leftovers; Fyers UC/LC or prev_close±10%; `CIRCUIT_FLAT` only on Fyers bands.

---

## Later / parked

- [ ] Dual ML **capital** book (only if marked ML beats rules + rand1 on forward)
- [x] Re-train meta on **fee-only replay** panel — 2026-08-12 `meta_lgbm_v0` (`replay_fees: true`; no bid/ask)
- [ ] Measured spreads in replay (only where `friction_spreads` exist; no guessed bps)
- [ ] True cross-sectional replay for cluster C (today: per-symbol-then-aggregate)
- [ ] Real ADX / VIX (stubs in `build_state`)
- [ ] Promote meta `mode` off `rules` — **explicit decision only**

---

## Done (recent)

- [x] Offline + forward metrics table (CAGR / Max DD / Sharpe / trades / turnover)
- [x] Sharpe 5d **evaluation** column (model still t+1)
- [x] Forward bake-off reset; live track starts next EOD
- [x] EOD auto `forward_metrics`; no new timers
- [x] Replay **fee-only** drag (registry STT/brokerage; no spread) + MIS flatten at date change
