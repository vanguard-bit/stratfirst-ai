# Paper notes — limitations, methods, claims

**Living register** for the research write-up. Update after material code/train changes.  
**Freeze** a dated copy under `docs/paper/freeze/` only after a **clean** post-audit meta-train.

**Related artifacts (do not duplicate):**

- Strategy audit: [`docs/strategy-audit-matrix.md`](../strategy-audit-matrix.md)
- Re-train gate: [`docs/meta-train-green-gate.md`](../meta-train-green-gate.md)
- Meta design: [`docs/superpowers/specs/2026-08-11-lightgbm-meta-bootstrap-design.md`](../superpowers/specs/2026-08-11-lightgbm-meta-bootstrap-design.md)
- Live paper design: [`docs/superpowers/specs/2026-08-11-live-multi-strategy-paper-design.md`](../superpowers/specs/2026-08-11-live-multi-strategy-paper-design.md)
- Train artifacts: `data/store/experiments/meta_train/`, `data/store/models/meta_lgbm_v0.*`
- Lab backlog: [`docs/lab-backlog.md`](../lab-backlog.md)

---

## 1. Claim (what the paper is)

**Working title framing:** Walk-forward bake-off of equal/rules allocation vs a LightGBM meta-selector over **21 hand-specified NSE equity strategies** in a **paper** (simulated) book.

**Research question:** Conditional on this strategy zoo and simulation assumptions, does a daily top-K meta-model improve next-day strategy selection vs simple baselines?

**Not claimed:** Live broker alpha, “AI that trades,” production allocator promotion, or that cluster-C is a true cross-sectional portfolio in v0 labels.

---

## 2. System (implementation context)

| Item | Choice |
|---|---|
| Venue / book | NSE cash equities; **paper only** (no live broker orders) |
| Universe | Nifty 50 list in `config/nifty50.yaml` (incl. TMPV post demerger, TRENT, etc.) |
| Ingest | Fyers websocket → 1m bars + measured spreads; placeholder path skips paper-live |
| Schedule | User systemd: ingest / paper / eod / llm / fyers-refresh (IST Mon–Fri) |
| Execution sim | `SimPipeline` + measured bid/ask friction when quotes valid; CNC overnight; MIS flat ~15:20 |
| Strategy routing | Same `on_bar` + `features.bar_state.build_state` for paper-live and historical replay |
| Positions | Keyed `strategy_id:symbol` — strategies do not exit each other’s CNC |

---

## 3. Strategy zoo (21)

Clusters A–G in `config/strategies.yaml` (trend, mean-reversion, cross-sectional, vol/regime, intraday, calendar, defensive).

- Full TF/product/state contract: audit matrix.
- Overlays (D1/D2, F3, G1/G2) often emit HOLD/FLAT with scaled exposure rather than standalone alpha engines.
- Audit P0/P1 (Donchian, lookback HOLD, ORB TF, etc.) fixed **2026-08-11**; any model trained on the **pre-fix** worker code is **not** paper-grade.

---

## 4. Features / state

Central builder: `features/bar_state.py`.

**Produced for contracts:** RSI/z-score, MAs, `ma_cross`, Donchian (prior window), returns, gap, ATR ratio, ORB, VWAP, calendar flags, crude CS ranks when multi-symbol frame present.

**Intentional stubs / injections (see §6):**

| Key | Behaviour |
|---|---|
| `adx` | Constant ≈22 until real ADX |
| `vix_above_median` | False in bare `build_state`; paper overlays regime |
| `beta_rank` | Proxied from momentum-style rank |
| `momentum_rank` / `reversion_rank` | Last-close rank proxy, not lookback return rank |
| `in_position` / `portfolio_drawdown` | Paper books / caller |

---

## 5. Replay, labels, meta

| Item | v0 choice |
|---|---|
| History | ~3y EOD (yfinance) + Fyers 1m chunks → DuckDB; parquet replay cache |
| Aggregation | Per-symbol replay → **fixed notional** book returns (`N0 = capital/50`) |
| Label | Binary: strategy in **top-5** by next-day book return |
| Model | LightGBM pointwise classifier; walk-forward folds + embargo |
| Live | **Shadow only**; allocator mode stays `rules` until explicit promotion |
| Costs in meta labels | Replay: close-to-close ± **fee registry** on trades; **no measured spread**. Live/measure still uses bid/ask. Re-train to bake fees into y |

**CS note:** Universe job is **per-symbol-then-aggregate**, not a single true cross-sectional backtest. Cluster C labels inherit that approximation.

---

## 6. Shortcomings (paper Limitations draft)

Update severity if fixed. Impact = effect on **interpretation of results**, not code quality alone.

| ID | Shortcoming | Severity | Impact on claim |
|---|---|---|---|
| L1 | **CS rank quality** — last-close / proxies; single-symbol replay weak for C1–C3/G3 | P2 | Cluster-C (and G3) meta signals partly noise; bake-off should report C separately or sensitivity without C |
| L2 | **ADX stub** — G1 rarely uses true ADX&lt;20 | P2 | “Trend-absent cash” ≈ MA filter; do not cite as ADX regime |
| L3 | **F2 half-FLAT** — paper flattens full qty on FLAT despite 0.5 exposure | P2 | Calendar reduce days overstated as exits in live paper path |
| L4 | **Replay speed O(n²)** — mitigated 2026-08-11 via vectorized `build_state_frame` in replay; live still uses `build_state` | P2→mitigated | Ops only for iteration speed; re-train after upgrade |

| L5 | **VIX offline** — D2 often inactive in pure replay | P2 | Vol-regime overlay under-represented in bootstrap labels |
| L6 | **Meta labels without full friction** | Partial | Replay subtracts **fee-table** drag (no bid/ask). `meta_lgbm_v0` retrained 2026-08-12 on fee-only panel. Live/measure still uses bid/ask |
| L7 | **No true CS portfolio simulation** | Design | Do not claim CS arb / long-short book fidelity |
| L8 | **Pre-audit train runs** | Process | Discard for paper; require green-gate re-train |
| L9 | **Measured quotes / circuits** — TMPV/TRENT gaps mitigated by `_resolve_quotes`. UC/LC: Fyers when >0 else prev_close±10% (`circuit_fallback_pct`); CIRCUIT_FLAT only on Fyers bands. MIS strat flat 15:15 (`MIS_STRAT_FLAT`); BROKER 15:20 leftovers |
| L10 | **Symbol events** — TMPV demerger continuity vs TATAMOTORS history | Data | State clearly in data section |

---

## 7. Threats to validity (checklist for Discussion)

- Lookahead / leakage in fold design (embargo days — cite manifest).
- Multiple testing across 21 strategies + meta.
- Regime dependence (2023–2026 sample).
- Liquidity / borrow (paper ignores hard-to-borrow / impact).
- Survivorship / index membership churn in Nifty50 list.
- Paper CNC inventory vs realistic capital and taxes.

---

## 8. What we will not claim

- LightGBM (or any strat) is live-trading ready.
- Results generalize outside this 50-name paper sim.
- Cluster C is a faithful cross-sectional strategy in v0.
- Meta labels equal net-of-cost live PnL.
- The in-flight / RELIANCE-only / pre-audit models are final.

---

## 9. Numbers log (current frozen train)

| Field | Value |
|---|---|
| Train finished (UTC) | 2026-08-24T13:32:53Z (`written_at` in manifest) |
| Code / audit freeze | post-audit + fee-table replay + overlay disables |
| Symbols / workers / notional | 50 / (default) / total_capital÷n_symbols |
| Panel | 737 dates, 21 strategies (13 enabled in live paper) |
| `n_folds` | 33 |
| `mean_auc` | **0.629** |
| `mean_top5_precision` | **0.596** |
| Manifest path | `data/store/experiments/meta_train/manifest.json` (`replay_fees: true`) |
| Model path | `data/store/models/meta_lgbm_v0.txt` |
| Bake-off vs BH (summary) | Cat1 OOF fee-net: `model_top5_eq` CAGR ≈+1.4% / MaxDD −12% / Sh 0.22 vs `bh_nifty50` +12.9% / −19% / 0.92; `eq_all` ≈−33% (high-turnover MIS crushed by fees); P(top5>rand1)≈0.59 |
| Notes | Fee-table only in Cat-1 (no spread). Earlier 2026-08-12 row (AUC 0.739 / top5 0.476) is **superseded** by this freeze. Still shadow-only. Public summary: [`docs/results/RESULTS.md`](../results/RESULTS.md). |

### Bake-off (two categories)

| Cat | What | Where |
|---|---|---|
| 1 Offline | OOF policies (top5_eq, score_all, eq, rand1, rand5, rules_proxy, oracle) + **metrics table** | `meta_bakeoff/offline_*.json`, `offline_metrics.{json,csv}` — **fee-table** (no spread) |
| 2 Forward | Measure ledger (unit-notional sim fills for all 21) + mark ML/rand/eq; rules capital separate + **metrics table** | EOD → `forward_daily.parquet`, `forward_metrics.{json,csv}`, `meta_bakeoff_glance.json` |

**Metrics table columns:** sleeve, CAGR, Max DD, Sharpe (daily, ann.), **Sharpe 5d**, n_days, trades, turnover.  
- Offline: trades always n/a; `model_top5_eq` turnover = **selection** (top-5 membership churn).  
- Forward: trades/turnover from `measure_fills` when present (`turnover_kind: notional`).  
- `sharpe_5d`: annualized Sharpe of **non-overlapping 5-day** compounded returns (needs ≥10 days / 2 blocks). Evaluation only — the model still predicts **t+1**.  
- `sparse: true` when n_days &lt; 20 (noisy CAGR/Sharpe). Equity = `(1+r).cumprod()`.
- If daily `|r|` is large (measure MTM artifacts / absolute INR), CAGR/Sharpe are skipped; MaxDD uses additive cumsum.

**Automation:** forward metrics refresh inside EOD `refresh_forward_and_glance` (no new timers). Offline rebuild is CLI-only after meta-train: `meta-bakeoff`.

**Armed checklist (unattended week):** timers on → Fyers healthy → pytest/`main.py test`/`diagnose` green → one smoke `forward_metrics` (`sparse` OK) → offline baseline once → glance warnings only known babysit risks.

**Quick check:** `python main.py meta-status`  
**Come-back:** `diagnose` → `meta-status` → dashboard metrics tab / `offline_metrics.csv` + `forward_metrics.csv`

---

## 10. Paper outline (when §9 filled)

1. Introduction — bake-off framing  
2. Related work — meta-labeling / mixture policies (cite AFML-style, arXiv-style refs as chosen)  
3. Data & universe  
4. Strategy specifications (point to audit; appendix table in appendix)  
5. Simulation assumptions (friction, CNC/MIS, aggregation)  
6. Meta-learning setup (labels, features, walk-forward)  
7. Results  
8. Limitations — lift §6  
9. Conclusion / future work (true CS, costs in labels, ADX, faster replay)

---

## Changelog

| Date | Change |
|---|---|
| 2026-08-11 | Initial scaffold; P2 shortcomings L1–L4 + design L5–L10; audit linked |
| 2026-08-11 | L4 mitigated: `features/state_series.build_state_frame` + O(n) replay walk; TF parquet cache 5m/15m/1H |
| 2026-08-11 | Metrics table offline+forward (CAGR/MaxDD/Sharpe/trades/turnover); EOD auto forward_metrics; diagnose hooks |
| 2026-08-12 | Added Sharpe 5d (non-overlapping 5-day compounds; eval only, model still t+1) |
| 2026-08-11 | Shadow journal: history parquet + JSONL + English reasons (pred_contrib drivers) |
| 2026-08-11 | Bake-off Cat1 offline + Cat2 measure ledger / glance / meta-status; EOD isolated |
| 2026-08-12 | Re-train `meta_lgbm_v0` on fee-only replay (`replay_fees: true`); interim mean_auc ≈0.739 / top5 ≈0.476 |
| 2026-08-24 | Current freeze: mean_auc 0.629 / top5 0.596; public RESULTS.md aligned |
| 2026-08-29 | Public-release packaging: offline `demo`, portable systemd templates, health metric daily rebalances |

