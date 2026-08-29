# What broke—and how StratFirst AI recovered

StratFirst AI became useful through failures in data, execution assumptions, scheduling, and model evaluation. This document records the plumbing behind the project and the evidence that changed its design.

The recurring principle is:

> **When an input or model cannot be trusted, preserve the audit trail and reduce automation—not realism.**

## Pitch-ready answer

The most instructive failure happened near market close. Our one-shot Fyers websocket jobs normally completed, but several 15:29 runs hit the systemd timeout. The final post-market skip then succeeded, so a simple “last service result” health check reported green even though the daily strategies had never received their closing bar.

We traced the hang to the Fyers SDK shutdown path: `close_connection()` can wait indefinitely while joining websocket threads blocked in `recv()`. We wrapped shutdown in a bounded daemon thread, removed a redundant non-daemon keep-alive loop, increased the service’s explicit resource budget, and made daily strategies use the timestamp of the last persisted bar with a 15:29–15:35 catch-up window. Finally, diagnostics now scan the entire day’s journal for timeout events instead of trusting only the latest service result.

The lesson was bigger than a timeout: a trading-research system can look healthy while silently skipping its most important decision. Recovery required fixing the runtime, the market-time semantics, and the health signal together.

## Failure register

### 1. A broker secret is not a usable access token

**Symptom:** Ingest could not authenticate consistently, and a configured app secret was initially easy to mistake for a market-data access token.

**Root cause:** Fyers uses an approved OAuth login to mint a short-lived access token. Its refresh endpoint may return the SEBI-disabled `-16` response, so a naïve daily refresh is not dependable.

**Correction:**

- Added an explicit one-time `fyers-login` flow.
- Added proactive token-expiry checks before data calls.
- Added TOTP + PIN remint as the fallback when refresh is unavailable.
- Added one remint-and-retry attempt for authentication failures.
- Kept all credentials in `.env`, outside source control.

**Verification:** Authentication status and refresh logs expose field presence and token lifetime without printing token values.

### 2. Websocket shutdown blocked the whole market-close pipeline

**Symptom:** The 15:29 ingest service sometimes exhausted its timeout; paper-live did not evaluate the daily close.

**Root cause:** The Fyers SDK can perform unbounded thread joins during `close_connection()` while a socket thread remains blocked. A redundant keep-alive loop could also keep the process alive.

**Correction:**

- Wrapped close in `close_fyers_bounded(..., timeout_sec=2)`.
- Continued after the bounded wait using a daemon thread.
- Removed the redundant keep-alive path.
- Raised the one-shot systemd budget to cover websocket collection, bounded shutdown, persistence, and paper evaluation.

**Verification:** Dedicated tests simulate an SDK close that never returns and assert the caller regains control.

### 3. Wall-clock timing dropped daily signals

**Symptom:** A job started at 15:29 could finish after 15:30 and conclude that no 1D bar had just closed.

**Root cause:** Strategy scheduling used the process wall clock after a long ingest call instead of the timestamp of the persisted market bar.

**Correction:**

- Closed-timeframe detection now uses the latest bar timestamp.
- A 15:29–15:35 catch-up window allows a persisted daily close to be evaluated once.
- Idempotent day keys prevent duplicate processing.

**Verification:** Tests cover 15:29 close, delayed completion, catch-up, and pre-close rejection.

### 4. Placeholder and incomplete quotes looked too real

**Symptom:** Development needed an offline ingest path, but placeholder prices could contaminate paper evidence if treated like market quotes. Some live ticks also lacked valid bid/ask depth.

**Root cause:** Availability and validity are different properties. A numeric last price does not prove that a fill was executable.

**Correction:**

- Placeholder ingest remains available for plumbing tests.
- Paper-live runs only after a successful Fyers websocket result.
- Missing, crossed, or non-positive bid/ask quotes cause the symbol to be skipped.
- The simulator never invents a spread to make a trade pass.

**Verification:** Data, friction, and paper-live tests cover crossed markets, absent asks, and placeholder gating.

### 5. Circuit limits and EOD attribution were unrealistic

**Symptom:** Early paper fills used wide `ltp × 1.2 / 0.8` circuit bands, so circuit checks almost never bound. MIS positions flattened under a generic broker identity, hiding which strategy created the risk.

**Root cause:** Exchange constraints and operational square-off had been implemented as safety stubs rather than measurement-quality behaviour.

**Correction:**

- Prefer valid upper/lower circuit values from Fyers.
- Fall back to documented previous-close bands only when necessary.
- Never force a fill through a locked circuit.
- Flatten MIS positions under the owning strategy at 15:15.
- Retain a broker-style 15:20 square-off only for leftovers.

**Verification:** Tests cover buys and sells at upper/lower circuits, allowed and rejected exits, strategy attribution, and idempotent flattening.

### 6. Replay and paper-live disagreed about short selling

**Symptom:** Historical replay interpreted `SELL` as negative exposure for every product, while paper-live ignored `SELL` from a flat book. This trained the model on trades the live paper path could not reproduce.

**Root cause:** Signal semantics were shared, but product semantics were not.

**Correction:**

- MIS may open a short and must cover before the intraday cutoff.
- CNC remains long-only; `SELL` from flat cannot create overnight short exposure.
- Paper-live, measurement, and historical replay now apply the same product rules.
- The LightGBM model was retrained after parity corrections.

**Verification:** Replay and live tests assert MIS open-short/cover behaviour, CNC flatten-only behaviour, and end-of-day risk cuts.

### 7. “More strategies” produced worse evidence

**Symptom:** Several sleeves generated excessive turnover or deeply negative paper behaviour despite correct-looking strategy names.

**Root cause:** A known strategy family is not automatically a valid implementation, parameterization, or execution fit for this universe.

**Correction:**

- Kept all 21 implementations available for audit and replay.
- Disabled eight sleeves whose current evidence was weak or whose role was only an overlay.
- Retained explicit comments explaining why some intraday strategies were parked.
- Added drawdown-based zero weighting as an additional runtime guard.

**Verification:** Strategy wiring and contract tests still exercise all 21 implementations; the live registry loads only enabled sleeves.

### 8. Good classification metrics did not imply good investment results

**Symptom:** Walk-forward AUC and top-five precision were respectable, but the LightGBM top-five policy underperformed the Nifty benchmark on return and Sharpe.

**Root cause:** Predicting relative next-day winners is not the same as producing an economically superior, fully costed portfolio.

**Correction:**

- Added policy-level bake-offs against equal, random, rules, oracle, and Nifty baselines.
- Added fee-table drag, turnover, drawdown, daily Sharpe, and non-overlapping five-day Sharpe.
- Kept LightGBM in shadow mode.
- Added human-readable contribution reasons for shadow rankings.
- Made promotion an explicit decision rather than an automatic consequence of training.

**Verification:** The frozen artifacts report 33 walk-forward folds over 737 sessions; the model remains unpromoted because the economic evidence is insufficient.

### 9. News features were late, quota-sensitive, and failure-prone

**Symptom:** A 16:00 sentiment extract missed global and company information arriving overnight. Calling stock-level enrichment for the entire Nifty 50 would also exhaust provider quotas.

**Root cause:** The original job optimized for “after Indian close,” not “before the next allocation,” and did not treat external API calls as a budgeted resource.

**Correction:**

- Moved the primary extract to 08:40 IST before the first allocation.
- Added one FinStack morning brief and one Tapetide market-news call.
- Enforced maximums of two MCP calls/day and 40/month.
- Kept per-symbol Google News RSS as the uncapped fallback.
- Gemini failures return neutral features with an `extract_error` event.
- A successful morning extract invalidates the weight cache so the next paper tick recomputes with fresh context.

**Verification:** Budget, rollover, fail-soft, prompt-enrichment, and cache-bust behaviour have dedicated tests and append-only usage logs.

### 10. The last green service result hid earlier failures

**Symptom:** Diagnostics passed after a successful post-window skip even when multiple ingest runs had timed out earlier that day.

**Root cause:** `systemctl show` exposes the latest unit result, not a complete session history.

**Correction:** `diagnose` scans the current day’s user journal and counts timeout failures independently of the latest result.

**Verification:** Tests feed a mixed success/timeout journal and require the timeouts to remain visible.

## Plumbing that makes the experiment credible

| Concern | Implementation |
|---|---|
| Repeatable strategy interface | Shared `on_bar` contracts and required-state audit |
| Live/replay parity | Same state builder and explicit MIS/CNC semantics |
| Execution realism | Fees, measured spreads, quote validation, circuits, square-off |
| AI boundaries | Rules live; LightGBM shadow; Gemini meta-only |
| Counterfactual | Daily with-LLM and no-LLM weight vectors |
| External quotas | Persistent daily/monthly MCP budget |
| Crash containment | Per-strategy exception isolation and fail-soft enrichment |
| Scheduling | Short-lived user systemd jobs instead of a permanent process |
| Auditability | Parquet/JSON/JSONL artifacts, allocation snapshots, English reasons |
| Recovery | `diagnose --json` with findings, log tails, systemd state, and fix hints |

## What remains unresolved

- Forward evidence is still too short to establish economic uplift.
- Historical replay lacks complete measured bid/ask history.
- Cross-sectional replay is approximate.
- Some regime/ranking inputs use documented proxies.
- External headlines may be duplicated, stale, or incorrectly associated.
- A paper fill is still not proof of real-market capacity or execution.

These limitations are intentionally visible because hiding them would defeat the purpose of a gated research system.

