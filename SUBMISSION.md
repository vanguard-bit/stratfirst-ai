# StratFirst AI — Buildathon submission

**Track:** Razorpay AI Buildathon — Open Track  
**Repo name (recommended):** `stratfirst-ai`  
**Public repository:** https://github.com/vanguard-bit/stratfirst-ai  
**Pitch video:** _TBD — unlisted ≤5 min link_

## Problem (≈120 words)

Retail and student builders are flooded with “AI trading” demos that ask a model to invent trades or weight stocks directly. When results go wrong, a non-expert cannot tell whether the strategy, data, execution assumptions, or the model failed. Opaque systems also encourage premature live deployment. The Open Track needs an honest alternative: an explainable multi-strategy research lab where AI is allowed only behind hard gates, paper execution is realistic, and weak evidence is refused promotion—so a novice can run a disciplined experiment without unproven models touching money.

## Solution (≈130 words)

StratFirst AI implements 21 systematic NSE strategy sleeves (13 enabled) and routes capital with a deterministic rules meta-allocator. Gemini compresses headlines into bounded sentiment features; LightGBM ranks next-day top-five strategies in **shadow only**. Daily dual dumps compare with-LLM versus no-LLM weights. A measured friction simulator applies fees, spreads, circuits, and MIS/CNC rules—never live broker orders. Offline bake-offs show the model reducing drawdown versus some baselines while underperforming Nifty buy-and-hold on return/Sharpe; that underperformance is why shadow gating stays on. Judges can clone the repo and run `python main.py demo` with no credentials.

## Architecture

Strategies produce intents → rules allocator (regime + optional Gemini tilt) → paper simulator. Historical replay trains LightGBM with walk-forward folds and embargo; predictions never move the paper book while `meta_allocator.mode=rules`. See README Mermaid diagram and `docs/FAILURE_RECOVERY.md`.

## Measured evidence

| Metric | Value |
|---|---:|
| Mean fold AUC | 0.629 |
| Mean top-five precision | 0.596 |
| Model top-five CAGR / MaxDD / Sharpe | +1.4% / −12.2% / 0.22 |
| Nifty 50 BH CAGR / MaxDD / Sharpe | +12.9% / −19.0% / 0.92 |
| Forward sample | sparse (few days) — not promotion-ready |

Full write-up: [`docs/results/RESULTS.md`](docs/results/RESULTS.md).

## What broke and how we recovered

Primary application story: Fyers websocket `close_connection()` hung at the cash close, so systemd killed the oneshot **before** end-of-day 1D paper-live catch-up. Fix: bounded close in a daemon thread with timeout, plus wall-clock catch-up for missed daily bars. Lesson: third-party cleanup must not own your process lifetime. Secondary: model underperformance vs benchmark kept LightGBM in shadow—AI judgment as refusal.

## Demo commands

```bash
# Prefer uv (fast); or: python -m venv .venv && pip install -r requirements.txt
uv sync && source .venv/bin/activate
python main.py refresh-fees --offline
python main.py test
python main.py demo
# open data/demo/dashboard.html
```

Optional live paper (credentials required): see README.

## Future plan (summary)

Remain paper-only until pre-registered forward evidence clears economic and ops gates (see README “Future plan”). SEBI-relevant simulator rules improve honesty of fills; they are not a go-live. Next concrete steps: lock deps with `uv`, grow non-sparse forward samples, true CS replay, spread-aware labels, then human-gated promotion of shadow ML — broker sandbox only after that.
