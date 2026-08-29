# Results summary (public, redacted)

Frozen offline bake-off and walk-forward training metrics for StratFirst AI.
**These numbers do not establish profitable alpha.** LightGBM remains shadow-only;
paper capital stays on the rules allocator.

Source artifacts (local, not committed): `meta_train/manifest.json` (written
2026-08-24), `meta_bakeoff/offline_metrics.json`, `offline_summary.json`.

## Training window

| Field | Value |
|---|---|
| Universe | Nifty 50 (50 symbols) |
| Walk-forward folds | 33 |
| Panel dates | 737 sessions |
| OOF policy days | 694 (2023-10-17 → 2026-08-10) |
| Label / costs | Fee-table drag in replay (`replay_fees: true`); no bid/ask in Cat-1 offline |
| Embargo | Expanding folds with gap between train end and test start (see fold rows in manifest) |
| Mean fold AUC | **0.629** |
| Mean top-five precision | **0.596** |

## Offline policy comparison (Cat-1)

| Sleeve | CAGR | Max DD | Sharpe | Notes |
|---|---:|---:|---:|---|
| `model_top5_eq` | +1.4% | −12.2% | 0.22 | Probability model top-five equal weight |
| `bh_nifty50` | +12.9% | −19.0% | 0.92 | Buy-and-hold Nifty 50 benchmark |
| `rules_proxy` | −7.6% | −24.8% | −0.92 | Rules allocator proxy on OOF days |
| `eq_all` | −32.7% | −66.3% | −15.2 | Equal all strategies (fee-crushed) |
| `rand5_eq` | −30.6% | −63.4% | −8.72 | Random five baseline |

Additional offline hits:

- P(model top-five beats random-one) ≈ **0.593**
- Hit rate vs equal-all ≈ **0.772**

## Forward sample (Cat-2)

Forward metrics currently cover only a **few sparse sessions** (`sparse: true`).
Treat them as smoke checks, not promotion evidence. That is why `meta_allocator.mode`
stays `rules` and LightGBM stays in shadow.

## With-LLM vs no-LLM weights

Live paper dumps dual weights daily (`weights` vs `weights_no_llm`). Judges can
reproduce the counterfactual offline:

```bash
python main.py demo
# open data/demo/dashboard.html
```

The offline demo uses a synthetic bullish LLM regime to show the tilt; it is not
a live news call and is not capital advice.

## Execution assumptions

- Paper fills only; no live broker order API.
- Measured bid/ask when present; missing or crossed quotes skip rather than invent slippage.
- Fee registry from official/broker tables (`refresh-fees --offline` for seed).
- MIS shorts intraday only; CNC long-only; circuit-locked exits may reject.
- Placeholder Fyers quotes never drive paper-live fills.

## Known limitations

- Model top-five underperforms the Nifty buy-and-hold on return and Sharpe in the frozen offline window.
- Offline Cat-1 has no spread friction; live paper uses measured quotes.
- Forward sample size is too small for promotion.
- See [`docs/paper/limitations-and-methods.md`](../paper/limitations-and-methods.md).

## Visual

See [`demo-dashboard.svg`](demo-dashboard.svg) for a public-safe illustration of the
offline demo report layout (no host paths, IPs, or account identifiers).
