"""Offline judge-safe demo — no broker, Gemini, MCP, Tailscale, or network."""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from meta.allocator import AllocatorConstraints, MetaAllocator
from meta.drawdown_kill import apply_strategy_drawdown_zero
from meta.features import RegimeFeatures
from meta.regime import l1_weight_distance, weight_delta_top
from nse_trader.config import ROOT, load_yaml
from sim.exchange.circuits import circuits_or_open
from sim.exchange.rules import evaluate_order
from sim.friction.measured import Quote
from sim.orders import OrderIntent, OrderSide, OrderType, Product

IST = ZoneInfo("Asia/Kolkata")
DEMO_DIR = ROOT / "data" / "demo"
DEMO_SEED = 42


def _enabled_strategies() -> tuple[list[str], dict[str, str]]:
    cfg = load_yaml("strategies.yaml")
    strategies = cfg.get("strategies", {})
    enabled = [sid for sid, meta in strategies.items() if meta.get("enabled", True)]
    cluster_of = {sid: meta["cluster"] for sid, meta in strategies.items()}
    return sorted(enabled), cluster_of


def _constraints() -> AllocatorConstraints:
    port = load_yaml("portfolio.yaml")
    raw = (port.get("meta_allocator") or {}).get("constraints") or {}
    return AllocatorConstraints(
        max_strategy_weight=float(raw.get("max_strategy_weight", 0.25)),
        max_cluster_weight=float(raw.get("max_cluster_weight", 0.40)),
        min_cash=float(raw.get("min_cash", 0.05)),
        max_cash=float(raw.get("max_cash", 0.30)),
    )


def _llm_tilt() -> dict:
    port = load_yaml("portfolio.yaml")
    return dict((port.get("meta_allocator") or {}).get("llm_tilt") or {})


def _demo_regime(*, with_llm: bool) -> RegimeFeatures:
    return RegimeFeatures(
        adx=22.0,
        vix=14.5,
        vix_above_median=False,
        expiry_week=False,
        nifty_return_20d=0.012,
        llm_sentiment_mean=0.42 if with_llm else 0.0,
        llm_high_materiality=7 if with_llm else 0,
        llm_as_of=date(2026, 8, 28).isoformat() if with_llm else "",
    )


def _allocate(enabled: list[str], cluster_of: dict[str, str], *, with_llm: bool) -> dict[str, float]:
    alloc = MetaAllocator(
        strategy_ids=enabled,
        cluster_of=cluster_of,
        constraints=_constraints(),
        mode="rules",
        llm_tilt=_llm_tilt(),
    )
    return apply_strategy_drawdown_zero(alloc.allocate(_demo_regime(with_llm=with_llm)))


def _strategy_intents(enabled: list[str]) -> list[dict]:
    preferred = [
        ("A1", "RELIANCE", "BUY", Product.CNC),
        ("B1", "TCS", "SELL", Product.MIS),
        ("C1", "INFY", "BUY", Product.CNC),
        ("G3", "HDFCBANK", "FLAT", Product.CNC),
    ]
    out = []
    for sid, symbol, side, product in preferred:
        use_sid = sid if sid in enabled else enabled[0]
        out.append(
            {
                "strategy_id": use_sid,
                "symbol": symbol,
                "side": side,
                "product": product.value,
            }
        )
    return out


def _simulate_fills() -> list[dict]:
    """Exercise exchange safety: one accept, one circuit reject, one missing-quote reject."""
    cases: list[tuple[str, OrderIntent, Quote]] = [
        (
            "fill",
            OrderIntent(
                strategy_id="A1",
                symbol="RELIANCE",
                side=OrderSide.BUY,
                quantity=1,
                order_type=OrderType.MARKET,
                product=Product.CNC,
            ),
            Quote(symbol="RELIANCE", ltp=2500.0, bid=2499.5, ask=2500.5, prev_close=2480.0),
        ),
        (
            "circuit_reject",
            OrderIntent(
                strategy_id="B1",
                symbol="TCS",
                side=OrderSide.BUY,
                quantity=1,
                order_type=OrderType.MARKET,
                product=Product.MIS,
            ),
            Quote(
                symbol="TCS",
                ltp=4000.0,
                bid=3999.0,
                ask=4001.0,
                prev_close=3600.0,
                upper_ckt=4000.0,
                lower_ckt=3200.0,
            ),
        ),
        (
            "missing_quote_reject",
            OrderIntent(
                strategy_id="C1",
                symbol="INFY",
                side=OrderSide.SELL,
                quantity=1,
                order_type=OrderType.MARKET,
                product=Product.MIS,
            ),
            Quote(symbol="INFY", ltp=1500.0, bid=None, ask=None, prev_close=1490.0),
        ),
    ]
    results = []
    for label, intent, quote in cases:
        uc, lc = circuits_or_open(quote, prev_close=quote.prev_close, fallback_pct=0.10)
        result = evaluate_order(
            side=intent.side.value,
            order_type=intent.order_type.value,
            ltp=quote.ltp,
            uc=uc,
            lc=lc,
            bid=quote.bid,
            ask=quote.ask,
        )
        results.append(
            {
                "case": label,
                "symbol": intent.symbol,
                "side": intent.side.value,
                "product": intent.product.value,
                "allowed": bool(result.allowed),
                "reason": None if result.allowed else result.reason,
            }
        )
    return results


def _shadow_sample(enabled: list[str]) -> dict:
    ranked = list(enabled[:5])
    while len(ranked) < 5 and enabled:
        ranked.append(enabled[len(ranked) % len(enabled)])
    return {
        "mode": "sample_shadow_only",
        "disclaimer": (
            "Synthetic ranking for demo layout only — not model output or capital advice."
        ),
        "top5": ranked[:5],
        "as_of": date(2026, 8, 28).isoformat(),
    }


def _write_html(out_dir: Path, payload: dict) -> Path:
    weights = payload["weights_with_llm"]
    no_llm = payload["weights_no_llm"]
    deltas = payload["weight_deltas"]
    fills = payload["execution_cases"]
    shadow = payload["shadow_sample"]
    intents = payload["strategy_intents"]

    def _rows(d: dict[str, float], n: int = 8) -> str:
        items = sorted(d.items(), key=lambda kv: -kv[1])[:n]
        return "".join(f"<tr><td>{sid}</td><td>{w:.3f}</td></tr>" for sid, w in items)

    delta_rows = "".join(
        f"<tr><td>{r['strategy_id']}</td><td>{float(r['delta']):+.4f}</td></tr>" for r in deltas
    )
    fill_rows = "".join(
        f"<tr><td>{r['case']}</td><td>{r['symbol']}</td>"
        f"<td>{'ALLOW' if r['allowed'] else 'REJECT'}</td>"
        f"<td>{r['reason'] or '—'}</td></tr>"
        for r in fills
    )
    intent_rows = "".join(
        f"<tr><td>{r['strategy_id']}</td><td>{r['symbol']}</td>"
        f"<td>{r['side']}</td><td>{r['product']}</td></tr>"
        for r in intents
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>StratFirst AI — offline demo</title>
<style>
  :root {{ --bg:#0f1419; --fg:#e7ecf1; --muted:#9aa7b5; --acc:#3d9cf0; }}
  body {{ margin:0; font-family:"IBM Plex Sans", "Segoe UI", sans-serif;
    background:linear-gradient(160deg,#0f1419,#1a2330 55%,#121820); color:var(--fg); }}
  main {{ max-width:960px; margin:0 auto; padding:2.5rem 1.25rem 4rem; }}
  h1 {{ font-size:1.75rem; letter-spacing:-0.02em; margin:0 0 0.35rem; }}
  .sub {{ color:var(--muted); margin-bottom:1.75rem; line-height:1.45; }}
  .banner {{ border:1px solid #334155; background:#1e293b88; padding:0.85rem 1rem;
    border-radius:8px; margin-bottom:1.75rem; color:#fbbf24; }}
  section {{ margin:1.75rem 0; }}
  h2 {{ font-size:1.05rem; color:var(--acc); margin:0 0 0.6rem;
    text-transform:uppercase; letter-spacing:0.06em; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.92rem; }}
  th, td {{ text-align:left; padding:0.4rem 0.5rem; border-bottom:1px solid #243041; }}
  th {{ color:var(--muted); font-weight:500; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1.25rem; }}
  @media (max-width:720px) {{ .grid {{ grid-template-columns:1fr; }} }}
  code {{ color:#93c5fd; }}
</style>
</head>
<body>
<main>
  <h1>StratFirst AI — offline demo</h1>
  <p class="sub">Deterministic paper-research walkthrough. No broker, Gemini, MCP, or network calls.
  Generated {payload['generated_at']}.</p>
  <div class="banner">Synthetic demo figures are <strong>not</strong> investment evidence.
  LightGBM remains shadow-only; live paper capital stays on rules.</div>

  <section>
    <h2>Strategy sleeve sample</h2>
    <table><thead><tr><th>Strategy</th><th>Symbol</th><th>Intent</th><th>Product</th></tr></thead>
    <tbody>{intent_rows}</tbody></table>
  </section>

  <section class="grid">
    <div>
      <h2>Weights with LLM tilt</h2>
      <table><thead><tr><th>Strategy</th><th>Weight</th></tr></thead>
      <tbody>{_rows(weights)}</tbody></table>
    </div>
    <div>
      <h2>Weights without LLM</h2>
      <table><thead><tr><th>Strategy</th><th>Weight</th></tr></thead>
      <tbody>{_rows(no_llm)}</tbody></table>
    </div>
  </section>

  <section>
    <h2>Counterfactual deltas (with − without)</h2>
    <table><thead><tr><th>Strategy</th><th>Δ weight</th></tr></thead>
    <tbody>{delta_rows}</tbody></table>
    <p class="sub">L1 distance: <code>{payload['l1_distance']:.4f}</code></p>
  </section>

  <section>
    <h2>Execution safety cases</h2>
    <table><thead><tr><th>Case</th><th>Symbol</th><th>Result</th><th>Reason</th></tr></thead>
    <tbody>{fill_rows}</tbody></table>
  </section>

  <section>
    <h2>LightGBM shadow sample</h2>
    <p class="sub">{shadow['disclaimer']}</p>
    <p>Top-5 (sample): <code>{', '.join(shadow['top5'])}</code></p>
  </section>

  <section>
    <h2>Enabled / implemented</h2>
    <p>{payload['n_enabled']} enabled of {payload['n_implemented']} implemented strategies.
    Allocator mode: <code>rules</code>.</p>
  </section>
</main>
</body>
</html>
"""
    path = out_dir / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    return path


def run_demo(*, out_dir: Path | None = None) -> dict:
    """Run the offline demo into a disposable directory (default ``data/demo``)."""
    out = out_dir or DEMO_DIR
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    enabled, cluster_of = _enabled_strategies()
    cfg = load_yaml("strategies.yaml")
    n_implemented = len(cfg.get("strategies", {}))

    weights = _allocate(enabled, cluster_of, with_llm=True)
    no_llm = _allocate(enabled, cluster_of, with_llm=False)
    deltas = weight_delta_top(weights, no_llm, k=5)
    fills = _simulate_fills()
    shadow = _shadow_sample(enabled)
    intents = _strategy_intents(enabled)

    payload = {
        "generated_at": datetime.now(tz=IST).isoformat(),
        "seed": DEMO_SEED,
        "disclaimer": (
            "Offline synthetic demo. Not investment advice. "
            "Demo figures are not evidence of profitability."
        ),
        "n_implemented": n_implemented,
        "n_enabled": len(enabled),
        "enabled": enabled,
        "strategy_intents": intents,
        "weights_with_llm": weights,
        "weights_no_llm": no_llm,
        "weight_deltas": deltas,
        "l1_distance": l1_weight_distance(weights, no_llm),
        "execution_cases": fills,
        "shadow_sample": shadow,
        "meta_allocator_mode": "rules",
    }

    (out / "demo_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path = _write_html(out, payload)
    try:
        payload["dashboard_html"] = str(html_path.relative_to(ROOT))
        payload["report_json"] = str((out / "demo_report.json").relative_to(ROOT))
    except ValueError:
        payload["dashboard_html"] = str(html_path)
        payload["report_json"] = str(out / "demo_report.json")
    return payload
