"""Local PnL + EOD markers dashboard (static HTML, optional serve)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from nse_trader.config import ROOT

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger(__name__)

DASHBOARD_HTML = ROOT / "data" / "state" / "dashboard.html"
GLANCE_PATH = ROOT / "data" / "state" / "meta_bakeoff_glance.json"
OFFLINE_DAILY = ROOT / "data" / "store" / "experiments" / "meta_bakeoff" / "offline_daily.parquet"
FORWARD_DAILY = ROOT / "data" / "store" / "experiments" / "meta_bakeoff" / "forward_daily.parquet"
OFFLINE_SUMMARY = ROOT / "data" / "store" / "experiments" / "meta_bakeoff" / "offline_summary.json"
OFFLINE_METRICS = ROOT / "data" / "store" / "experiments" / "meta_bakeoff" / "offline_metrics.json"
FORWARD_METRICS = ROOT / "data" / "store" / "experiments" / "meta_bakeoff" / "forward_metrics.json"
EOD_GLOB = ROOT / "data" / "logs"


def _cum_series(daily: pd.DataFrame, col: str) -> list[dict[str, Any]]:
    if daily is None or daily.empty or col not in daily.columns:
        return []
    s = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)
    cum = s.cumsum()
    out = []
    for d, v in zip(daily["date"].astype(str), cum):
        out.append({"date": d, "v": float(v)})
    return out


def _level_series(daily: pd.DataFrame, col: str) -> list[dict[str, Any]]:
    """Already a level return (e.g. buyhold from fixed buy date) — do not cumsum."""
    if daily is None or daily.empty or col not in daily.columns:
        return []
    s = pd.to_numeric(daily[col], errors="coerce")
    out = []
    for d, v in zip(daily["date"].astype(str), s):
        if pd.isna(v):
            continue
        out.append({"date": d, "v": float(v)})
    return out


def _load_eod_markers(limit: int = 40) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(EOD_GLOB.glob("eod_*.json"), reverse=True)[:limit]
    for p in paths:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        shadow = payload.get("meta_shadow") or {}
        bake = payload.get("meta_bakeoff") or {}
        rows.append(
            {
                "date": payload.get("date") or p.stem.replace("eod_", ""),
                "created_at": payload.get("created_at"),
                "health_ok": payload.get("health_ok"),
                "n_trades": payload.get("n_trades"),
                "health_warnings": payload.get("health_warnings"),
                "health_errors": payload.get("health_errors"),
                "ml_top5": shadow.get("top5") or bake.get("ml_top5"),
                "shadow_error": shadow.get("error"),
                "bakeoff_error": (bake.get("error") if isinstance(bake, dict) else None),
                "paper_source": payload.get("paper_source"),
            }
        )
    rows.sort(key=lambda r: str(r.get("date")))
    return rows


def build_dashboard_payload() -> dict[str, Any]:
    glance: dict[str, Any] = {}
    if GLANCE_PATH.exists():
        try:
            glance = json.loads(GLANCE_PATH.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning("glance read failed: %s", e)

    offline_summary: dict[str, Any] = {}
    if OFFLINE_SUMMARY.exists():
        try:
            offline_summary = json.loads(OFFLINE_SUMMARY.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            offline_summary = {}

    offline_daily = pd.DataFrame()
    if OFFLINE_DAILY.exists():
        offline_daily = pd.read_parquet(OFFLINE_DAILY)

    forward_daily = pd.DataFrame()
    if FORWARD_DAILY.exists():
        forward_daily = pd.read_parquet(FORWARD_DAILY)

    offline_metrics: dict[str, Any] = {}
    if OFFLINE_METRICS.exists():
        try:
            offline_metrics = json.loads(OFFLINE_METRICS.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            offline_metrics = {}
    forward_metrics: dict[str, Any] = {}
    if FORWARD_METRICS.exists():
        try:
            forward_metrics = json.loads(FORWARD_METRICS.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            forward_metrics = {}

    series = {
        "offline": {
            "model_top5_eq": _cum_series(offline_daily, "model_top5_eq"),
            "eq_all": _cum_series(offline_daily, "eq_all"),
            "rand1_E": _cum_series(offline_daily, "rand1_E"),
            "rules_proxy": _cum_series(offline_daily, "rules_proxy"),
            "bh_nifty50": _level_series(offline_daily, "bh_nifty50"),
        },
        "forward": {
            "ml_top5_eq": _cum_series(forward_daily, "ml_top5_eq"),
            "eq_all": _cum_series(forward_daily, "eq_all"),
            "rand1_E": _cum_series(forward_daily, "rand1_E"),
            "rules_capital": _cum_series(forward_daily, "rules_capital"),
        },
    }

    return {
        "generated_at": datetime.now(tz=IST).isoformat(),
        "glance": glance,
        "offline_summary": offline_summary,
        "offline_metrics": offline_metrics,
        "forward_metrics": forward_metrics,
        "eod_markers": _load_eod_markers(),
        "series": series,
    }


def render_dashboard_html(payload: dict[str, Any] | None = None) -> str:
    payload = payload or build_dashboard_payload()
    data_json = json.dumps(payload, default=str)
    # Self-contained page; data embedded. Brand-first local ops surface.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NSE Trader — PnL &amp; EOD</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,560;9..144,700&family=IBM+Plex+Mono:wght@400;500&family=Schibsted+Grotesk:wght@400;600&display=swap" rel="stylesheet"/>
<style>
:root {{
  --bg: #0e1412;
  --bg2: #16201c;
  --ink: #e8f0ea;
  --muted: #8fa396;
  --line: #2a3a32;
  --ok: #3dba7a;
  --bad: #e07060;
  --warn: #d4a017;
  --ml: #7ec8a3;
  --rand: #6a8faf;
  --rules: #c4a574;
  --eq: #9aa39a;
}}
* {{ box-sizing: border-box; }}
html, body {{
  margin: 0; min-height: 100%;
  background:
    radial-gradient(1200px 600px at 10% -10%, #1a2e24 0%, transparent 55%),
    radial-gradient(900px 500px at 100% 0%, #1c2430 0%, transparent 50%),
    var(--bg);
  color: var(--ink);
  font-family: "Schibsted Grotesk", sans-serif;
}}
.wrap {{ max-width: 1100px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }}
.brand {{
  font-family: Fraunces, serif;
  font-weight: 700;
  font-size: clamp(2.4rem, 6vw, 3.6rem);
  letter-spacing: -0.03em;
  line-height: 1.05;
  margin: 0 0 0.35rem;
}}
.lede {{
  color: var(--muted);
  max-width: 36rem;
  font-size: 1.05rem;
  margin: 0 0 2rem;
}}
.meta {{
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.75rem;
  color: var(--muted);
  margin-bottom: 1.75rem;
}}
.hero-grid {{
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.75rem;
  margin-bottom: 1.75rem;
}}
@media (max-width: 900px) {{
  .hero-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}
.kpi {{
  background: linear-gradient(160deg, var(--bg2), #101816);
  border: 1px solid var(--line);
  padding: 1rem 1.1rem;
}}
.kpi .lbl {{
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}}
.kpi .val {{
  font-family: Fraunces, serif;
  font-size: 1.65rem;
  margin-top: 0.35rem;
}}
.kpi .sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.2rem; }}
.pos {{ color: var(--ok); }}
.neg {{ color: var(--bad); }}
section {{ margin: 2.5rem 0; }}
h2 {{
  font-family: Fraunces, serif;
  font-size: 1.35rem;
  font-weight: 560;
  margin: 0 0 0.35rem;
}}
.sec-lede {{ color: var(--muted); font-size: 0.92rem; margin: 0 0 1rem; }}
.chart-box {{
  background: var(--bg2);
  border: 1px solid var(--line);
  padding: 1rem;
  position: relative;
}}
svg.chart {{ width: 100%; height: 340px; display: block; }}
.chart-tip {{
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.72rem;
  color: var(--muted);
  min-height: 1.2rem;
  margin-top: 0.5rem;
}}
.legend {{
  display: flex; flex-wrap: wrap; gap: 1rem;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.72rem;
  margin-top: 0.5rem;
  color: var(--muted);
}}
.legend i {{ display: inline-block; width: 12px; height: 3px; margin-right: 6px; vertical-align: middle; }}
.eod-rail {{
  display: flex; gap: 0.5rem; overflow-x: auto; padding-bottom: 0.5rem;
}}
.eod-chip {{
  flex: 0 0 auto;
  min-width: 9.5rem;
  border: 1px solid var(--line);
  background: var(--bg2);
  padding: 0.75rem 0.85rem;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.72rem;
}}
.eod-chip .d {{ font-family: Fraunces, serif; font-size: 1rem; margin-bottom: 0.35rem; }}
.dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; }}
.dot.ok {{ background: var(--ok); }}
.dot.bad {{ background: var(--bad); }}
.dot.warn {{ background: var(--warn); }}
table {{
  width: 100%; border-collapse: collapse;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.78rem;
}}
th, td {{
  text-align: left; padding: 0.45rem 0.4rem;
  border-bottom: 1px solid var(--line);
}}
th {{ color: var(--muted); font-weight: 500; }}
.star {{ color: var(--ml); }}
.warns {{
  border-left: 3px solid var(--warn);
  padding: 0.5rem 0.85rem;
  color: var(--warn);
  font-size: 0.88rem;
  margin-top: 1rem;
}}
.tabs {{ display: flex; gap: 0.5rem; margin-bottom: 0.75rem; }}
.tab {{
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.72rem;
  border: 1px solid var(--line);
  background: transparent;
  color: var(--muted);
  padding: 0.35rem 0.7rem;
  cursor: pointer;
}}
.tab.active {{ color: var(--ink); border-color: var(--ml); }}
table.metrics td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
.sparse-tag {{ color: var(--warn); font-size: 0.75rem; margin-left: 0.5rem; }}
</style>
</head>
<body>
<div class="wrap">
  <h1 class="brand">NSE Trader</h1>
  <p class="lede">Paper meta bake-off — cumulative PnL tracks and end-of-day markers. Rules capital stays live; ML is shadow-marked until promote.</p>
  <div class="meta" id="gen"></div>
  <div class="hero-grid" id="kpis"></div>
  <div id="warns"></div>

  <section>
    <h2>Cumulative PnL</h2>
    <p class="sec-lede">Offline OOF (fee-table, no spread) vs forward measure marks when the ledger has days.</p>
    <div class="tabs">
      <button class="tab active" data-series="offline">Offline OOF</button>
      <button class="tab" data-series="forward">Forward (live)</button>
    </div>
    <div class="chart-box">
      <svg class="chart" id="chart" viewBox="0 0 1000 340" preserveAspectRatio="xMidYMid meet"></svg>
      <div class="chart-tip" id="chartTip">Hover the chart for date + values</div>
      <div class="legend" id="legend"></div>
    </div>
  </section>

  <section>
    <h2>Metrics table</h2>
    <p class="sec-lede">CAGR / Max DD / Sharpe (daily) / Sharpe 5d (less noisy eval horizon). Model still predicts t+1. Offline trades n/a; forward uses measure fills when present.</p>
    <div class="tabs">
      <button class="tab active" data-metrics="offline">Offline</button>
      <button class="tab" data-metrics="forward">Forward</button>
    </div>
    <div id="metricsSparse" class="sec-lede"></div>
    <table class="metrics" id="metrics"><thead>
      <tr><th>Sleeve</th><th>Block</th><th>n</th><th>CAGR</th><th>Max DD</th><th>Sharpe</th><th>Sh 5d</th><th>Trades</th><th>T/O</th></tr>
    </thead><tbody></tbody></table>
  </section>

  <section>
    <h2>EOD markers</h2>
    <p class="sec-lede">Each session’s health, trades, and ML top-5 when shadow ran.</p>
    <div class="eod-rail" id="eod"></div>
  </section>

  <section>
    <h2>Clusters &amp; strategies</h2>
    <p class="sec-lede">From latest glance — star = in ML top-5.</p>
    <div style="display:grid;grid-template-columns:1fr 1.4fr;gap:1.25rem">
      <div><table id="clusters"><thead><tr><th>Cluster</th><th>Today</th><th>Cumulative</th><th>ML</th></tr></thead><tbody></tbody></table></div>
      <div><table id="strats"><thead><tr><th></th><th>Strat</th><th>Cl</th><th>Today</th><th>Cumulative</th></tr></thead><tbody></tbody></table></div>
    </div>
  </section>
</div>
<script>
const DATA = {data_json};

function fmt(x) {{
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  const n = Number(x);
  const s = (n >= 0 ? "+" : "") + n.toFixed(5);
  return s;
}}
function cls(x) {{
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "";
  return Number(x) >= 0 ? "pos" : "neg";
}}

document.getElementById("gen").textContent =
  "generated " + (DATA.generated_at || "") +
  " · glance as_of " + ((DATA.glance && DATA.glance.as_of) || "—");

const tracks = (DATA.glance && DATA.glance.tracks) || {{}};
const off = (DATA.offline_summary && DATA.offline_summary.policies) || {{}};
const bh = (DATA.offline_summary && DATA.offline_summary.buyhold_nifty50) || {{}};
const kpis = [
  {{ label: "ML top5 (glance today)", val: tracks.ml_top5_eq && tracks.ml_top5_eq.today, sub: "cumulative " + fmt(tracks.ml_top5_eq && tracks.ml_top5_eq.cum) }},
  {{ label: "Rand1 E (glance)", val: tracks.rand1_E && tracks.rand1_E.today, sub: "cumulative " + fmt(tracks.rand1_E && tracks.rand1_E.cum) }},
  {{ label: "Rules capital", val: tracks.rules_capital && tracks.rules_capital.today, sub: "cumulative " + fmt(tracks.rules_capital && tracks.rules_capital.cum) }},
  {{ label: "OOF ML mean/day", val: off.model_top5_eq && off.model_top5_eq.mean_daily, sub: "vs rand1 " + fmt(off.rand1_E && off.rand1_E.mean_daily) }},
  {{ label: "Buy Nifty50 → sell end", val: bh.total_return, sub: (bh.buy_date || "?") + " → " + (bh.sell_date || "?") + " eq-wt hold" }},
];
document.getElementById("kpis").innerHTML = kpis.map(k => `
  <div class="kpi">
    <div class="lbl">${{k.label}}</div>
    <div class="val ${{cls(k.val)}}">${{fmt(k.val)}}</div>
    <div class="sub">${{k.sub}}</div>
  </div>`).join("");

const warns = (DATA.glance && DATA.glance.warnings) || [];
if (warns.length) {{
  document.getElementById("warns").innerHTML =
    `<div class="warns">${{warns.map(w => "· " + w).join("<br/>")}}</div>`;
}}

const COLORS = {{
  model_top5_eq: "#7ec8a3",
  ml_top5_eq: "#7ec8a3",
  eq_all: "#9aa39a",
  rand1_E: "#6a8faf",
  rules_proxy: "#c4a574",
  rules_capital: "#c4a574",
  bh_nifty50: "#e8c547",
}};

function shortDate(d) {{
  if (!d) return "";
  const p = String(d).slice(0, 10).split("-");
  if (p.length !== 3) return String(d).slice(0, 10);
  return p[0].slice(2) + "-" + p[1] + "-" + p[2];
}}

function niceTicks(min, max, count) {{
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0];
  if (min === max) {{
    const e = Math.abs(min) * 0.1 || 0.01;
    return [min - e, min, min + e];
  }}
  const span = max - min;
  const raw = span / Math.max(count - 1, 1);
  const pow = Math.pow(10, Math.floor(Math.log10(Math.abs(raw))));
  const err = raw / pow;
  const step = (err >= 5 ? 5 : err >= 2 ? 2 : 1) * pow;
  const start = Math.floor(min / step) * step;
  const ticks = [];
  for (let v = start; v <= max + step * 0.5; v += step) ticks.push(v);
  return ticks;
}}

let chartState = null;

function draw(seriesKey) {{
  const bundle = (DATA.series && DATA.series[seriesKey]) || {{}};
  const keys = Object.keys(bundle).filter(k => (bundle[k] || []).length > 1);
  const svg = document.getElementById("chart");
  const tip = document.getElementById("chartTip");
  svg.innerHTML = "";
  const legend = document.getElementById("legend");
  chartState = null;
  if (!keys.length) {{
    svg.innerHTML = `<text x="80" y="160" fill="#8fa396" font-size="14" font-family="IBM Plex Mono, monospace">No series yet for ${{seriesKey}}</text>`;
    legend.innerHTML = "";
    tip.textContent = "No data";
    return;
  }}
  const xs = bundle[keys[0]];
  let all = [];
  keys.forEach(k => all = all.concat(bundle[k].map(p => p.v)));
  const min = Math.min(...all, 0);
  const max = Math.max(...all, 0);
  const pad = (max - min) * 0.08 || 0.01;
  const y0 = min - pad, y1 = max + pad;
  const W = 1000, H = 340, L = 78, R = 24, T = 18, B = 52;
  const n = xs.length;
  const x = i => L + (i / Math.max(n - 1, 1)) * (W - L - R);
  const y = v => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);

  const NS = "http://www.w3.org/2000/svg";
  const el = (name, attrs) => {{
    const node = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    return node;
  }};

  // plot frame
  svg.appendChild(el("rect", {{
    x: L, y: T, width: W - L - R, height: H - T - B,
    fill: "none", stroke: "#2a3a32"
  }}));

  // Y grid + labels
  const yTicks = niceTicks(y0, y1, 6);
  yTicks.forEach(v => {{
    const yy = y(v);
    svg.appendChild(el("line", {{
      x1: L, x2: W - R, y1: yy, y2: yy,
      stroke: "#24332c", "stroke-width": "1"
    }}));
    const label = el("text", {{
      x: L - 8, y: yy + 3,
      fill: "#8fa396",
      "font-size": "11",
      "font-family": "IBM Plex Mono, monospace",
      "text-anchor": "end"
    }});
    label.textContent = (v >= 0 ? "+" : "") + v.toFixed(3);
    svg.appendChild(label);
  }});

  // zero emphasis
  if (y0 < 0 && y1 > 0) {{
    svg.appendChild(el("line", {{
      x1: L, x2: W - R, y1: y(0), y2: y(0),
      stroke: "#3a4f44", "stroke-dasharray": "4 4"
    }}));
  }}

  // X date ticks (start, end, ~even spaced)
  const xCount = Math.min(8, n);
  const xIdx = [];
  for (let t = 0; t < xCount; t++) {{
    xIdx.push(Math.round(t * (n - 1) / Math.max(xCount - 1, 1)));
  }}
  [...new Set(xIdx)].forEach(i => {{
    const xx = x(i);
    svg.appendChild(el("line", {{
      x1: xx, x2: xx, y1: T, y2: H - B,
      stroke: "#1f2b25", "stroke-width": "1"
    }}));
    svg.appendChild(el("line", {{
      x1: xx, x2: xx, y1: H - B, y2: H - B + 5,
      stroke: "#8fa396"
    }}));
    const label = el("text", {{
      x: xx, y: H - B + 20,
      fill: "#8fa396",
      "font-size": "11",
      "font-family": "IBM Plex Mono, monospace",
      "text-anchor": "middle"
    }});
    label.textContent = shortDate(xs[i].date);
    svg.appendChild(label);
  }});

  // axis titles
  const yTitle = el("text", {{
    x: 14, y: T + (H - T - B) / 2,
    fill: "#8fa396",
    "font-size": "11",
    "font-family": "IBM Plex Mono, monospace",
    transform: `rotate(-90 14 ${{T + (H - T - B) / 2}})`
  }});
  yTitle.textContent = "cumulative return";
  svg.appendChild(yTitle);
  const xTitle = el("text", {{
    x: L + (W - L - R) / 2, y: H - 8,
    fill: "#8fa396",
    "font-size": "11",
    "font-family": "IBM Plex Mono, monospace",
    "text-anchor": "middle"
  }});
  xTitle.textContent = "date (session)";
  svg.appendChild(xTitle);

  // series
  keys.forEach(k => {{
    const pts = bundle[k].map((p, i) => `${{x(i)}},${{y(p.v)}}`).join(" ");
    svg.appendChild(el("polyline", {{
      fill: "none",
      stroke: COLORS[k] || "#ccc",
      "stroke-width": "2.2",
      points: pts
    }}));
  }});

  // hover cursor
  const vline = el("line", {{
    x1: L, x2: L, y1: T, y2: H - B,
    stroke: "#d4a017", "stroke-width": "1", opacity: "0"
  }});
  svg.appendChild(vline);

  chartState = {{ bundle, keys, xs, x, y, L, R, T, B, W, H, n, vline, seriesKey }};

  const onMove = (evt) => {{
    if (!chartState) return;
    const rect = svg.getBoundingClientRect();
    const px = (evt.clientX - rect.left) / rect.width * W;
    let i = Math.round((px - L) / (W - L - R) * (n - 1));
    i = Math.max(0, Math.min(n - 1, i));
    const xx = x(i);
    vline.setAttribute("x1", xx);
    vline.setAttribute("x2", xx);
    vline.setAttribute("opacity", "0.9");
    const d = xs[i].date;
    const bits = keys.map(k => {{
      const row = bundle[k][i];
      const v = row ? row.v : NaN;
      return k + "=" + fmt(v);
    }});
    tip.textContent = shortDate(d) + " · " + bits.join(" · ");
  }};
  const onLeave = () => {{
    vline.setAttribute("opacity", "0");
    tip.textContent = "Hover the chart for date + values · "
      + shortDate(xs[0].date) + " → " + shortDate(xs[n - 1].date)
      + " (" + n + " sessions)";
  }};
  svg.onmousemove = onMove;
  svg.onmouseleave = onLeave;
  onLeave();

  legend.innerHTML = keys.map(k =>
    `<span><i style="background:${{COLORS[k] || '#ccc'}}"></i>${{k}}</span>`
  ).join("");
}}

document.querySelectorAll(".tab[data-series]").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".tab[data-series]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    draw(btn.dataset.series);
  }});
}});
draw("offline");

function pct(x) {{
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return (Number(x) * 100).toFixed(1) + "%";
}}
function num(x, d) {{
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return Number(x).toFixed(d === undefined ? 2 : d);
}}
function renderMetrics(key) {{
  const payload = key === "forward" ? (DATA.forward_metrics || {{}}) : (DATA.offline_metrics || {{}});
  const rows = payload.rows || [];
  const sparseEl = document.getElementById("metricsSparse");
  sparseEl.textContent = rows.length
    ? ("schema " + (payload.metrics_schema || "?") +
       (payload.sparse ? " · sparse (n_days < 20 — noisy CAGR/Sharpe)" : "") +
       " · " + rows.length + " rows")
    : "No metrics file yet — run meta-bakeoff (offline) or wait for EOD (forward).";
  const tb = document.querySelector("#metrics tbody");
  tb.innerHTML = rows.map(r => `
    <tr>
      <td>${{r.sleeve}}</td>
      <td>${{r.block}}</td>
      <td class="num">${{r.n_days ?? "—"}}</td>
      <td class="num">${{pct(r.cagr)}}</td>
      <td class="num ${{cls(r.max_dd)}}">${{pct(r.max_dd)}}</td>
      <td class="num">${{num(r.sharpe)}}</td>
      <td class="num">${{num(r.sharpe_5d)}}</td>
      <td class="num">${{r.trades == null ? "—" : Math.round(r.trades)}}</td>
      <td class="num">${{r.turnover == null ? "—" : num(r.turnover, 3)}}</td>
    </tr>`).join("") || `<tr><td colspan="9">—</td></tr>`;
}}
document.querySelectorAll(".tab[data-metrics]").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".tab[data-metrics]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    renderMetrics(btn.dataset.metrics);
  }});
}});
renderMetrics("offline");

document.getElementById("eod").innerHTML = (DATA.eod_markers || []).map(m => {{
  const ok = m.health_ok === true;
  const bad = m.health_ok === false;
  const dot = ok ? "ok" : (bad ? "bad" : "warn");
  const top = (m.ml_top5 || []).slice(0, 5).join(" ") || "—";
  return `<div class="eod-chip">
    <div class="d"><span class="dot ${{dot}}"></span>${{m.date}}</div>
    <div>trades ${{m.n_trades ?? "—"}} · w${{m.health_warnings ?? 0}}/e${{m.health_errors ?? 0}}</div>
    <div style="margin-top:0.35rem;color:#8fa396">ML ${{top}}</div>
  </div>`;
}}).join("") || `<div class="eod-chip"><div class="d">No EOD files</div></div>`;

const clusters = (DATA.glance && DATA.glance.clusters) || {{}};
document.querySelector("#clusters tbody").innerHTML = Object.keys(clusters).sort().map(cl => {{
  const b = clusters[cl];
  return `<tr><td>${{cl}}</td><td class="${{cls(b.today)}}">${{fmt(b.today)}}</td>
    <td class="${{cls(b.cum)}}">${{fmt(b.cum)}}</td>
    <td>${{(b.in_ml_top5 || []).join(" ") || "—"}}</td></tr>`;
}}).join("");

const strats = (DATA.glance && DATA.glance.strategies) || [];
document.querySelector("#strats tbody").innerHTML = strats.slice(0, 21).map(s => `
  <tr>
    <td class="star">${{s.in_ml_top5 ? "★" : ""}}</td>
    <td>${{s.strategy_id}}</td>
    <td>${{s.cluster}}</td>
    <td class="${{cls(s.today)}}">${{fmt(s.today)}}</td>
    <td class="${{cls(s.cum)}}">${{fmt(s.cum)}}</td>
  </tr>`).join("");
</script>
</body>
</html>
"""


def write_dashboard(path: Path | None = None) -> Path:
    path = Path(path or DASHBOARD_HTML)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = render_dashboard_html()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(path)
    return path


def serve_dashboard(*, host: str = "127.0.0.1", port: int = 18765) -> None:
    path = write_dashboard()
    root = path.parent

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/{path.name}"
    print(f"dashboard → {path}")
    print(f"open {url}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
