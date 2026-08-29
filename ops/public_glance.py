"""Redacted public glance for GitHub Pages — no paths, tokens, or books."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from nse_trader.config import ROOT, load_yaml

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger(__name__)

SITE_DIR = ROOT / "docs" / "site"
GLANCE_JSON = SITE_DIR / "glance.json"
INDEX_HTML = SITE_DIR / "index.html"
LOCAL_GLANCE = ROOT / "data" / "state" / "meta_bakeoff_glance.json"
WEIGHTS_CACHE = ROOT / "data" / "state" / "meta_weights_day.json"

_ABS_PATH = re.compile(r"(/home/|/Users/|[A-Za-z]:\\)")
_SECRETISH = re.compile(r"(eyJ[A-Za-z0-9_-]{10,}|AIza[0-9A-Za-z_-]{10,}|tpt_[A-Za-z0-9_]{10,})")


def _safe_str(value: Any) -> Any:
    if isinstance(value, str):
        if _ABS_PATH.search(value) or _SECRETISH.search(value):
            return "[redacted]"
        return value
    return value


def _frozen_offline() -> dict[str, Any]:
    """Judge-safe offline bake-off snapshot (matches docs/results/RESULTS.md)."""
    return {
        "source": "docs/results/RESULTS.md freeze",
        "mean_auc": 0.629,
        "mean_top5_precision": 0.596,
        "n_folds": 33,
        "oof_days": 694,
        "date_start": "2023-10-17",
        "date_end": "2026-08-10",
        "sleeves": [
            {"id": "model_top5_eq", "cagr": 0.014, "max_dd": -0.122, "sharpe": 0.22},
            {"id": "bh_nifty50", "cagr": 0.129, "max_dd": -0.190, "sharpe": 0.92},
            {"id": "rules_proxy", "cagr": -0.076, "max_dd": -0.248, "sharpe": -0.92},
            {"id": "eq_all", "cagr": -0.327, "max_dd": -0.663, "sharpe": -15.2},
            {"id": "rand5_eq", "cagr": -0.306, "max_dd": -0.634, "sharpe": -8.72},
        ],
        "p_top5_beats_rand1": 0.593,
        "hit_rate_vs_eq_all": 0.772,
        "disclaimer": "Offline fee-table costs only; not investment advice. LightGBM shadow-only.",
    }


def _redact_local_glance(raw: dict[str, Any]) -> dict[str, Any]:
    tracks = {}
    for name, block in (raw.get("tracks") or {}).items():
        if not isinstance(block, dict):
            continue
        tracks[str(name)] = {
            "today": block.get("today"),
            "cumulative": block.get("cum"),
        }
    clusters = {}
    for name, block in (raw.get("clusters") or {}).items():
        if not isinstance(block, dict):
            continue
        clusters[str(name)] = {
            "today": block.get("today"),
            "cumulative": block.get("cum"),
            "n": block.get("n"),
            "in_ml_top5": list(block.get("in_ml_top5") or [])[:5],
        }
    strategies = []
    for row in raw.get("strategies") or []:
        if not isinstance(row, dict):
            continue
        strategies.append(
            {
                "strategy_id": str(row.get("strategy_id")),
                "cluster": str(row.get("cluster")) if row.get("cluster") is not None else None,
                "today": row.get("today"),
                "cumulative": row.get("cum"),
            }
        )
    strategies = strategies[:21]
    warnings = []
    for w in raw.get("warnings") or []:
        warnings.append(_safe_str(str(w)))
    return {
        "as_of": _safe_str(raw.get("as_of")),
        "written_at": _safe_str(raw.get("written_at")),
        "ml_top5": [str(x) for x in (raw.get("ml_top5") or [])[:5]],
        "tracks": tracks,
        "clusters": clusters,
        "strategies": strategies,
        "warnings": warnings[:20],
    }


def _llm_dual_summary() -> dict[str, Any] | None:
    if not WEIGHTS_CACHE.exists():
        return None
    try:
        raw = json.loads(WEIGHTS_CACHE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    weights = {k: float(v) for k, v in (raw.get("weights") or {}).items()}
    no_llm = {k: float(v) for k, v in (raw.get("weights_no_llm") or {}).items()}
    if not weights and not no_llm:
        return None
    keys = set(weights) | set(no_llm)
    l1 = float(sum(abs(weights.get(k, 0.0) - no_llm.get(k, 0.0)) for k in keys))
    deltas = sorted(
        (
            {
                "strategy_id": sid,
                "delta": float(weights.get(sid, 0.0) - no_llm.get(sid, 0.0)),
            }
            for sid in keys
        ),
        key=lambda r: abs(r["delta"]),
        reverse=True,
    )[:5]
    return {
        "date": _safe_str(raw.get("date")),
        "l1_distance": l1,
        "top_deltas": deltas,
        "note": "Rules allocator with-LLM vs no-LLM counterfactual (paper only).",
    }


def _strategy_counts() -> dict[str, int]:
    cfg = load_yaml("strategies.yaml")
    strategies = cfg.get("strategies", {})
    enabled = sum(1 for m in strategies.values() if m.get("enabled", True))
    return {"implemented": len(strategies), "enabled": enabled}


def build_public_glance(*, include_local: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": 1,
        "project": "StratFirst AI",
        "generated_at": datetime.now(tz=IST).isoformat(),
        "meta_allocator_mode": "rules",
        "lightgbm": "shadow_only",
        "paper_only": True,
        "disclaimer": (
            "Paper research dashboard. Not investment advice. "
            "Synthetic or sparse forward figures are not evidence of live alpha."
        ),
        "counts": _strategy_counts(),
        "offline": _frozen_offline(),
        "forward_glance": None,
        "llm_dual": None,
        "links": {
            "repo": "https://github.com/vanguard-bit/stratfirst-ai",
            "results": "https://github.com/vanguard-bit/stratfirst-ai/blob/main/docs/results/RESULTS.md",
            "submission": "https://github.com/vanguard-bit/stratfirst-ai/blob/main/SUBMISSION.md",
        },
    }
    if include_local and LOCAL_GLANCE.exists():
        try:
            raw = json.loads(LOCAL_GLANCE.read_text(encoding="utf-8"))
            payload["forward_glance"] = _redact_local_glance(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("public glance: local glance read failed: %s", exc)
    payload["llm_dual"] = _llm_dual_summary()
    return payload


def render_index_html(payload: dict[str, Any] | None = None) -> str:
    """Static shell; live numbers come from glance.json next to this file."""
    _ = payload  # reserved for optional SSR later
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>StratFirst AI — public glance</title>
<style>
  :root { --bg:#0f1419; --fg:#e7ecf1; --muted:#9aa7b5; --acc:#3d9cf0; --warn:#fbbf24; }
  body { margin:0; font-family:"IBM Plex Sans","Segoe UI",sans-serif;
    background:linear-gradient(160deg,#0f1419,#1a2330 55%,#121820); color:var(--fg); }
  main { max-width:980px; margin:0 auto; padding:2rem 1.25rem 4rem; }
  h1 { font-size:1.7rem; margin:0 0 .35rem; letter-spacing:-.02em; }
  .sub,.muted { color:var(--muted); line-height:1.45; }
  .banner { margin:1.25rem 0; padding:.85rem 1rem; border:1px solid #334155;
    border-radius:8px; background:#1e293b88; color:var(--warn); }
  h2 { font-size:.95rem; color:var(--acc); text-transform:uppercase; letter-spacing:.06em; margin:1.6rem 0 .5rem; }
  table { width:100%; border-collapse:collapse; font-size:.92rem; }
  th,td { text-align:left; padding:.4rem .5rem; border-bottom:1px solid #243041; }
  th { color:var(--muted); font-weight:500; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
  @media (max-width:720px) { .grid { grid-template-columns:1fr; } }
  code { color:#93c5fd; }
  a { color:#93c5fd; }
  .err { color:#f07178; }
</style>
</head>
<body>
<main>
  <h1>StratFirst AI — public glance</h1>
  <p class="sub">Judge-safe paper research summary. Live ops stay on the private dashboard.</p>
  <div class="banner" id="disclaimer">Loading…</div>
  <p class="muted" id="meta"></p>

  <section>
    <h2>Offline bake-off (frozen)</h2>
    <div id="offline"></div>
  </section>

  <section class="grid">
    <div>
      <h2>Forward glance (redacted)</h2>
      <div id="forward"></div>
    </div>
    <div>
      <h2>LLM dual weights</h2>
      <div id="llm"></div>
    </div>
  </section>

  <section>
    <h2>Links</h2>
    <p id="links" class="muted"></p>
  </section>
</main>
<script>
async function load() {
  const el = (id) => document.getElementById(id);
  try {
    const res = await fetch('./glance.json', { cache: 'no-store' });
    if (!res.ok) throw new Error('glance.json HTTP ' + res.status);
    const g = await res.json();
    el('disclaimer').textContent = g.disclaimer || '';
    el('meta').innerHTML = [
      'Generated <code>' + (g.generated_at || '') + '</code>',
      'mode <code>' + (g.meta_allocator_mode || '') + '</code>',
      'LightGBM <code>' + (g.lightgbm || '') + '</code>',
      (g.counts ? (g.counts.enabled + '/' + g.counts.implemented + ' strategies enabled') : '')
    ].filter(Boolean).join(' · ');

    const off = g.offline || {};
    const sleeves = (off.sleeves || []).map(s =>
      '<tr><td>' + s.id + '</td><td>' + (s.cagr*100).toFixed(1) + '%</td><td>' +
      (s.max_dd*100).toFixed(1) + '%</td><td>' + Number(s.sharpe).toFixed(2) + '</td></tr>'
    ).join('');
    el('offline').innerHTML =
      '<p class="muted">AUC ' + off.mean_auc + ' · top5 precision ' + off.mean_top5_precision +
      ' · OOF days ' + off.oof_days + '</p>' +
      '<table><thead><tr><th>Sleeve</th><th>CAGR</th><th>MaxDD</th><th>Sharpe</th></tr></thead><tbody>' +
      sleeves + '</tbody></table>';

    const fg = g.forward_glance;
    if (!fg) {
      el('forward').innerHTML = '<p class="muted">No forward glance export yet (normal on clean clone).</p>';
    } else {
      const tracks = Object.entries(fg.tracks || {}).map(([k,v]) =>
        '<tr><td>' + k + '</td><td>' + (v.today==null?'—':Number(v.today).toFixed(4)) +
        '</td><td>' + (v.cumulative==null?'—':Number(v.cumulative).toFixed(4)) + '</td></tr>'
      ).join('');
      el('forward').innerHTML =
        '<p class="muted">as_of <code>' + (fg.as_of||'') + '</code> · shadow top5 <code>' +
        (fg.ml_top5||[]).join(', ') + '</code></p>' +
        '<table><thead><tr><th>Track</th><th>Today</th><th>Cumulative</th></tr></thead><tbody>' +
        tracks + '</tbody></table>';
    }

    const llm = g.llm_dual;
    if (!llm) {
      el('llm').innerHTML = '<p class="muted">No dual-weight dump for today.</p>';
    } else {
      const rows = (llm.top_deltas||[]).map(r =>
        '<tr><td>' + r.strategy_id + '</td><td>' + (r.delta>=0?'+':'') + Number(r.delta).toFixed(4) + '</td></tr>'
      ).join('');
      el('llm').innerHTML =
        '<p class="muted">L1 <code>' + Number(llm.l1_distance).toFixed(4) + '</code> · ' + (llm.note||'') + '</p>' +
        '<table><thead><tr><th>Strategy</th><th>Δ weight</th></tr></thead><tbody>' + rows + '</tbody></table>';
    }

    const L = g.links || {};
    el('links').innerHTML = [
      L.repo ? '<a href="' + L.repo + '">Repository</a>' : '',
      L.results ? '<a href="' + L.results + '">RESULTS.md</a>' : '',
      L.submission ? '<a href="' + L.submission + '">SUBMISSION.md</a>' : ''
    ].filter(Boolean).join(' · ');
  } catch (e) {
    el('disclaimer').innerHTML = '<span class="err">Failed to load glance.json: ' + e + '</span>';
  }
}
load();
</script>
</body>
</html>
"""


def export_public_glance(*, include_local: bool = True, site_dir: Path | None = None) -> dict[str, Any]:
    """Write ``docs/site/glance.json`` + ``index.html`` for GitHub Pages."""
    out = site_dir or SITE_DIR
    out.mkdir(parents=True, exist_ok=True)
    payload = build_public_glance(include_local=include_local)
    (out / "glance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "index.html").write_text(render_index_html(payload), encoding="utf-8")
    # GitHub Pages needs no jekyll sometimes for underscore files; keep clean.
    (out / ".nojekyll").write_text("", encoding="utf-8")
    glance_path = out / "glance.json"
    index_path = out / "index.html"
    try:
        payload["_paths"] = {
            "glance_json": str(glance_path.relative_to(ROOT)),
            "index_html": str(index_path.relative_to(ROOT)),
        }
    except ValueError:
        payload["_paths"] = {
            "glance_json": str(glance_path),
            "index_html": str(index_path),
        }
    return payload
