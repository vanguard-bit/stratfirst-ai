"""Zero strategy weight when glance cumulative return breaches drawdown threshold."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from nse_trader.config import ROOT, load_yaml

logger = logging.getLogger(__name__)

DEFAULT_GLANCE = ROOT / "data" / "state" / "meta_bakeoff_glance.json"


def load_glance_cum(path: Path | None = None) -> dict[str, float]:
    p = path or DEFAULT_GLANCE
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("glance read failed for drawdown kill: %s", exc)
        return {}
    out: dict[str, float] = {}
    for row in raw.get("strategies") or []:
        sid = row.get("strategy_id")
        if not sid:
            continue
        try:
            out[str(sid)] = float(row.get("cum", 0.0))
        except (TypeError, ValueError):
            continue
    return out


def drawdown_zero_threshold(portfolio: dict | None = None) -> float:
    port = portfolio if portfolio is not None else load_yaml("portfolio.yaml")
    risk = port.get("risk") or {}
    return float(risk.get("strategy_drawdown_zero", 0.15))


def apply_strategy_drawdown_zero(
    weights: dict[str, float],
    *,
    cum_by_sid: dict[str, float] | None = None,
    threshold: float | None = None,
    glance_path: Path | None = None,
) -> dict[str, float]:
    """
    Set weight to 0 when glance `cum` <= -threshold, then renormalize.
    If every strategy is zeroed (or weights empty), fall back to equal weight.
    Missing glance / missing sid → leave weight unchanged.
    """
    if not weights:
        return {}
    thr = float(threshold) if threshold is not None else drawdown_zero_threshold()
    cum = cum_by_sid if cum_by_sid is not None else load_glance_cum(glance_path)
    out = {sid: float(w) for sid, w in weights.items()}
    for sid in list(out):
        if sid not in cum:
            continue
        if cum[sid] <= -thr:
            out[sid] = 0.0
    total = sum(max(w, 0.0) for w in out.values())
    n = len(out)
    if total <= 0:
        eq = 1.0 / n
        return {sid: eq for sid in out}
    return {sid: max(w, 0.0) / total for sid, w in out.items()}
