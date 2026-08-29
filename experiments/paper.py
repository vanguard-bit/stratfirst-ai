from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.allocation_log import log_allocation_snapshot
from experiments.log import log_trades
from meta.allocator import MetaAllocator
from meta.features import RegimeFeatures
from nse_trader.config import PortfolioConfig, ROOT, load_yaml
from sim.friction.measured import Quote
from sim.orders import OrderIntent, OrderSide, OrderType, Product
from sim.pipeline import SimPipeline
from strategies.registry import load_enabled_strategies

IST = ZoneInfo("Asia/Kolkata")

DEFAULT_STATE: dict = {
    "last_ts": None,
    "positions": {},
    "virtual_books": {},
}


def _state_dir() -> Path:
    raw = load_yaml("ops.yaml")
    rel = raw.get("persistence", {}).get("state_dir", "data/state")
    return ROOT / rel


def _cluster_of() -> dict[str, str]:
    cfg = load_yaml("strategies.yaml")
    return {sid: meta["cluster"] for sid, meta in cfg["strategies"].items()}


def reconcile_state(state_file: Path) -> dict:
    """Reload checkpoint on startup; normalize schema and persist."""
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    else:
        state = dict(DEFAULT_STATE)

    state.setdefault("positions", {})
    state.setdefault("virtual_books", {})
    state["reconciled_at"] = datetime.now(tz=IST).isoformat()

    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def _meta_rebalance() -> dict[str, float]:
    portfolio = load_yaml("portfolio.yaml")
    meta_cfg = portfolio.get("meta_allocator", {})
    constraints_raw = meta_cfg.get("constraints", {})
    from meta.allocator import AllocatorConstraints

    constraints = AllocatorConstraints(
        max_strategy_weight=float(constraints_raw.get("max_strategy_weight", 0.25)),
        max_cluster_weight=float(constraints_raw.get("max_cluster_weight", 0.40)),
        min_cash=float(constraints_raw.get("min_cash", 0.05)),
        max_cash=float(constraints_raw.get("max_cash", 0.30)),
    )
    strategies = load_enabled_strategies()
    ids = list(strategies.keys())
    cluster_of = _cluster_of()
    alloc = MetaAllocator(
        ids,
        cluster_of,
        constraints=constraints,
        mode=meta_cfg.get("mode", "equal_weight"),
    )
    regime = RegimeFeatures(adx=22.0, vix=16.0, vix_above_median=False, expiry_week=False)
    from meta.drawdown_kill import apply_strategy_drawdown_zero

    weights = apply_strategy_drawdown_zero(alloc.allocate(regime))
    log_allocation_snapshot(
        weights,
        cluster_of=cluster_of,
        regime={"adx": regime.adx, "vix": regime.vix, "expiry_week": regime.expiry_week},
        mode=meta_cfg.get("mode", "equal_weight"),
    )
    return weights


def _demo_paper_trade(pipeline: SimPipeline, strategy_id: str) -> dict | None:
    """Single illustrative fill when pipeline + registry available."""
    quote = Quote("RELIANCE", ltp=2500.0, bid=2499.5, ask=2500.5)
    cfg = load_yaml("strategies.yaml")
    product_name = cfg["strategies"].get(strategy_id, {}).get("product", "CNC")
    product = Product.MIS if product_name == "MIS" else Product.CNC
    intent = OrderIntent(
        strategy_id=strategy_id,
        symbol="RELIANCE",
        side=OrderSide.BUY,
        quantity=1,
        order_type=OrderType.MARKET,
        product=product,
    )
    result = pipeline.process(intent, quote=quote, uc=3000.0, lc=2000.0)
    if result.status != "FILLED" or result.fill is None:
        return None
    cluster = cfg["strategies"][strategy_id]["cluster"]
    return {
        "trade_id": str(uuid.uuid4()),
        "ts": datetime.now(tz=IST).isoformat(),
        "strategy_id": strategy_id,
        "cluster": cluster,
        "symbol": "RELIANCE",
        "side": "BUY",
        "qty": 1,
        "signal_price": result.fill.signal_price,
        "fill_price": result.fill.fill_price,
        "total_cost": result.fill.charges.total,
        "regime_adx": 22.0,
        "regime_vix": 16.0,
        "expiry_flag": False,
    }


def run_paper_day(
    date: str,
    out_dir: Path | None = None,
    *,
    state_file: Path | None = None,
) -> dict:
    """Run one forward paper session — meta rebalance, optional sim fills, EOD state."""
    cfg = PortfolioConfig.load()
    out = out_dir or (cfg.store_path / "experiments" / "paper")
    out.mkdir(parents=True, exist_ok=True)

    state_path = state_file or (out / "portfolio.json")
    state = reconcile_state(state_path)

    _meta_rebalance()

    trades: list[dict] = []
    registry = cfg.fees_registry_path
    if registry.exists():
        pipeline = SimPipeline(registry_path=registry)
        for sid in list(load_enabled_strategies())[:1]:
            row = _demo_paper_trade(pipeline, sid)
            if row:
                trades.append(row)

        # MIS EOD square-off on open paper positions
        eod_state = {"positions": state.get("positions", {})}
        square_offs = pipeline.end_of_day(
            datetime.fromisoformat(f"{date}T15:20:00+05:30"),
            eod_state,
        )
        for action in square_offs:
            trades.append(
                {
                    "trade_id": str(uuid.uuid4()),
                    "ts": f"{date}T15:20:00+05:30",
                    "strategy_id": "BROKER",
                    "cluster": "G",
                    "symbol": action.symbol,
                    "side": action.side,
                    "qty": action.quantity,
                    "signal_price": 0.0,
                    "fill_price": 0.0,
                    "total_cost": 0.0,
                    "regime_adx": 22.0,
                    "regime_vix": 16.0,
                    "expiry_flag": False,
                    "reason": action.reason,
                }
            )

    run_id = f"paper-{date}"
    if trades:
        log_trades(trades, run_id, base_dir=out)

    state["last_ts"] = f"{date}T15:30:00+05:30"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    summary = {
        "date": date,
        "trades": trades,
        "n_trades": len(trades),
        "run_id": run_id,
    }
    (out / f"day_{date}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_paper(run_id: str, mode: str = "sim") -> dict:
    """CLI entry — sim runs one paper day; ingest records spread snapshot stub."""
    if mode == "ingest":
        from data.ingest.live import run_live_ingest

        return run_live_ingest()

    today = datetime.now(tz=IST).date().isoformat()
    result = run_paper_day(today)
    result["run_id"] = run_id
    return result
