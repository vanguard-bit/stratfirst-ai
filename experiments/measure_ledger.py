"""Unit-notional friction measure ledger (bake-off; not capital book)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from nse_trader.config import ROOT
from sim.exchange.circuits import circuits_or_open
from sim.friction.measured import Quote
from sim.orders import OrderIntent, OrderSide, OrderType, Product
from sim.pipeline import SimPipeline

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger(__name__)

UNIT_NOTIONAL = 10_000.0  # INR per measure entry
MEASURE_BOOKS = ROOT / "data" / "state" / "measure_books.json"
MEASURE_FILLS = (
    ROOT / "data" / "store" / "experiments" / "meta_bakeoff" / "measure_fills.parquet"
)
STRAT_DAILY = (
    ROOT / "data" / "store" / "experiments" / "meta_bakeoff" / "bakeoff_strat_daily.parquet"
)


def _position_key(strategy_id: str, symbol: str) -> str:
    return f"{strategy_id}:{symbol}"


def load_measure_books(path: Path | None = None) -> dict[str, Any]:
    path = Path(path or MEASURE_BOOKS)
    if not path.exists():
        return {"positions": {}, "cash_pnl": {}, "updated_at": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_measure_books(books: dict[str, Any], path: Path | None = None) -> None:
    path = Path(path or MEASURE_BOOKS)
    path.parent.mkdir(parents=True, exist_ok=True)
    books["updated_at"] = datetime.now(tz=IST).isoformat()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(books, indent=2), encoding="utf-8")
    tmp.replace(path)


def unit_qty(price: float, *, notional: float = UNIT_NOTIONAL) -> int:
    if price <= 0:
        return 0
    return max(int(notional // price), 1)


def append_measure_fills(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    if not rows:
        return
    path = Path(path or MEASURE_FILLS)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    tmp = path.with_suffix(path.suffix + ".tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(path)


def process_measure_signal(
    *,
    pipeline: SimPipeline,
    quote: Quote,
    strategy_id: str,
    cluster: str,
    action: str,
    product_name: str,
    ts: datetime,
    books: dict[str, Any] | None = None,
    books_path: Path | None = None,
    fills_path: Path | None = None,
    unit_notional: float = UNIT_NOTIONAL,
) -> dict[str, Any] | None:
    """
    Sim-fill a strategy signal at unit notional (ignores meta weight).
    Updates measure books only — never touches capital virtual_books.
    """
    action = (action or "HOLD").upper()
    if action == "HOLD":
        return None

    books = books if books is not None else load_measure_books(books_path)
    positions = books.setdefault("positions", {})
    cash_pnl = books.setdefault("cash_pnl", {})
    key = _position_key(strategy_id, quote.symbol)
    cur_qty = int(positions.get(key, {}).get("qty", 0))
    product = Product.MIS if product_name == "MIS" else Product.CNC
    price = float(quote.ltp or 0.0)
    if price <= 0:
        return None

    if action == "FLAT":
        if cur_qty == 0:
            return None
        target = 0
    elif action == "BUY":
        target = unit_qty(price, notional=unit_notional)
    elif action == "SELL":
        if str(product_name).upper() == "MIS":
            target = -unit_qty(price, notional=unit_notional)
        else:
            if cur_qty <= 0:
                return None
            target = 0
    else:
        return None
    delta = target - cur_qty
    if delta == 0:
        return None
    side = OrderSide.BUY if delta > 0 else OrderSide.SELL
    qty = abs(delta)

    intent = OrderIntent(
        strategy_id=f"MEASURE:{strategy_id}",
        symbol=quote.symbol,
        side=side,
        quantity=qty,
        order_type=OrderType.MARKET,
        product=product,
    )
    uc, lc = circuits_or_open(quote, fallback_pct=0.10)
    pr = pipeline.process(intent, quote=quote, uc=uc, lc=lc)
    fill = getattr(pr, "fill", None) or getattr(pr, "fills", None)
    if pr.status != "FILLED" or fill is None:
        logger.debug(
            "measure fill rejected %s/%s %s %s",
            strategy_id,
            quote.symbol,
            action,
            getattr(pr, "reject_reason", None) or pr.status,
        )
        return None

    signed = qty if side == OrderSide.BUY else -qty
    new_qty = cur_qty + signed
    positions[key] = {
        "qty": new_qty,
        "product": product_name,
        "avg_px": float(fill.fill_price),
    }

    notional = float(fill.fill_price) * qty
    cost = float(fill.charges.total)
    delta = (-notional if side == OrderSide.BUY else notional) - cost
    cash_pnl[strategy_id] = float(cash_pnl.get(strategy_id, 0.0)) + delta

    row = {
        "ts": ts.isoformat() if isinstance(ts, datetime) else str(ts),
        "date": (ts.date().isoformat() if isinstance(ts, datetime) else str(ts)[:10]),
        "strategy_id": strategy_id,
        "cluster": cluster,
        "symbol": quote.symbol,
        "side": side.value,
        "qty": qty,
        "fill_price": float(fill.fill_price),
        "signal_price": float(fill.signal_price),
        "total_cost": cost,
        "unit_notional": float(unit_notional),
        "meta_weight_ignored": True,
    }
    append_measure_fills([row], path=fills_path)
    save_measure_books(books, path=books_path)
    return row


def _row_field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


MARK_VS_AVG_MAX = 0.15


def _mark_px(symbol: str, avg_px: float, marks: dict[str, float]) -> float:
    """Use session marks only when they agree with fill/avg; else stay on the book px."""
    avg = float(avg_px or 0.0)
    mark = marks.get(symbol)
    if mark is None:
        return avg
    mark_f = float(mark)
    if avg > 0 and mark_f > 0 and abs(mark_f / avg - 1.0) > MARK_VS_AVG_MAX:
        return avg
    return mark_f


def apply_measure_fill_to_books(books: dict[str, Any], row: Any) -> bool:
    """Replay one measure fill onto books. Returns False if the fill is skipped."""
    sid = str(_row_field(row, "strategy_id"))
    symbol = str(_row_field(row, "symbol"))
    side = str(_row_field(row, "side")).upper()
    qty = int(_row_field(row, "qty"))
    px = float(_row_field(row, "fill_price"))
    key = _position_key(sid, symbol)
    positions = books.setdefault("positions", {})
    cur = int(positions.get(key, {}).get("qty", 0))
    if side != "BUY" and cur < qty:
        return False
    signed = qty if side == "BUY" else -qty
    new_qty = cur + signed
    product = str(
        positions.get(key, {}).get("product")
        or _row_field(row, "product")
        or "CNC"
    )
    if new_qty == 0:
        positions.pop(key, None)
    else:
        positions[key] = {"qty": new_qty, "product": product, "avg_px": px}
    return True


def valid_measure_fills(fills: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and sells with no matching long."""
    if fills is None or fills.empty:
        return fills if fills is not None else pd.DataFrame()
    df = fills.copy().drop_duplicates(keep="first")
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    if "ts" in df.columns:
        df = df.sort_values("ts")
    else:
        df = df.sort_values(["date", "strategy_id", "symbol"])
    books: dict[str, Any] = {"positions": {}}
    keep: list[int] = []
    for idx, r in df.iterrows():
        if apply_measure_fill_to_books(books, r):
            keep.append(idx)
    return df.loc[keep]


def reconstruct_measure_books_from_fills(
    fills: pd.DataFrame,
    *,
    through_date: str | None = None,
) -> dict[str, Any]:
    books: dict[str, Any] = {"positions": {}, "cash_pnl": {}}
    if fills is None or fills.empty:
        return books
    df = valid_measure_fills(fills)
    if df is None or df.empty:
        return books
    if through_date:
        df = df[df["date"].astype(str).str.slice(0, 10) <= str(through_date)[:10]]
    for r in df.itertuples():
        apply_measure_fill_to_books(books, r)
    return books


def _bod_by_sid_from_snapshot_or_fills(
    *,
    positions: dict,
    day_fills: pd.DataFrame,
    prev_mtm: dict,
    use_prev: bool,
    marks: dict[str, float],
) -> dict[str, float]:
    """
    Beginning-of-day inventory value.
    Prefer yesterday's MTM snapshot. If missing, reverse today's fills so leftover
    inventory is not booked as profit and the opening day's buys keep BOD=0.
    """
    if use_prev:
        return {str(k): float(v) for k, v in (prev_mtm or {}).items()}

    state: dict[str, dict[str, float]] = {}
    for key, pos in (positions or {}).items():
        qty = int(pos.get("qty", 0))
        if qty == 0:
            continue
        sid, symbol = str(key).split(":", 1)
        state[str(key)] = {
            "qty": float(qty),
            "avg_px": float(pos.get("avg_px") or 0.0),
            "sid": sid,
            "symbol": symbol,
        }
    if day_fills is not None and not day_fills.empty:
        work = day_fills
        if "ts" in work.columns:
            work = work.sort_values("ts")
        for r in reversed(list(work.itertuples())):
            key = _position_key(str(r.strategy_id), str(r.symbol))
            qty = int(r.qty)
            px = float(r.fill_price)
            side = str(r.side).upper()
            cur = state.get(key) or {
                "qty": 0.0,
                "avg_px": px,
                "sid": str(r.strategy_id),
                "symbol": str(r.symbol),
            }
            if side == "BUY":
                new_qty = cur["qty"] - qty
                if new_qty <= 0:
                    state.pop(key, None)
                else:
                    cur["qty"] = new_qty
                    state[key] = cur
            else:
                new_qty = cur["qty"] + qty
                if cur["qty"] <= 0:
                    cur["avg_px"] = px
                else:
                    cur["avg_px"] = (cur["qty"] * cur["avg_px"] + qty * px) / new_qty
                cur["qty"] = new_qty
                state[key] = cur

    bod: dict[str, float] = {}
    for st in state.values():
        px = _mark_px(st["symbol"], float(st["avg_px"]), marks)
        sid = str(st["sid"])
        bod[sid] = bod.get(sid, 0.0) + float(st["qty"]) * px
    return bod


def aggregate_strat_day(
    *,
    day: str,
    marks: dict[str, float] | None = None,
    books_path: Path | None = None,
    fills_path: Path | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Per-strategy day return from measure fills + MTM of open positions.

    PnL = cash_delta + EOD_mtm − BOD_mtm.
    Without a prior snapshot, BOD is inventory at avg (not full MTM as profit).
    Return divides by deployed capital (BOD, else cost/EOD), never a single 10k unit
    when many names are open.
    """
    books = load_measure_books(books_path)
    fills_path = Path(fills_path or MEASURE_FILLS)
    out_path = Path(out_path or STRAT_DAILY)
    marks = marks or {}
    positions: dict = books.setdefault("positions", {})

    day_fills = pd.DataFrame()
    if fills_path.exists():
        all_f = valid_measure_fills(pd.read_parquet(fills_path))
        if all_f is not None and not all_f.empty:
            day_fills = all_f[all_f["date"].astype(str).str.slice(0, 10) == str(day)]

    cash_today: dict[str, float] = {}
    clusters: dict[str, str] = {}
    if not day_fills.empty:
        for r in day_fills.itertuples():
            sid = str(r.strategy_id)
            clusters[sid] = str(getattr(r, "cluster", "?") or "?")
            notional = float(r.fill_price) * int(r.qty)
            delta = (
                (-notional if str(r.side).upper() == "BUY" else notional)
                - float(r.total_cost)
            )
            cash_today[sid] = cash_today.get(sid, 0.0) + delta

    mtm: dict[str, float] = {}
    cost: dict[str, float] = {}
    for key, pos in positions.items():
        qty = int(pos.get("qty", 0))
        if qty == 0:
            continue
        sid, symbol = key.split(":", 1)
        px = _mark_px(symbol, float(pos.get("avg_px") or 0.0), marks)
        mtm[sid] = mtm.get(sid, 0.0) + qty * px
        cost[sid] = cost.get(sid, 0.0) + abs(qty) * float(pos.get("avg_px") or 0.0)
        if sid not in clusters:
            clusters[sid] = "?"

    prev_mtm = books.get("mtm_snapshot") or {}
    prev_date = str(books.get("mtm_snapshot_date") or "")
    use_prev = prev_date != "" and prev_date != str(day)
    bod_by_sid = _bod_by_sid_from_snapshot_or_fills(
        positions=positions,
        day_fills=day_fills,
        prev_mtm=prev_mtm,
        use_prev=use_prev,
        marks=marks,
    )
    sids = sorted(set(cash_today) | set(mtm) | set(bod_by_sid))
    rows = []
    for sid in sids:
        cash_d = float(cash_today.get(sid, 0.0))
        mtm_now = float(mtm.get(sid, 0.0))
        cost_now = float(cost.get(sid, 0.0))
        bod = float(bod_by_sid.get(sid, 0.0))
        pnl = cash_d + mtm_now - bod
        deployed = max(abs(bod), abs(cost_now), abs(mtm_now), UNIT_NOTIONAL)
        ret = pnl / deployed if deployed else 0.0
        rows.append(
            {
                "date": str(day),
                "strategy_id": sid,
                "cluster": clusters.get(sid, "?"),
                "pnl": pnl,
                "ret": ret,
                "deployed": deployed,
                "cash_delta": cash_d,
                "mtm": mtm_now,
            }
        )

    books["mtm_snapshot"] = mtm
    books["mtm_snapshot_date"] = str(day)
    save_measure_books(books, path=books_path)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        old = pd.read_parquet(out_path)
        old = old[old["date"].astype(str) != str(day)]
        combined = pd.concat([old, df], ignore_index=True)
    else:
        combined = df
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return df


def rebuild_strat_daily_from_fills(
    *,
    days: list[str],
    fills_path: Path | None = None,
    books_path: Path | None = None,
    out_path: Path | None = None,
    marks_by_day: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """
    Replay measure fills into a fresh book, then aggregate each day in order.
    Days with no fills still mark leftover inventory vs the prior snapshot.
    """
    fills_path = Path(fills_path or MEASURE_FILLS)
    books_path = Path(books_path or MEASURE_BOOKS)
    out_path = Path(out_path or STRAT_DAILY)
    fills = pd.read_parquet(fills_path) if fills_path.exists() else pd.DataFrame()
    if out_path.exists():
        out_path.unlink()
    snapshot_mtm: dict[str, float] = {}
    snapshot_date = ""
    parts: list[pd.DataFrame] = []
    for day in days:
        books = reconstruct_measure_books_from_fills(fills, through_date=day)
        books["mtm_snapshot"] = snapshot_mtm
        books["mtm_snapshot_date"] = snapshot_date
        save_measure_books(books, path=books_path)
        df = aggregate_strat_day(
            day=str(day),
            marks=(marks_by_day or {}).get(str(day)) or {},
            books_path=books_path,
            fills_path=fills_path,
            out_path=out_path,
        )
        after = load_measure_books(books_path)
        snapshot_mtm = after.get("mtm_snapshot") or {}
        snapshot_date = str(after.get("mtm_snapshot_date") or day)
        if not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
