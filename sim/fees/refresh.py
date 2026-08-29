from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from sim.fees.sources.official import (
    FeeRegistry,
    fetch_broker_charges,
    fetch_nse_stamp_duty,
    fetch_nse_stt,
    fetch_nse_txn_circular_pdf,
    merge_components,
    SegmentFees,
)
from sim.fees.sources.seed import load_seed_registry
from nse_trader.config import ROOT


def _load_costs_config() -> dict:
    with (ROOT / "config" / "costs_nse.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _try(name: str, fn) -> tuple[str, object, str | None]:
    try:
        return name, fn(), None
    except Exception as exc:  # noqa: BLE001
        return name, None, f"{name}: {exc}"


def refresh_registry(
    out_path: Path | None = None,
    broker_profile: str | None = None,
    *,
    offline: bool = False,
    skip_nse_html: bool = True,
) -> FeeRegistry:
    """
    Build fee registry. Default: skip slow NSE HTML, try broker+PDF sequentially
    with 8s timeouts. On any failure → official seed (cited, not guessed).
    """
    cfg = _load_costs_config()
    broker = broker_profile or cfg.get("default_broker_profile", "zerodha")
    out_path = out_path or (ROOT / "data" / "fees" / "registry.json")

    if offline:
        reg = load_seed_registry(broker)
        reg.save(out_path)
        return reg

    sources_cfg = cfg["sources"]
    broker_cfg = sources_cfg["broker_profiles"].get(broker) or sources_cfg["broker_profiles"]["zerodha"]
    nse_pdf_url = sources_cfg["regulatory"]["nse_txn_circular"]["url"]

    fetch_errors: list[str] = []
    broker_segs = None
    nse_txn: list = []
    nse_stt: list = []
    nse_stamp: list = []

    _, broker_segs, err = _try("broker", lambda: fetch_broker_charges(broker_cfg["url"], broker))
    if err:
        fetch_errors.append(err)

    _, nse_txn, err = _try("nse_txn_pdf", lambda: fetch_nse_txn_circular_pdf(nse_pdf_url))
    if err:
        fetch_errors.append(err)
    else:
        nse_txn = nse_txn or []

    if not skip_nse_html:
        nse_stt_url = sources_cfg["regulatory"]["nse_stt"]["url"]
        nse_stamp_url = sources_cfg["regulatory"]["nse_stamp_duty"]["url"]
        _, stt, err = _try("nse_stt", lambda: fetch_nse_stt(nse_stt_url))
        if err:
            fetch_errors.append(err)
        else:
            nse_stt = stt or []
        _, stamp, err = _try("nse_stamp", lambda: fetch_nse_stamp_duty(nse_stamp_url))
        if err:
            fetch_errors.append(err)
        else:
            nse_stamp = stamp or []

    if not broker_segs:
        reg = load_seed_registry(broker)
        reg.sources.append({"id": "fetch_fallback", "reason": "broker_unreachable", "errors": fetch_errors})
        reg.save(out_path)
        return reg

    reg = FeeRegistry(
        updated_at=datetime.now(timezone.utc).isoformat(),
        broker_profile=broker,
    )
    reg.sources = [
        {"id": broker, "url": broker_cfg["url"], "live": True},
        {"id": "nse_txn_pdf", "url": nse_pdf_url, "count": len(nse_txn)},
    ]
    if fetch_errors:
        reg.sources.append({"id": "fetch_warnings", "messages": fetch_errors})

    delivery = merge_components(
        [c for c in nse_stt if "delivery" in c.source_label.lower()],
        [c for c in nse_stamp if "delivery" in c.source_label.lower()],
        nse_txn,
        broker_segs.get("equity_delivery", []),
    )
    intraday = merge_components(
        [c for c in nse_stt if "intraday" in c.source_label.lower()],
        [
            c
            for c in nse_stamp
            if "intraday" in c.source_label.lower() or "non-delivery" in c.source_label.lower()
        ],
        nse_txn,
        broker_segs.get("equity_intraday", []),
    )

    if len(delivery) < 4 or len(intraday) < 4:
        reg = load_seed_registry(broker)
        reg.sources.append({"id": "fetch_fallback", "reason": "incomplete_parse", "errors": fetch_errors})
        reg.save(out_path)
        return reg

    reg.segments["equity_delivery"] = SegmentFees(
        segment="equity_delivery", exchange="NSE", components=delivery
    )
    reg.segments["equity_intraday"] = SegmentFees(
        segment="equity_intraday", exchange="NSE", components=intraday
    )
    reg.save(out_path)
    return reg
