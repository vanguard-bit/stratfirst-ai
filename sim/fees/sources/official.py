from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from sim.fees.sources.http import get_url

CONNECT_TIMEOUT = 4
READ_TIMEOUT = 8
_nse_warmed = False
_nse_session = requests.Session()
_nse_session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
)


def _get(url: str) -> requests.Response:
    return get_url(url)


def _get_nse(url: str) -> requests.Response:
    global _nse_warmed
    if not _nse_warmed:
        _nse_session.get("https://www.nseindia.com/", timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        _nse_session.headers["Referer"] = "https://www.nseindia.com/"
        _nse_warmed = True
    resp = _nse_session.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    resp.raise_for_status()
    return resp


@dataclass
class FeeComponent:
    name: str
    rate: float
    rate_unit: str  # percent | per_crore | flat_inr | formula
    side: str  # buy | sell | both | na
    source_url: str
    source_label: str
    fetched_at: str
    raw_text: str = ""


@dataclass
class SegmentFees:
    segment: str
    exchange: str
    components: list[FeeComponent] = field(default_factory=list)


@dataclass
class FeeRegistry:
    version: int = 1
    updated_at: str = ""
    broker_profile: str = "fyers"
    segments: dict[str, SegmentFees] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "broker_profile": self.broker_profile,
            "sources": self.sources,
            "segments": {
                k: {
                    "segment": v.segment,
                    "exchange": v.exchange,
                    "components": [asdict(c) for c in v.components],
                }
                for k, v in self.segments.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeeRegistry:
        reg = cls(
            version=data.get("version", 1),
            updated_at=data.get("updated_at", ""),
            broker_profile=data.get("broker_profile", "fyers"),
            sources=data.get("sources", []),
        )
        for key, seg in data.get("segments", {}).items():
            reg.segments[key] = SegmentFees(
                segment=seg["segment"],
                exchange=seg["exchange"],
                components=[FeeComponent(**c) for c in seg["components"]],
            )
        return reg

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: Path) -> FeeRegistry:
        with path.open(encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pct_from_text(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|per cent|percent)", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1))
    return None


def _per_crore_from_text(text: str) -> float | None:
    m = re.search(r"₹?\s*(\d+(?:\.\d+)?)\s*/?\s*crore", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"Rs\.?\s*(\d+(?:\.\d+)?)\s*/?\s*crore", text, re.I)
    if m:
        return float(m.group(1))
    return None


def fetch_nse_stamp_duty(url: str) -> list[FeeComponent]:
    """Parse NSE official stamp duty table."""
    resp = _get_nse(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    fetched = _now_iso()
    out: list[FeeComponent] = []

    for row in soup.select("table tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        if len(cells) < 2:
            continue
        label, rate_text = cells[0], cells[1]
        low = label.lower()
        pct = _pct_from_text(rate_text)
        if pct is None:
            continue
        if "delivery" in low and "debenture" not in low and "non-delivery" not in low:
            out.append(
                FeeComponent(
                    name="stamp_duty",
                    rate=pct,
                    rate_unit="percent",
                    side="buy",
                    source_url=url,
                    source_label="NSE stamp duty (delivery)",
                    fetched_at=fetched,
                    raw_text=f"{label} | {rate_text}",
                )
            )
        elif "non-delivery" in low:
            out.append(
                FeeComponent(
                    name="stamp_duty",
                    rate=pct,
                    rate_unit="percent",
                    side="buy",
                    source_url=url,
                    source_label="NSE stamp duty (intraday/non-delivery)",
                    fetched_at=fetched,
                    raw_text=f"{label} | {rate_text}",
                )
            )
    return out


def fetch_nse_stt(url: str) -> list[FeeComponent]:
    """Parse NSE STT table for equity delivery buy/sell and intraday sell."""
    resp = _get_nse(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    fetched = _now_iso()
    out: list[FeeComponent] = []

    for row in soup.select("table tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        if len(cells) < 3:
            continue
        desc = cells[1].lower() if len(cells) > 1 else ""
        rate_col = cells[2] if len(cells) > 2 else cells[1]
        pct = _pct_from_text(rate_col)
        if pct is None:
            continue
        if "purchase" in desc and "delivery" in desc:
            out.append(
                FeeComponent(
                    name="stt",
                    rate=pct,
                    rate_unit="percent",
                    side="buy",
                    source_url=url,
                    source_label="NSE STT equity delivery buy",
                    fetched_at=fetched,
                    raw_text=" | ".join(cells),
                )
            )
        elif "sale" in desc and "delivery" in desc and "otherwise" not in desc:
            out.append(
                FeeComponent(
                    name="stt",
                    rate=pct,
                    rate_unit="percent",
                    side="sell",
                    source_url=url,
                    source_label="NSE STT equity delivery sell",
                    fetched_at=fetched,
                    raw_text=" | ".join(cells),
                )
            )
        elif "otherwise than by the actual delivery" in desc:
            out.append(
                FeeComponent(
                    name="stt",
                    rate=pct,
                    rate_unit="percent",
                    side="sell",
                    source_url=url,
                    source_label="NSE STT equity intraday",
                    fetched_at=fetched,
                    raw_text=" | ".join(cells),
                )
            )
    return out


def fetch_nse_txn_circular_pdf(url: str) -> list[FeeComponent]:
    """Extract cash market txn charge from NSE finance circular PDF."""
    from pypdf import PdfReader
    import io

    resp = _get(url)
    reader = PdfReader(io.BytesIO(resp.content))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    fetched = _now_iso()
    out: list[FeeComponent] = []

    m = re.search(
        r"Cash Market:\s*Rs\.?\s*(\d+(?:\.\d+)?)\s*each side per lakh",
        text,
        re.I,
    )
    if m:
        per_lakh = float(m.group(1))
        pct = per_lakh / 100_000 * 100
        out.append(
            FeeComponent(
                name="exchange_txn",
                rate=pct,
                rate_unit="percent",
                side="both",
                source_url=url,
                source_label=f"NSE circular cash txn Rs {per_lakh}/lakh each side",
                fetched_at=fetched,
                raw_text=m.group(0),
            )
        )
    return out


def fetch_broker_charges(url: str, broker: str) -> dict[str, list[FeeComponent]]:
    """Parse broker charges page (Fyers/Zerodha publish MII rates as %)."""
    resp = _get(url)
    text = resp.text
    soup = BeautifulSoup(text, "html.parser")
    fetched = _now_iso()
    segments: dict[str, list[FeeComponent]] = {
        "equity_delivery": [],
        "equity_intraday": [],
    }

    def add(seg: str, name: str, rate: float, unit: str, side: str, label: str, raw: str):
        segments[seg].append(
            FeeComponent(
                name=name,
                rate=rate,
                rate_unit=unit,
                side=side,
                source_url=url,
                source_label=f"{broker}: {label}",
                fetched_at=fetched,
                raw_text=raw,
            )
        )

    if "fyers" in broker.lower():
        # Regulatory table on Fyers charges-list
        for row in soup.select("table tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
            if len(cells) < 5:
                continue
            charge = cells[0].upper()
            delivery = cells[1]
            intraday = cells[2]

            for seg_key, cell in [("equity_delivery", delivery), ("equity_intraday", intraday)]:
                if not cell or cell in {"-", "0", "0%"}:
                    continue
                if "STT" in charge:
                    side = "both" if "Buy and Sell" in cell else "sell"
                    pct = _pct_from_text(cell)
                    if pct:
                        add(seg_key, "stt", pct, "percent", side, charge, cell)
                elif "EXCHANGE" in charge or "TRANSACTION" in charge:
                    m = re.search(r"NSE:\s*(\d+(?:\.\d+)?)\s*%", cell, re.I)
                    if m:
                        add(seg_key, "exchange_txn", float(m.group(1)), "percent", "both", charge, cell)
                elif "SEBI" in charge:
                    pc = _per_crore_from_text(cell)
                    if pc:
                        add(seg_key, "sebi", pc, "per_crore", "both", charge, cell)
                elif "STAMP" in charge:
                    pct = _pct_from_text(cell)
                    if pct:
                        add(seg_key, "stamp_duty", pct, "percent", "buy", charge, cell)
                elif "IPFT" in charge:
                    pc = _per_crore_from_text(cell)
                    if pc:
                        add(seg_key, "ipft", pc, "per_crore", "both", charge, cell)
                elif "GST" in charge:
                    add(seg_key, "gst", 18.0, "percent_on_taxable", "both", charge, cell)

        # Brokerage headline
        add("equity_delivery", "brokerage", 0.0, "flat_inr", "both", "Fyers delivery headline", "₹0 mutual funds; delivery ₹20 or 0.3%")
        add("equity_intraday", "brokerage", 20.0, "flat_inr_or_pct", "both", "Fyers intraday headline", "₹20 or 0.03%")

    elif "zerodha" in broker.lower():
        tables = soup.select("table")
        for table in tables:
            headers = [h.get_text(" ", strip=True).lower() for h in table.select("tr th")]
            if "equity delivery" not in headers and "equity intraday" not in " ".join(headers):
                continue
            for row in table.select("tr"):
                cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
                if len(cells) < 3:
                    continue
                charge = cells[0]
                delivery = cells[1] if len(cells) > 1 else ""
                intraday = cells[2] if len(cells) > 2 else ""
                cl = charge.lower()
                for seg_key, cell in [("equity_delivery", delivery), ("equity_intraday", intraday)]:
                    if not cell:
                        continue
                    if "brokerage" in cl:
                        if "zero" in cell.lower():
                            add(seg_key, "brokerage", 0.0, "flat_inr", "both", charge, cell)
                        else:
                            add(seg_key, "brokerage", 20.0, "flat_inr_or_pct", "both", charge, cell)
                    elif "stt" in cl:
                        if "buy & sell" in cell.lower() or "buy and sell" in cell.lower():
                            pct = _pct_from_text(cell)
                            if pct:
                                add(seg_key, "stt", pct, "percent", "buy", charge, cell)
                                add(seg_key, "stt", pct, "percent", "sell", charge, cell)
                        else:
                            pct = _pct_from_text(cell)
                            if pct:
                                add(seg_key, "stt", pct, "percent", "sell", charge, cell)
                    elif "transaction" in cl:
                        m = re.search(r"NSE:\s*(\d+(?:\.\d+)?)\s*%", cell, re.I)
                        if m:
                            add(seg_key, "exchange_txn", float(m.group(1)), "percent", "both", charge, cell)
                    elif "sebi" in cl:
                        pc = _per_crore_from_text(cell)
                        if pc:
                            add(seg_key, "sebi", pc, "per_crore", "both", charge, cell)
                    elif "stamp" in cl:
                        pct = _pct_from_text(cell)
                        if pct:
                            add(seg_key, "stamp_duty", pct, "percent", "buy", charge, cell)
                    elif "gst" in cl:
                        add(seg_key, "gst", 18.0, "percent_on_taxable", "both", charge, cell)
    return segments


def merge_components(*groups: list[FeeComponent]) -> list[FeeComponent]:
    """Prefer broker-published rates for txn/STT when available; NSE for stamp."""
    by_key: dict[tuple[str, str], FeeComponent] = {}
    priority = {"NSE": 0, "fyers": 1, "zerodha": 2}

    def pri(c: FeeComponent) -> int:
        for k, v in priority.items():
            if k.lower() in c.source_label.lower():
                return v
        return 3

    for group in groups:
        for c in group:
            key = (c.name, c.side)
            if key not in by_key or pri(c) < pri(by_key[key]):
                by_key[key] = c
    return list(by_key.values())
