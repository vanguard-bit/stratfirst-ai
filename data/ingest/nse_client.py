from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def _looks_like_index(symbol: str) -> bool:
    s = symbol.strip().upper()
    return s.startswith("NIFTY") or s in {"SENSEX", "BANKNIFTY", "FINNIFTY"}


def _yfinance_ticker(symbol: str) -> str:
    s = symbol.strip().upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    return f"{s}.NS"


def get_historical_equity_yf(symbol: str, start: date, end: date) -> pd.DataFrame:
    """EOD OHLC via Yahoo Finance (NSE equities as SYMBOL.NS)."""
    import yfinance as yf

    # yfinance end is exclusive
    end_excl = end + timedelta(days=1)
    raw = yf.download(
        _yfinance_ticker(symbol),
        start=start.isoformat(),
        end=end_excl.isoformat(),
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    df = raw.copy()
    # Flatten MultiIndex columns from newer yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).strip().lower() for c in df.columns]
    else:
        df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.reset_index()
    # Date column name
    date_col = "date" if "date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    keep = ["date", "open", "high", "low", "close", "volume"]
    for c in keep:
        if c not in df.columns:
            raise ValueError(f"yfinance frame missing {c!r} for {symbol}")
    return df[keep]


def get_historical_index(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch EOD index OHLC via nsedata (nse-archives package)."""
    from nsedata import nse

    return nse.get_historical_index(
        symbol,
        start.strftime("%d-%b-%Y"),
        end.strftime("%d-%b-%Y"),
    )


def get_historical_eod(symbol: str, start: date, end: date) -> pd.DataFrame:
    """
    Equity → yfinance (.NS); index-like names → nsedata niftyindices.
    """
    if _looks_like_index(symbol):
        return get_historical_index(symbol, start, end)
    return get_historical_equity_yf(symbol, start, end)
