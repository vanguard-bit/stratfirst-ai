from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from nse_trader.config import PortfolioConfig


class DataStore:
    """DuckDB store for bars, spreads, and experiment outputs."""

    def __init__(self, db_path: Path | None = None):
        cfg = PortfolioConfig.load()
        store = cfg.store_path
        store.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (store / "market.duckdb")
        self.con = duckdb.connect(str(self.db_path))

    def __enter__(self) -> DataStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def init_schema(self) -> None:
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS bars_1m (
                ts TIMESTAMP,
                symbol VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE
            );
            CREATE TABLE IF NOT EXISTS bars_1d (
                date DATE,
                symbol VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE
            );
            CREATE TABLE IF NOT EXISTS friction_spreads (
                ts TIMESTAMP,
                symbol VARCHAR,
                bid DOUBLE,
                ask DOUBLE,
                ltp DOUBLE,
                half_spread_bps DOUBLE,
                upper_ckt DOUBLE,
                lower_ckt DOUBLE,
                prev_close DOUBLE
            );
            CREATE TABLE IF NOT EXISTS bars_1w (
                date DATE,
                symbol VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE
            );
        """)
        self.ensure_friction_spread_columns()

    def ensure_friction_spread_columns(self) -> None:
        """Migrate older DBs that lack circuit columns on friction_spreads."""
        cols = {
            r[0]
            for r in self.con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'friction_spreads'"
            ).fetchall()
        }
        for name in ("upper_ckt", "lower_ckt", "prev_close"):
            if name not in cols:
                self.con.execute(f"ALTER TABLE friction_spreads ADD COLUMN {name} DOUBLE")

    def ensure_bars_1w(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS bars_1w (
                date DATE,
                symbol VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE
            );
            """
        )

    @staticmethod
    def normalize_bars_1d(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Map provider frames to bars_1d schema."""
        out = df.copy()
        out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

        if "date" not in out.columns and "timestamp" in out.columns:
            out["date"] = pd.to_datetime(out["timestamp"]).dt.date
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"]).dt.date

        rename = {
            "open_price": "open",
            "high_price": "high",
            "low_price": "low",
            "close_price": "close",
            "ltp": "close",
            "tottrdqty": "volume",
            "total_traded_quantity": "volume",
        }
        out = out.rename(columns=rename)

        for col in ("open", "high", "low", "close"):
            if col not in out.columns:
                raise ValueError(f"EOD frame missing {col!r} for {symbol}")

        if "volume" not in out.columns:
            out["volume"] = 0.0

        out["symbol"] = symbol
        return out[["date", "symbol", "open", "high", "low", "close", "volume"]]

    @staticmethod
    def normalize_bars_1m(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        out = df.copy()
        out.columns = [str(c).strip().lower() for c in out.columns]
        if "ts" not in out.columns and "timestamp" in out.columns:
            out["ts"] = out["timestamp"]
        out["ts"] = pd.to_datetime(out["ts"])
        out["symbol"] = symbol
        return out[["ts", "symbol", "open", "high", "low", "close", "volume"]]

    def _insert_df(self, table: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        self.con.register("_insert_df", df)
        before = self.row_count(table)
        self.con.execute(f"INSERT INTO {table} SELECT * FROM _insert_df")
        self.con.unregister("_insert_df")
        return self.row_count(table) - before

    def write_bars_1d(self, df: pd.DataFrame) -> int:
        return self._insert_df("bars_1d", df)

    def write_bars_1w(self, df: pd.DataFrame) -> int:
        self.ensure_bars_1w()
        return self._insert_df("bars_1w", df)

    def write_bars_1m(self, df: pd.DataFrame) -> int:
        return self._insert_df("bars_1m", df)

    def write_friction_spreads(self, df: pd.DataFrame) -> int:
        self.ensure_friction_spread_columns()
        out = df.copy()
        for col in ("upper_ckt", "lower_ckt", "prev_close"):
            if col not in out.columns:
                out[col] = None
        cols = [
            "ts",
            "symbol",
            "bid",
            "ask",
            "ltp",
            "half_spread_bps",
            "upper_ckt",
            "lower_ckt",
            "prev_close",
        ]
        return self._insert_df("friction_spreads", out[cols])

    def row_count(self, table: str) -> int:
        return int(self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def read_bars_1d(self, symbol: str | None = None) -> pd.DataFrame:
        if symbol:
            return self.con.execute(
                "SELECT * FROM bars_1d WHERE symbol = ? ORDER BY date",
                [symbol],
            ).df()
        return self.con.execute("SELECT * FROM bars_1d ORDER BY date, symbol").df()

    def read_bars_1w(self, symbol: str | None = None) -> pd.DataFrame:
        self.ensure_bars_1w()
        if symbol:
            return self.con.execute(
                "SELECT * FROM bars_1w WHERE symbol = ? ORDER BY date",
                [symbol],
            ).df()
        return self.con.execute("SELECT * FROM bars_1w ORDER BY date, symbol").df()

    def read_bars_1m(
        self,
        symbol: str | None = None,
        *,
        since: object | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        params: list[object] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM bars_1m {where} ORDER BY ts, symbol"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return self.con.execute(sql, params).df()

    def read_latest_spreads(
        self,
        symbols: list[str] | None = None,
        *,
        valid_only: bool = False,
    ) -> pd.DataFrame:
        """Latest friction_spreads row per symbol (one row each)."""
        clauses: list[str] = []
        params: list[object] = []
        if symbols:
            placeholders = ", ".join(["?"] * len(symbols))
            clauses.append(f"symbol IN ({placeholders})")
            params.extend(symbols)
        if valid_only:
            clauses.append("bid IS NOT NULL AND ask IS NOT NULL AND ask >= bid AND bid > 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.con.execute(
            f"""
            SELECT * FROM friction_spreads
            {where}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts DESC) = 1
            """,
            params,
        ).df()

    def read_prev_closes(self, symbols: list[str] | None = None) -> dict[str, float]:
        """Latest bars_1d close per symbol (proxy for previous close)."""
        clauses: list[str] = []
        params: list[object] = []
        if symbols:
            placeholders = ", ".join(["?"] * len(symbols))
            clauses.append(f"symbol IN ({placeholders})")
            params.extend(symbols)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        df = self.con.execute(
            f"""
            SELECT symbol, close FROM bars_1d
            {where}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) = 1
            """,
            params,
        ).df()
        if df is None or df.empty:
            return {}
        return {str(r.symbol): float(r.close) for r in df.itertuples() if float(r.close) > 0}

    def close(self) -> None:
        self.con.close()
