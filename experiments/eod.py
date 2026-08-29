"""End-of-day pipeline — paper day + health audit (no live broker orders)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.paper import reconcile_state, run_paper_day
from nse_trader.config import PortfolioConfig, ROOT, load_yaml
from ops.monitor.runner import run_health_checks, write_report_json

IST = ZoneInfo("Asia/Kolkata")


def run_eod(*, date: str | None = None) -> dict:
    """
    EOD job for systemd/cron:
    1. reconcile portfolio state
    2. run paper day if not already written for this date (paper.timer may have done it)
    3. health check + write logs
    """
    cfg = PortfolioConfig.load()
    logs = ROOT / load_yaml("ops.yaml")["persistence"]["logs_dir"]
    logs.mkdir(parents=True, exist_ok=True)

    day = date or datetime.now(tz=IST).date().isoformat()
    state_file = ROOT / load_yaml("ops.yaml")["persistence"]["state_dir"] / "portfolio.json"
    reconcile_state(state_file)

    paper_out = cfg.store_path / "experiments" / "eod" / "paper"
    paper_out.mkdir(parents=True, exist_ok=True)
    day_file = paper_out / f"day_{day}.json"
    if day_file.exists():
        paper = json.loads(day_file.read_text(encoding="utf-8"))
        paper_source = "existing"
    else:
        # Prefer shared paper experiment dir if paper.timer already wrote today
        alt = cfg.store_path / "experiments" / "paper" / f"day_{day}.json"
        if alt.exists():
            paper = json.loads(alt.read_text(encoding="utf-8"))
            paper_source = "paper_timer"
        else:
            paper = run_paper_day(day, out_dir=paper_out, state_file=state_file)
            paper_source = "eod"

    health = run_health_checks()
    health_path = logs / "health.json"
    write_report_json(health, health_path)

    shadow_info: dict | None = None
    try:
        from meta.shadow import feature_rows_for_shadow, run_meta_shadow

        feature_rows = None
        try:
            import pandas as pd

            cache = ROOT / "data" / "state" / "meta_shadow_features.parquet"
            feature_rows = None
            if cache.exists():
                try:
                    cached = pd.read_parquet(cache)
                    # Reuse only if cache was built for this EOD date
                    if (
                        not cached.empty
                        and "as_of" in cached.columns
                        and (cached["as_of"].astype(str) == str(day)).all()
                    ):
                        feature_rows = cached.drop(columns=["as_of"], errors="ignore")
                except Exception:  # noqa: BLE001
                    feature_rows = None
            if feature_rows is None or feature_rows.empty:
                from experiments.strategy_replay import replay_all_enabled

                returns = replay_all_enabled()
                feature_rows = feature_rows_for_shadow(returns, as_of=day)
                if feature_rows is not None and not feature_rows.empty:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    to_store = feature_rows.copy()
                    to_store["as_of"] = str(day)
                    to_store.to_parquet(cache, index=False)
        except Exception:  # noqa: BLE001
            feature_rows = None
        shadow_info = run_meta_shadow(as_of=day, feature_rows=feature_rows)
    except Exception as e:  # noqa: BLE001
        # Never fail EOD because of shadow scoring
        shadow_info = {"error": str(e)}

    bakeoff_info: dict | None = None
    try:
        from experiments.measure_ledger import aggregate_strat_day
        from experiments.meta_bakeoff import refresh_forward_and_glance

        marks: dict[str, float] = {}
        try:
            from data.ingest.store import DataStore

            with DataStore() as store:
                store.init_schema()
                # best-effort last closes for MTM
                import pandas as pd

                for tbl, col in (("bars_1d", "date"), ("bars_1m", "ts")):
                    try:
                        df = store.con.execute(
                            f"SELECT symbol, close FROM {tbl} ORDER BY {col} DESC LIMIT 500"
                        ).fetchdf()
                        if df is not None and not df.empty:
                            for r in df.itertuples():
                                marks.setdefault(str(r.symbol), float(r.close))
                            break
                    except Exception:  # noqa: BLE001
                        continue
        except Exception:  # noqa: BLE001
            marks = {}

        aggregate_strat_day(day=day, marks=marks)
        warns: list[str] = []
        if shadow_info is None:
            warns.append("meta shadow skipped (no model or empty features)")
        elif shadow_info.get("error"):
            warns.append(f"meta shadow error: {shadow_info.get('error')}")
        glance = refresh_forward_and_glance(day=day, warnings=warns)
        bakeoff_info = {
            "glance_as_of": glance.get("as_of"),
            "ml_top5": glance.get("ml_top5"),
            "warnings": glance.get("warnings"),
        }
        try:
            from ops.public_glance import export_public_glance

            pub = export_public_glance(include_local=True)
            bakeoff_info["public_glance"] = (pub.get("_paths") or {}).get("glance_json")
        except Exception as pub_exc:  # noqa: BLE001
            bakeoff_info["public_glance_error"] = str(pub_exc)
    except Exception as e:  # noqa: BLE001
        bakeoff_info = {"error": str(e)}

    summary = {
        "date": day,
        "created_at": datetime.now(tz=IST).isoformat(),
        "n_trades": paper.get("n_trades", 0),
        "run_id": paper.get("run_id", f"paper-{day}"),
        "paper_source": paper_source,
        "health_ok": health.ok,
        "health_warnings": health.warn_count,
        "health_errors": health.error_count,
        "health_path": str(health_path),
        "meta_shadow": (
            None
            if shadow_info is None
            else {
                "mode": shadow_info.get("mode"),
                "top5": shadow_info.get("top5"),
                "top5_reasons": shadow_info.get("top5_reasons"),
                "error": shadow_info.get("error"),
            }
        ),
        "meta_bakeoff": bakeoff_info,
    }
    out = logs / f"eod_{day}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["path"] = str(out)
    return summary
