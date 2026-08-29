from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="NSE paper trading research platform")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rf = sub.add_parser("refresh-fees", help="Fetch official/broker fee tables into registry")
    p_rf.add_argument(
        "--with-nse-html",
        action="store_true",
        help="Also fetch NSE HTML tables (slow, may timeout)",
    )
    p_rf.add_argument(
        "--offline",
        action="store_true",
        help="Use cited official seed only (instant, no network)",
    )

    p_bf = sub.add_parser("backfill", help="Bulk EOD historical download")
    p_bf.add_argument("--years", type=int, default=5)

    p_bh = sub.add_parser(
        "backfill-history",
        help="Multi-year EOD(+1W)+1m history for meta bootstrap (foreground, long)",
    )
    p_bh.add_argument("--years", type=int, default=3)
    p_bh.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Symbols (default: Nifty50)",
    )
    p_bh.add_argument(
        "--skip-eod",
        action="store_true",
        help="Skip 1d/1W download (only Fyers 1m)",
    )
    p_bh.add_argument(
        "--skip-1m",
        action="store_true",
        help="Skip Fyers 1m history (EOD only)",
    )

    p_mt = sub.add_parser(
        "meta-train",
        help="Walk-forward LightGBM meta train (foreground, long)",
    )
    p_mt.add_argument("--years", type=int, default=3, help="History depth hint (for logs)")
    p_mt.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Symbols for universe replay (default: Nifty50)",
    )
    p_mt.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process pool size (default: min(10, cpu-2))",
    )
    p_mt.add_argument(
        "--notional-per-symbol",
        type=float,
        default=None,
        help="Fixed notional per name (default: total_capital / n_symbols)",
    )
    p_mt.add_argument(
        "--force-export",
        action="store_true",
        help="Re-export DuckDB → parquet replay cache",
    )
    sub.add_parser(
        "meta-eval",
        help="Print last meta-train fold metrics from experiment manifest",
    )
    p_mb = sub.add_parser(
        "meta-bakeoff",
        help="Offline OOF bake-off and/or forward summary",
    )
    p_mb.add_argument(
        "--offline",
        action="store_true",
        help="Recompute Cat1 offline table from oof_preds",
    )
    p_mb.add_argument(
        "--forward",
        action="store_true",
        help="Print Cat2 forward bake-off summary",
    )
    p_ms = sub.add_parser(
        "meta-status",
        help="Quick glance: strat/cluster/ML/rules tracks (no EOD dig)",
    )
    p_ms.add_argument("--json", action="store_true", help="Print raw glance JSON")
    p_ms.add_argument("--date", help="As-of date YYYY-MM-DD (default today IST)")
    p_dash = sub.add_parser(
        "dashboard",
        help="Write/serve local PnL + EOD markers HTML dashboard",
    )
    p_dash.add_argument(
        "--serve",
        action="store_true",
        help="Serve on localhost (default write HTML only)",
    )
    p_dash.add_argument("--host", default="127.0.0.1")
    p_dash.add_argument("--port", type=int, default=18765)

    sub.add_parser("backtest", help="Run walk-forward backtest")
    p_llm = sub.add_parser("llm-extract", help="Gemini batch — compress headlines to structured features")
    p_llm.add_argument("--offline", action="store_true", help="Neutral sample features, no API")
    p_llm.add_argument(
        "--no-headlines",
        action="store_true",
        help="Skip Google News RSS; send empty/neutral headline text",
    )
    p_llm.add_argument(
        "--slot",
        choices=["morning", "evening", "manual"],
        default="manual",
        help="morning: enrich+MCP+bust weights cache; evening/manual: optional",
    )
    p_llm.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip FinStack/Tapetide MCP enrichment",
    )
    p_llm.add_argument(
        "symbols",
        nargs="*",
        default=None,
        help="Symbols (default: full Nifty50 list from config)",
    )
    p_paper = sub.add_parser("paper", help="Run forward paper sim")
    p_paper.add_argument("--mode", choices=["sim", "ingest"], default="sim")
    p_paper.add_argument("--duration", type=int, default=55, help="Ingest websocket seconds (systemd tick)")
    p_fwd = sub.add_parser("forward", help="Run forward validation (paper + health audit)")
    p_fwd.add_argument("--days", type=int, default=1)
    p_fwd.add_argument("--start", help="Start date YYYY-MM-DD (default today)")
    p_eod = sub.add_parser("eod", help="End-of-day: paper day + health (systemd/cron)")
    p_eod.add_argument("--date", help="Session date YYYY-MM-DD (default today IST)")
    p_health = sub.add_parser("health", help="Runtime checks: logs, strategies, meta allocator diversity")
    p_health.add_argument("--json", action="store_true", help="Write report to data/logs/health.json")
    p_diag = sub.add_parser(
        "diagnose",
        help="Agent triage: health + systemd timers/services + EOD/log artifacts",
    )
    p_diag.add_argument("--date", help="Session date YYYY-MM-DD for EOD artifact (default today IST)")
    p_diag.add_argument(
        "--json",
        action="store_true",
        help="Write data/logs/diagnose.json (always prints human summary)",
    )
    p_fl = sub.add_parser(
        "fyers-login",
        help="One-time browser OAuth → write FYERS_ACCESS_TOKEN (+ refresh) to .env",
    )
    p_fl.add_argument(
        "--auth-code",
        help="Auth code or full redirect URL (otherwise prompts)",
    )
    p_fl.add_argument(
        "--url-only",
        action="store_true",
        help="Only print the login URL (no exchange)",
    )
    sub.add_parser(
        "fyers-refresh",
        help="Mint new daily access token from FYERS_REFRESH_TOKEN + FYERS_PIN",
    )
    sub.add_parser("fyers-status", help="Show which Fyers .env fields are set (no secrets)")
    p_test = sub.add_parser("test", help="Run pytest suite")
    p_test.add_argument("pytest_args", nargs="*", help="Extra args e.g. -m phase0")
    sub.add_parser(
        "demo",
        help="Offline judge-safe demo (no broker/Gemini/MCP/network)",
    )
    sub.add_parser(
        "public-glance",
        help="Export redacted docs/site glance for GitHub Pages",
    )

    args = parser.parse_args()

    if args.cmd == "refresh-fees":
        from sim.fees.refresh import refresh_registry

        reg = refresh_registry(
            offline=args.offline,
            skip_nse_html=not args.with_nse_html,
        )
        mode = "offline seed" if args.offline else "live"
        print(f"Updated fee registry ({mode}): {len(reg.segments)} segments")
        for key, seg in reg.segments.items():
            names = sorted({c.name for c in seg.components})
            print(f"  {key}: {names}")
    elif args.cmd == "backfill":
        from data.ingest.backfill import backfill_eod

        rows = backfill_eod(["NIFTY 50"], years=args.years)
        print(f"Backfill complete: {rows} EOD bar(s) written")
    elif args.cmd == "backfill-history":
        from data.ingest.backfill_history import backfill_history

        tfs: list[str] = []
        if not args.skip_eod:
            tfs.extend(["1d", "1W"])
        if not args.skip_1m:
            tfs.append("1m")
        if not tfs:
            print("Nothing to do: both --skip-eod and --skip-1m set")
            raise SystemExit(2)
        stats = backfill_history(
            symbols=args.symbols,
            years=args.years,
            tfs=tuple(tfs),
            skip_eod=args.skip_eod,
        )
        print(f"backfill-history complete: {stats}")
    elif args.cmd == "meta-train":
        from meta.train_lgbm import run_meta_train_cli

        manifest = run_meta_train_cli(
            years=args.years,
            symbols=args.symbols,
            workers=args.workers,
            notional_per_symbol=args.notional_per_symbol,
            force_export=args.force_export,
        )
        print(
            f"meta-train complete: folds={manifest['n_folds']} "
            f"mean_auc={manifest.get('mean_auc')} symbols={manifest.get('symbols')} "
            f"agg={manifest.get('agg')} → {manifest['model_path']}"
        )
    elif args.cmd == "meta-eval":
        import json
        from nse_trader.config import ROOT

        man = ROOT / "data" / "store" / "experiments" / "meta_train" / "manifest.json"
        if not man.exists():
            print(f"No manifest at {man}; run meta-train first")
            raise SystemExit(1)
        payload = json.loads(man.read_text())
        print(
            f"model={payload.get('model_name')} folds={payload.get('n_folds')} "
            f"mean_auc={payload.get('mean_auc')} "
            f"mean_top5_precision={payload.get('mean_top5_precision')}"
        )
        for fold in payload.get("fold_metrics") or []:
            print(
                f"  fold {fold['fold']}: auc={fold.get('auc'):.4f} "
                f"top5={fold.get('top5_precision'):.4f} "
                f"pnl_score={fold.get('pnl_score'):.6f} "
                f"pnl_rules={fold.get('pnl_rules'):.6f} "
                f"pnl_eq={fold.get('pnl_eq'):.6f}"
            )
    elif args.cmd == "meta-bakeoff":
        from experiments.meta_bakeoff import (
            format_forward_summary,
            format_offline_summary,
            run_offline_bakeoff,
        )

        if not args.offline and not args.forward:
            args.offline = True
            args.forward = True
        if args.offline:
            summary = run_offline_bakeoff()
            print(format_offline_summary(summary))
            print(f"→ {summary.get('summary_path')}")
        if args.forward:
            print(format_forward_summary())
    elif args.cmd == "meta-status":
        import json
        from experiments.meta_bakeoff import (
            GLANCE_PATH,
            build_glance,
            format_glance,
            refresh_forward_and_glance,
        )

        day = args.date
        if GLANCE_PATH.exists() and not args.date:
            glance = json.loads(GLANCE_PATH.read_text(encoding="utf-8"))
        else:
            glance = build_glance(day=day) if day else refresh_forward_and_glance()
        if args.json:
            print(json.dumps(glance, indent=2, default=str))
        else:
            print(format_glance(glance))
    elif args.cmd == "dashboard":
        from ops.dashboard import serve_dashboard, write_dashboard

        if args.serve:
            serve_dashboard(host=args.host, port=args.port)
        else:
            path = write_dashboard()
            print(f"wrote {path}")
            print("open it in a browser, or: python main.py dashboard --serve")
    elif args.cmd == "backtest":
        from experiments.backtest import run_backtest

        manifest = run_backtest("manual")
        print(f"Backtest complete: {manifest['run_id']} ({manifest['n_splits']} splits)")
    elif args.cmd == "llm-extract":
        from datetime import date

        from data.ingest.symbols import load_nifty50_symbols
        from features.headlines import fetch_headlines_map
        from features.llm_gemini import (
            extract_features_live,
            extract_features_offline_sample,
            write_features_parquet,
        )
        from features.mcp_enrich import bust_meta_weights_cache, fetch_market_context
        from nse_trader.config import PortfolioConfig
        from nse_trader.env import load_dotenv

        load_dotenv()

        symbols = list(args.symbols) if args.symbols else load_nifty50_symbols()
        cfg = PortfolioConfig.load()
        out = cfg.store_path / "features" / f"llm_{date.today().isoformat()}.parquet"
        market_context = ""
        if args.slot == "morning" and not args.offline and not args.no_enrich:
            market_context = fetch_market_context()
            if market_context:
                print(f"MCP enrichment: {len(market_context)} chars")
            else:
                print("MCP enrichment: skipped/empty (budget or fail-soft)")
        if args.offline:
            rows = extract_features_offline_sample(symbols)
        else:
            if args.no_headlines:
                headlines = {s: "No material news." for s in symbols}
            else:
                headlines = fetch_headlines_map(symbols)
            rows = extract_features_live(
                symbols, headlines=headlines, market_context=market_context
            )
        path = write_features_parquet(rows, out)
        print(f"Wrote {len(rows)} feature row(s) → {path}")
        if args.slot == "morning" and not args.offline:
            if bust_meta_weights_cache():
                print("Busted meta_weights_day.json for next paper-live allocate")
            else:
                print("No meta_weights_day.json to bust")
    elif args.cmd == "paper":
        if args.mode == "ingest":
            from data.ingest.live import run_live_ingest
            from ops.market_hours import in_ingest_window

            if not in_ingest_window():
                print("Ingest: skipped (outside Mon–Fri 09:10–15:35 IST)")
                return
            result = run_live_ingest(duration_sec=args.duration)
            print(f"Ingest: {result.get('mode')} — spreads={result.get('spread_rows', 0)}")

            # Multi-strategy paper-live only on real Fyers ticks (never placeholders)
            if result.get("mode") == "fyers_websocket":
                from datetime import datetime, timedelta
                from zoneinfo import ZoneInfo

                from data.ingest.store import DataStore
                from experiments.paper_live import run_paper_live_tick
                from sim.friction.measured import Quote

                IST = ZoneInfo("Asia/Kolkata")
                since = datetime.now(tz=IST) - timedelta(days=5)
                with DataStore() as store:
                    store.init_schema()
                    bars = store.read_bars_1m(since=since)
                    spreads = store.read_latest_spreads(valid_only=True)

                quotes = result.get("quotes") or []
                # Always resolve: usable live quotes win; DuckDB valid spreads
                # fill gaps (TMPV/TRENT missed the last WS batch on Aug-11).
                if bars is not None and not bars.empty:
                    from experiments.paper_live import _resolve_quotes

                    syms = sorted(set(bars["symbol"].astype(str)))
                    quotes = list(_resolve_quotes(quotes, syms).values())
                elif not spreads.empty:
                    quotes = [
                        Quote(
                            symbol=str(r.symbol),
                            ltp=float(r.ltp),
                            bid=float(r.bid) if r.bid is not None else None,
                            ask=float(r.ask) if r.ask is not None else None,
                        )
                        for r in spreads.itertuples()
                    ]
                paper = run_paper_live_tick(
                    bars_1m=bars,
                    quotes=quotes,
                    mode=result["mode"],
                )
                print(
                    f"Paper-live: signals={paper.get('signals', 0)} "
                    f"fills={paper.get('fills', 0)} tfs={paper.get('closed_tfs')} "
                    f"errors={len(paper.get('errors') or [])}"
                )
        else:
            from experiments.paper import run_paper

            result = run_paper("manual", mode="sim")
            print(f"Paper day {result['date']}: {result['n_trades']} trade(s)")
    elif args.cmd == "forward":
        from experiments.forward import run_forward_validation

        manifest = run_forward_validation(days=args.days, start_date=args.start)
        print(
            f"Forward validation {manifest['start_date']}→{manifest['end_date']}: "
            f"health_ok={manifest['health_last_ok']}"
        )
        for day in manifest["days_detail"]:
            print(f"  {day['date']}: trades={day['n_trades']} health={'ok' if day['health_ok'] else 'FAIL'}")
    elif args.cmd == "eod":
        from experiments.eod import run_eod

        summary = run_eod(date=args.date)
        print(
            f"EOD {summary['date']}: trades={summary['n_trades']} "
            f"health_ok={summary['health_ok']} → {summary['path']}"
        )
        raise SystemExit(0 if summary["health_ok"] else 1)
    elif args.cmd == "health":
        from nse_trader.config import ROOT
        from ops.monitor.runner import format_report, run_health_checks, write_report_json

        report = run_health_checks()
        print(format_report(report))
        if args.json:
            out = ROOT / "data" / "logs" / "health.json"
            write_report_json(report, out)
            print(f"\nWrote {out}")
        raise SystemExit(0 if report.ok else 1)
    elif args.cmd == "diagnose":
        from ops.monitor.diagnose import format_diagnose, run_diagnose, write_diagnose_json

        payload = run_diagnose(day=args.date)
        print(format_diagnose(payload))
        if args.json:
            out = write_diagnose_json(payload)
            print(f"\nWrote {out}")
        raise SystemExit(0 if payload["ok"] else 1)
    elif args.cmd == "fyers-status":
        from data.ingest.fyers_auth import token_status

        st = token_status()
        for k, v in st.items():
            print(f"{k}: {v}")
        ready = st["has_app_id"] and st["has_secret"] and st["has_access_token"]
        raise SystemExit(0 if ready else 1)
    elif args.cmd == "fyers-login":
        from data.ingest.fyers_auth import auth_url, run_login_interactive

        if args.url_only:
            print(auth_url())
            return
        run_login_interactive(auth_code=args.auth_code)
    elif args.cmd == "fyers-refresh":
        from data.ingest.fyers_auth import refresh_access_token

        result = refresh_access_token()
        print(
            f"OK — {result.get('method', 'refresh')} access token "
            f"(len={result['access_token_len']}) → {result['env_file']}"
        )
    elif args.cmd == "test":
        import pytest

        extra = getattr(args, "pytest_args", None) or []
        raise SystemExit(pytest.main(["-v", "tests", *extra]))
    elif args.cmd == "demo":
        from experiments.demo import run_demo

        result = run_demo()
        print("Offline demo complete (synthetic — not investment evidence).")
        print(f"  enabled: {result['n_enabled']}/{result['n_implemented']} strategies")
        print(f"  L1 weight distance (LLM vs no-LLM): {result['l1_distance']:.4f}")
        print(f"  report: {result['report_json']}")
        print(f"  open:   {result['dashboard_html']}")
    elif args.cmd == "public-glance":
        from ops.public_glance import export_public_glance

        payload = export_public_glance()
        paths = payload.get("_paths") or {}
        print("Public glance exported (redacted — safe for GitHub Pages).")
        print(f"  {paths.get('index_html')}")
        print(f"  {paths.get('glance_json')}")
        print("Publish: ./deploy/publish-public-glance.sh")


if __name__ == "__main__":
    main()
